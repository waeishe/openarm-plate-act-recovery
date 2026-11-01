"""Versioned, bounded contracts shared by simulation and policy processes."""

import hashlib
import io
import json
import os
import subprocess
from pathlib import Path
from urllib.parse import urlparse

import numpy as np

PROJECT = Path(__file__).resolve().parents[2]
ASSETS = PROJECT / "assets/openarm_mujoco"
JOINT_NAMES = [
    f"openarm_{s}_{j}"
    for s in ("right", "left")
    for j in [*(f"joint{i}" for i in range(1, 8)), "finger_joint1"]
]
CAMERA_MAP = {
    f"observation.images.{n}": f"camera_{n}"
    for n in ("ceiling", "head_left", "head_right", "wrist_left", "wrist_right")
}
PROTOCOL = "openarm-policy/1"
MAX_BODY = 64 * 1024 * 1024


def command_speed_limits(arm_max_speed, gripper_max_speed):
    """Command slew limits in canonical joint order, not measured joint velocities."""
    for value in (arm_max_speed, gripper_max_speed):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0.01 <= value <= 2:
            raise ValueError("invalid_command_speed_limits")
    return np.array(
        [gripper_max_speed if name.endswith("finger_joint1") else arm_max_speed for name in JOINT_NAMES],
        dtype=np.float64,
    )


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2))
    os.replace(tmp, path)


def jsonl(file, value):
    file.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")
    file.flush()


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def tree_digest(root):
    h = hashlib.sha256()
    for file in sorted(Path(root).rglob("*")):
        if file.is_file() and file.suffix in {".xml", ".obj", ".stl", ".mtl", ".png", ".jpg"}:
            h.update(str(file.relative_to(root)).encode())
            h.update(digest(file).encode())
    return h.hexdigest()


def source_revision():
    def git(*args):
        return subprocess.check_output(["git", "-C", str(PROJECT), *args], text=True).strip()

    try:
        return {"git_commit": git("rev-parse", "HEAD"), "git_dirty": bool(git("status", "--porcelain"))}
    except (OSError, subprocess.CalledProcessError):
        return {"git_commit": None, "git_dirty": None}


def local_url(url):
    parsed = urlparse(url)
    if (
        parsed.scheme != "http"
        or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("Use a loopback HTTP URL; tunnel remote GPU services with SSH")
    return url.rstrip("/")


def pack_observation(meta, state, images):
    f = io.BytesIO()
    np.savez_compressed(
        f,
        metadata=np.frombuffer(json.dumps(meta).encode(), dtype=np.uint8),
        state=np.asarray(state, dtype=np.float32),
        **images,
    )
    return f.getvalue()


def unpack_observation(body):
    if len(body) > MAX_BODY:
        raise ValueError("observation_too_large")
    # Bound the expanded archive too; do not unpickle arrays supplied by a client.
    import zipfile

    with zipfile.ZipFile(io.BytesIO(body)) as z:
        if sum(i.file_size for i in z.infolist()) > MAX_BODY or len(z.infolist()) > 10:
            raise ValueError("expanded_observation_too_large")
    with np.load(io.BytesIO(body), allow_pickle=False) as data:
        meta = json.loads(data["metadata"].tobytes())
        state = data["state"].copy()
        images = {k: data[k].copy() for k in data.files if k not in {"state", "metadata"}}
    if meta.get("protocol") != PROTOCOL or state.shape != (16,) or not np.isfinite(state).all():
        raise ValueError("invalid_observation_contract")
    if any(im.dtype != np.uint8 or im.ndim != 3 or im.shape[2] != 3 for im in images.values()):
        raise ValueError("images_must_be_RGB_uint8")
    return meta, state, images


def validate_chunk(response, *, executable=True):
    if response.get("protocol") != PROTOCOL:
        raise ValueError("policy_protocol_mismatch")
    a = np.asarray(response.get("positions"), dtype=np.float64)
    if a.ndim != 2 or not 1 <= a.shape[0] <= 1000 or not 1 <= a.shape[1] <= 64 or not np.isfinite(a).all():
        raise ValueError("invalid_policy_actions")
    if executable and (
        a.shape[1] != 16
        or response.get("joint_names") != JOINT_NAMES
        or response.get("action_space") != "joint_position_rad"
    ):
        raise ValueError("incompatible_robot_action_space")
    dt = response.get("interval_s", 0)
    if not isinstance(dt, (float, int)) or not 0.005 <= dt <= 0.2:
        raise ValueError("invalid_action_interval")
    return a, float(dt)
