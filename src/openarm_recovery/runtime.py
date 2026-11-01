"""Snapshot and logging helpers used by the recovery controller."""
import json
from datetime import datetime,timezone
from pathlib import Path
import mujoco
import numpy as np
from openarm_vla.common import atomic_json,digest
def now():
    return datetime.now(timezone.utc).isoformat()

def read(path):
    return json.loads(Path(path).read_text())

def restore(backend, snapshot, assets_hash):
    """Restore all integration variables without advancing simulation time."""
    expected = backend.snapshot()
    for key in ("format", "backend", "scene", "mujoco_version", "spec", "joint_names", "joint_layout"):
        if snapshot.get(key) != expected[key]:
            raise ValueError(f"incompatible_snapshot:{key}")
    if assets_hash != backend.asset_hash:
        raise ValueError("model_assets_hash_mismatch")
    state = np.asarray(snapshot["integration"], dtype=float)
    if state.shape != (mujoco.mj_stateSize(backend.model, snapshot["spec"]),) or not np.isfinite(state).all():
        raise ValueError("invalid_integration_state")
    mujoco.mj_setState(backend.model, backend.data, state, snapshot["spec"])
    # Refresh derived kinematics; preserve warm-start/applied-force integration fields.
    mujoco.mj_forward(backend.model, backend.data)
    mujoco.mj_setState(backend.model, backend.data, state, snapshot["spec"])
    backend.target = backend.data.ctrl[backend.act_ids].copy()
    backend.target_time = float(backend.data.time)
    actual = backend.snapshot()
    for key in ("integration", "qpos", "qvel", "ctrl", "sim_time"):
        if actual[key] != snapshot[key]:
            raise ValueError(f"restore_not_exact:{key}")
    return {"exact": True, "fields": ["integration", "qpos", "qvel", "ctrl", "sim_time"], "sim_time": actual["sim_time"]}
