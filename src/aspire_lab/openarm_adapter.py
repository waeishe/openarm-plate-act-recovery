"""The sole dependency boundary to the existing OpenArm project.

Physics state and the evaluator stay inside this adapter. Policy-visible object
positions come from RGB-D only. This is an API boundary, not a security sandbox.
"""

import math
import time
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image
from openarm_vla.backend import NativeBackend
from openarm_vla.common import atomic_json, jsonl
from openarm_vla.evaluator import Evaluator

from .vision import detect


def rotation(pitch=-0.5, yaw=0.0):
    cp, sp, cy, sy = math.cos(pitch), math.sin(pitch), math.cos(yaw), math.sin(yaw)
    return np.array([[cy * cp, -sy, cy * sp], [sy * cp, cy, sy * sp], [-sp, 0, cp]])


class OpenArmAdapter:
    def __init__(self, directory, seed, jitter, max_seconds=30, arm_speed=1, gripper_speed=2):
        self.directory = Path(directory)
        self.backend = NativeBackend("v2/cell/demo.xml", seed, jitter)
        self.evaluator = Evaluator(self.backend, "cube_in_tray")
        self.ik_data = mujoco.MjData(self.backend.model)
        self.renderer = mujoco.Renderer(self.backend.model, height=480, width=640)
        self.frame = self.observations = 0
        self.start = float(self.backend.data.time)
        self.deadline = time.monotonic() + 180
        self.max_seconds, self.arm_speed, self.gripper_speed = max_seconds, arm_speed, gripper_speed
        self.handles = {
            n: (self.directory / f"{n}.jsonl").open("w")
            for n in ("trajectory", "actions", "evaluation", "events")
        }
        atomic_json(self.directory / "initial-state.json", self.backend.snapshot())
        jsonl(self.handles["trajectory"], self.backend.sample(0))

    def event(self, kind, **data):
        jsonl(self.handles["events"], {"type": kind, "sim_time": float(self.backend.data.time), **data})

    def check(self):
        if (self.directory / "STOP").exists():
            raise InterruptedError("stop_requested")
        if time.monotonic() > self.deadline:
            raise TimeoutError("wall_time_limit")
        if self.backend.data.time - self.start >= self.max_seconds - 1e-6:
            raise TimeoutError("simulation_time_limit")

    def step(self, target, dt=1 / 30):
        self.check()
        b = self.backend
        applied = b.step(target, dt, self.arm_speed, self.gripper_speed)
        self.frame += 1
        jsonl(self.handles["trajectory"], b.sample(self.frame))
        jsonl(
            self.handles["actions"],
            {"frame": self.frame, "requested": np.asarray(target).tolist(), "applied": applied.tolist()},
        )
        jsonl(
            self.handles["evaluation"],
            {"frame": self.frame, "sim_time": float(b.data.time), **self.evaluator.update(dt)},
        )

    def wait(self, seconds=0.5):
        if not 0.03 <= seconds <= 10:
            raise ValueError("wait_seconds_out_of_bounds")
        for _ in range(math.ceil(seconds * 30)):
            self.step(self.backend.target.copy())

    def pose(self, arm="left"):
        if arm not in {"left", "right"}:
            raise ValueError("invalid_arm")
        b = self.backend
        site = b.model.site(f"{arm}_ee_control_point").id
        rot = b.data.site_xmat[site].reshape(3, 3)
        # Center of finger pads relative to the wrist site, from robot geometry.
        tip = b.data.site_xpos[site] + rot @ np.array([-0.012, 0, -0.12])
        return {"position": tip.tolist(), "rotation": rot.tolist(), "frame": "world"}

    def observe(self):
        self.check()
        b, r = self.backend, self.renderer
        camera = "camera_ceiling"
        r.disable_depth_rendering()
        r.update_scene(b.data, camera=camera, scene_option=b.option)
        rgb = r.render().copy()
        r.enable_depth_rendering()
        depth = r.render().copy()
        r.disable_depth_rendering()
        cam = b.model.camera(camera).id
        calibration = {
            "focal_px": 240 / math.tan(math.radians(b.model.cam_fovy[cam]) / 2),
            "rotation": b.data.cam_xmat[cam].tolist(),
            "position": b.data.cam_xpos[cam].tolist(),
        }
        found = detect(rgb, depth, calibration)
        name = f"obs-{self.observations:04d}"
        dest = self.directory / "observations" / name
        dest.mkdir(parents=True)
        Image.fromarray(rgb).save(dest / "rgb.png")
        np.save(dest / "depth.npy", depth, allow_pickle=False)
        obs = {
            "id": name,
            "sim_time": float(b.data.time),
            "coordinate_frame": "world",
            "calibration": calibration,
            "perception": found,
            "joints": b.position().tolist(),
            "left": self.pose("left"),
            "right": self.pose("right"),
        }
        atomic_json(dest / "observation.json", obs)
        self.observations += 1
        self.event("observation", observation=obs)
        return obs

    def solve(self, arm, position, rot):
        b, d, m = self.backend, self.ik_data, self.backend.model
        start = 8 if arm == "left" else 0
        indices = np.arange(start, start + 7)
        qa, dofs = b.qadr[indices], b.dadr[indices]
        d.qpos[:] = b.data.qpos
        site = m.site(f"{arm}_ee_control_point").id
        body = m.site_bodyid[site]
        jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        desired_q = np.zeros(4)
        mujoco.mju_mat2Quat(desired_q, rot.ravel())
        for _ in range(220):
            mujoco.mj_forward(m, d)
            current_rot = d.site_xmat[site].reshape(3, 3)
            tip = d.site_xpos[site] + current_rot @ np.array([-0.012, 0, -0.12])
            current_q = np.zeros(4)
            mujoco.mju_mat2Quat(current_q, current_rot.ravel())
            angular = np.zeros(3)
            # mju_subQuat returns a local rotation vector; transform to world.
            mujoco.mju_subQuat(angular, desired_q, current_q)
            angular = current_rot @ angular
            error = np.r_[np.asarray(position) - tip, 0.25 * angular]
            if np.linalg.norm(error[:3]) < 0.001 and np.linalg.norm(angular) < 0.025:
                result = b.target.copy()
                result[indices] = d.qpos[qa]
                return result
            mujoco.mj_jac(m, d, jp, jr, tip, body)
            j = np.vstack((jp[:, dofs], 0.25 * jr[:, dofs]))
            dq = j.T @ np.linalg.solve(j @ j.T + 0.002**2 * np.eye(6), error)
            dq *= min(1, 0.12 / max(np.max(np.abs(dq)), 1e-9))
            d.qpos[qa] = np.clip(d.qpos[qa] + dq, b.limits[indices, 0], b.limits[indices, 1])
        raise RuntimeError(f"ik_unreachable: position_error={np.linalg.norm(error[:3]):.4f} m")

    def move(self, arm, position, seconds=1.5, pitch=-0.5, yaw=0.0, orientation=None):
        if arm not in {"left", "right"} or not 0.1 <= seconds <= 10:
            raise ValueError("invalid_move")
        p = np.asarray(position, dtype=float)
        rot = rotation(pitch, yaw) if orientation is None else np.asarray(orientation, dtype=float)
        if (
            p.shape != (3,)
            or not np.isfinite(p).all()
            or rot.shape != (3, 3)
            or not np.isfinite(rot).all()
            or not np.allclose(rot.T @ rot, np.eye(3), atol=1e-5)
            or not np.isclose(np.linalg.det(rot), 1, atol=1e-5)
        ):
            raise ValueError("invalid_pose")
        target = self.solve(arm, p, rot)
        initial = self.backend.target.copy()
        steps = math.ceil(max(seconds, np.max(np.abs(target - initial)) / self.arm_speed) * 30)
        self.event("move", arm=arm, position=p.tolist(), seconds=steps / 30, rotation=rot.tolist())
        for i in range(steps):
            self.step(initial + (target - initial) * (i + 1) / steps)
        self.wait(0.3)
        actual = self.pose(arm)["position"]
        self.event("move_result", actual=actual, error_m=float(np.linalg.norm(np.asarray(actual) - p)))

    def grip(self, arm, closed, seconds=0.7):
        if arm not in {"left", "right"} or not isinstance(closed, bool) or not 0.03 <= seconds <= 10:
            raise ValueError("invalid_gripper")
        target = self.backend.target.copy()
        index = 15 if arm == "left" else 7
        target[index] = (-0.15 if closed else 0.7854) * (1 if arm == "left" else -1)
        self.event("grip", arm=arm, closed=closed)
        for _ in range(math.ceil(seconds * 30)):
            self.step(target)

    def finish(self):
        self.backend.hold()
        atomic_json(self.directory / "final-state.json", self.backend.snapshot())
        for camera in ("camera_ceiling", "camera_head_left"):
            rgb = self.backend.capture({"im": {"camera": camera, "height": 480, "width": 640}})["im"]
            Image.fromarray(rgb).save(self.directory / f"final-{camera}.png")
        return self.evaluator.result()

    def close(self):
        for handle in self.handles.values():
            handle.close()
        self.renderer.close()
        self.backend.close()
