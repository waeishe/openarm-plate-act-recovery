"""Single-owner native MuJoCo backend; rendering and physics stay on its main thread."""

import time

import mujoco
import numpy as np

from .common import ASSETS, JOINT_NAMES, command_speed_limits, tree_digest


class NativeBackend:
    def __init__(self, scene="v2/cell/demo.xml", seed=0, jitter=0.0):
        file = (ASSETS / scene).resolve()
        if not file.is_relative_to(ASSETS.resolve()) or file.suffix != ".xml":
            raise ValueError("invalid_scene")
        self.scene = scene
        self.model = m = mujoco.MjModel.from_xml_path(str(file))
        self.data = d = mujoco.MjData(m)
        self.renderers = {}
        ids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name) for name in JOINT_NAMES]
        if min(ids) < 0:
            raise ValueError("scene_missing_canonical_OpenArm_joints")
        self.joint_ids = np.array(ids)
        self.qadr, self.dadr = m.jnt_qposadr[ids], m.jnt_dofadr[ids]
        acts = []
        for j in ids:
            matches = np.where(m.actuator_trnid[:, 0] == j)[0]
            if len(matches) != 1:
                raise ValueError("ambiguous_position_actuator")
            acts.append(matches[0])
        self.act_ids = np.array(acts)
        self.limits = m.actuator_ctrlrange[self.act_ids].copy()
        # Fingers have intentional overtravel in the supplied position actuator range.
        for i in [j for j in range(16) if j not in (7, 15)]:
            self.limits[i, 0] = max(self.limits[i, 0], m.jnt_range[ids[i], 0])
            self.limits[i, 1] = min(self.limits[i, 1], m.jnt_range[ids[i], 1])
        self.bias_dofs = [int(self.dadr[i]) for i in range(16) if i not in (7, 15)]
        lifter = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "openarm_lifter_joint")
        if lifter >= 0:
            self.bias_dofs.append(int(m.jnt_dofadr[lifter]))
        key = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "home")
        if key >= 0:
            mujoco.mj_resetDataKeyframe(m, d, key)
        for i in range(m.nu):
            j = m.actuator_trnid[i, 0]
            if j >= 0:
                d.ctrl[i] = d.qpos[m.jnt_qposadr[j]]
        cube = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "orange_cube")
        if jitter:
            if not 0 <= jitter <= 0.03 or cube < 0:
                raise ValueError("cube_jitter_requires_demo_scene_and_0_to_0.03_m")
            adr = m.jnt_qposadr[m.body_jntadr[cube]]
            d.qpos[adr : adr + 2] += np.random.default_rng(seed).uniform(-jitter, jitter, 2)
        self.target = d.ctrl[self.act_ids].copy()
        self.target_time = float(d.time)
        # Native cameras use the original MJCF lighting and visible geometry groups.
        m.vis.global_.offwidth = max(1280, m.vis.global_.offwidth)
        m.vis.global_.offheight = max(960, m.vis.global_.offheight)
        self.option = mujoco.MjvOption()
        self.option.geomgroup[3:] = 0
        mujoco.mj_forward(m, d)
        self.step(self.target, 0.3)  # let free objects settle before the initial observation
        self.initial_cube = self.object_position("orange_cube")
        self.asset_hash = tree_digest(ASSETS / scene.split("/")[0])

    def position(self):
        return self.data.qpos[self.qadr].copy()

    def hold(self):
        self.target = np.clip(self.position(), self.limits[:, 0], self.limits[:, 1])
        self.data.ctrl[self.act_ids] = self.target

    def step(self, requested, dt, max_speed=1.0, gripper_max_speed=None):
        # Preserve the scalar API for older callers. New runners pass both limits.
        speeds = command_speed_limits(
            max_speed, max_speed if gripper_max_speed is None else gripper_max_speed
        )
        target = np.asarray(requested, dtype=np.float64)
        if target.shape != (16,) or not np.isfinite(target).all():
            raise ValueError("invalid_action")
        limited = np.clip(target, self.limits[:, 0], self.limits[:, 1])
        applied = self.target + np.clip(limited - self.target, -speeds * dt, speeds * dt)
        self.data.ctrl[self.act_ids] = applied
        self.target = applied.copy()
        self.target_time += dt
        while self.data.time + 1e-10 < self.target_time:
            self.data.qfrc_applied[self.bias_dofs] = self.data.qfrc_bias[self.bias_dofs]
            mujoco.mj_step(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        if not np.isfinite(self.data.qpos).all() or not np.isfinite(self.data.qvel).all():
            raise RuntimeError("nonfinite_physics")
        for warning in self.data.warning:
            if warning.number:
                raise RuntimeError("mujoco_warning")
        return applied

    def capture(self, cameras):
        images = {}
        for key, spec in cameras.items():
            h, w = spec["height"], spec["width"]
            if not 32 <= h <= 960 or not 32 <= w <= 1280:
                raise ValueError("camera_resolution_out_of_bounds")
            shape = (h, w)
            if shape not in self.renderers:
                self.renderers[shape] = mujoco.Renderer(self.model, height=h, width=w)
            renderer = self.renderers[shape]
            renderer.update_scene(self.data, camera=spec["camera"], scene_option=self.option)
            images[key] = renderer.render().copy()
        return images

    def sample(self, frame):
        d = self.data
        return {
            "frame": frame,
            "sim_time": float(d.time),
            "wall_time_ns": time.time_ns(),
            "qpos": d.qpos.tolist(),
            "qvel": d.qvel.tolist(),
            "ctrl": d.ctrl.tolist(),
            "position": self.position().tolist(),
        }

    def snapshot(self):
        spec = mujoco.mjtState.mjSTATE_INTEGRATION
        state = np.empty(mujoco.mj_stateSize(self.model, spec))
        mujoco.mj_getState(self.model, self.data, state, spec)
        return {
            "format": 2,
            "backend": "native",
            "scene": self.scene,
            "mujoco_version": mujoco.__version__,
            "spec": int(spec),
            "integration": state.tolist(),
            "joint_names": [self.model.joint(i).name for i in range(self.model.njnt)],
            "joint_layout": [
                {
                    "type": int(self.model.jnt_type[i]),
                    "qpos": int(self.model.jnt_qposadr[i]),
                    "dof": int(self.model.jnt_dofadr[i]),
                    "body": self.model.body(int(self.model.jnt_bodyid[i])).name,
                }
                for i in range(self.model.njnt)
            ],
            **self.sample(0),
        }

    def object_position(self, name):
        idx = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
        return None if idx < 0 else self.data.xpos[idx].copy()

    def close(self):
        for renderer in self.renderers.values():
            renderer.close()
