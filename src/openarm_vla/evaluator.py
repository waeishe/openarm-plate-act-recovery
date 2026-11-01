"""Privileged ground-truth evaluation; its values never enter policy observations."""

import numpy as np


class Evaluator:
    def __init__(self, backend, task):
        if task not in {"none", "cube_lift", "cube_in_tray"}:
            raise ValueError("unknown_evaluation_task")
        self.backend, self.task = backend, task
        if task != "none" and backend.initial_cube is None:
            raise ValueError("evaluation_requires_orange_cube")
        if task == "cube_in_tray" and backend.object_position("black_frame") is None:
            raise ValueError("evaluation_requires_black_frame")
        self.held_for = 0.0
        self.succeeded = False
        self.max_lift = 0.0
        self.latest = {}

    def update(self, dt):
        b = self.backend
        cube, tray = b.object_position("orange_cube"), b.object_position("black_frame")
        lifted = 0.0 if cube is None else float(cube[2] - b.initial_cube[2])
        self.max_lift = max(self.max_lift, lifted)
        stable = False
        contained = False
        finger_contact = False
        if cube is not None:
            body = b.model.body("orange_cube").id
            adr = b.model.jnt_dofadr[b.model.body_jntadr[body]]
            stable = bool(np.linalg.norm(b.data.qvel[adr : adr + 6]) < 0.05)
            for contact in b.data.contact:
                bodies = [int(b.model.geom_bodyid[g]) for g in contact.geom]
                if body in bodies:
                    finger_contact |= any(
                        "finger" in b.model.body(other).name for other in bodies if other != body
                    )
        released = bool(abs(b.position()[7]) > 0.5 and abs(b.position()[15]) > 0.5)
        if cube is not None and tray is not None:
            # Task-specific 4-cm cube / 12-cm inner tray; geometry contract is the checked demo scene.
            contained = bool(
                np.all(np.abs(cube[:2] - tray[:2]) < 0.038) and 0.025 < cube[2] - tray[2] < 0.045
            )
        condition = (
            (lifted > 0.05 and finger_contact)
            if self.task == "cube_lift"
            else contained and stable and released
        )
        self.held_for = self.held_for + dt if condition else 0.0
        # Placement requires the final stable condition; a past success alone is insufficient.
        if self.task == "cube_in_tray":
            self.succeeded = self.held_for >= 0.5
        elif self.task == "cube_lift":
            self.succeeded |= self.held_for >= 0.25
        self.latest = {
            "cube_position": cube.tolist() if cube is not None else None,
            "tray_position": tray.tolist() if tray is not None else None,
            "lift_m": lifted,
            "inside_tray": contained,
            "stable": stable,
            "finger_contact": finger_contact,
            "grippers_open": released,
            "condition_duration_s": self.held_for,
            "success": self.succeeded if self.task != "none" else None,
        }
        return self.latest

    def result(self):
        return {
            "task": self.task,
            "source": "privileged_simulator_ground_truth",
            "success": None if self.task == "none" else bool(self.succeeded),
            "max_lift_m": self.max_lift,
            "final": self.latest,
            "definition": {
                "cube_lift": "cube center > initial z + 0.05 m with finger contact for 0.25 s",
                "cube_in_tray": "final cube containment, low velocity and open grippers for 0.5 s",
                "none": "no task success predicate",
            }[self.task],
        }
