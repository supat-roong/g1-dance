"""Single owner of the G1 model's joint ordering and index maps.

Everything that needs to know which slot a joint occupies asks here, rather
than hardcoding a list. A mismatch between the retargeter's ordering and the
environment's produces a plausible looking but wrong dance, which is both the
most expensive and the least visible bug available in this project.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path

import mujoco
import numpy as np

ASSETS = Path(__file__).resolve().parents[3] / "assets" / "g1"
SCENE = ASSETS / "scene.xml"
CONFIG = ASSETS / "model_config.json"

LEG_JOINTS = [
    f"{s}_{j}_joint"
    for s in ("left", "right")
    for j in ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll")
]
WAIST_JOINTS = ["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"]
ARM_JOINTS = [
    f"{s}_{j}_joint"
    for s in ("left", "right")
    for j in ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow",
              "wrist_roll", "wrist_pitch", "wrist_yaw")
]
#: the 29 actuated body DoF, in the order used everywhere downstream
BODY_JOINTS = LEG_JOINTS + WAIST_JOINTS + ARM_JOINTS
#: the 17 DoF tracked in mode A (upper body); legs balance instead
UPPER_JOINTS = WAIST_JOINTS + ARM_JOINTS

#: the 12 independently actuated Inspire DoF, 6 per hand. Names are
#: dex-retargeting's Inspire target_joint_names under an L_/R_ prefix.
FINGER_ORDER = ["pinky_proximal", "ring_proximal", "middle_proximal",
                "index_proximal", "thumb_proximal_pitch", "thumb_proximal_yaw"]
HAND_JOINTS = [f"{s}_{j}_joint" for s in ("L", "R") for j in FINGER_ORDER]

#: bodies whose pelvis-frame positions feed the link tracking reward
KEY_BODIES = ["left_wrist_yaw_link", "right_wrist_yaw_link",
              "left_elbow_link", "right_elbow_link",
              "left_ankle_roll_link", "right_ankle_roll_link", "torso_link"]

TRACK_SETS = {"upper_body": UPPER_JOINTS, "whole_body": BODY_JOINTS}


@dataclass
class G1Model:
    """Loads the model once and exposes every index map derived from it."""

    scene: Path = SCENE
    _model: mujoco.MjModel = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self._model is None:
            self._model = mujoco.MjModel.from_xml_path(str(self.scene))
        missing = [j for j in BODY_JOINTS + HAND_JOINTS if self._jid(j) < 0]
        if missing:
            raise ValueError(f"model is missing expected joints: {missing}")

    @property
    def model(self) -> mujoco.MjModel:
        return self._model

    def data(self) -> mujoco.MjData:
        return mujoco.MjData(self._model)

    def _jid(self, name: str) -> int:
        return mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_JOINT, name)

    def _aid(self, name: str) -> int:
        return mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)

    def body_id(self, name: str) -> int:
        return mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, name)

    def qpos_adr(self, names: list[str]) -> np.ndarray:
        return np.array([self._model.jnt_qposadr[self._jid(n)] for n in names], int)

    def dof_adr(self, names: list[str]) -> np.ndarray:
        return np.array([self._model.jnt_dofadr[self._jid(n)] for n in names], int)

    def act_id(self, names: list[str]) -> np.ndarray:
        return np.array([self._aid(n) for n in names], int)

    def joint_range(self, names: list[str]) -> np.ndarray:
        """(n, 2) lower/upper limits, in the given order."""
        return np.array([self._model.jnt_range[self._jid(n)] for n in names], float)

    # index maps used on every control step -------------------------------
    @cached_property
    def body_qpos(self) -> np.ndarray:
        return self.qpos_adr(BODY_JOINTS)

    @cached_property
    def body_dof(self) -> np.ndarray:
        return self.dof_adr(BODY_JOINTS)

    @cached_property
    def body_act(self) -> np.ndarray:
        return self.act_id(BODY_JOINTS)

    @cached_property
    def hand_qpos(self) -> np.ndarray:
        return self.qpos_adr(HAND_JOINTS)

    @cached_property
    def hand_act(self) -> np.ndarray:
        return self.act_id(HAND_JOINTS)

    @cached_property
    def key_body_ids(self) -> np.ndarray:
        return np.array([self.body_id(b) for b in KEY_BODIES], int)

    @cached_property
    def pelvis_id(self) -> int:
        return self.body_id("pelvis")

    @cached_property
    def config(self) -> dict:
        return json.loads(CONFIG.read_text())

    @cached_property
    def default_qpos(self) -> np.ndarray:
        """Default stance over BODY_JOINTS."""
        d = self.config["default_joint_pos"]
        return np.array([d.get(j, 0.0) for j in BODY_JOINTS], float)

    @cached_property
    def action_scales(self) -> np.ndarray:
        d = self.config["action_scales"]
        return np.array([d.get(j, 0.25) for j in BODY_JOINTS], float)

    def tracked(self, mode: str) -> list[str]:
        if mode not in TRACK_SETS:
            raise ValueError(f"mode must be one of {sorted(TRACK_SETS)}, got {mode!r}")
        return TRACK_SETS[mode]

    def tracked_index(self, mode: str) -> np.ndarray:
        """Positions of the tracked joints within BODY_JOINTS."""
        pos = {j: i for i, j in enumerate(BODY_JOINTS)}
        return np.array([pos[j] for j in self.tracked(mode)], int)

    def hash(self) -> str:
        """Identity of the compiled model, stamped into artifacts."""
        import hashlib
        h = hashlib.sha256()
        h.update(np.asarray(self._model.jnt_range).tobytes())
        h.update(str(self._model.nq).encode())
        h.update(",".join(BODY_JOINTS + HAND_JOINTS).encode())
        return h.hexdigest()[:16]

    def reset(self, d: mujoco.MjData, height: float = 0.793) -> None:
        """Default stance at a given pelvis height."""
        mujoco.mj_resetData(self._model, d)
        d.qpos[:3] = [0.0, 0.0, height]
        d.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
        d.qpos[self.body_qpos] = self.default_qpos
        d.ctrl[self.body_act] = self.default_qpos
        mujoco.mj_forward(self._model, d)
