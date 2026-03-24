"""State recorder for capturing simulation state and skill programs.

Saves demonstrations in HDF5 format with structured groups::

    episode.hdf5
    ├── metadata/          attrs: seed, env_id, solver, success,
    │                             control_freq, num_frames
    ├── robot/
    │   ├── qpos           (num_frames, 9)
    │   ├── tcp_pos        (num_frames, 3)
    │   ├── tcp_quat       (num_frames, 4)
    │   └── gripper_qpos   (num_frames, 2)
    ├── objects/
    │   ├── cube_0/
    │   │   ├── pos        (num_frames, 3)
    │   │   └── quat       (num_frames, 4)
    │   └── cube_1/
    │       ├── pos        (num_frames, 3)
    │       └── quat       (num_frames, 4)
    ├── skill              (num_frames,)  per-frame skill labels
    └── program/
        ├── skill          ["pick", "place", ...]  skill names
        ├── args           ['{"obj_name": "cube_1"}', ...]  JSON kwargs
        └── start_frame    [0, 142, ...]  frame index where each call started

To replay a recorded program::

    import h5py, json
    with h5py.File("data/episode.hdf5") as f:
        seed = int(f["metadata"].attrs["seed"])
        env_id = f["metadata"].attrs["env_id"]
        skills = list(f["program/skill"])
        args = [json.loads(a) for a in f["program/args"]]

    env = gym.make(env_id, ...)
    env.reset(seed=seed)
    ctx = SkillContext(env)
    ctx.reset(seed=seed)
    for skill_name, kwargs in zip(skills, args):
        getattr(ctx, skill_name)(**kwargs)

Usage::

    from taskbench.recorder import StateRecorder

    recorder = StateRecorder(
        env,
        objects={"cube_0": cube0, "cube_1": cube1},
        robot_fields=["qpos", "tcp_pos", "tcp_quat", "gripper_qpos"],
    )
    recorder.record()  # initial state

    recorder.record_skill_call("pick", {"obj_name": "cube_1", "lift_height": 0.13})
    ctx.pick("cube_1", lift_height=0.13)

    recorder.record_skill_call("place", {"target_pose": [...], "retract_height": 0.2})
    ctx.place(target_pose, retract_height=0.2)

    recorder.save("data/episode.hdf5", metadata={"seed": 42, "success": True})

Available robot_fields:
    - ``qpos``         — all joint positions (arm + fingers)
    - ``qvel``         — all joint velocities
    - ``qf``           — all joint generalized forces / motor effort
    - ``arm_force_limit`` — commanded arm drive force limit
    - ``joint_load_l2`` — aggregate norm of incoming joint forces / load proxy
    - ``tcp_pos``      — end-effector position (3,)
    - ``tcp_quat``     — end-effector orientation (4,)
    - ``gripper_qpos`` — finger joint positions (2,)
    - ``gripper_contact_force`` — estimated peak gripper contact force (N)
"""

import json
import logging
import os
from typing import Any, Dict, List, Optional

import h5py
import numpy as np

from taskbench.skills.motion import (
    get_arm_drive_settings,
    get_gripper_contact_summary,
    get_robot_contact_summary,
)

logger = logging.getLogger("taskbench.recorder")


def _extract_gripper_contact_force(raw):
    """Return the current peak gripper contact force estimate in Newtons."""
    return float(get_gripper_contact_summary(raw)["peak_force"])


def _extract_shelf_contact_force(raw):
    """Return the current peak robot-shelf contact force estimate in Newtons."""
    summary = get_robot_contact_summary(
        raw,
        entity_filter=lambda name: "shelf" in name.lower(),
    )
    return float(summary["peak_force"])


def _extract_joint_load_l2(raw):
    """Return an aggregate internal joint-load proxy for the current step."""
    incoming = raw.agent.robot.get_link_incoming_joint_forces().detach().cpu().numpy()
    return float(np.linalg.norm(incoming))


def _extract_arm_force_limit(raw):
    """Return the current commanded arm drive force limit."""
    settings = get_arm_drive_settings(raw)
    return 0.0 if settings is None else float(settings["force_limit"])


# Each extractor takes (raw_env,) and returns an array-like value.
_ROBOT_FIELD_EXTRACTORS = {
    "qpos": lambda raw: raw.agent.robot.get_qpos()[0].detach().cpu().numpy(),
    "qvel": lambda raw: raw.agent.robot.get_qvel()[0].detach().cpu().numpy(),
    "qf": lambda raw: raw.agent.robot.get_qf()[0].detach().cpu().numpy(),
    "arm_force_limit": _extract_arm_force_limit,
    "joint_load_l2": _extract_joint_load_l2,
    "tcp_pos": lambda raw: raw.agent.tcp.pose.p[0].detach().cpu().numpy(),
    "tcp_quat": lambda raw: raw.agent.tcp.pose.q[0].detach().cpu().numpy(),
    "gripper_qpos": lambda raw: raw.agent.robot.get_qpos()[0].detach().cpu().numpy()[-2:],
    "gripper_contact_force": _extract_gripper_contact_force,
    "shelf_contact_force": _extract_shelf_contact_force,
}


def _serialize_arg(value):
    """Convert a skill argument to a JSON-serializable form."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if hasattr(value, "p") and hasattr(value, "q"):
        # sapien.Pose
        p = np.asarray(value.p, dtype=np.float64).flatten().tolist()
        q = np.asarray(value.q, dtype=np.float64).flatten().tolist()
        return {"_type": "pose", "p": p, "q": q}
    if isinstance(value, (tuple, list)):
        return [_serialize_arg(v) for v in value]
    return value


class StateRecorder:
    """Records simulation state and skill programs at every control step.

    Args:
        env: Gym env (raw or wrapped — accesses ``env.unwrapped``).
        objects: Dict mapping name → SAPIEN actor to track poses of.
            If None, no object poses are recorded.
        robot_fields: List of robot state fields to record.
            See module docstring for available fields.
            If None, nothing about the robot is recorded.
    """

    def __init__(
        self,
        env,
        objects: Optional[Dict[str, object]] = None,
        robot_fields: Optional[List[str]] = None,
    ):
        self.env = env
        self.raw = env.unwrapped
        self.objects = objects or {}
        self.robot_fields = robot_fields or []
        self.frames = []
        self._skill = ""
        self._program_steps = []

        # Validate robot fields
        for field in self.robot_fields:
            if field not in _ROBOT_FIELD_EXTRACTORS:
                available = ", ".join(sorted(_ROBOT_FIELD_EXTRACTORS))
                raise ValueError(
                    f"Unknown robot field {field!r}. Available: {available}"
                )

    def set_skill(self, label):
        """Set the current skill label stamped on subsequent frames.

        Args:
            label: String like ``"pick(cube_1)"`` or ``"settle"``.
        """
        self._skill = label

    def record_skill_call(self, skill_name, kwargs=None):
        """Record a skill call in the program trace.

        Call this right before executing the skill. The current frame
        count is stored as the start_frame for this step.

        Args:
            skill_name: Name of the skill (e.g. "pick", "place", "push").
            kwargs: Dict of keyword arguments passed to the skill call.
                Values are serialized to JSON (numpy arrays and sapien.Pose
                are converted automatically).
        """
        serialized = {}
        if kwargs:
            for k, v in kwargs.items():
                serialized[k] = _serialize_arg(v)

        self._program_steps.append({
            "skill": skill_name,
            "args": serialized,
            "start_frame": len(self.frames),
        })
        # Also update the per-frame skill label
        self.set_skill(skill_name)

    def record(self):
        """Capture one frame of state. Call after every env.step()."""
        raw = self.raw
        frame = {"skill": self._skill}

        # Robot fields
        for field in self.robot_fields:
            frame[field] = _ROBOT_FIELD_EXTRACTORS[field](raw)

        # Tracked object poses
        for name, actor in self.objects.items():
            frame[f"{name}_pos"] = actor.pose.p[0].detach().cpu().numpy()
            frame[f"{name}_quat"] = actor.pose.q[0].detach().cpu().numpy()

        self.frames.append(frame)

    def save(self, path, metadata: Optional[Dict[str, Any]] = None,
             hydra_cfg=None):
        """Save all recorded frames and the skill program to an HDF5 file.

        Args:
            path: Output file path (should end in .hdf5).
            metadata: Optional dict of episode metadata to store as
                attributes on the metadata group. Common keys:
                seed, env_id, solver, success, failure_reason.
            hydra_cfg: Optional resolved Hydra DictConfig. Stored as a
                YAML string so the demo is fully self-contained and
                reproducible (like saving config.yaml with checkpoints).
        """
        if not self.frames:
            logger.warning("No frames recorded, skipping save")
            return

        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        num_frames = len(self.frames)

        with h5py.File(path, "w") as f:
            # Metadata
            meta = f.create_group("metadata")
            meta.attrs["num_frames"] = num_frames
            meta.attrs["control_freq"] = self.raw.control_freq
            meta.attrs["env_id"] = self.raw.spec.id if self.raw.spec else ""
            if metadata:
                for key, value in metadata.items():
                    meta.attrs[key] = value
            if hydra_cfg is not None:
                from omegaconf import OmegaConf
                meta.attrs["hydra_config"] = OmegaConf.to_yaml(hydra_cfg)

            # Robot state
            if self.robot_fields:
                robot_grp = f.create_group("robot")
                for field in self.robot_fields:
                    data = np.stack([frame[field] for frame in self.frames])
                    robot_grp.create_dataset(field, data=data, compression="gzip")

            # Object poses
            if self.objects:
                obj_grp = f.create_group("objects")
                for name in self.objects:
                    name_grp = obj_grp.create_group(name)
                    pos = np.stack([frame[f"{name}_pos"] for frame in self.frames])
                    quat = np.stack([frame[f"{name}_quat"] for frame in self.frames])
                    name_grp.create_dataset("pos", data=pos, compression="gzip")
                    name_grp.create_dataset("quat", data=quat, compression="gzip")

            # Per-frame skill labels
            skills = [frame["skill"] for frame in self.frames]
            dt = h5py.string_dtype()
            f.create_dataset("skill", data=skills, dtype=dt)

            # Skill program
            if self._program_steps:
                prog_grp = f.create_group("program")
                prog_grp.create_dataset(
                    "skill",
                    data=[s["skill"] for s in self._program_steps],
                    dtype=dt,
                )
                prog_grp.create_dataset(
                    "args",
                    data=[json.dumps(s["args"]) for s in self._program_steps],
                    dtype=dt,
                )
                prog_grp.create_dataset(
                    "start_frame",
                    data=[s["start_frame"] for s in self._program_steps],
                )

        logger.info("Saved %d frames, %d skill calls to %s",
                     num_frames, len(self._program_steps), path)

    def clear(self):
        """Discard all recorded frames and program steps."""
        self.frames = []
        self._program_steps = []
