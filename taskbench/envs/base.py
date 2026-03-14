"""Base class for taskbench environments."""

from abc import ABCMeta, abstractmethod

import numpy as np
import sapien

from mani_skill.envs.sapien_env import BaseEnv


# Init configs for robots not supported by ManiSkill's TableSceneBuilder.
# Each entry: (rest_qpos, base_pose).
_CUSTOM_ROBOT_INIT = {
    "ur5_robotiq": dict(
        qpos=np.array([
            0, -np.pi / 2, np.pi / 2, -np.pi / 2, -np.pi / 2, 0,
            0, 0, 0, 0, 0, 0,
        ]),
        # UR5 internal frame has X+ backward; rotate base by π around Z
        base_pose=sapien.Pose(
            [-0.56, 0, 0],
            [0, 0, 0, 1],  # 180° around Z: q = (w=0, x=0, y=0, z=1)
        ),
    ),
    "ur5e_robotiq": dict(
        qpos=np.array([
            0, -np.pi / 2, np.pi / 2, -np.pi / 2, -np.pi / 2, 0,
            # 8 gripper joints (all zero = open)
            0, 0, 0, 0, 0, 0, 0, 0,
        ]),
        # Menagerie MJCF has 180° Z on the base body; apply another 180° Z
        # on the root to cancel it so the arm faces +X toward the table.
        base_pose=sapien.Pose(
            [-0.56, 0, 0],
            [0, 0, 0, 1],  # 180° around Z
        ),
    ),
}


class TaskEnv(BaseEnv, metaclass=ABCMeta):
    """Base class for all taskbench custom environments.

    Subclasses must implement ``get_objects()`` to expose their
    manipulable objects with canonical names.
    """

    @abstractmethod
    def get_objects(self) -> dict[str, object]:
        """Return a name→actor mapping for all manipulable objects."""
        ...

    def _after_table_scene_init(self, env_idx):
        """Apply init config for robots that TableSceneBuilder doesn't know about."""
        uid = self.robot_uids
        if uid not in _CUSTOM_ROBOT_INIT:
            return
        cfg = _CUSTOM_ROBOT_INIT[uid]
        b = len(env_idx)
        noise = getattr(self, "robot_init_qpos_noise", 0)
        qpos = cfg["qpos"].copy()
        if noise > 0:
            qpos = self._episode_rng.normal(0, noise, (b, len(qpos))) + qpos
        self.agent.reset(qpos)
        self.agent.robot.set_pose(cfg["base_pose"])
