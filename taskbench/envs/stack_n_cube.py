"""Parameterized N-cube stacking environment for ManiSkill3.

Spawns N cubes (2-6) on the table with collision-free random placement.
Success requires all N cubes stacked in a single ordered tower.
"""

from typing import Union

import numpy as np
import sapien
import torch
from mani_skill.agents.robots import Fetch, Panda
from taskbench.envs.base import TaskEnv
from mani_skill.envs.utils import randomization
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import common
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.structs.types import SceneConfig, SimConfig

CUBE_COLORS = [
    [0, 1, 0, 1],  # green (cube_0 is always the base)
    [1, 0, 0, 1],  # red
    [0, 0, 1, 1],  # blue
    [1, 1, 0, 1],  # yellow
    [1, 0, 1, 1],  # magenta
    [0, 1, 1, 1],  # cyan
]


@register_env("StackNCube-v1", max_episode_steps=250)
class StackNCubeEnv(TaskEnv):
    """Parameterized N-cube stacking environment.

    Creates N cubes on a table. Success requires stacking all cubes in
    a single ordered tower: cube_0 on table, cube_1 on cube_0, etc.

    Args:
        num_cubes: Number of cubes to spawn (2-6). Default: 3.
        robot_uids: Robot to use. Default: "panda_wristcam".
        robot_init_qpos_noise: Noise added to robot initial joint positions.
    """

    SUPPORTED_ROBOTS = ["panda_wristcam", "panda", "fetch", "ur5_robotiq", "ur5e_robotiq"]
    SUPPORTED_REWARD_MODES = ["sparse", "none"]
    agent: Union[Panda, Fetch]

    def __init__(
        self,
        *args,
        robot_uids="panda_wristcam",
        robot_init_qpos_noise=0.02,
        num_cubes: int = 3,
        **kwargs,
    ):
        self.num_cubes = num_cubes
        self.robot_init_qpos_noise = robot_init_qpos_noise
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sim_config(self):
        return SimConfig(
            scene_config=SceneConfig(
                solver_position_iterations=20,
                solver_velocity_iterations=4,
            )
        )

    @property
    def _default_sensor_configs(self):
        """Camera configurations for rendering and observation."""
        pose = sapien.Pose(p=[0, 0, 0], q=[1, 0, 0, 0])
        return [CameraConfig("base_camera", pose, 128, 128, np.pi / 2, 0.01, 100)]

    @property
    def _default_human_render_camera_configs(self):
        """Camera configuration for human-readable rendering."""
        pose = sapien.Pose(p=[-0.4, 0, 0.4], q=[0.9238795, 0, 0.3826834, 0])
        return CameraConfig("render_camera", pose, 512, 512, 1, 0.01, 100)

    def _load_scene(self, options: dict):
        self.cube_half_size = common.to_tensor([0.02] * 3, device=self.device)
        self.table_scene = TableSceneBuilder(
            env=self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()

        self.cubes = []
        for i in range(self.num_cubes):
            cube = actors.build_cube(
                self.scene,
                half_size=0.02,
                color=CUBE_COLORS[i % len(CUBE_COLORS)],
                name=f"cube_{i}",
                initial_pose=sapien.Pose(p=[i * 0.1, 0, 0.1]),
            )
            self.cubes.append(cube)

    def get_objects(self) -> dict[str, object]:
        """Return a name→actor mapping for all manipulable objects."""
        return {f"cube_{i}": cube for i, cube in enumerate(self.cubes)}

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)
            self._after_table_scene_init(env_idx)

            xyz = torch.zeros((b, 3))
            xyz[:, 2] = 0.02  # half cube height above table

            region = [[-0.1, -0.2], [0.1, 0.2]]
            sampler = randomization.UniformPlacementSampler(
                bounds=region, batch_size=b, device=self.device
            )
            radius = torch.linalg.norm(torch.tensor([0.02, 0.02])) + 0.001

            for i, cube in enumerate(self.cubes):
                cube_xy = sampler.sample(radius, 100, verbose=False)
                xyz[:, :2] = cube_xy
                qs = randomization.random_quaternions(
                    b, lock_x=True, lock_y=True, lock_z=False
                )
                if i < self.num_cubes - 1:
                    cube.set_pose(Pose.create_from_pq(p=xyz.clone(), q=qs))
                else:
                    cube.set_pose(Pose.create_from_pq(p=xyz, q=qs))

    def evaluate(self):
        # Sort cubes by Z height — any ordering is valid as long as
        # cube_0 is the base and all cubes form a single vertical tower.
        all_pos = torch.stack([cube.pose.p for cube in self.cubes], dim=1)  # (B, N, 3)
        zs = all_pos[:, :, 2]
        order = torch.argsort(zs, dim=1)
        sorted_pos = torch.gather(all_pos, 1, order.unsqueeze(-1).expand(-1, -1, 3))

        half_size = self.cube_half_size
        xy_thresh = torch.linalg.norm(half_size[:2]) + 0.005
        z_target = half_size[2] * 2  # full cube height

        all_pairs_stacked = torch.ones(
            self.num_envs, device=self.device, dtype=torch.bool
        )
        all_static = torch.ones(self.num_envs, device=self.device, dtype=torch.bool)
        any_grasped = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        base_is_cube_0 = order[:, 0] == 0

        for k in range(self.num_cubes):
            all_static &= self.cubes[k].is_static(lin_thresh=1e-2, ang_thresh=0.5)
            any_grasped |= self.agent.is_grasping(self.cubes[k])

        for k in range(1, self.num_cubes):
            offset = sorted_pos[:, k] - sorted_pos[:, k - 1]
            xy_ok = torch.linalg.norm(offset[..., :2], dim=1) <= xy_thresh
            z_ok = torch.abs(offset[..., 2] - z_target) <= 0.005
            all_pairs_stacked &= xy_ok & z_ok

        success = base_is_cube_0 & all_pairs_stacked & all_static & (~any_grasped)
        return {
            "base_is_cube_0": base_is_cube_0,
            "all_pairs_stacked": all_pairs_stacked,
            "all_static": all_static,
            "any_grasped": any_grasped,
            "success": success.bool(),
        }

    def _get_obs_extra(self, info: dict):
        obs = dict(tcp_pose=self.agent.tcp.pose.raw_pose)
        if "state" in self.obs_mode:
            for i, cube in enumerate(self.cubes):
                obs[f"cube_{i}_pose"] = cube.pose.raw_pose
                obs[f"tcp_to_cube_{i}_pos"] = cube.pose.p - self.agent.tcp.pose.p
            # Pairwise distances between adjacent cubes
            for i in range(1, self.num_cubes):
                obs[f"cube_{i-1}_to_cube_{i}_pos"] = (
                    self.cubes[i].pose.p - self.cubes[i - 1].pose.p
                )
        return obs

    def compute_sparse_reward(self, obs, action, info):
        return info["success"].float()
