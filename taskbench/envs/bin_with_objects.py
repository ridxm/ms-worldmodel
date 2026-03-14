"""Bin-with-objects environment for ManiSkill3.

A bin sits on the table filled with a mix of colorful primitive shapes
(cubes, spheres, cylinders, boxes in small/medium/large sizes) and YCB
mesh objects, gravity-settled each reset.  CPU-only, single-env.
"""

from typing import Any, Union

import numpy as np
import sapien
import sapien.render
import torch

from mani_skill.agents.robots import Panda
from taskbench.envs.base import TaskEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.types import SceneConfig, SimConfig

# Distinct colors for primitive objects (RGBA)
OBJECT_COLORS = [
    [0.85, 0.15, 0.15, 1.0],  # red
    [0.15, 0.65, 0.15, 1.0],  # green
    [0.15, 0.35, 0.85, 1.0],  # blue
    [0.95, 0.75, 0.10, 1.0],  # yellow
    [0.85, 0.45, 0.10, 1.0],  # orange
    [0.65, 0.15, 0.75, 1.0],  # purple
    [0.10, 0.80, 0.80, 1.0],  # cyan
    [0.90, 0.40, 0.60, 1.0],  # pink
    [0.55, 0.35, 0.15, 1.0],  # brown
    [0.40, 0.75, 0.40, 1.0],  # light green
]

# Size presets: (min_half_size, max_half_size) for small / medium / large
SIZE_PRESETS = {
    "small": (0.012, 0.02),
    "medium": (0.025, 0.035),
    "large": (0.04, 0.055),
}

# Simple graspable YCB objects
YCB_MODEL_IDS = [
    "005_tomato_soup_can",
    "006_mustard_bottle",
    "007_tuna_fish_can",
    "008_pudding_box",
    "009_gelatin_box",
    "010_potted_meat_can",
    "013_apple",
    "014_lemon",
    "017_orange",
    "024_bowl",
    "025_mug",
    "036_wood_block",
    "056_tennis_ball",
    "061_foam_brick",
    "077_rubiks_cube",
]


@register_env("BinWithObjects-v1", max_episode_steps=200, asset_download_ids=["ycb"])
class BinWithObjectsEnv(TaskEnv):
    """Bin on table filled with a mix of colorful primitives and YCB objects.

    Each reset: new random objects are created, dropped into the bin one by
    one with gravity settling, then the episode begins.  CPU-only, single-env.

    Args:
        num_objects: Number of objects to spawn (default: 6).
        robot_uids: Robot to use (default: "panda").
    """

    SUPPORTED_ROBOTS = ["panda", "ur5_robotiq", "ur5e_robotiq"]
    SUPPORTED_REWARD_MODES = ["none"]
    agent: Union[Panda]

    # Bin dimensions (interior half-sizes)
    BIN_BX = 0.20  # 40cm x-extent
    BIN_BY = 0.20  # 40cm y-extent
    BIN_BZ = 0.15  # 30cm height
    BIN_WALL_T = 0.02  # wall/floor thickness
    BIN_CENTER = [0.25, 0.0]  # (x, y) on table, in front of robot

    def __init__(
        self,
        *args,
        robot_uids="panda",
        robot_init_qpos_noise=0.02,
        num_objects: int = 30,
        **kwargs,
    ):
        self.num_objects = num_objects
        self.robot_init_qpos_noise = robot_init_qpos_noise
        super().__init__(
            *args,
            robot_uids=robot_uids,
            reconfiguration_freq=1,
            **kwargs,
        )

    @property
    def _default_sim_config(self):
        return SimConfig(
            scene_config=SceneConfig(
                solver_position_iterations=40,
                solver_velocity_iterations=10,
            )
        )

    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at(eye=[0.5, 0, 0.6], target=[0.25, 0, 0.1])
        return [CameraConfig("base_camera", pose, 128, 128, np.pi / 2, 0.01, 100)]

    @property
    def _default_human_render_camera_configs(self):
        # Front-right, looking down at ~45 deg — good overview of bin contents
        pose = sapien_utils.look_at([0.65, -0.35, 0.55], [0.25, 0.0, 0.05])
        return CameraConfig("render_camera", pose, 1024, 1024, 1, 0.01, 100)

    def _build_bin(self):
        """Build a kinematic bin from 5 box shapes (bottom + 4 walls)."""
        builder = self.scene.create_actor_builder()
        bx, by, bz = self.BIN_BX, self.BIN_BY, self.BIN_BZ
        t = self.BIN_WALL_T
        mat = sapien.render.RenderMaterial(base_color=[0.2, 0.3, 0.8, 1.0])

        # Bottom plate
        builder.add_box_collision(sapien.Pose([0, 0, 0]), half_size=[bx, by, t / 2])
        builder.add_box_visual(sapien.Pose([0, 0, 0]), half_size=[bx, by, t / 2], material=mat)

        # +X wall
        builder.add_box_collision(sapien.Pose([bx, 0, bz / 2]), half_size=[t / 2, by, bz / 2])
        builder.add_box_visual(sapien.Pose([bx, 0, bz / 2]), half_size=[t / 2, by, bz / 2], material=mat)
        # -X wall
        builder.add_box_collision(sapien.Pose([-bx, 0, bz / 2]), half_size=[t / 2, by, bz / 2])
        builder.add_box_visual(sapien.Pose([-bx, 0, bz / 2]), half_size=[t / 2, by, bz / 2], material=mat)
        # +Y wall
        builder.add_box_collision(sapien.Pose([0, by, bz / 2]), half_size=[bx, t / 2, bz / 2])
        builder.add_box_visual(sapien.Pose([0, by, bz / 2]), half_size=[bx, t / 2, bz / 2], material=mat)
        # -Y wall
        builder.add_box_collision(sapien.Pose([0, -by, bz / 2]), half_size=[bx, t / 2, bz / 2])
        builder.add_box_visual(sapien.Pose([0, -by, bz / 2]), half_size=[bx, t / 2, bz / 2], material=mat)

        return builder.build_kinematic(name="bin")

    def _build_primitive(self, idx: int):
        """Build a random colorful primitive with random size (small/medium/large)."""
        shape = self.np_random.choice(["cube", "sphere", "cylinder", "box"])
        size_cat = self.np_random.choice(["small", "medium", "large"])
        lo, hi = SIZE_PRESETS[size_cat]
        color = OBJECT_COLORS[idx % len(OBJECT_COLORS)]
        mat = sapien.render.RenderMaterial(base_color=color)
        builder = self.scene.create_actor_builder()

        if shape == "cube":
            hs = self.np_random.uniform(lo, hi)
            builder.add_box_collision(half_size=[hs, hs, hs])
            builder.add_box_visual(half_size=[hs, hs, hs], material=mat)
        elif shape == "sphere":
            r = self.np_random.uniform(lo, hi)
            builder.add_sphere_collision(radius=r)
            builder.add_sphere_visual(radius=r, material=mat)
        elif shape == "cylinder":
            r = self.np_random.uniform(lo * 0.7, hi * 0.7)
            hl = self.np_random.uniform(lo, hi)
            builder.add_cylinder_collision(radius=r, half_length=hl)
            builder.add_cylinder_visual(radius=r, half_length=hl, material=mat)
        else:  # box
            hx = self.np_random.uniform(lo, hi)
            hy = self.np_random.uniform(lo * 0.6, hi)
            hz = self.np_random.uniform(lo * 0.6, hi)
            builder.add_box_collision(half_size=[hx, hy, hz])
            builder.add_box_visual(half_size=[hx, hy, hz], material=mat)

        builder.initial_pose = sapien.Pose(p=[0, 0, 0.5 + idx * 0.1])
        return builder.build(name=f"prim_{idx}")

    def _build_ycb(self, idx: int):
        """Build a random YCB mesh object."""
        model_id = self.np_random.choice(YCB_MODEL_IDS)
        builder = actors.get_actor_builder(self.scene, id=f"ycb:{model_id}")
        builder.initial_pose = sapien.Pose(p=[0, 0, 0.5 + idx * 0.1])
        return builder.build(name=f"ycb_{idx}")

    def get_objects(self) -> dict[str, object]:
        """Return a name→actor mapping for all manipulable objects."""
        return {obj.name: obj for obj in self.objects}

    def _load_scene(self, options: dict):
        self.table_scene = TableSceneBuilder(
            env=self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()
        self.bin = self._build_bin()

        # Mix of primitives and YCB — roughly 50/50
        self.objects = []
        for i in range(self.num_objects):
            if self.np_random.random() < 0.5:
                self.objects.append(self._build_primitive(i))
            else:
                self.objects.append(self._build_ycb(i))

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            self.table_scene.initialize(env_idx)
            self._after_table_scene_init(env_idx)

            # Move robot out of the way during settling
            self.agent.robot.set_root_pose(sapien.Pose(p=[-5, 0, 0]))
            n_joints = len(self.agent.robot.get_active_joints())
            park_qpos = np.zeros(n_joints)
            self.agent.reset(park_qpos)

            # Place bin on table
            cx, cy = self.BIN_CENTER
            bin_z = self.BIN_WALL_T / 2
            self.bin.set_pose(sapien.Pose(p=[cx, cy, bin_z]))

            # Park all objects on the floor, spread out away from the table
            for i, obj in enumerate(self.objects):
                row, col = divmod(i, 6)
                obj.set_pose(sapien.Pose(p=[-1.0 - row * 0.15, -0.4 + col * 0.15, 0.05]))

            # Drop objects one at a time, gently from just above the bin floor
            spawn_r = 0.75
            drop_z = bin_z + 0.05  # just 5cm above the bin floor

            def _is_inside(obj):
                p = obj.pose.p[0].cpu().numpy()
                return (abs(p[0] - cx) < self.BIN_BX
                        and abs(p[1] - cy) < self.BIN_BY
                        and p[2] > -0.05)

            for obj in self.objects:
                for attempt in range(3):
                    x = cx + self.np_random.uniform(-self.BIN_BX * spawn_r, self.BIN_BX * spawn_r)
                    y = cy + self.np_random.uniform(-self.BIN_BY * spawn_r, self.BIN_BY * spawn_r)
                    obj.set_pose(sapien.Pose(p=[x, y, drop_z]))
                    escaped = False
                    for step in range(100):
                        self.scene.step()
                        if step % 10 == 9 and not _is_inside(obj):
                            escaped = True
                            break
                    if not escaped and _is_inside(obj):
                        # Raise drop height for next object
                        drop_z = max(drop_z, obj.pose.p[0, 2].item() + 0.05)
                        break
                else:
                    obj.set_pose(sapien.Pose(p=[cx, cy, bin_z + 0.03]))

            # Re-drop any objects knocked out by later drops
            for obj in self.objects:
                if not _is_inside(obj):
                    obj.set_pose(sapien.Pose(p=[cx, cy, drop_z]))
                    for _ in range(100):
                        self.scene.step()

            # Move robot back and reset to default pose
            self.agent.robot.set_root_pose(sapien.Pose(p=[-0.615, 0, 0]))
            self.table_scene.initialize(env_idx)

    def evaluate(self):
        return {
            "success": torch.zeros(self.num_envs, device=self.device, dtype=torch.bool),
        }

    def _get_obs_extra(self, info: dict):
        obs = dict(tcp_pose=self.agent.tcp.pose.raw_pose)
        if "state" in self.obs_mode:
            obj_poses = torch.cat([obj.pose.p for obj in self.objects], dim=-1)
            obs["obj_poses"] = obj_poses
        return obs

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        return torch.zeros(self.num_envs, device=self.device)

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: dict
    ):
        return torch.zeros(self.num_envs, device=self.device)


if __name__ == "__main__":
    import gymnasium as gym

    env = gym.make(
        "BinWithObjects-v1",
        num_envs=1,
        render_mode="human",
        robot_uids="panda",
    )
    obs, _ = env.reset()
    for _ in range(300):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        if terminated or truncated:
            obs, _ = env.reset()
    env.close()
