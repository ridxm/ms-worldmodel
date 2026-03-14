"""Cluttered shelf environment for ManiSkill3.

An enclosed shelf on legs sits in front of the Panda robot, open toward
the robot (-X face).  Inside: 19 blue cylinders and 1 red target.  The
robot must reach in from the front, push blue cylinders aside, and extract
the red one.  CPU-only, single-env.
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
from mani_skill.utils.building.ground import build_ground
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.types import SceneConfig, SimConfig

# Colors
BLUE_COLOR = [0.20, 0.40, 0.85, 1.0]
RED_COLOR = [0.90, 0.10, 0.10, 1.0]
WOOD_COLOR = [0.55, 0.35, 0.10, 1.0]

# ── Shelf geometry ──────────────────────────────────────────────────
# Robot at origin faces +X.  Shelf sits in front, open toward -X.
#
#   Depth runs along +X (into the shelf, away from robot).
#   Width runs along Y.
#   Height runs along Z.
#
SHELF_FRONT_X = 0.42       # front edge X (closest to robot)
SHELF_DEPTH = 0.30         # depth along +X
SHELF_BACK_X = SHELF_FRONT_X + SHELF_DEPTH
SHELF_CENTER_X = SHELF_FRONT_X + SHELF_DEPTH / 2
SHELF_HALF_W = 0.50        # half-width along Y (100cm total)
SHELF_FLOOR_Z = 0.55       # bottom board height above ground
SHELF_T = 0.01             # board/wall thickness
SHELF_SURFACE_Z = SHELF_FLOOR_Z + SHELF_T
SHELF_INNER_H = 0.50       # interior height
SHELF_CEIL_Z = SHELF_FLOOR_Z + 2 * SHELF_T + SHELF_INNER_H
LEG_HEIGHT = SHELF_FLOOR_Z  # legs from ground to bottom board

# ── Cylinders ───────────────────────────────────────────────────────
NUM_OBJECTS = 20
CYL_RADIUS = 0.018         # 3.6cm diameter
CYL_HALF_LENGTH = 0.045    # 9cm tall upright
CYL_UPRIGHT_Q = [0.7071068, 0, 0.7071068, 0]  # rotate +X axis -> +Z

SUCCESS_LIFT_Z = SHELF_CEIL_Z + 0.05


@register_env("ShelfEnv-v1", max_episode_steps=300)
class ShelfEnv(TaskEnv):
    """Enclosed shelf with 19 blue cylinders and 1 red target.

    The shelf faces the robot (open toward -X).  The robot reaches in
    to find and extract the red cylinder.

    Args:
        num_objects: Total number of cylinders (default: 20).
        robot_uids: Robot to use (default: "panda").
    """

    SUPPORTED_ROBOTS = ["panda", "ur5_robotiq", "ur5e_robotiq"]
    SUPPORTED_REWARD_MODES = ["none"]
    agent: Union[Panda]

    def __init__(self, *args, robot_uids="panda", num_objects: int = NUM_OBJECTS, **kwargs):
        self.num_objects = num_objects
        self.target_idx = 0
        super().__init__(
            *args,
            robot_uids=robot_uids,
            reconfiguration_freq=1,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # Sim / camera configs
    # ------------------------------------------------------------------

    @property
    def _default_sim_config(self):
        return SimConfig(
            scene_config=SceneConfig(
                solver_position_iterations=20,
                solver_velocity_iterations=5,
            )
        )

    @property
    def _default_sensor_configs(self):
        # Looking from behind the robot toward the shelf
        pose = sapien_utils.look_at(eye=[0.10, 0.0, 0.55], target=[0.55, 0.0, 0.42])
        return [CameraConfig("base_camera", pose, 128, 128, np.pi / 2, 0.01, 100)]

    @property
    def _default_human_render_camera_configs(self):
        # Front-right view: see both the robot and the open face of the shelf
        pose = sapien_utils.look_at([-0.15, -0.70, 0.65], [0.40, 0.0, 0.38])
        return CameraConfig("render_camera", pose, 1024, 1024, 1, 0.01, 100)

    # ------------------------------------------------------------------
    # Collision geometry (for motion planner)
    # ------------------------------------------------------------------

    def get_collision_boxes(self):
        """Return shelf collision geometry as a list of (name, center, half_size).

        Each entry is a box: (str, [x,y,z], [hx,hy,hz]).
        Solvers can pass these to the motion planner as obstacles.
        """
        cx = SHELF_CENTER_X
        hw = SHELF_HALF_W
        fz = SHELF_FLOOR_Z
        ih = SHELF_INNER_H
        t = SHELF_T
        d = SHELF_DEPTH

        boxes = [
            ("shelf_bottom", [cx, 0, fz], [d / 2, hw, t]),
            ("shelf_top", [cx, 0, fz + 2 * t + ih], [d / 2, hw, t]),
            ("shelf_back", [SHELF_BACK_X, 0, fz + t + ih / 2], [t, hw, ih / 2]),
            ("shelf_left", [cx, -hw, fz + t + ih / 2], [d / 2, t, ih / 2]),
            ("shelf_right", [cx, hw, fz + t + ih / 2], [d / 2, t, ih / 2]),
        ]

        leg_r = 0.015
        lh = LEG_HEIGHT / 2
        for li, (dx, dy) in enumerate([(-d / 2 + 0.02, -hw + 0.02),
                                         (-d / 2 + 0.02, hw - 0.02),
                                         (d / 2 - 0.02, -hw + 0.02),
                                         (d / 2 - 0.02, hw - 0.02)]):
            boxes.append((f"leg_{li}", [cx + dx, dy, lh], [leg_r, leg_r, lh]))

        return boxes

    # ------------------------------------------------------------------
    # Scene building
    # ------------------------------------------------------------------

    def _build_shelf(self):
        """Build enclosed shelf on legs, open toward -X (facing robot)."""
        mat = sapien.render.RenderMaterial(base_color=WOOD_COLOR)
        parts = []

        cx = SHELF_CENTER_X
        hw = SHELF_HALF_W
        fz = SHELF_FLOOR_Z
        ih = SHELF_INNER_H
        t = SHELF_T
        d = SHELF_DEPTH

        def _box(name, center, half_size):
            b = self.scene.create_actor_builder()
            b.add_box_collision(half_size=half_size)
            b.add_box_visual(half_size=half_size, material=mat)
            b.initial_pose = sapien.Pose(p=center)
            parts.append(b.build_static(name=name))

        # Bottom board
        _box("shelf_bottom", [cx, 0, fz], [d / 2, hw, t])
        # Top board (ceiling)
        _box("shelf_top", [cx, 0, fz + 2 * t + ih], [d / 2, hw, t])
        # Back wall (+X side, far from robot)
        _box("shelf_back", [SHELF_BACK_X, 0, fz + t + ih / 2], [t, hw, ih / 2])
        # Left wall (-Y)
        _box("shelf_left", [cx, -hw, fz + t + ih / 2], [d / 2, t, ih / 2])
        # Right wall (+Y)
        _box("shelf_right", [cx, hw, fz + t + ih / 2], [d / 2, t, ih / 2])

        # 4 legs
        leg_r = 0.015  # leg radius approximated as thin box
        lh = LEG_HEIGHT / 2
        for li, (dx, dy) in enumerate([(-d / 2 + 0.02, -hw + 0.02),
                                         (-d / 2 + 0.02, hw - 0.02),
                                         (d / 2 - 0.02, -hw + 0.02),
                                         (d / 2 - 0.02, hw - 0.02)]):
            _box(f"leg_{li}", [cx + dx, dy, lh], [leg_r, leg_r, lh])

        return parts

    def get_objects(self) -> dict[str, object]:
        """Return a name→actor mapping for all manipulable objects."""
        return {f"cyl_{i}": obj for i, obj in enumerate(self.shelf_objects)}

    def _load_scene(self, options: dict):
        build_ground(self.scene, altitude=0.0)
        self.shelf_parts = self._build_shelf()

        # Pick target before building so we can color it red
        self.shelf_objects = []
        self.target_object = None
        if self.num_objects > 0:
            self.target_idx = self.np_random.integers(0, self.num_objects)
            for i in range(self.num_objects):
                color = RED_COLOR if i == self.target_idx else BLUE_COLOR
                cyl_mat = sapien.render.RenderMaterial(base_color=color)
                builder = self.scene.create_actor_builder()
                builder.add_cylinder_collision(radius=CYL_RADIUS, half_length=CYL_HALF_LENGTH)
                builder.add_cylinder_visual(radius=CYL_RADIUS, half_length=CYL_HALF_LENGTH, material=cyl_mat)
                builder.initial_pose = sapien.Pose(p=[0, 0, 1.0 + i * 0.1])
                self.shelf_objects.append(builder.build(name=f"cyl_{i}"))
            self.target_object = self.shelf_objects[self.target_idx]

    # ------------------------------------------------------------------
    # Episode init
    # ------------------------------------------------------------------

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            # Scatter cylinders randomly inside the shelf
            margin = CYL_RADIUS + 0.01
            x_lo = SHELF_FRONT_X + margin
            x_hi = SHELF_BACK_X - margin
            y_lo = -SHELF_HALF_W + margin
            y_hi = SHELF_HALF_W - margin
            z = SHELF_SURFACE_Z + CYL_HALF_LENGTH

            for i, obj in enumerate(self.shelf_objects):
                x = self.np_random.uniform(x_lo, x_hi)
                y = self.np_random.uniform(y_lo, y_hi)
                obj.set_pose(sapien.Pose(p=[x, y, z], q=CYL_UPRIGHT_Q))

    # ------------------------------------------------------------------
    # Evaluation / obs / reward
    # ------------------------------------------------------------------

    def evaluate(self):
        if self.target_object is None:
            success = False
        else:
            target_z = self.target_object.pose.p[0, 2].item()
            success = target_z > SUCCESS_LIFT_Z
        return {
            "success": torch.tensor(
                [success], device=self.device, dtype=torch.bool
            ),
        }

    def _get_obs_extra(self, info: dict):
        obs = dict(tcp_pose=self.agent.tcp.pose.raw_pose)
        if self.shelf_objects:
            obj_poses = torch.cat(
                [obj.pose.p for obj in self.shelf_objects], dim=-1
            )
            obs["obj_poses"] = obj_poses
            obs["target_idx"] = torch.tensor(
                [[self.target_idx]], device=self.device, dtype=torch.float32
            )
            obs["target_pos"] = self.target_object.pose.p
        return obs

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        return torch.zeros(self.num_envs, device=self.device)

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: dict
    ):
        return torch.zeros(self.num_envs, device=self.device)


if __name__ == "__main__":
    import gymnasium as gym

    import taskbench.envs  # noqa: F401

    env = gym.make(
        "ShelfEnv-v1",
        num_envs=1,
        render_mode="human",
        robot_uids="panda",
    )
    obs, _ = env.reset()
    for _ in range(300):
        action = env.action_space.sample()
        obs, rew, term, trunc, info = env.step(action)
        env.render()
        if term or trunc:
            obs, _ = env.reset()
    env.close()
