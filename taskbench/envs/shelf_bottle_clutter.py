"""Bottle clutter shelf environment with replayable random layouts."""

from __future__ import annotations

from typing import Any

import numpy as np
import sapien
import sapien.render
import torch
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.building.ground import build_ground
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.types import SceneConfig, SimConfig
from transforms3d.euler import euler2quat
from transforms3d.quaternions import qmult

from taskbench.envs.base import TaskEnv
from taskbench.envs.bottle_builder import (
    BOTTLE_UPRIGHT_Q,
    ObjectGeometry,
    build_bottle_actor,
    get_visual_footprint_radius,
)
from taskbench.envs.open_table_defaults import (
    BOTTLE_BALLAST_HALF_LENGTH_BASE,
    BOTTLE_BALLAST_OFFSET_BASE,
    BOTTLE_BALLAST_RADIUS_BASE,
    BOTTLE_BODY_HALF_LENGTH_BASE,
    BOTTLE_BODY_RADIUS_BASE,
    BOTTLE_NECK_HALF_LENGTH_BASE,
    BOTTLE_NECK_OFFSET_BASE,
    BOTTLE_NECK_RADIUS_BASE,
    BOTTLE_SCALE,
    BOTTLE_VISUAL_STYLE,
    DENSE_LAYOUT_PROB,
    EDGE_MARGIN,
    MIXED_LAYOUT_PROB,
    PLACEMENT_CLEARANCE,
    PLACEMENT_JITTER,
    PLACEMENT_SPACING,
)
from taskbench.envs.placement import (
    PlacementGrid,
    build_rect_grid,
    clamp_jitter,
    jitter_positions,
    sample_frontier_cells,
)
from taskbench.envs.shelf_env import ShelfGeometry

BLUE_COLOR = [0.20, 0.40, 0.85, 1.0]
RED_COLOR = [0.90, 0.10, 0.10, 1.0]
WOOD_COLOR = [0.55, 0.35, 0.10, 1.0]
DEFAULT_SHELF_BOTTLE_GEOMETRY = {
    "front_x": 0.26,
    "depth": 0.22,
    "half_w": 0.35,
    "floor_z": 0.40,
    "thickness": 0.01,
    "inner_h": 0.35,
}


@register_env(
    "ShelfBottleClutter-v1",
    max_episode_steps=800,
    asset_download_ids=["ycb"],
)
class ShelfBottleClutterEnv(TaskEnv):
    """Shelf scene populated with random bottle clutter.

    The shelf is static geometry. Bottles reuse the same composite collision
    model and YCB mustard bottle visual path as the open-table clutter env so
    mass and contact behavior stay aligned across tasks.
    """

    SUPPORTED_REWARD_MODES = ["none"]

    def __init__(
        self,
        *args,
        robot_uids: str = "panda",
        num_bottles: int = 12,
        shelf: dict | None = None,
        bottle_scale: float = BOTTLE_SCALE,
        object_density: float = 1800.0,
        edge_margin: float = EDGE_MARGIN,
        placement_spacing: float = PLACEMENT_SPACING,
        placement_clearance: float = PLACEMENT_CLEARANCE,
        placement_jitter: float = PLACEMENT_JITTER,
        layout_mode: str = "auto",
        dense_layout_prob: float = DENSE_LAYOUT_PROB,
        mixed_layout_prob: float = MIXED_LAYOUT_PROB,
        random_yaw: bool = True,
        movement_success_threshold: float = 0.02,
        robot_init_qpos_noise: float = 0.0,
        **kwargs,
    ):
        shelf_cfg = dict(DEFAULT_SHELF_BOTTLE_GEOMETRY)
        if shelf is not None:
            shelf_cfg.update(shelf)
        self.shelf_geom = ShelfGeometry(**shelf_cfg)
        self.num_bottles = int(num_bottles)
        if self.num_bottles <= 0:
            raise ValueError("num_bottles must be positive")
        self.target_idx = self.num_bottles - 1
        self.robot_init_qpos_noise = float(robot_init_qpos_noise)
        self.bottle_scale = float(bottle_scale)
        if self.bottle_scale <= 0:
            raise ValueError("bottle_scale must be positive")

        self.edge_margin = float(edge_margin)
        self.placement_spacing = float(placement_spacing)
        self.placement_clearance = float(placement_clearance)
        self.requested_placement_jitter = float(placement_jitter)
        self.layout_mode = str(layout_mode)
        self.dense_layout_prob = float(dense_layout_prob)
        self.mixed_layout_prob = float(mixed_layout_prob)
        self.random_yaw = bool(random_yaw)
        self.movement_success_threshold = float(movement_success_threshold)

        self.obj = ObjectGeometry(
            kind="bottle",
            density=float(object_density),
            body_radius=BOTTLE_BODY_RADIUS_BASE * self.bottle_scale,
            body_half_length=BOTTLE_BODY_HALF_LENGTH_BASE * self.bottle_scale,
            neck_radius=BOTTLE_NECK_RADIUS_BASE * self.bottle_scale,
            neck_half_length=BOTTLE_NECK_HALF_LENGTH_BASE * self.bottle_scale,
            neck_offset=BOTTLE_NECK_OFFSET_BASE * self.bottle_scale,
            ballast_radius=BOTTLE_BALLAST_RADIUS_BASE * self.bottle_scale,
            ballast_half_length=BOTTLE_BALLAST_HALF_LENGTH_BASE * self.bottle_scale,
            ballast_offset=BOTTLE_BALLAST_OFFSET_BASE * self.bottle_scale,
            visual_style=BOTTLE_VISUAL_STYLE,
        )
        self.bottle_body_radius = float(self.obj.body_radius)
        self.bottle_body_half_length = float(self.obj.body_half_length)

        self.min_center_distance = (
            2.0 * self.bottle_body_radius + self.placement_clearance
        )
        if self.placement_spacing < self.min_center_distance:
            raise ValueError(
                "placement_spacing must be at least 2 * bottle_body_radius + "
                f"placement_clearance ({self.min_center_distance:.4f})"
            )

        self._workspace_lo_xy = np.zeros((2,), dtype=np.float32)
        self._workspace_hi_xy = np.zeros((2,), dtype=np.float32)
        self._placement_lo_xy = np.zeros((2,), dtype=np.float32)
        self._placement_hi_xy = np.zeros((2,), dtype=np.float32)
        self._placement_grid: PlacementGrid | None = None
        self._grid_centers_xy = np.empty((0, 2), dtype=np.float32)
        self.placement_jitter = 0.0
        self.scene_layout: dict[str, object] = {}

        super().__init__(
            *args,
            robot_uids=robot_uids,
            reconfiguration_freq=1,
            **kwargs,
        )

        self.initial_positions_xy = np.zeros((self.num_bottles, 2), dtype=np.float32)
        self.initial_positions_xy_batched = torch.zeros(
            self.num_envs, self.num_bottles, 2, device=self.device
        )

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
        pose = sapien_utils.look_at(
            eye=[0.10, 0.0, 0.55],
            target=[0.55, 0.0, 0.42],
        )
        return [CameraConfig("base_camera", pose, 128, 128, np.pi / 2, 0.01, 100)]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at(
            eye=[0.37, 0.0, 1.42],
            target=[0.37, 0.0, 0.45],
        )
        return CameraConfig("render_camera", pose, 1024, 1024, 0.95, 0.01, 100)

    def get_collision_boxes(self):
        g = self.shelf_geom
        cx = g.center_x
        hw = g.half_w
        fz = g.floor_z
        ih = g.inner_h
        t = g.thickness
        d = g.depth

        boxes = [
            ("shelf_bottom", [cx, 0, fz], [d / 2, hw, t]),
            ("shelf_top", [cx, 0, fz + 2 * t + ih], [d / 2, hw, t]),
            ("shelf_back", [g.back_x, 0, fz + t + ih / 2], [t, hw, ih / 2]),
            ("shelf_left", [cx, -hw, fz + t + ih / 2], [d / 2, t, ih / 2]),
            ("shelf_right", [cx, hw, fz + t + ih / 2], [d / 2, t, ih / 2]),
        ]

        leg_r = 0.015
        leg_h = g.leg_height / 2
        for idx, (dx, dy) in enumerate(
            [
                (-d / 2 + 0.02, -hw + 0.02),
                (-d / 2 + 0.02, hw - 0.02),
                (d / 2 - 0.02, -hw + 0.02),
                (d / 2 - 0.02, hw - 0.02),
            ]
        ):
            boxes.append((f"leg_{idx}", [cx + dx, dy, leg_h], [leg_r, leg_r, leg_h]))
        return boxes

    def _build_shelf(self):
        material = sapien.render.RenderMaterial(base_color=WOOD_COLOR)
        parts = []

        g = self.shelf_geom
        cx = g.center_x
        hw = g.half_w
        fz = g.floor_z
        ih = g.inner_h
        t = g.thickness
        d = g.depth

        def _box(name, center, half_size):
            builder = self.scene.create_actor_builder()
            builder.add_box_collision(half_size=half_size)
            builder.add_box_visual(half_size=half_size, material=material)
            builder.initial_pose = sapien.Pose(p=center)
            parts.append(builder.build_static(name=name))

        _box("shelf_bottom", [cx, 0, fz], [d / 2, hw, t])
        _box("shelf_top", [cx, 0, fz + 2 * t + ih], [d / 2, hw, t])
        _box("shelf_back", [g.back_x, 0, fz + t + ih / 2], [t, hw, ih / 2])
        _box("shelf_left", [cx, -hw, fz + t + ih / 2], [d / 2, t, ih / 2])
        _box("shelf_right", [cx, hw, fz + t + ih / 2], [d / 2, t, ih / 2])

        leg_r = 0.015
        leg_h = g.leg_height / 2
        for idx, (dx, dy) in enumerate(
            [
                (-d / 2 + 0.02, -hw + 0.02),
                (-d / 2 + 0.02, hw - 0.02),
                (d / 2 - 0.02, -hw + 0.02),
                (d / 2 - 0.02, hw - 0.02),
            ]
        ):
            _box(f"leg_{idx}", [cx + dx, dy, leg_h], [leg_r, leg_r, leg_h])
        return parts

    def _configure_placement_workspace(self) -> None:
        g = self.shelf_geom
        self._workspace_lo_xy = np.array([g.front_x, -g.half_w], dtype=np.float32)
        self._workspace_hi_xy = np.array([g.back_x, g.half_w], dtype=np.float32)

        visual_margin = self.edge_margin + get_visual_footprint_radius(self.obj)
        front_margin_x = self.edge_margin + self.bottle_body_radius
        self._placement_lo_xy = np.array(
            [
                float(self._workspace_lo_xy[0] + front_margin_x),
                float(self._workspace_lo_xy[1] + visual_margin),
            ],
            dtype=np.float32,
        )
        self._placement_hi_xy = np.array(
            [
                float(self._workspace_hi_xy[0] - visual_margin),
                float(self._workspace_hi_xy[1] - visual_margin),
            ],
            dtype=np.float32,
        )
        if np.any(self._placement_hi_xy < self._placement_lo_xy):
            raise ValueError("Shelf geometry leaves no usable bottle placement area")

        self._placement_grid = build_rect_grid(
            self._placement_lo_xy, self._placement_hi_xy, self.placement_spacing
        )
        self.placement_jitter = clamp_jitter(
            self.requested_placement_jitter,
            grid_spacing=self._placement_grid.spacing,
            min_center_distance=self.min_center_distance,
        )
        self._grid_centers_xy = self._placement_grid.centers_xy.copy()
        if len(self._grid_centers_xy) < self.num_bottles:
            raise ValueError(
                f"Shelf workspace only supports {len(self._grid_centers_xy)} bottles at "
                f"spacing {self.placement_spacing:.3f}, but num_bottles={self.num_bottles}"
            )

    def _build_bottle(self, idx: int):
        color = RED_COLOR if idx == self.target_idx else BLUE_COLOR
        return build_bottle_actor(self.scene, self.obj, idx, color)

    def _load_scene(self, options: dict):
        build_ground(self.scene, altitude=0.0)
        self.shelf_parts = self._build_shelf()
        self.bottles = [self._build_bottle(i) for i in range(self.num_bottles)]
        self.target_object = self.bottles[self.target_idx]
        self._configure_placement_workspace()

    def get_objects(self) -> dict[str, object]:
        return {obj.name: obj for obj in self.bottles}

    def get_workspace_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        return self._workspace_lo_xy.copy(), self._workspace_hi_xy.copy()

    def get_grid_centers(self) -> np.ndarray:
        return self._grid_centers_xy.copy()

    def get_scene_layout(self) -> dict[str, object]:
        layout = dict(self.scene_layout)
        for key, dtype in (
            ("occupied_indices", np.int32),
            ("seed_indices", np.int32),
            ("positions_xy", np.float32),
            ("cluster_target_sizes", np.int32),
        ):
            if key in layout:
                layout[key] = np.asarray(layout[key], dtype=dtype).copy()
        return layout

    def is_inside_workspace(self, xy, margin: float = 0.0) -> bool:
        xy = np.asarray(xy, dtype=np.float32).reshape(-1)[:2]
        margin = float(margin)
        return bool(
            np.all(xy >= self._workspace_lo_xy + margin)
            and np.all(xy <= self._workspace_hi_xy - margin)
        )

    def get_bottle_positions_xy(self) -> tuple[list[str], np.ndarray]:
        names = [obj.name for obj in self.bottles]
        positions = np.stack(
            [
                obj.pose.p[0, :2].detach().cpu().numpy().astype(np.float32)
                for obj in self.bottles
            ],
            axis=0,
        )
        return names, positions

    def _bottle_positions_xy_batched(self) -> torch.Tensor:
        return torch.stack([obj.pose.p[:, :2] for obj in self.bottles], dim=1)

    def get_scene_spec(self) -> dict[str, object]:
        object_names = [obj.name for obj in self.bottles]
        object_positions = np.stack(
            [
                obj.pose.p[0].detach().cpu().numpy().astype(np.float32)
                for obj in self.bottles
            ],
            axis=0,
        )
        object_quats = np.stack(
            [
                obj.pose.q[0].detach().cpu().numpy().astype(np.float32)
                for obj in self.bottles
            ],
            axis=0,
        )
        robot_root_position = (
            self.agent.robot.pose.p[0].detach().cpu().numpy().astype(np.float32)
        )
        robot_root_quat = (
            self.agent.robot.pose.q[0].detach().cpu().numpy().astype(np.float32)
        )
        robot_qpos = (
            self.agent.robot.get_qpos()[0].detach().cpu().numpy().astype(np.float32)
        )
        workspace_lo, workspace_hi = self.get_workspace_bounds()
        return {
            "env_id": "ShelfBottleClutter-v1",
            "num_bottles": int(self.num_bottles),
            "target_idx": int(self.target_idx),
            "object_names": list(object_names),
            "object_positions_xyz": object_positions,
            "object_quats_wxyz": object_quats,
            "robot_root_position_xyz": robot_root_position,
            "robot_root_quat_wxyz": robot_root_quat,
            "robot_qpos": robot_qpos,
            "workspace_lo_xy": workspace_lo,
            "workspace_hi_xy": workspace_hi,
            "layout": self.get_scene_layout(),
        }

    def apply_scene_spec(self, scene_spec: dict[str, object]) -> None:
        target_idx = int(scene_spec.get("target_idx", self.target_idx))
        if target_idx != self.target_idx:
            raise ValueError(
                f"scene target_idx={target_idx} does not match env target_idx={self.target_idx}"
            )

        object_positions = np.asarray(
            scene_spec["object_positions_xyz"], dtype=np.float32
        )
        object_quats = np.asarray(scene_spec["object_quats_wxyz"], dtype=np.float32)
        if object_positions.shape != (self.num_bottles, 3):
            raise ValueError(
                "scene object_positions_xyz must have shape "
                f"({self.num_bottles}, 3), got {object_positions.shape}"
            )
        if object_quats.shape != (self.num_bottles, 4):
            raise ValueError(
                "scene object_quats_wxyz must have shape "
                f"({self.num_bottles}, 4), got {object_quats.shape}"
            )

        robot_root_position = np.asarray(
            scene_spec["robot_root_position_xyz"], dtype=np.float32
        )
        robot_root_quat = np.asarray(
            scene_spec["robot_root_quat_wxyz"], dtype=np.float32
        )
        robot_qpos = np.asarray(scene_spec["robot_qpos"], dtype=np.float32)

        self.agent.robot.set_root_pose(
            sapien.Pose(robot_root_position.tolist(), robot_root_quat.tolist())
        )
        robot_qpos_t = torch.as_tensor(
            robot_qpos, dtype=torch.float32, device=self.device
        ).reshape(1, -1)
        self.agent.robot.set_qpos(robot_qpos_t)
        if hasattr(self.agent.robot, "set_qvel"):
            self.agent.robot.set_qvel(torch.zeros_like(robot_qpos_t))

        zero_twist = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)
        for obj, position, quat in zip(self.bottles, object_positions, object_quats):
            obj.set_pose(sapien.Pose(position.tolist(), quat.tolist()))
            if hasattr(obj, "set_linear_velocity"):
                obj.set_linear_velocity(zero_twist)
            if hasattr(obj, "set_angular_velocity"):
                obj.set_angular_velocity(zero_twist)

        self.initial_positions_xy_batched = self._bottle_positions_xy_batched()
        _, self.initial_positions_xy = self.get_bottle_positions_xy()
        layout = scene_spec.get("layout", {})
        self.scene_layout = dict(layout) if isinstance(layout, dict) else {}

    def _sample_bottle_quaternion(self) -> np.ndarray:
        if not self.random_yaw:
            return np.asarray(BOTTLE_UPRIGHT_Q, dtype=np.float32)
        yaw = float(self.np_random.uniform(-np.pi, np.pi))
        yaw_q = euler2quat(0.0, 0.0, yaw)
        return np.asarray(qmult(yaw_q, BOTTLE_UPRIGHT_Q), dtype=np.float32)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        if self._placement_grid is None:
            raise RuntimeError("Placement grid is not configured")

        self._reset_robot(env_idx)
        with torch.device(self.device):
            z = self.shelf_geom.surface_z + self.bottle_body_half_length
            occupied_indices, layout_meta = sample_frontier_cells(
                self.np_random,
                self._placement_grid,
                num_cells=self.num_bottles,
                mode=self.layout_mode,
                dense_layout_prob=self.dense_layout_prob,
                mixed_layout_prob=self.mixed_layout_prob,
            )
            self.np_random.shuffle(occupied_indices)
            centers_xy = self._placement_grid.centers_xy[occupied_indices]
            positions_xy = jitter_positions(
                self.np_random,
                centers_xy,
                max_jitter=self.placement_jitter,
                lo_xy=self._placement_lo_xy,
                hi_xy=self._placement_hi_xy,
            )

            zero_twist = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)
            for idx, xy in enumerate(positions_xy):
                q = self._sample_bottle_quaternion()
                self.bottles[idx].set_pose(sapien.Pose([xy[0], xy[1], z], q))
                if hasattr(self.bottles[idx], "set_linear_velocity"):
                    self.bottles[idx].set_linear_velocity(zero_twist)
                if hasattr(self.bottles[idx], "set_angular_velocity"):
                    self.bottles[idx].set_angular_velocity(zero_twist)

            self.initial_positions_xy_batched = self._bottle_positions_xy_batched()
            _, self.initial_positions_xy = self.get_bottle_positions_xy()
            self.scene_layout = {
                **layout_meta,
                "occupied_indices": occupied_indices.astype(np.int32),
                "positions_xy": positions_xy.astype(np.float32),
            }

    def evaluate(self):
        current_xy = self._bottle_positions_xy_batched()
        deltas = current_xy - self.initial_positions_xy_batched
        displacement = torch.linalg.norm(deltas, dim=-1)
        max_displacement = displacement.max(dim=-1).values
        moved_bottles = (displacement > self.movement_success_threshold).sum(dim=-1)
        success = max_displacement > self.movement_success_threshold
        return {
            "success": success.to(dtype=torch.bool),
            "max_displacement": max_displacement.to(dtype=torch.float32),
            "moved_bottles": moved_bottles.to(dtype=torch.int32),
        }

    def _get_obs_extra(self, info: dict):
        obs = dict(tcp_pose=self.agent.tcp.pose.raw_pose)
        for obj in self.bottles:
            obs[f"{obj.name}_pose"] = obj.pose.raw_pose
        return obs

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        return torch.zeros(self.num_envs, device=self.device)

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: dict
    ):
        return torch.zeros(self.num_envs, device=self.device)
