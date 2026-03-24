"""Shared bottle geometry builder for open-table environments.

Contains ``ObjectGeometry`` (the canonical dataclass for object shape/physics)
and pure functions that add collision/visual geometry to SAPIEN actor builders.
Used by both ``TabletopRetrievalEnv`` and ``OpenTableBottleClutterEnv`` without
requiring inheritance between them.
"""

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import sapien
import sapien.physx as physx
import sapien.render
from mani_skill import ASSET_DIR
from mani_skill.utils.io_utils import load_json
from transforms3d.quaternions import qinverse

from taskbench.envs.open_table_defaults import (
    BOTTLE_BALLAST_HALF_LENGTH_BASE, BOTTLE_BALLAST_OFFSET_BASE,
    BOTTLE_BALLAST_RADIUS_BASE, BOTTLE_BODY_HALF_LENGTH_BASE,
    BOTTLE_BODY_RADIUS_BASE, BOTTLE_NECK_HALF_LENGTH_BASE,
    BOTTLE_NECK_OFFSET_BASE, BOTTLE_NECK_RADIUS_BASE, BOTTLE_VISUAL_STYLE)

BOTTLE_UPRIGHT_Q = [0.7071068, 0.0, -0.7071068, 0.0]
YCB_MUSTARD_BOTTLE_ID = "006_mustard_bottle"


@dataclass
class ObjectGeometry:
    """Shape, physics, and visual style for objects in the push row."""
    kind: str = "bottle"
    density: float = 1800.0
    static_friction: float = 0.10
    dynamic_friction: float = 0.07
    restitution: float = 0.02
    cylinder_radius: float = 0.018
    cylinder_half_length: float = 0.045
    body_radius: float = BOTTLE_BODY_RADIUS_BASE
    body_half_length: float = BOTTLE_BODY_HALF_LENGTH_BASE
    neck_radius: float = BOTTLE_NECK_RADIUS_BASE
    neck_half_length: float = BOTTLE_NECK_HALF_LENGTH_BASE
    neck_offset: float = BOTTLE_NECK_OFFSET_BASE
    neck_density_scale: float = 0.5
    ballast_radius: float = BOTTLE_BALLAST_RADIUS_BASE
    ballast_half_length: float = BOTTLE_BALLAST_HALF_LENGTH_BASE
    ballast_offset: float = BOTTLE_BALLAST_OFFSET_BASE
    ballast_density_scale: float = 4.0
    visual_style: str = BOTTLE_VISUAL_STYLE
    ycb_model_id: str = YCB_MUSTARD_BOTTLE_ID


@lru_cache(maxsize=None)
def load_ycb_metadata(model_id: str) -> dict:
    metadata_path = Path(ASSET_DIR) / "assets" / "mani_skill2_ycb" / "info_pick_v0.json"
    model_db = load_json(metadata_path)
    return model_db[model_id]


def add_bottle_collision(builder, obj: ObjectGeometry, material=None) -> None:
    """Add bottle collision geometry (body + neck + ballast) to an actor builder."""
    builder.add_cylinder_collision(
        radius=obj.body_radius,
        half_length=obj.body_half_length,
        material=material,
        density=obj.density,
    )
    neck_pose = sapien.Pose([obj.neck_offset, 0, 0])
    builder.add_cylinder_collision(
        pose=neck_pose,
        radius=obj.neck_radius,
        half_length=obj.neck_half_length,
        material=material,
        density=obj.density * obj.neck_density_scale,
    )
    ballast_pose = sapien.Pose([-obj.ballast_offset, 0, 0])
    builder.add_cylinder_collision(
        pose=ballast_pose,
        radius=obj.ballast_radius,
        half_length=obj.ballast_half_length,
        material=material,
        density=obj.density * obj.ballast_density_scale,
    )


def add_bottle_visual(builder, obj: ObjectGeometry, material) -> None:
    """Add bottle visual geometry to an actor builder."""
    if obj.visual_style == "primitive":
        builder.add_cylinder_visual(
            radius=obj.body_radius,
            half_length=obj.body_half_length,
            material=material,
        )
        neck_pose = sapien.Pose([obj.neck_offset, 0, 0])
        builder.add_cylinder_visual(
            pose=neck_pose,
            radius=obj.neck_radius,
            half_length=obj.neck_half_length,
            material=material,
        )
        return
    if obj.visual_style != "ycb_mustard":
        raise ValueError(f"Unsupported bottle_visual_style={obj.visual_style!r}")

    scale = get_ycb_bottle_visual_scale(obj)
    mesh_pose = get_ycb_bottle_visual_pose(obj, scale=scale)
    mesh_path = (
        Path(ASSET_DIR) / "assets" / "mani_skill2_ycb" / "models"
        / obj.ycb_model_id / "textured.obj"
    )
    builder.add_visual_from_file(
        filename=str(mesh_path),
        pose=mesh_pose,
        scale=[scale] * 3,
        material=material,
    )


def build_bottle_actor(scene, obj: ObjectGeometry, idx: int, color) -> object:
    """Build a complete bottle actor with collision + visual geometry."""
    builder = scene.create_actor_builder()
    phys_mat = physx.PhysxMaterial(
        static_friction=float(obj.static_friction),
        dynamic_friction=float(obj.dynamic_friction),
        restitution=float(obj.restitution),
    )
    add_bottle_collision(builder, obj, material=phys_mat)
    add_bottle_visual(builder, obj, sapien.render.RenderMaterial(base_color=color))
    builder.initial_pose = sapien.Pose([0, 0, 1.0 + idx * 0.1])
    return builder.build(name=f"bottle_{idx}")


def get_ycb_bottle_visual_scale(obj: ObjectGeometry) -> float:
    """Return the scale factor for the YCB mesh to match the body radius."""
    meta = load_ycb_metadata(obj.ycb_model_id)
    bbox = meta["bbox"]
    half_extent_xy = max(
        abs(float(bbox["min"][0])),
        abs(float(bbox["max"][0])),
        abs(float(bbox["min"][1])),
        abs(float(bbox["max"][1])),
    )
    if half_extent_xy <= 0:
        raise ValueError(f"Invalid YCB bottle metadata for {obj.ycb_model_id!r}")
    return float(obj.body_radius / half_extent_xy)


def get_ycb_bottle_visual_pose(obj: ObjectGeometry, *, scale: float) -> sapien.Pose:
    """Return the local pose offset for the YCB mesh."""
    meta = load_ycb_metadata(obj.ycb_model_id)
    bbox = meta["bbox"]
    bottom_z = float(bbox["min"][2]) * scale
    world_z_offset = -obj.body_half_length - bottom_z
    return sapien.Pose(p=[world_z_offset, 0.0, 0.0], q=qinverse(BOTTLE_UPRIGHT_Q))


def get_visual_footprint_radius(obj: ObjectGeometry) -> float:
    """Return the visual footprint radius (for placement spacing)."""
    radius = float(obj.body_radius)
    if obj.visual_style != "ycb_mustard":
        return radius
    meta = load_ycb_metadata(obj.ycb_model_id)
    bbox = meta["bbox"]
    half_extent_x = max(abs(float(bbox["min"][0])), abs(float(bbox["max"][0])))
    half_extent_y = max(abs(float(bbox["min"][1])), abs(float(bbox["max"][1])))
    scale = get_ycb_bottle_visual_scale(obj)
    circumscribed_radius = scale * float(np.hypot(half_extent_x, half_extent_y))
    return max(radius, circumscribed_radius)
