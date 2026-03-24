"""Low-level motion planning helpers using mplib 0.2.x.

All functions are module-level (no class state) and operate on a raw
gym env with ``num_envs=1`` and ``sim_backend="cpu"``.

Robot-specific constants (move group, finger length, etc.) come from
``RobotConfig`` — see ``taskbench.skills.robot_config``.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import TYPE_CHECKING, Tuple, Union

import mplib
import numpy as np
import sapien

if TYPE_CHECKING:
    from taskbench.skills.robot_config import RobotConfig

logger = logging.getLogger("taskbench.skills.motion")

# A pose can be a sapien.Pose or a (position, quaternion) tuple of array-likes.
PoseLike = Union[sapien.Pose, Tuple]


@dataclass(frozen=True)
class LinearPushPlan:
    """Task-space description of a straight-line push.

    This keeps the API in Cartesian terms instead of exposing robot-specific
    joint choices. IK/planning is responsible for mapping these poses to a
    feasible robot configuration.
    """

    approach_pose: sapien.Pose
    push_pose: sapien.Pose
    hover_pose: sapien.Pose | None = None
    staging_pose: sapien.Pose | None = None
    contact_position: np.ndarray | None = None
    approach_position: np.ndarray | None = None
    push_position: np.ndarray | None = None
    push_direction: np.ndarray | None = None
    wrist_orientation: str = "vertical"
    tool_spin_deg: float = 0.0

    def as_skill_kwargs(self) -> dict[str, PoseLike]:
        """Return kwargs that can be passed directly to ``ctx.push(...)``."""
        return {
            "staging_pose": self.staging_pose,
            "hover_pose": self.hover_pose,
            "approach_pose": self.approach_pose,
            "push_pose": self.push_pose,
        }


def _unwrap_env(env_or_raw):
    """Accept either a wrapped env or a raw ManiSkill env."""
    return env_or_raw.unwrapped if hasattr(env_or_raw, "unwrapped") else env_or_raw


_DRIVE_SCALAR_WARNED: set[str] = set()


def _drive_value_as_scalar(value, *, name):
    """Convert a controller drive parameter to a representative scalar."""
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        raise ValueError(f"Controller drive parameter {name!r} is empty")
    if arr.size > 1 and not np.allclose(arr, arr[0]):
        if name not in _DRIVE_SCALAR_WARNED:
            _DRIVE_SCALAR_WARNED.add(name)
            logger.warning(
                "Controller drive parameter %s varies per joint;"
                " using the first value %.3f",
                name,
                arr[0],
            )
    return float(arr[0])


def _normalize_vector(vec, *, name):
    """Return a unit vector, raising on near-zero input."""
    arr = np.asarray(vec, dtype=np.float64).reshape(-1)
    norm = np.linalg.norm(arr)
    if norm <= 1e-8:
        raise ValueError(f"{name} must be non-zero")
    return arr / norm


def _orthogonal_unit_vector(primary, secondary=None):
    """Return a stable unit vector orthogonal to ``primary``.

    If ``secondary`` is provided and not parallel to ``primary``, the result
    is proportional to ``cross(primary, secondary)``. Otherwise a world-frame
    fallback axis is used.
    """
    primary = _normalize_vector(primary, name="primary")
    if secondary is not None:
        secondary = np.asarray(secondary, dtype=np.float64).reshape(-1)
        cross = np.cross(primary, secondary)
        norm = np.linalg.norm(cross)
        if norm > 1e-8:
            return cross / norm

    fallback_axes = (
        np.array([0.0, 0.0, 1.0], dtype=np.float64),
        np.array([1.0, 0.0, 0.0], dtype=np.float64),
        np.array([0.0, 1.0, 0.0], dtype=np.float64),
    )
    for axis in fallback_axes:
        cross = np.cross(axis, primary)
        norm = np.linalg.norm(cross)
        if norm > 1e-8:
            return cross / norm
    raise ValueError("Failed to find a vector orthogonal to primary")


def resolve_push_direction(push_direction=None, *, push_angle_deg=None):
    """Resolve a planar push heading to a 3D unit vector."""
    if push_direction is not None:
        arr = np.asarray(push_direction, dtype=np.float64).reshape(-1)
        if arr.size < 2:
            raise ValueError("push_direction must have at least 2 components")
        return _normalize_vector([arr[0], arr[1], 0.0], name="push_direction")
    if push_angle_deg is None:
        raise ValueError("Either push_direction or push_angle_deg must be provided")
    angle_rad = np.deg2rad(float(push_angle_deg))
    return np.array([np.cos(angle_rad), np.sin(angle_rad), 0.0], dtype=np.float64)


def rotate_vector_about_axis(vec, axis, angle_rad):
    """Rotate a vector about an axis with Rodrigues' formula."""
    vec = np.asarray(vec, dtype=np.float64).reshape(-1)
    axis = _normalize_vector(axis, name="axis")
    c = np.cos(angle_rad)
    s = np.sin(angle_rad)
    return (
        vec * c
        + np.cross(axis, vec) * s
        + axis * np.dot(axis, vec) * (1.0 - c)
    )


def to_sapien_pose(pose: PoseLike) -> sapien.Pose:
    """Convert a pose-like input to sapien.Pose.

    Accepts:
        - ``sapien.Pose`` — returned as-is.
        - ``(p, q)`` tuple — position (3,) and quaternion (4,) array-likes.
    """
    if isinstance(pose, sapien.Pose):
        return pose
    p, q = pose
    return sapien.Pose(
        np.asarray(p, dtype=np.float32),
        np.asarray(q, dtype=np.float32),
    )


def build_push_pose(agent, position, push_direction_xy, *,
                    wrist_orientation="vertical", tool_spin_deg=0.0,
                    approach_axis=None):
    """Build an end-effector pose for a push from intuitive parameters.

    Args:
        agent: ManiSkill robot agent exposing ``build_grasp_pose``.
        position: World-space TCP position.
        push_direction_xy: 2D or 3D vector describing the push direction in
            the tabletop plane.
        wrist_orientation: ``"vertical"`` keeps the tool normal along ``-Z``.
            ``"horizontal"`` points the tool along the push direction. Ignored
            when ``approach_axis`` is provided explicitly.
        tool_spin_deg: Rotation about the tool's approach axis.
        approach_axis: Optional explicit tool approach axis. This is the more
            general robotics interface: specify the end-effector orientation in
            task space and let IK choose the joints.
    """
    position = np.asarray(position, dtype=np.float32).reshape(-1)[:3]
    push_dir = resolve_push_direction(push_direction_xy)
    if approach_axis is not None:
        approaching = _normalize_vector(approach_axis, name="approach_axis")
    elif wrist_orientation == "vertical":
        approaching = np.array([0.0, 0.0, -1.0], dtype=np.float64)
    elif wrist_orientation == "horizontal":
        approaching = push_dir
    else:
        raise ValueError(
            f"Unsupported wrist_orientation={wrist_orientation!r}; expected 'vertical' or 'horizontal'"
        )

    # Prefer a closing axis induced by the push direction when possible, but
    # fall back to a stable world-frame axis when the tool axis is parallel to
    # the push direction.
    closing_base = _orthogonal_unit_vector(approaching, push_dir)

    closing = rotate_vector_about_axis(
        closing_base, approaching, np.deg2rad(float(tool_spin_deg))
    )
    pose = agent.build_grasp_pose(
        np.asarray(approaching, dtype=np.float32),
        np.asarray(closing, dtype=np.float32),
        position,
    )
    quat = np.asarray(pose.q, dtype=np.float32).reshape(-1)[:4]
    return sapien.Pose(position, quat)


def _pose_matrix(position, quaternion):
    """Return a homogeneous transform matrix from pose components."""
    pose = sapien.Pose(
        np.asarray(position, dtype=np.float64).reshape(-1)[:3],
        np.asarray(quaternion, dtype=np.float64).reshape(-1)[:4],
    )
    return np.asarray(pose.to_transformation_matrix(), dtype=np.float64)


def _sample_collision_shape_points(shape):
    """Return representative geometry points in the collision-shape frame."""
    if hasattr(shape, "get_half_size"):
        half_size = np.asarray(shape.get_half_size(), dtype=np.float64).reshape(-1)[:3]
        return np.array(
            [
                [sx, sy, sz]
                for sx in (-half_size[0], half_size[0])
                for sy in (-half_size[1], half_size[1])
                for sz in (-half_size[2], half_size[2])
            ],
            dtype=np.float64,
        )

    if hasattr(shape, "get_vertices"):
        vertices = np.asarray(shape.get_vertices(), dtype=np.float64)
        scale = np.asarray(shape.get_scale(), dtype=np.float64).reshape(-1)
        if scale.size >= 3:
            vertices = vertices * scale[:3]
        return vertices

    if hasattr(shape, "get_radius") and hasattr(shape, "get_half_length"):
        radius = float(shape.get_radius())
        half_length = float(shape.get_half_length())
        return np.array(
            [
                [-half_length, 0.0, 0.0],
                [half_length, 0.0, 0.0],
                [0.0, radius, 0.0],
                [0.0, -radius, 0.0],
                [0.0, 0.0, radius],
                [0.0, 0.0, -radius],
            ],
            dtype=np.float64,
        )

    if hasattr(shape, "get_radius"):
        radius = float(shape.get_radius())
        return np.array(
            [
                [radius, 0.0, 0.0],
                [-radius, 0.0, 0.0],
                [0.0, radius, 0.0],
                [0.0, -radius, 0.0],
                [0.0, 0.0, radius],
                [0.0, 0.0, -radius],
            ],
            dtype=np.float64,
        )

    raise TypeError(f"Unsupported collision shape type: {type(shape).__name__}")


def get_gripper_collision_points_in_tcp(agent):
    """Sample the gripper collision geometry in the TCP frame."""
    from taskbench.skills.robot_config import _discover_config, _BUILTIN_CONFIGS

    robot_config = _discover_config(agent.uid)
    if robot_config is None:
        robot_config = _BUILTIN_CONFIGS.get(agent.uid)
    if robot_config is None:
        raise KeyError(f"No RobotConfig for robot {agent.uid!r}")

    links_by_name = {link.name: link for link in agent.robot.get_links()}
    if agent.ee_link_name not in links_by_name:
        raise KeyError(f"Robot is missing ee link {agent.ee_link_name!r}")

    tcp_link = links_by_name[agent.ee_link_name]
    tcp_world = _pose_matrix(tcp_link.pose.p[0], tcp_link.pose.q[0])
    tcp_world_inv = np.linalg.inv(tcp_world)

    tcp_points = []
    for link_name in sorted(robot_config.gripper_link_names):
        link = links_by_name.get(link_name)
        if link is None:
            continue
        world_link = _pose_matrix(link.pose.p[0], link.pose.q[0])
        if not hasattr(link, "_objs") or not link._objs:
            raise AttributeError(
                f"Link {link_name!r} has no '_objs' attribute. "
                "This uses a private ManiSkill/SAPIEN API — check your ManiSkill version."
            )
        body = link._objs[0]
        for shape in body.get_collision_shapes():
            local_shape = shape.get_local_pose()
            link_shape = _pose_matrix(local_shape.p, local_shape.q)
            tcp_shape = tcp_world_inv @ world_link @ link_shape
            points = _sample_collision_shape_points(shape)
            points_h = np.concatenate(
                [points, np.ones((len(points), 1), dtype=np.float64)], axis=1
            )
            tcp_points.append((tcp_shape @ points_h.T).T[:, :3])

    if not tcp_points:
        raise RuntimeError("No collision geometry found for gripper links")
    return np.concatenate(tcp_points, axis=0)


def tcp_height_for_table_clearance(
    agent,
    push_direction_xy,
    *,
    wrist_orientation="vertical",
    tool_spin_deg=0.0,
    approach_axis=None,
    table_z=0.0,
    table_clearance=0.0,
):
    """Return the TCP z needed to keep the gripper above the table plane."""
    pose = build_push_pose(
        agent,
        [0.0, 0.0, 0.0],
        push_direction_xy,
        wrist_orientation=wrist_orientation,
        tool_spin_deg=tool_spin_deg,
        approach_axis=approach_axis,
    )
    rotation = _pose_matrix([0.0, 0.0, 0.0], pose.q)[:3, :3]
    tcp_points = get_gripper_collision_points_in_tcp(agent)
    min_z_rel = float((rotation @ tcp_points.T).T[:, 2].min())
    return float(table_z + table_clearance - min_z_rel)


def tcp_height_for_pose_clearance(
    agent,
    tcp_quat_wxyz,
    *,
    plane_z=0.0,
    plane_clearance=0.0,
):
    """Return the TCP z needed to keep gripper collision geometry above a plane."""
    quat = np.asarray(tcp_quat_wxyz, dtype=np.float32).reshape(-1)[:4]
    rotation = _pose_matrix([0.0, 0.0, 0.0], quat)[:3, :3]
    tcp_points = get_gripper_collision_points_in_tcp(agent)
    min_z_rel = float((rotation @ tcp_points.T).T[:, 2].min())
    return float(plane_z + plane_clearance - min_z_rel)


def make_linear_push_plan(
    agent,
    *,
    contact_position,
    push_distance,
    push_direction=None,
    push_angle_deg=None,
    contact_height=None,
    wrist_orientation="vertical",
    tool_spin_deg=0.0,
    approach_axis=None,
    approach_gap=0.05,
    hover_height=None,
    staging_height=None,
    staging_backoff=0.0,
    staging_wrist_orientation=None,
    staging_tool_spin_deg=None,
    staging_approach_axis=None,
):
    """Build a straight-line tabletop push from task-space parameters.

    This is the intended abstraction for most push tasks: choose a contact
    point, heading, distance, contact height, and tool orientation. The robot
    joints are solved later by IK/planning.
    """
    push_dir = resolve_push_direction(
        push_direction, push_angle_deg=push_angle_deg
    )

    contact_position = np.asarray(contact_position, dtype=np.float64).reshape(-1)
    if contact_position.size < 2:
        raise ValueError("contact_position must have at least 2 components")
    if contact_height is None:
        if contact_position.size < 3:
            raise ValueError(
                "contact_height is required when contact_position has only x/y"
            )
        contact_height = float(contact_position[2])
    contact_p = np.array(
        [contact_position[0], contact_position[1], float(contact_height)],
        dtype=np.float32,
    )
    approach_p = np.array(
        [
            contact_p[0] - push_dir[0] * float(approach_gap),
            contact_p[1] - push_dir[1] * float(approach_gap),
            contact_p[2],
        ],
        dtype=np.float32,
    )
    push_p = np.array(
        [
            contact_p[0] + push_dir[0] * float(push_distance),
            contact_p[1] + push_dir[1] * float(push_distance),
            contact_p[2],
        ],
        dtype=np.float32,
    )

    if staging_wrist_orientation is None:
        staging_wrist_orientation = wrist_orientation
    if staging_tool_spin_deg is None:
        staging_tool_spin_deg = tool_spin_deg
    if staging_approach_axis is None:
        staging_approach_axis = approach_axis

    hover_pose = None
    if hover_height is not None:
        hover_p = np.array(
            [approach_p[0], approach_p[1], float(hover_height)],
            dtype=np.float32,
        )
        hover_pose = build_push_pose(
            agent,
            hover_p,
            push_dir,
            wrist_orientation=wrist_orientation,
            tool_spin_deg=tool_spin_deg,
            approach_axis=approach_axis,
        )

    staging_pose = None
    if staging_height is not None:
        staging_p = np.array(
            [
                approach_p[0] - push_dir[0] * float(staging_backoff),
                approach_p[1] - push_dir[1] * float(staging_backoff),
                float(staging_height),
            ],
            dtype=np.float32,
        )
        if hover_pose is not None and np.allclose(staging_p, hover_pose.p):
            staging_pose = None
        else:
            staging_pose = build_push_pose(
                agent,
                staging_p,
                push_dir,
                wrist_orientation=staging_wrist_orientation,
                tool_spin_deg=staging_tool_spin_deg,
                approach_axis=staging_approach_axis,
            )

    approach_pose = build_push_pose(
        agent,
        approach_p,
        push_dir,
        wrist_orientation=wrist_orientation,
        tool_spin_deg=tool_spin_deg,
        approach_axis=approach_axis,
    )
    push_pose = build_push_pose(
        agent,
        push_p,
        push_dir,
        wrist_orientation=wrist_orientation,
        tool_spin_deg=tool_spin_deg,
        approach_axis=approach_axis,
    )

    return LinearPushPlan(
        approach_pose=approach_pose,
        push_pose=push_pose,
        hover_pose=hover_pose,
        staging_pose=staging_pose,
        contact_position=contact_p.astype(np.float64),
        approach_position=approach_p.astype(np.float64),
        push_position=push_p.astype(np.float64),
        push_direction=push_dir.astype(np.float64),
        wrist_orientation=wrist_orientation,
        tool_spin_deg=float(tool_spin_deg),
    )


def get_arm_controller(env):
    """Return the active arm controller if the env exposes one."""
    raw = _unwrap_env(env)
    controller = raw.agent.controller
    if not hasattr(controller, "controllers"):
        return None
    return controller.controllers.get("arm")


def get_arm_drive_settings(env):
    """Read the active arm controller drive settings.

    Returns both scalar summaries (first-element representative values for
    reporting) and raw config values that may be per-joint arrays.  The
    ``_*_raw`` keys preserve the original config type so callers can scale
    proportionally without destroying per-joint differentiation.
    """
    arm_controller = get_arm_controller(env)
    if arm_controller is None:
        return None
    cfg = arm_controller.config
    return {
        "stiffness": _drive_value_as_scalar(cfg.stiffness, name="stiffness"),
        "damping": _drive_value_as_scalar(cfg.damping, name="damping"),
        "force_limit": _drive_value_as_scalar(cfg.force_limit, name="force_limit"),
        "_stiffness_raw": cfg.stiffness,
        "_damping_raw": cfg.damping,
        "_force_limit_raw": cfg.force_limit,
    }


def set_arm_drive_settings(env, *, stiffness=None, damping=None, force_limit=None):
    """Update the active arm controller drive settings in-place.

    Accepts scalars or per-joint arrays/tuples.  ``set_drive_property()``
    uses ``np.broadcast_to`` internally so both forms work.
    """
    arm_controller = get_arm_controller(env)
    if arm_controller is None:
        raise RuntimeError("Active control mode does not expose an arm controller")
    if stiffness is not None:
        arm_controller.config.stiffness = stiffness
    if damping is not None:
        arm_controller.config.damping = damping
    if force_limit is not None:
        arm_controller.config.force_limit = force_limit
    arm_controller.set_drive_property()


def build_action(env, qpos, gripper_state, qvel=None):
    """Build an action array from joint positions and gripper state.

    Handles ``pd_joint_pos`` vs ``pd_joint_pos_vel`` control modes.

    Args:
        qvel: Joint velocities for ``pd_joint_pos_vel`` mode.
            Defaults to zero if not provided.
    """
    control_mode = env.unwrapped.control_mode
    if control_mode == "pd_joint_pos_vel":
        if qvel is None:
            qvel = qpos * 0
        return np.hstack([qpos, qvel, gripper_state])
    return np.hstack([qpos, gripper_state])


def sapien_to_mplib_pose(pose: sapien.Pose) -> mplib.pymp.Pose:
    """Convert a SAPIEN Pose to an mplib Pose (handles batched tensors)."""
    p = np.asarray(pose.p, dtype=np.float64).flatten()[:3]
    q = np.asarray(pose.q, dtype=np.float64).flatten()[:4]
    return mplib.pymp.Pose(p=p, q=q)


def _add_table_collision(env, planner):
    """Add the table surface as a point cloud collision object."""
    import sapien.physx as physx

    raw = env.unwrapped
    for actor in raw.scene.get_all_actors():
        if "table" not in actor.name:
            continue
        comp = actor.find_component_by_type(physx.PhysxRigidDynamicComponent)
        if comp is None:
            comp = actor.find_component_by_type(physx.PhysxRigidStaticComponent)
        if comp is None:
            continue
        shape = comp.get_collision_shapes()[0]
        half = np.asarray(shape.half_size, dtype=np.float64)
        # Table top in world frame
        world_pose = actor.pose * shape.get_local_pose()
        center = np.asarray(world_pose.p, dtype=np.float64).flatten()
        top_z = center[2] + half[2]
        # Place points slightly below the true surface so that grasps
        # near the table remain feasible (the planner inflates obstacles
        # by the collision margin, so exact z=0 blocks nearby IK).
        table_z = top_z - 0.02
        # Generate grid of points on table surface
        xs = np.linspace(center[0] - half[0], center[0] + half[0], 50)
        ys = np.linspace(center[1] - half[1], center[1] + half[1], 50)
        xx, yy = np.meshgrid(xs, ys)
        zz = np.full_like(xx, table_z)
        points = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=-1)
        planner.update_point_cloud(points, resolution=0.02, name="table")
        logger.debug(
            "Added table collision: %d points at z=%.4f", len(points), top_z
        )
        return True
    return False


def add_collision_boxes(planner, boxes, resolution=0.01):
    """Add box obstacles to the planner as point clouds.

    Args:
        planner: mplib.Planner instance.
        boxes: List of (name, center, half_size) tuples, e.g. from
            ``env.unwrapped.get_collision_boxes()``.
        resolution: Point spacing on box surfaces (meters).
    """
    for name, center, half_size in boxes:
        center = np.asarray(center, dtype=np.float64)
        hs = np.asarray(half_size, dtype=np.float64)
        points = _box_surface_points(center, hs, resolution)
        planner.update_point_cloud(points, resolution=resolution, name=name)
        logger.debug("Added collision box '%s': %d points", name, len(points))


def _add_env_collision_boxes(env, planner, *, resolution=0.02):
    """Add static env collision boxes when the env exposes them."""
    raw = env.unwrapped
    if not hasattr(raw, "get_collision_boxes"):
        return False
    boxes = raw.get_collision_boxes()
    if not boxes:
        return False
    add_collision_boxes(planner, boxes, resolution=resolution)
    return True


def _box_surface_points(center, half_size, res):
    """Generate a point cloud covering the 6 faces of an axis-aligned box."""
    cx, cy, cz = center
    hx, hy, hz = half_size
    faces = []

    # +/- X faces
    ys = np.arange(cy - hy, cy + hy + res, res)
    zs = np.arange(cz - hz, cz + hz + res, res)
    yy, zz = np.meshgrid(ys, zs)
    for sign in [+1, -1]:
        xx = np.full_like(yy, cx + sign * hx)
        faces.append(np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=-1))

    # +/- Y faces
    xs = np.arange(cx - hx, cx + hx + res, res)
    zs = np.arange(cz - hz, cz + hz + res, res)
    xx, zz = np.meshgrid(xs, zs)
    for sign in [+1, -1]:
        yy = np.full_like(xx, cy + sign * hy)
        faces.append(np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=-1))

    # +/- Z faces
    xs = np.arange(cx - hx, cx + hx + res, res)
    ys = np.arange(cy - hy, cy + hy + res, res)
    xx, yy = np.meshgrid(xs, ys)
    for sign in [+1, -1]:
        zz = np.full_like(xx, cz + sign * hz)
        faces.append(np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=-1))

    return np.concatenate(faces, axis=0)


def setup_planner(env, robot_config: RobotConfig) -> mplib.Planner:
    """Create an mplib Planner from the env's robot.

    Args:
        env: Gym env (num_envs=1, sim_backend="cpu").
        robot_config: Robot-specific constants (move group, etc.).
    """
    raw = env.unwrapped
    agent = raw.agent
    robot = agent.robot

    link_names = [link.get_name() for link in robot.get_links()]
    joint_names = [joint.get_name() for joint in robot.get_active_joints()]

    planner = mplib.Planner(
        urdf=agent.urdf_path,
        srdf=agent.urdf_path.replace(".urdf", ".srdf"),
        user_link_names=link_names,
        user_joint_names=joint_names,
        move_group=robot_config.move_group,
    )

    base_pose = sapien_to_mplib_pose(agent.robot.pose)
    planner.set_base_pose(base_pose)

    planner.joint_vel_limits = np.asarray(planner.joint_vel_limits) * 0.9
    planner.joint_acc_limits = np.asarray(planner.joint_acc_limits) * 0.9

    table_added = _add_table_collision(env, planner)
    boxes_added = _add_env_collision_boxes(env, planner)
    if not table_added and not boxes_added:
        logger.warning("No static collision geometry found for planner setup")

    return planner


def plan_to_pose(
    env,
    planner,
    pose,
    *,
    start_qpos=None,
    time_step_scale=1.0,
    allow_pose_planner_fallback=False,
    pose_planning_time=1.0,
):
    """Plan a motion to ``pose`` from ``start_qpos``.

    The fast path uses ``plan_screw()``. When ``allow_pose_planner_fallback``
    is enabled and the screw planner fails, this falls back to the more
    general ``plan_pose()`` search in mplib.
    """
    time_step_scale = float(time_step_scale)
    if time_step_scale <= 0:
        raise ValueError("time_step_scale must be > 0")

    goal = sapien_to_mplib_pose(pose)
    if start_qpos is None:
        start_qpos = env.unwrapped.agent.robot.get_qpos().cpu().numpy()[0]
    current_qpos = np.asarray(start_qpos, dtype=np.float64)
    time_step = env.unwrapped.control_timestep * time_step_scale

    result = planner.plan_screw(
        goal,
        current_qpos,
        time_step=time_step,
    )
    if result["status"] == "Success":
        return result

    if not allow_pose_planner_fallback:
        logger.warning("plan_screw failed: %s", result["status"])
        return None

    logger.info("plan_screw failed (%s); falling back to plan_pose", result["status"])
    if current_qpos.shape[0] != planner.joint_limits.shape[0]:
        current_qpos = planner.pad_move_group_qpos(current_qpos.copy())
    fallback = planner.plan_pose(
        goal,
        current_qpos,
        time_step=time_step,
        planning_time=float(pose_planning_time),
        simplify=True,
    )
    if fallback["status"] != "Success":
        logger.warning(
            "plan_pose fallback failed after screw failure: %s",
            fallback["status"],
        )
        return None
    return fallback


def _get_robot_contacts(env, robot_link_names=None):
    """Return contacts between robot links and non-robot entities."""
    raw = _unwrap_env(env)
    all_robot_link_names = {link.get_name() for link in raw.agent.robot.get_links()}
    monitored_links = (
        all_robot_link_names if robot_link_names is None else set(robot_link_names)
    )

    results = []
    for contact in raw.scene.px.get_contacts():
        names = [contact.bodies[i].entity.name for i in range(2)]
        for idx in range(2):
            if (
                names[idx] in monitored_links
                and names[1 - idx] not in all_robot_link_names
            ):
                results.append((contact, names[idx], names[1 - idx]))
    return results


def get_robot_contact_violation(
    env,
    *,
    allowed_contact_links=None,
    force_threshold=0.01,
    robot_link_names=None,
):
    """Return the strongest disallowed robot contact above threshold, if any."""
    allowed_contact_links = set(allowed_contact_links or ())
    strongest = None
    for contact, link_name, other in _get_robot_contacts(
        env, robot_link_names=robot_link_names
    ):
        if link_name in allowed_contact_links:
            continue
        force = sum(np.linalg.norm(pt.impulse) for pt in contact.points)
        force /= env.unwrapped.control_timestep
        if force <= float(force_threshold):
            continue
        if strongest is None or force > strongest["force"]:
            strongest = {
                "link_name": link_name,
                "entity_name": other,
                "force": float(force),
            }
    return strongest


def _get_gripper_contacts(env, robot_config: RobotConfig):
    """Return contacts involving gripper links (hand + fingers)."""
    return _get_robot_contacts(env, robot_link_names=robot_config.gripper_link_names)


def get_gripper_contact_summary(env, robot_config: RobotConfig | None = None):
    """Estimate gripper contact force from PhysX impulses for the current step.

    The returned forces are per-step estimates in Newtons computed as
    ``|impulse| / control_timestep`` for each active contact manifold.
    """
    raw = _unwrap_env(env)
    if robot_config is None:
        from taskbench.skills.robot_config import get_robot_config

        robot_config = get_robot_config(env)

    peak_force = 0.0
    total_force = 0.0
    other_entities = set()

    for contact, _finger, other in _get_gripper_contacts(raw, robot_config):
        force = sum(
            np.linalg.norm(pt.impulse) for pt in contact.points
        ) / raw.control_timestep
        if force <= 0:
            continue
        peak_force = max(peak_force, float(force))
        total_force += float(force)
        other_entities.add(other)

    return {
        "peak_force": peak_force,
        "total_force": total_force,
        "other_entities": tuple(sorted(other_entities)),
    }


def get_robot_contact_summary(
    env,
    *,
    robot_link_names=None,
    entity_filter=None,
):
    """Summarize current robot contacts as peak/total force and entities."""
    raw = _unwrap_env(env)
    peak_force = 0.0
    total_force = 0.0
    other_entities = set()

    for contact, _link_name, other in _get_robot_contacts(
        raw, robot_link_names=robot_link_names
    ):
        if entity_filter is not None and not entity_filter(other):
            continue
        force = sum(
            np.linalg.norm(pt.impulse) for pt in contact.points
        ) / raw.control_timestep
        if force <= 0:
            continue
        peak_force = max(peak_force, float(force))
        total_force += float(force)
        other_entities.add(other)

    return {
        "peak_force": peak_force,
        "total_force": total_force,
        "other_entities": tuple(sorted(other_entities)),
    }


def _record_motion_diagnostics(env, robot_config: RobotConfig, diagnostics):
    """Accumulate per-step effort/contact telemetry during execution."""
    if diagnostics is None:
        return

    raw = env.unwrapped
    qf = raw.agent.robot.get_qf()[0].detach().cpu().numpy()
    joint_load = raw.agent.robot.get_link_incoming_joint_forces().detach().cpu().numpy()
    contact = get_gripper_contact_summary(raw, robot_config)

    diagnostics.setdefault("joint_effort_samples", []).append(qf.copy())
    diagnostics.setdefault("joint_effort_l2_samples", []).append(
        float(np.linalg.norm(qf))
    )
    diagnostics.setdefault("joint_load_samples", []).append(joint_load.copy())
    diagnostics.setdefault("joint_load_l2_samples", []).append(
        float(np.linalg.norm(joint_load))
    )
    diagnostics.setdefault("contact_force_samples", []).append(
        float(contact["peak_force"])
    )
    diagnostics.setdefault("contact_entities", set()).update(
        contact["other_entities"]
    )


def follow_path(env, result, gripper_state, robot_config: RobotConfig,
                refine_steps=0, monitor_contacts=False, diagnostics=None,
                allowed_contact_links=None, control_hook=None, step_callback=None,
                contact_force_threshold=0.01, stop_hook=None):
    """Execute a planned path, returning the last step result.

    Args:
        robot_config: Robot-specific constants (for contact detection).
        monitor_contacts: If True, log warnings when gripper fingers
            contact non-robot objects during trajectory execution.
        diagnostics: Optional dict populated with per-step contact-force
            and joint-effort samples during execution.
        allowed_contact_links: Optional set of robot link names that may
            contact non-robot objects during the motion. When
            ``monitor_contacts=True``, contacts on other robot links abort.
        control_hook: Optional callable receiving
            ``(progress, step_idx, num_steps)`` before each control step.
        step_callback: Optional callable invoked after each env.step()
            (e.g. ``env.render_human`` for live viewer updates).
        contact_force_threshold: Minimum contact magnitude treated as a
            disallowed collision when ``monitor_contacts=True``.
        stop_hook: Optional callable receiving
            ``(env, robot_config, diagnostics, step_idx, num_steps)`` and
            returning a string reason to end the motion early without
            treating it as a failure.
    """
    if allowed_contact_links is None:
        allowed_contact_links = set()
    else:
        allowed_contact_links = set(allowed_contact_links)
    if monitor_contacts:
        violation = get_robot_contact_violation(
            env,
            allowed_contact_links=allowed_contact_links,
            force_threshold=contact_force_threshold,
        )
        if violation is not None:
            if diagnostics is not None:
                diagnostics["failure_reason"] = "contact_violation_start"
                diagnostics["collision_link"] = violation["link_name"]
                diagnostics["collision_entity"] = violation["entity_name"]
                diagnostics["collision_force"] = violation["force"]
            logger.warning(
                "Starting move in contact: %s -> %s (%.2f N), aborting",
                violation["link_name"],
                violation["entity_name"],
                violation["force"],
            )
            return None
    n_step = result["position"].shape[0]
    if n_step == 0:
        logger.warning("Planned path has zero waypoints; treating as planning failure")
        return None
    has_velocity = "velocity" in result
    for i in range(n_step + refine_steps):
        idx = min(i, n_step - 1)
        progress = float(idx) / max(n_step - 1, 1)
        if control_hook is not None:
            control_hook(progress, idx, n_step)
        qpos = result["position"][idx]
        qvel = result["velocity"][idx] if has_velocity else None
        action = build_action(env, qpos, gripper_state, qvel=qvel)
        obs, reward, terminated, truncated, info = env.step(action)

        if step_callback is not None:
            step_callback()

        _record_motion_diagnostics(env, robot_config, diagnostics)

        if stop_hook is not None:
            stop_reason = stop_hook(env, robot_config, diagnostics, i, n_step)
            if stop_reason:
                if diagnostics is not None:
                    diagnostics["stop_reason"] = str(stop_reason)
                    diagnostics["stop_step"] = int(i)
                return obs, reward, terminated, truncated, info

        if monitor_contacts:
            violation = get_robot_contact_violation(
                env,
                allowed_contact_links=allowed_contact_links,
                force_threshold=contact_force_threshold,
            )
            if violation is not None:
                if diagnostics is not None:
                    diagnostics["failure_reason"] = "contact_violation"
                    diagnostics["collision_link"] = violation["link_name"]
                    diagnostics["collision_entity"] = violation["entity_name"]
                    diagnostics["collision_force"] = violation["force"]
                logger.warning(
                    "Collision at step %d/%d: %s -> %s (%.2f N), aborting",
                    i,
                    n_step,
                    violation["link_name"],
                    violation["entity_name"],
                    violation["force"],
                )
                return None
    return obs, reward, terminated, truncated, info


def actuate_gripper(env, planner, gripper_state, steps=6, step_callback=None):
    """Open or close the gripper for a number of steps."""
    if steps <= 0:
        return None, None, None, None, None
    robot = env.unwrapped.agent.robot
    qpos = robot.get_qpos()[0, : len(planner.joint_vel_limits)].cpu().numpy()
    for _ in range(steps):
        action = build_action(env, qpos, gripper_state)
        obs, reward, terminated, truncated, info = env.step(action)
        if step_callback is not None:
            step_callback()
    return obs, reward, terminated, truncated, info


def hold_current_pose(env, planner, gripper_state, steps=1, step_callback=None):
    """Hold the robot at its current joint target for a number of steps."""
    return actuate_gripper(
        env,
        planner,
        gripper_state,
        steps=steps,
        step_callback=step_callback,
    )


def attach_object(planner, size, pose=None):
    """Tell the planner a box is attached to the end effector.

    Args:
        planner: mplib.Planner instance.
        size: (3,) array-like — full extents (x, y, z) of the box.
        pose: mplib.pymp.Pose — relative pose from the end-effector link
            to the object center.  Defaults to identity (centered on TCP).
    """
    if pose is None:
        pose = mplib.pymp.Pose()
    planner.update_attached_box(size, pose)
    logger.debug("Attached box (%.3f, %.3f, %.3f) to end effector", *size)


def detach_object(planner):
    """Remove the attached object from the planner."""
    planner.detach_object("attached_geom", also_remove=True)
    logger.debug("Detached object from end effector")


def move_to_pose(env, planner, pose, gripper_state, robot_config: RobotConfig,
                 dry_run=False, monitor_contacts=False, diagnostics=None,
                 allowed_contact_links=None, control_hook=None,
                 step_callback=None, time_step_scale=1.0,
                 refine_steps=0,
                 contact_force_threshold=0.01,
                 stop_hook=None,
                 allow_pose_planner_fallback=False,
                 pose_planning_time=1.0):
    """Plan and execute a straight-line motion to target pose.

    Uses ``plan_screw()`` (Cartesian straight-line interpolation).

    Returns None on planning failure, the plan dict if dry_run=True,
    or the last (obs, reward, terminated, truncated, info) tuple.
    When ``diagnostics`` is provided it is populated in-place with
    contact-force and joint-effort samples recorded during execution.
    ``control_hook`` can be used to adjust controller settings online
    during trajectory execution. When ``monitor_contacts=True``, the move is
    also rejected if it starts already in disallowed contact.
    """
    time_step_scale = float(time_step_scale)
    if time_step_scale <= 0:
        raise ValueError("time_step_scale must be > 0")
    result = plan_to_pose(
        env,
        planner,
        pose,
        time_step_scale=time_step_scale,
        allow_pose_planner_fallback=allow_pose_planner_fallback,
        pose_planning_time=pose_planning_time,
    )
    if result is None:
        return None
    if dry_run:
        return result
    return follow_path(env, result, gripper_state, robot_config,
                       refine_steps=refine_steps,
                       monitor_contacts=monitor_contacts,
                       diagnostics=diagnostics,
                       allowed_contact_links=allowed_contact_links,
                       control_hook=control_hook,
                       step_callback=step_callback,
                       contact_force_threshold=contact_force_threshold,
                       stop_hook=stop_hook)
