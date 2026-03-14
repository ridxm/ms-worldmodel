"""Low-level motion planning helpers using mplib 0.2.x.

All functions are module-level (no class state) and operate on a raw
gym env with ``num_envs=1`` and ``sim_backend="cpu"``.

Robot-specific constants (move group, finger length, etc.) come from
``RobotConfig`` — see ``taskbench.skills.robot_config``.
"""

from __future__ import annotations

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
        # Exclude points near the robot base to avoid false collisions.
        base_p = np.asarray(raw.agent.robot.pose.p, dtype=np.float64).flatten()[:3]
        dist_sq = (points[:, 0] - base_p[0]) ** 2 + (points[:, 1] - base_p[1]) ** 2
        points = points[dist_sq > 0.15**2]
        planner.update_point_cloud(points, resolution=0.02, name="table")
        logger.debug(
            "Added table collision: %d points at z=%.4f", len(points), top_z
        )
        return
    logger.warning("No table actor found in scene")


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


def _get_srdf_path(agent) -> str:
    """Get or auto-generate an SRDF file for the agent's URDF.

    For MJCF-based agents whose URDF is auto-exported to a tempfile,
    there is no adjacent SRDF. In that case, auto-generate one using
    mplib's SRDF exporter and write it next to the URDF.
    """
    import os
    srdf_path = agent.urdf_path.replace(".urdf", ".srdf")
    if os.path.exists(srdf_path):
        return srdf_path

    # Auto-generate SRDF from the loaded articulation
    import re as _re
    import xml.etree.ElementTree as ET
    from mplib.sapien_utils.srdf_exporter import export_srdf_xml

    srdf_xml = export_srdf_xml(agent.robot._objs[0])
    srdf_str = ET.tostring(srdf_xml, encoding="unicode")
    # Strip scene prefix from link names to match the cleaned URDF.
    # The exporter prefixes names with "scene-{N}-{agent_uid}_".
    uid = agent.uid
    srdf_str = _re.sub(rf'scene-\d+-{_re.escape(uid)}_', '', srdf_str)
    with open(srdf_path, "w") as f:
        f.write(srdf_str)
    logger.info("Auto-generated SRDF at %s", srdf_path)
    return srdf_path


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

    srdf_path = _get_srdf_path(agent)

    planner = mplib.Planner(
        urdf=agent.urdf_path,
        srdf=srdf_path,
        user_link_names=link_names,
        user_joint_names=joint_names,
        move_group=robot_config.move_group,
    )

    base_pose = sapien_to_mplib_pose(agent.robot.pose)
    planner.set_base_pose(base_pose)

    planner.joint_vel_limits = np.asarray(planner.joint_vel_limits) * 0.9
    planner.joint_acc_limits = np.asarray(planner.joint_acc_limits) * 0.9

    _add_table_collision(env, planner)

    return planner


def _get_gripper_contacts(env, robot_config: RobotConfig):
    """Return contacts involving gripper links (hand + fingers).

    Returns a list of ``(contact, gripper_link_name, other_entity_name)``
    tuples for any contact where a gripper link touches a non-robot entity.
    """
    raw = env.unwrapped
    robot_link_names = {link.get_name() for link in raw.agent.robot.get_links()}
    gripper_links = robot_config.gripper_link_names

    results = []
    for contact in raw.scene.px.get_contacts():
        names = [contact.bodies[i].entity.name for i in range(2)]
        for idx in range(2):
            if names[idx] in gripper_links and names[1 - idx] not in robot_link_names:
                results.append((contact, names[idx], names[1 - idx]))
    return results


def follow_path(env, result, gripper_state, robot_config: RobotConfig,
                refine_steps=0, monitor_contacts=False, step_callback=None):
    """Execute a planned path, returning the last step result.

    Args:
        robot_config: Robot-specific constants (for contact detection).
        monitor_contacts: If True, log warnings when gripper fingers
            contact non-robot objects during trajectory execution.
        step_callback: Optional callable invoked after each env.step()
            (e.g. ``env.render_human`` for live viewer updates).
    """
    n_step = result["position"].shape[0]
    has_velocity = "velocity" in result
    for i in range(n_step + refine_steps):
        idx = min(i, n_step - 1)
        qpos = result["position"][idx]
        qvel = result["velocity"][idx] if has_velocity else None
        action = build_action(env, qpos, gripper_state, qvel=qvel)
        obs, reward, terminated, truncated, info = env.step(action)

        if step_callback is not None:
            step_callback()

        if monitor_contacts:
            for contact, finger, other in _get_gripper_contacts(env, robot_config):
                force = sum(
                    np.linalg.norm(pt.impulse) for pt in contact.points
                ) / env.unwrapped.control_timestep
                if force > 0.01:  # low threshold to catch brushing contacts
                    logger.warning(
                        "Collision at step %d/%d: %s -> %s (%.2f N), aborting",
                        i, n_step, finger, other, force,
                    )
                    return None
    return obs, reward, terminated, truncated, info


def actuate_gripper(env, planner, gripper_state, steps=6, step_callback=None):
    """Open or close the gripper for a number of steps."""
    robot = env.unwrapped.agent.robot
    qpos = robot.get_qpos()[0, : len(planner.joint_vel_limits)].cpu().numpy()
    for _ in range(steps):
        action = build_action(env, qpos, gripper_state)
        obs, reward, terminated, truncated, info = env.step(action)
        if step_callback is not None:
            step_callback()
    return obs, reward, terminated, truncated, info


def attach_object(planner, size, pose=None):
    """Tell the planner a box is attached to the end effector.

    Args:
        planner: mplib.Planner instance.
        size: (3,) array-like — full extents (x, y, z) of the box.
            Clamped to a maximum of 0.06m per axis to avoid false
            collisions with the robot's own links (OBB can overestimate
            due to object rotation).
        pose: mplib.pymp.Pose — relative pose from the end-effector link
            to the object center.  Defaults to identity (centered on TCP).
    """
    if pose is None:
        pose = mplib.pymp.Pose()
    size = np.clip(np.asarray(size, dtype=np.float64), 0, 0.06)
    planner.update_attached_box(size, pose)
    logger.debug("Attached box (%.3f, %.3f, %.3f) to end effector", *size)


def detach_object(planner):
    """Remove the attached object from the planner."""
    planner.detach_object("attached_geom", also_remove=True)
    logger.debug("Detached object from end effector")


def move_to_pose(env, planner, pose, gripper_state, robot_config: RobotConfig,
                 dry_run=False, monitor_contacts=False, step_callback=None):
    """Plan and execute a straight-line motion to target pose.

    Uses ``plan_screw()`` (Cartesian straight-line interpolation).

    Returns None on planning failure, the plan dict if dry_run=True,
    or the last (obs, reward, terminated, truncated, info) tuple.
    """
    # If the target pose has identity orientation (default), use the current
    # TCP orientation instead.  This avoids forcing a large rotation that
    # hits joint limits or collisions for robots whose TCP frame at rest is
    # far from identity (e.g. UR5).
    raw = env.unwrapped
    if np.allclose(pose.q, [1, 0, 0, 0], atol=1e-6):
        tcp_q = raw.agent.tcp.pose.q.flatten().cpu().numpy()
        pose = sapien.Pose(p=pose.p, q=tcp_q)
    goal = sapien_to_mplib_pose(pose)
    current_qpos = raw.agent.robot.get_qpos().cpu().numpy()[0]
    # Clamp qpos to joint limits — SAPIEN physics can push joints slightly
    # outside URDF limits (e.g. gripper joints), which makes mplib reject.
    limits = np.asarray(planner.joint_limits)
    current_qpos = np.clip(current_qpos, limits[:, 0], limits[:, 1])
    result = planner.plan_screw(
        goal,
        current_qpos,
        time_step=env.unwrapped.control_timestep,
    )
    if result["status"] != "Success":
        logger.warning("plan_screw failed: %s", result["status"])
        return None
    if dry_run:
        return result
    return follow_path(env, result, gripper_state, robot_config,
                       monitor_contacts=monitor_contacts,
                       step_callback=step_callback)
