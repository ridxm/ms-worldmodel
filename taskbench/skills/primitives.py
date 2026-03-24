"""Reusable manipulation skills as composable objects.

Each skill binds shared context (env, planner, robot_config, step_callback)
at construction, exposing only task-specific parameters in ``__call__``:

    pick = Pick(env, planner, robot_config=rc, objects=objects)
    result = pick("cube_1", lift_height=0.1)

Base class ``Skill`` provides the common interface. All skills return a
dataclass result with ``success`` and ``failure_reason`` fields.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import sapien
from transforms3d.euler import euler2quat

from mani_skill.examples.motionplanning.base_motionplanner.utils import (
    compute_grasp_info_by_obb,
    get_actor_obb,
)

from taskbench.skills.motion import (
    PoseLike,
    actuate_gripper,
    attach_object,
    detach_object,
    get_arm_drive_settings,
    get_robot_contact_summary,
    hold_current_pose,
    move_to_pose,
    set_arm_drive_settings,
    to_sapien_pose,
)
from taskbench.skills.robot_config import RobotConfig, get_robot_config

logger = logging.getLogger("taskbench.skills.primitives")


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class SkillResult:
    """Base result for all skills."""
    success: bool
    failure_reason: Optional[str] = None
    step_result: Optional[tuple] = None


@dataclass
class MoveResult(SkillResult):
    contact_link: Optional[str] = None
    contact_entity: Optional[str] = None
    contact_force: float = 0.0


@dataclass
class PickResult(SkillResult):
    grasp_pose: Optional[sapien.Pose] = None
    lift_pose: Optional[sapien.Pose] = None
    obj_size: Optional[np.ndarray] = None


@dataclass
class PlaceResult(SkillResult):
    pass


@dataclass
class PushResult(SkillResult):
    approach_pose: Optional[sapien.Pose] = None
    push_pose: Optional[sapien.Pose] = None
    push_distance: float = 0.0
    planar_push_distance: float = 0.0
    effort_scale_start: float = 1.0
    effort_scale_end: float = 1.0
    arm_force_limit_start: float = 0.0
    arm_force_limit_end: float = 0.0
    arm_force_limit_peak: float = 0.0
    arm_force_limit_mean: float = 0.0
    contact_steps: int = 0
    contact_force_peak: float = 0.0
    contact_force_mean: float = 0.0
    joint_effort_l2_peak: float = 0.0
    joint_effort_l2_mean: float = 0.0
    joint_load_l2_peak: float = 0.0
    joint_load_l2_mean: float = 0.0
    contact_objects: tuple[str, ...] = ()
    executed_push_distance: float = 0.0
    executed_push_fraction: float = 0.0
    cutoff_reason: Optional[str] = None
    shelf_contact_force_peak: float = 0.0
    shelf_contact_force_mean: float = 0.0
    shelf_contact_objects: tuple[str, ...] = ()


# ---------------------------------------------------------------------------
# Base skill
# ---------------------------------------------------------------------------

class Skill(ABC):
    """Base class for manipulation skills.

    Binds shared context (env, planner, robot_config, objects, step_callback)
    so that ``__call__`` only receives task-specific parameters.

    Args:
        env: Gym env (raw or wrapped).
        planner: mplib.Planner instance.
        robot_config: Robot-specific constants. If None, auto-detected
            from the env's agent.
        objects: Dict mapping string names to scene actors.
            Skills that need actors (e.g. Pick) resolve names through this.
        step_callback: Optional callable invoked after each env.step().
    """

    def __init__(self, env, planner, *, robot_config: Optional[RobotConfig] = None,
                 objects: Optional[dict[str, object]] = None,
                 step_callback: Optional[Callable] = None):
        self.env = env
        self.planner = planner
        self.robot_config = robot_config or get_robot_config(env)
        self.objects = objects or {}
        self.step_callback = step_callback

    @abstractmethod
    def __call__(self, *args, **kwargs) -> SkillResult:
        ...


# ---------------------------------------------------------------------------
# Move
# ---------------------------------------------------------------------------

class Move(Skill):
    """Move the arm to a target pose.

    Args (at call time):
        target_pose: PoseLike to move the end effector to.
        gripper_open: Gripper state during motion (default True).
        monitor_contacts: Abort on collision during execution (default True).
        allowed_contact_links: Optional set of robot link names allowed to
            touch external objects during this motion. Typical use is to
            allow only the pusher links during a push phase.
        time_step_scale: Multiplier on the planner/control waypoint spacing.
            Values > 1 execute faster with fewer waypoints.
        refine_steps: Number of extra hold steps at the end of the move to
            let the controller settle at the final target.
        contact_force_threshold: Minimum contact magnitude treated as a
            collision when ``monitor_contacts=True``.
    """

    def __call__(self, target_pose: PoseLike, *, gripper_open=True,
                 monitor_contacts=True, diagnostics=None,
                 allowed_contact_links=None, control_hook=None,
                 stop_hook=None,
                 time_step_scale=1.0,
                 refine_steps=0,
                 contact_force_threshold=0.01,
                 allow_pose_planner_fallback=False,
                 pose_planning_time=1.0) -> MoveResult:
        target_pose = to_sapien_pose(target_pose)
        rc = self.robot_config
        gripper_state = rc.gripper_open if gripper_open else rc.gripper_closed
        local_diagnostics = diagnostics if diagnostics is not None else {}
        res = move_to_pose(self.env, self.planner, target_pose, gripper_state,
                           rc, monitor_contacts=monitor_contacts,
                           diagnostics=local_diagnostics,
                           allowed_contact_links=allowed_contact_links,
                           control_hook=control_hook,
                           stop_hook=stop_hook,
                           step_callback=self.step_callback,
                           time_step_scale=time_step_scale,
                           refine_steps=refine_steps,
                           contact_force_threshold=contact_force_threshold,
                           allow_pose_planner_fallback=allow_pose_planner_fallback,
                           pose_planning_time=pose_planning_time)
        if res is None:
            failure_reason = local_diagnostics.get("failure_reason", "move_plan_failed")
            return MoveResult(
                success=False,
                failure_reason=failure_reason,
                contact_link=local_diagnostics.get("collision_link"),
                contact_entity=local_diagnostics.get("collision_entity"),
                contact_force=float(local_diagnostics.get("collision_force", 0.0)),
            )
        return MoveResult(success=True, step_result=res)


# ---------------------------------------------------------------------------
# Pick
# ---------------------------------------------------------------------------

class Pick(Skill):
    """Grasp an object and lift it.

    Internally: compute grasp from OBB, search rotation candidates,
    reach, approach, close gripper, verify grasp, lift.

    Args (at call time):
        obj_name: String name of the object to grasp (resolved via
            ``self.objects``).
        lift_height: Height above grasp pose to lift to (default 0.1m).
        verify_grasp: Check ``agent.is_grasping()`` after closing (default True).
    """

    def __call__(self, obj_name: str, *, lift_height=0.1,
                 verify_grasp=True) -> PickResult:
        obj = self.objects[obj_name]
        env, planner, rc = self.env, self.planner, self.robot_config
        raw = env.unwrapped
        move = Move(env, planner, robot_config=rc, step_callback=self.step_callback)

        # Compute grasp pose from OBB
        obb = get_actor_obb(obj)
        obj_size = np.asarray(obb.extents, dtype=np.float64)
        approaching = np.array([0, 0, -1])
        target_closing = (
            raw.agent.tcp.pose.to_transformation_matrix()[0, :3, 1].cpu().numpy()
        )
        grasp_info = compute_grasp_info_by_obb(
            obb,
            approaching=approaching,
            target_closing=target_closing,
            depth=rc.finger_length,
        )
        closing, center = grasp_info["closing"], grasp_info["center"]
        grasp_pose = raw.agent.build_grasp_pose(approaching, closing, center)

        # Search 6 rotation candidates for collision-free orientation
        angles = np.array([0, np.pi/6, -np.pi/6, np.pi/3, -np.pi/3, np.pi/2])

        grasp_found = False
        for angle in angles:
            delta_pose = sapien.Pose(q=euler2quat(0, 0, angle))
            candidate = grasp_pose * delta_pose
            res = move_to_pose(env, planner, candidate, rc.gripper_open, rc,
                               dry_run=True)
            if res is None:
                continue
            grasp_pose = candidate
            grasp_found = True
            break

        if not grasp_found:
            logger.warning("Failed to find a valid grasp pose")
            return PickResult(success=False, failure_reason="grasp_plan_failed")

        # Reach: approach from 0.05m behind grasp pose
        reach_pose = grasp_pose * sapien.Pose([0, 0, -0.05])
        result = move(reach_pose)
        if not result.success:
            return PickResult(success=False, failure_reason="reach_failed")

        # Grasp: move to grasp pose
        result = move(grasp_pose)
        if not result.success:
            return PickResult(success=False, failure_reason="grasp_approach_failed")

        # Close gripper
        actuate_gripper(env, planner, rc.gripper_closed,
                        step_callback=self.step_callback)

        # Verify grasp
        if verify_grasp:
            is_holding = raw.agent.is_grasping(obj)
            if not bool(is_holding.cpu().numpy().item()):
                logger.warning("Grasp verification failed")
                return PickResult(success=False,
                                  failure_reason="grasp_verification_failed")

        # Lift (contacts off — gripper is holding the object)
        lift_pose = sapien.Pose([0, 0, lift_height]) * grasp_pose
        result = move(lift_pose, gripper_open=False, monitor_contacts=False)
        if not result.success:
            return PickResult(
                success=False,
                failure_reason="lift_failed",
                grasp_pose=grasp_pose,
            )

        # Tell planner about the held object for collision-aware planning
        attach_object(planner, obj_size)

        return PickResult(
            success=True,
            grasp_pose=grasp_pose,
            lift_pose=lift_pose,
            obj_size=obj_size,
            step_result=result.step_result,
        )


# ---------------------------------------------------------------------------
# Place
# ---------------------------------------------------------------------------

class Place(Skill):
    """Move to target pose, release the held object, and retract upward.

    Args (at call time):
        target_pose: PoseLike where the gripper moves before releasing.
        settling_steps: Steps to let physics settle after release (default 10).
        retract_height: Absolute Z height to retract to after release.
            If None, retracts 0.1m above the release pose.
    """

    def __call__(self, target_pose: PoseLike, *, settling_steps=10,
                 retract_height=None) -> PlaceResult:
        target_pose = to_sapien_pose(target_pose)
        env, planner, rc = self.env, self.planner, self.robot_config
        move = Move(env, planner, robot_config=rc, step_callback=self.step_callback)

        # Move to target pose (contacts off — gripper is holding an object)
        result = move(target_pose, gripper_open=False, monitor_contacts=False)
        if not result.success:
            return PlaceResult(success=False, failure_reason="place_move_failed")

        # Save step_result from the place move — the retract may fail, so
        # this is the last guaranteed-good observation.
        place_step_result = result.step_result

        # Release gripper
        actuate_gripper(env, planner, rc.gripper_open,
                        step_callback=self.step_callback)

        # Object released — remove from planner
        detach_object(planner)

        # Settle
        actuate_gripper(env, planner, rc.gripper_open, steps=settling_steps,
                        step_callback=self.step_callback)

        # Retract: move straight up to clear before next action
        if retract_height is None:
            retract_height = target_pose.p[2] + 0.1
        retract_pose = sapien.Pose(
            [target_pose.p[0], target_pose.p[1], retract_height],
            target_pose.q,
        )
        result = move(retract_pose)
        if not result.success:
            logger.warning("Retract failed, continuing anyway")

        return PlaceResult(
            success=True,
            step_result=result.step_result if result.success else place_step_result,
        )


# ---------------------------------------------------------------------------
# Push
# ---------------------------------------------------------------------------

class Push(Skill):
    """Lift for clearance, close gripper, approach, sweep, lift, open.

    Args (at call time):
        staging_pose: Optional pre-push pose. Useful for vertical pushes that
            stage above the contact point before descending.
        hover_pose: Optional pre-descent pose directly above the approach pose.
            Useful when you want a smooth vertical lowering phase with more
            clearance around nearby objects.
        transit_poses: Optional intermediate free-space waypoints executed
            before the final approach. Useful for breaking a hard insertion
            into shorter planner segments.
        approach_pose: PoseLike to move to before pushing (no contact).
        push_pose: PoseLike to sweep toward using a straight-line Cartesian
            motion (contact expected).
        clearance_height: Height to lift above current position before
            approaching (default 0.1m).
        lift_height: Height to lift above push_pose after pushing (default 0.1m).
        effort_scale: Scalar multiplier on the arm's current drive force limit
            during the push sweep.
        effort_scale_end: Optional end multiplier for a linear ramp across the
            push sweep. If omitted, the commanded effort stays constant.
        min_contact_force: Optional minimum peak contact force (N) required
            for the push to count as successful.
        *_speed_scale: Optional planner/control waypoint spacing multipliers
            for each phase. Values > 1 run faster with fewer waypoints.
        staging_settle_steps: Optional hold steps after the staging move.
            Useful when a later insertion plan is sensitive to small tracking
            error at the staging waypoint.
        transit_allowed_contact_links: Optional set of robot links that may
            touch external objects during intermediate free-space waypoints.
        approach_allowed_contact_links: Optional set of robot links that may
            touch external objects during the final approach move.
    """

    def __call__(self, approach_pose: PoseLike, push_pose: PoseLike, *,
                 staging_pose: PoseLike | None = None,
                 hover_pose: PoseLike | None = None,
                 transit_poses: list[PoseLike] | tuple[PoseLike, ...] | None = None,
                 clearance_height=0.1, lift_height=0.1,
                 effort_scale=1.0, effort_scale_end=None,
                 min_contact_force=0.0,
                 open_gripper_after_push=True,
                 staging_speed_scale=1.0,
                 staging_settle_steps=0,
                 hover_speed_scale=1.0,
                 transit_speed_scale=1.0,
                 clearance_speed_scale=1.0,
                 approach_speed_scale=1.0,
                 push_speed_scale=1.0,
                 lift_speed_scale=1.0,
                 transit_allowed_contact_links=None,
                 approach_allowed_contact_links=None,
                 push_abort_on_contact=True,
                 push_cutoff_force_threshold=None,
                 push_cutoff_entity_substrings=None,
                 free_space_pose_planner_fallback=False,
                 free_space_pose_planning_time=1.0) -> PushResult:
        if staging_pose is not None:
            staging_pose = to_sapien_pose(staging_pose)
        if hover_pose is not None:
            hover_pose = to_sapien_pose(hover_pose)
        if transit_poses is None:
            transit_poses = []
        else:
            transit_poses = [to_sapien_pose(pose) for pose in transit_poses]
        approach_pose = to_sapien_pose(approach_pose)
        push_pose = to_sapien_pose(push_pose)
        env, planner, rc = self.env, self.planner, self.robot_config
        raw = env.unwrapped
        move = Move(env, planner, robot_config=rc, step_callback=self.step_callback)
        if effort_scale_end is None:
            effort_scale_end = effort_scale
        effort_scale = float(effort_scale)
        effort_scale_end = float(effort_scale_end)
        if effort_scale <= 0 or effort_scale_end <= 0:
            return PushResult(success=False, failure_reason="invalid_effort_scale")
        approach_p = np.asarray(approach_pose.p, dtype=np.float64).flatten()[:3]
        push_p = np.asarray(push_pose.p, dtype=np.float64).flatten()[:3]
        push_delta = push_p - approach_p
        push_distance = float(np.linalg.norm(push_delta))
        planar_push_distance = float(np.linalg.norm(push_delta[:2]))
        base_drive = get_arm_drive_settings(env)
        if base_drive is None:
            return PushResult(
                success=False,
                failure_reason="arm_controller_unavailable",
                approach_pose=approach_pose,
                push_pose=push_pose,
                push_distance=push_distance,
                planar_push_distance=planar_push_distance,
            )
        base_force_limit_raw = base_drive["_force_limit_raw"]
        arm_force_limit_start = base_drive["force_limit"] * effort_scale
        arm_force_limit_end = base_drive["force_limit"] * effort_scale_end

        # Common fields for all early-exit PushResults.
        base = dict(
            approach_pose=approach_pose,
            push_pose=push_pose,
            push_distance=push_distance,
            planar_push_distance=planar_push_distance,
            effort_scale_start=effort_scale,
            effort_scale_end=effort_scale_end,
            arm_force_limit_start=arm_force_limit_start,
            arm_force_limit_end=arm_force_limit_end,
        )

        # Optional pre-stage. Useful for vertical pushes that should descend
        # straight down to the contact start pose.
        if staging_pose is not None:
            result = move(
                staging_pose,
                gripper_open=False,
                time_step_scale=staging_speed_scale,
                allow_pose_planner_fallback=free_space_pose_planner_fallback,
                pose_planning_time=free_space_pose_planning_time,
            )
            if not result.success:
                return PushResult(success=False, failure_reason="staging_move_failed", **base)
            if int(staging_settle_steps) > 0:
                hold_current_pose(
                    env,
                    planner,
                    rc.gripper_closed,
                    steps=int(staging_settle_steps),
                    step_callback=self.step_callback,
                )

        if hover_pose is not None:
            result = move(
                hover_pose,
                gripper_open=False,
                time_step_scale=hover_speed_scale,
                allow_pose_planner_fallback=free_space_pose_planner_fallback,
                pose_planning_time=free_space_pose_planning_time,
            )
            if not result.success:
                return PushResult(success=False, failure_reason="hover_move_failed", **base)

        # Lift from current position for clearance when requested.
        if clearance_height > 0:
            tcp_pose = raw.agent.tcp.pose
            tcp_p = np.asarray(tcp_pose.p, dtype=np.float64).flatten()[:3]
            tcp_q = np.asarray(tcp_pose.q, dtype=np.float32).flatten()[:4]
            clearance_pose = sapien.Pose(
                np.array([tcp_p[0], tcp_p[1], tcp_p[2] + clearance_height],
                    dtype=np.float32),
                tcp_q,
            )
            result = move(
                clearance_pose,
                time_step_scale=clearance_speed_scale,
                allow_pose_planner_fallback=free_space_pose_planner_fallback,
                pose_planning_time=free_space_pose_planning_time,
            )
            if not result.success:
                return PushResult(success=False, failure_reason="clearance_lift_failed", **base)

        # Close gripper for the actual push surface after free-space transit.
        actuate_gripper(env, planner, rc.gripper_closed,
                        step_callback=self.step_callback)

        for transit_pose in transit_poses:
            result = move(
                transit_pose,
                gripper_open=False,
                time_step_scale=transit_speed_scale,
                allowed_contact_links=transit_allowed_contact_links,
                allow_pose_planner_fallback=free_space_pose_planner_fallback,
                pose_planning_time=free_space_pose_planning_time,
            )
            if not result.success:
                return PushResult(success=False, failure_reason="transit_move_failed", **base)

        # Approach — closed gripper, contact monitoring on
        result = move(
            approach_pose,
            gripper_open=False,
            time_step_scale=approach_speed_scale,
            allowed_contact_links=approach_allowed_contact_links,
            allow_pose_planner_fallback=free_space_pose_planner_fallback,
            pose_planning_time=free_space_pose_planning_time,
        )
        if not result.success:
            return PushResult(success=False, failure_reason="approach_failed", **base)

        # Sweep — closed gripper, contact monitoring off (contact is intentional)
        diagnostics = {}

        def _push_stop_hook(_env, _robot_config, _diagnostics, _step_idx, _num_steps):
            summary = get_robot_contact_summary(_env)
            _diagnostics.setdefault("robot_contact_force_samples", []).append(
                float(summary["peak_force"])
            )
            _diagnostics.setdefault("robot_contact_entities", set()).update(
                summary["other_entities"]
            )

            shelf_entities = ()
            if push_cutoff_entity_substrings:
                patterns = tuple(str(s).lower() for s in push_cutoff_entity_substrings)
                shelf_summary = get_robot_contact_summary(
                    _env,
                    entity_filter=lambda name: any(
                        pattern in name.lower() for pattern in patterns
                    ),
                )
                shelf_entities = shelf_summary["other_entities"]
                _diagnostics.setdefault("shelf_contact_force_samples", []).append(
                    float(shelf_summary["peak_force"])
                )
                _diagnostics.setdefault("shelf_contact_entities", set()).update(
                    shelf_entities
                )
                if shelf_entities:
                    return "entity_contact_cutoff"
            if push_cutoff_force_threshold is not None and (
                summary["peak_force"] >= float(push_cutoff_force_threshold)
            ):
                return "force_cutoff"
            return None

        sweep_stop_hook = None
        if push_cutoff_force_threshold is not None or push_cutoff_entity_substrings:
            sweep_stop_hook = _push_stop_hook

        def _push_effort_hook(progress, _idx, _num_steps):
            scale = effort_scale + (effort_scale_end - effort_scale) * progress
            force_limit = np.asarray(base_force_limit_raw) * scale
            diagnostics.setdefault("commanded_force_limit_samples", []).append(
                float(np.max(np.asarray(force_limit)))
            )
            set_arm_drive_settings(env, force_limit=force_limit)

        try:
            result = move(
                push_pose,
                gripper_open=False,
                monitor_contacts=push_abort_on_contact,
                allowed_contact_links=rc.gripper_link_names,
                diagnostics=diagnostics,
                control_hook=_push_effort_hook,
                stop_hook=sweep_stop_hook,
                time_step_scale=push_speed_scale,
            )
        finally:
            set_arm_drive_settings(
                env,
                stiffness=base_drive["_stiffness_raw"],
                damping=base_drive["_damping_raw"],
                force_limit=base_drive["_force_limit_raw"],
            )
        if not result.success:
            return PushResult(success=False, failure_reason="push_failed", **base)

        # Lift to disengage when requested — gripper stays closed to avoid snagging.
        if lift_height > 0:
            post_lift_pose = sapien.Pose(
                [push_pose.p[0], push_pose.p[1], push_pose.p[2] + lift_height],
                push_pose.q,
            )
            result = move(
                post_lift_pose,
                gripper_open=False,
                monitor_contacts=False,
                time_step_scale=lift_speed_scale,
                allow_pose_planner_fallback=free_space_pose_planner_fallback,
                pose_planning_time=free_space_pose_planning_time,
            )
            if not result.success:
                logger.warning("Push lift failed, continuing anyway")

        # Opening after a push is optional. Keeping the pusher geometry fixed
        # avoids end-of-demo chatter when the hand is still parked near objects.
        if open_gripper_after_push:
            actuate_gripper(env, planner, rc.gripper_open,
                            step_callback=self.step_callback)

        contact_force_samples = diagnostics.get("contact_force_samples", [])
        commanded_force_limit_samples = diagnostics.get(
            "commanded_force_limit_samples", []
        )
        nonzero_contact_forces = [f for f in contact_force_samples if f > 0]
        joint_effort_l2_samples = diagnostics.get("joint_effort_l2_samples", [])
        joint_load_l2_samples = diagnostics.get("joint_load_l2_samples", [])
        contact_force_peak = max(contact_force_samples, default=0.0)
        contact_force_mean = (
            float(np.mean(nonzero_contact_forces)) if nonzero_contact_forces else 0.0
        )
        joint_effort_l2_peak = max(joint_effort_l2_samples, default=0.0)
        joint_effort_l2_mean = (
            float(np.mean(joint_effort_l2_samples))
            if joint_effort_l2_samples else 0.0
        )
        joint_load_l2_peak = max(joint_load_l2_samples, default=0.0)
        joint_load_l2_mean = (
            float(np.mean(joint_load_l2_samples))
            if joint_load_l2_samples else 0.0
        )
        contact_objects = tuple(sorted(diagnostics.get("contact_entities", ())))
        robot_contact_force_samples = diagnostics.get("robot_contact_force_samples", [])
        shelf_contact_force_samples = diagnostics.get("shelf_contact_force_samples", [])
        shelf_contact_objects = tuple(sorted(diagnostics.get("shelf_contact_entities", ())))
        contact_steps = len(nonzero_contact_forces)
        success = contact_force_peak >= float(min_contact_force)
        failure_reason = None
        if not success:
            failure_reason = "insufficient_contact_force"
        cutoff_reason = diagnostics.get("stop_reason")
        stop_step = diagnostics.get("stop_step")
        executed_push_fraction = 1.0
        if stop_step is not None and push_distance > 1e-8:
            executed_push_fraction = float(stop_step + 1) / max(
                len(contact_force_samples), 1
            )
            executed_push_fraction = float(np.clip(executed_push_fraction, 0.0, 1.0))
        executed_push_distance = push_distance * executed_push_fraction

        return PushResult(
            success=success,
            failure_reason=failure_reason,
            step_result=result.step_result,
            approach_pose=approach_pose,
            push_pose=push_pose,
            push_distance=push_distance,
            planar_push_distance=planar_push_distance,
            effort_scale_start=effort_scale,
            effort_scale_end=effort_scale_end,
            arm_force_limit_start=arm_force_limit_start,
            arm_force_limit_end=arm_force_limit_end,
            arm_force_limit_peak=max(commanded_force_limit_samples, default=0.0),
            arm_force_limit_mean=(
                float(np.mean(commanded_force_limit_samples))
                if commanded_force_limit_samples else 0.0
            ),
            contact_steps=contact_steps,
            contact_force_peak=contact_force_peak,
            contact_force_mean=contact_force_mean,
            joint_effort_l2_peak=joint_effort_l2_peak,
            joint_effort_l2_mean=joint_effort_l2_mean,
            joint_load_l2_peak=joint_load_l2_peak,
            joint_load_l2_mean=joint_load_l2_mean,
            contact_objects=contact_objects,
            executed_push_distance=executed_push_distance,
            executed_push_fraction=executed_push_fraction,
            cutoff_reason=cutoff_reason,
            shelf_contact_force_peak=max(shelf_contact_force_samples, default=0.0),
            shelf_contact_force_mean=(
                float(np.mean(shelf_contact_force_samples))
                if shelf_contact_force_samples else 0.0
            ),
            shelf_contact_objects=shelf_contact_objects,
        )
