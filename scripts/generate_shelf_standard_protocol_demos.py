#!/usr/bin/env python3
"""Generate deterministic shelf demos from reachable insertion cells."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import gymnasium as gym
import imageio.v2 as imageio
import numpy as np
import sapien
from omegaconf import OmegaConf
from PIL import Image

import taskbench.envs  # noqa: F401
from taskbench.recorder import StateRecorder
from taskbench.skills.motion import (
    add_collision_boxes,
    get_robot_contact_summary,
    hold_current_pose,
    move_to_pose,
    plan_to_pose,
    setup_planner,
)
from taskbench.skills.primitives import Push
from taskbench.skills.robot_config import get_robot_config


Q_INTO_SHELF_VERTICAL = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32)
INSERTION_STAGE_SPEED = 1.8
INSERTION_TRANSIT_SPEED = 1.9
INSERTION_LOWER_SPEED = 2.1
SWEEP_SPEED = 0.8
IMPUDENCE_FORCE_CUTOFF = 18.0


@dataclass(frozen=True)
class InsertionCandidate:
    contact_x: float
    lane_y: float
    push_z: float
    insertion_z: float
    push_distance: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/shelf_demos"))
    parser.add_argument("--num-demos", type=int, default=10)
    parser.add_argument("--num-bottles", type=int, default=10)
    parser.add_argument("--seed-start", type=int, default=1)
    parser.add_argument("--max-seeds", type=int, default=200)
    parser.add_argument("--image-size", type=int, default=1024)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--robot-uids", type=str, default="ur5e_robotiq")
    parser.add_argument("--robot-base-x", type=float, default=-0.56)
    parser.add_argument(
        "--demo-mode",
        choices=("paired", "random-targets"),
        default="paired",
    )
    parser.add_argument("--push-distance-min", type=float, default=0.06)
    parser.add_argument("--push-distance-max", type=float, default=0.24)
    parser.add_argument("--push-sample-attempts", type=int, default=16)
    parser.add_argument("--random-seed", type=int, default=12345)
    return parser.parse_args()


def _coerce_render_frame(frame) -> np.ndarray:
    if hasattr(frame, "detach"):
        frame = frame.detach().cpu().numpy()
    frame = np.asarray(frame)
    if frame.ndim == 4:
        frame = frame[0]
    if frame.dtype != np.uint8:
        frame = np.clip(frame, 0, 255)
        if frame.max() <= 1.0:
            frame = frame * 255.0
        frame = frame.astype(np.uint8)
    return frame


def _resize_rgb(rgb: np.ndarray, image_size: int) -> np.ndarray:
    if rgb.shape[0] == image_size and rgb.shape[1] == image_size:
        return rgb.astype(np.uint8, copy=False)
    try:
        resample = Image.Resampling.BILINEAR
    except AttributeError:
        resample = Image.BILINEAR
    return np.asarray(
        Image.fromarray(rgb).resize((image_size, image_size), resample=resample),
        dtype=np.uint8,
    )


def _make_cfg(args: argparse.Namespace):
    return OmegaConf.create(
        {
            "seed": 0,
            "task": {
                "env_id": "ShelfBottleClutter-v1",
                "robot_uids": str(args.robot_uids),
                "robot_base_pose": [float(args.robot_base_x), 0.0, 0.0],
                "num_bottles": int(args.num_bottles),
                "bottle_scale": 1.45,
                "object_density": 1800.0,
                "edge_margin": 0.008,
                "placement_spacing": 0.062,
                "placement_clearance": 0.002,
                "placement_jitter": 0.003,
                "layout_mode": "auto",
                "dense_layout_prob": 0.45,
                "mixed_layout_prob": 0.45,
                "random_yaw": True,
                "movement_success_threshold": 0.0,
                "shelf": {
                    "front_x": 0.26,
                    "depth": 0.22,
                    "half_w": 0.35,
                    "floor_z": 0.40,
                    "thickness": 0.01,
                    "inner_h": 0.35,
                },
            },
            "runtime": {
                "obs_mode": "none",
                "control_mode": "pd_joint_pos",
                "reward_mode": "none",
                "record_video": False,
                "render_mode": "rgb_array",
                "num_envs": 1,
                "max_episode_steps": 800,
            },
            "run": {
                "solver": "shelf_random_push",
                "num_episodes": 1,
            },
        }
    )


def _make_env(args: argparse.Namespace, *, render_mode: str):
    return gym.make(
        "ShelfBottleClutter-v1",
        obs_mode="none",
        control_mode="pd_joint_pos",
        reward_mode="none",
        num_envs=1,
        max_episode_steps=800,
        sim_backend="cpu",
        render_mode=render_mode,
        robot_uids=str(args.robot_uids),
        robot_base_pose=[float(args.robot_base_x), 0.0, 0.0],
        num_bottles=int(args.num_bottles),
        bottle_scale=1.45,
        object_density=1800.0,
        edge_margin=0.008,
        placement_spacing=0.062,
        placement_clearance=0.002,
        placement_jitter=0.003,
        layout_mode="auto",
        dense_layout_prob=0.45,
        mixed_layout_prob=0.45,
        random_yaw=True,
        movement_success_threshold=0.0,
    )


def _pose(x: float, y: float, z: float) -> sapien.Pose:
    return sapien.Pose([float(x), float(y), float(z)], Q_INTO_SHELF_VERTICAL.copy())


def _write_video(path: Path, frames: list[np.ndarray], fps: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(str(path), fps=int(fps))
    try:
        for frame in frames:
            writer.append_data(frame)
    finally:
        writer.close()


def _make_capture(env, recorder: StateRecorder, image_size: int):
    raw = env.unwrapped
    top_actor = next(
        (
            actor
            for actor in getattr(raw, "shelf_parts", [])
            if getattr(actor, "name", "") == "shelf_top"
        ),
        None,
    )
    top_pose = None
    top_hidden_pose = None
    if top_actor is not None:
        top_p = top_actor.pose.p[0].detach().cpu().numpy().astype(np.float32)
        top_q = top_actor.pose.q[0].detach().cpu().numpy().astype(np.float32)
        top_pose = sapien.Pose(top_p.tolist(), top_q.tolist())
        hidden_p = top_p.copy()
        hidden_p[2] += 5.0
        top_hidden_pose = sapien.Pose(hidden_p.tolist(), top_q.tolist())

    frames: list[np.ndarray] = []

    def _capture():
        recorder.record()
        if top_actor is not None and top_hidden_pose is not None and top_pose is not None:
            top_actor.set_pose(top_hidden_pose)
        try:
            frames.append(_resize_rgb(_coerce_render_frame(env.render()), image_size))
        finally:
            if top_actor is not None and top_pose is not None:
                top_actor.set_pose(top_pose)

    return _capture, frames


def _make_bottle_collision_boxes(raw) -> list[tuple[str, list[float], list[float]]]:
    radius = float(raw.bottle_body_radius) + 0.002
    half_height = float(raw.bottle_body_half_length) + 0.01
    boxes = []
    for idx, bottle in enumerate(raw.bottles):
        center = bottle.pose.p[0].detach().cpu().numpy().astype(np.float32)
        boxes.append(
            (
                f"bottle_obs_{idx}",
                center.tolist(),
                [radius, radius, half_height],
            )
        )
    return boxes


def _iter_y_values(half_w: float) -> list[float]:
    ys = np.arange(-half_w + 0.08, half_w - 0.08 + 1e-6, 0.02, dtype=np.float32)
    return sorted((float(y) for y in ys), key=lambda y: (abs(y), y))


def _iter_insertion_candidates(raw) -> list[InsertionCandidate]:
    x_values = np.array(
        [
            min(raw.shelf_geom.front_x + 0.09, raw.shelf_geom.back_x - 0.06),
            min(raw.shelf_geom.front_x + 0.08, raw.shelf_geom.back_x - 0.07),
            min(raw.shelf_geom.front_x + 0.07, raw.shelf_geom.back_x - 0.08),
        ],
        dtype=np.float32,
    )
    z_values = np.array([0.57, 0.58], dtype=np.float32)
    insertion_z = min(raw.shelf_geom.ceil_z - 0.05, 0.61)
    candidates: list[InsertionCandidate] = []
    for x in x_values:
        for z in z_values:
            for y in _iter_y_values(float(raw.shelf_geom.half_w)):
                push_budget = min(0.28, float(raw.shelf_geom.half_w) - 0.05 - abs(y))
                if push_budget < 0.10:
                    continue
                candidates.append(
                    InsertionCandidate(
                        contact_x=float(x),
                        lane_y=float(y),
                        push_z=float(z),
                        insertion_z=float(insertion_z),
                        push_distance=float(push_budget),
                    )
                )
    return candidates


def _insertion_poses(raw, candidate: InsertionCandidate):
    stage_pose = _pose(raw.shelf_geom.front_x - 0.05, candidate.lane_y, candidate.insertion_z)
    deep_pose = _pose(candidate.contact_x, candidate.lane_y, candidate.insertion_z)
    approach_pose = _pose(candidate.contact_x, candidate.lane_y, candidate.push_z)
    return stage_pose, deep_pose, approach_pose


def _directional_push_distance(raw, candidate: InsertionCandidate, push_sign: float) -> float:
    edge_margin = 0.05
    if push_sign < 0:
        return max(0.0, candidate.lane_y - (-float(raw.shelf_geom.half_w) + edge_margin))
    return max(0.0, (float(raw.shelf_geom.half_w) - edge_margin) - candidate.lane_y)


def _candidate_priority(candidate: InsertionCandidate):
    return (
        -float(candidate.contact_x),
        abs(float(candidate.lane_y)),
        -float(candidate.push_z),
    )


def _sample_push_endpoint(
    env,
    planner,
    candidate: InsertionCandidate,
    push_sign: float,
    rng: np.random.Generator,
    *,
    push_distance_min: float,
    push_distance_max: float,
    sample_attempts: int,
):
    raw = env.unwrapped
    available = _directional_push_distance(raw, candidate, push_sign)
    capped_max = min(float(available), float(push_distance_max))
    min_required = float(push_distance_min)
    if capped_max < max(min_required, 1e-6):
        return None, 0.0

    lower = min_required

    # Randomize target distance but always include the far endpoint to avoid
    # pathological cases where all random samples are too small.
    samples = rng.uniform(lower, capped_max, size=max(int(sample_attempts), 1))
    samples = np.append(samples, capped_max)

    start_qpos = raw.agent.robot.get_qpos()[0].detach().cpu().numpy()
    for target_distance in samples:
        pose = _pose(
            candidate.contact_x,
            candidate.lane_y + push_sign * float(target_distance),
            candidate.push_z,
        )
        plan = plan_to_pose(
            env,
            planner,
            pose,
            start_qpos=start_qpos,
            time_step_scale=SWEEP_SPEED,
            allow_pose_planner_fallback=False,
        )
        if plan is not None:
            return pose, float(target_distance)
    return None, 0.0


def _execute_insertion(env, candidate: InsertionCandidate, *, step_callback=None, recorder=None):
    raw = env.unwrapped
    rc = get_robot_config(env)
    planner = setup_planner(env, rc)
    add_collision_boxes(planner, _make_bottle_collision_boxes(raw), resolution=0.015)

    if recorder is not None:
        recorder.record_skill_call("insert", asdict(candidate))

    stage_pose, deep_pose, approach_pose = _insertion_poses(raw, candidate)
    phase_specs = [
        ("stage", stage_pose, INSERTION_STAGE_SPEED, True),
        ("insert", deep_pose, INSERTION_TRANSIT_SPEED, False),
        ("lower", approach_pose, INSERTION_LOWER_SPEED, False),
    ]
    for phase_name, pose, scale, allow_fallback in phase_specs:
        step_result = move_to_pose(
            env,
            planner,
            pose,
            rc.gripper_closed,
            rc,
            monitor_contacts=True,
            step_callback=step_callback,
            time_step_scale=scale,
            allow_pose_planner_fallback=allow_fallback,
            pose_planning_time=1.0 if allow_fallback else 0.0,
        )
        if step_result is None:
            return False, f"{phase_name}_failed"
    return True, None


def _reachable_insertion_set(search_env, seed: int) -> list[InsertionCandidate]:
    search_env.reset(seed=seed)
    raw = search_env.unwrapped
    candidates = _iter_insertion_candidates(raw)
    reachable: list[InsertionCandidate] = []
    for candidate in candidates:
        search_env.reset(seed=seed)
        success, _failure_reason = _execute_insertion(search_env, candidate)
        if success:
            reachable.append(candidate)
    return reachable


def _select_demo_candidate(search_env, reachable: list[InsertionCandidate], seed: int):
    raw = search_env.unwrapped
    if not reachable:
        raise ValueError("reachable must be non-empty")

    max_x = max(float(c.contact_x) for c in reachable)
    deep = [c for c in reachable if float(c.contact_x) >= max_x - 0.015]
    if not deep:
        deep = list(reachable)

    min_room = 0.12
    balanced = [
        c
        for c in deep
        if min(
            _directional_push_distance(raw, c, -1.0),
            _directional_push_distance(raw, c, 1.0),
        )
        >= min_room
    ]
    pool = balanced if balanced else deep
    pool = sorted(pool, key=lambda c: (float(c.lane_y), -float(c.push_z), -float(c.contact_x)))
    selected_idx = int(seed % len(pool))
    return pool[selected_idx], selected_idx, len(pool)


def _find_demo_insertion(search_env, seed: int, args: argparse.Namespace):
    search_env.reset(seed=seed)
    raw = search_env.unwrapped
    lane_targets = [-0.14, -0.08, 0.0, 0.08, 0.14]
    target_lane = lane_targets[(seed - 1) % len(lane_targets)]
    candidates = sorted(
        _iter_insertion_candidates(raw),
        key=lambda c: (
            -float(c.contact_x),
            abs(float(c.lane_y) - float(target_lane)),
            -abs(float(c.lane_y)),
            -float(c.push_z),
        ),
    )

    for candidate_idx, candidate in enumerate(candidates):
        search_env.reset(seed=seed)
        raw = search_env.unwrapped
        recorder = StateRecorder(
            search_env,
            objects={obj.name: obj for obj in raw.bottles},
            robot_fields=[
                "qpos",
                "tcp_pos",
                "tcp_quat",
                "gripper_contact_force",
                "shelf_contact_force",
            ],
        )
        capture, frames = _make_capture(search_env, recorder, args.image_size)
        capture()
        success, failure_reason = _execute_insertion(
            search_env,
            candidate,
            step_callback=capture,
            recorder=None,
        )
        if success:
            return {
                "success": True,
                "candidate": candidate,
                "candidate_index": candidate_idx,
                "insertion_frames": frames,
                "inserted_scene_spec": raw.get_scene_spec(),
            }
    return {
        "success": False,
        "failure_reason": "no_renderable_insertion",
    }


def _render_insertion(env, seed: int, candidate: InsertionCandidate, args: argparse.Namespace):
    try:
        env.reset(seed=seed)
        raw = env.unwrapped
        recorder = StateRecorder(
            env,
            objects={obj.name: obj for obj in raw.bottles},
            robot_fields=[
                "qpos",
                "tcp_pos",
                "tcp_quat",
                "gripper_contact_force",
                "shelf_contact_force",
            ],
        )
        capture, frames = _make_capture(env, recorder, args.image_size)
        capture()
        insertion_ok, failure_reason = _execute_insertion(
            env,
            candidate,
            step_callback=capture,
            recorder=None,
        )
        if not insertion_ok:
            return {
                "success": False,
                "failure_reason": failure_reason,
            }
        inserted_scene_spec = raw.get_scene_spec()
        return {
            "success": True,
            "frames": frames,
            "inserted_scene_spec": inserted_scene_spec,
        }
    except Exception:
        raise


def _run_demo(
    seed: int,
    candidate: InsertionCandidate,
    inserted_scene_spec: dict[str, object],
    push_sign: float,
    args: argparse.Namespace,
    rng: np.random.Generator,
    *,
    prefix_frames: list[np.ndarray] | None = None,
):
    env = _make_env(args, render_mode="rgb_array")
    cfg = _make_cfg(args)
    cfg.seed = int(seed)
    try:
        env.reset(seed=seed)
        raw = env.unwrapped
        raw.apply_scene_spec(inserted_scene_spec)
        rc = get_robot_config(env)
        recorder = StateRecorder(
            env,
            objects={obj.name: obj for obj in raw.bottles},
            robot_fields=[
                "qpos",
                "tcp_pos",
                "tcp_quat",
                "gripper_contact_force",
                "shelf_contact_force",
            ],
        )
        capture, frames = _make_capture(env, recorder, args.image_size)
        capture()
        sweep_planner = setup_planner(env, rc)
        approach_pose = _pose(candidate.contact_x, candidate.lane_y, candidate.push_z)
        push_pose, selected_distance = _sample_push_endpoint(
            env,
            sweep_planner,
            candidate,
            push_sign,
            rng,
            push_distance_min=float(args.push_distance_min),
            push_distance_max=float(args.push_distance_max),
            sample_attempts=int(args.push_sample_attempts),
        )
        if push_pose is None:
            env.close()
            return {
                "success": False,
                "failure_reason": "out_of_range",
                "push_sign": float(push_sign),
            }

        recorder.record_skill_call(
            "push",
            {
                "push_sign": float(push_sign),
                "push_distance": float(selected_distance),
                "target_pose": {
                    "p": [float(v) for v in push_pose.p],
                    "q": [float(v) for v in push_pose.q],
                },
            },
        )
        push_skill = Push(
            env,
            sweep_planner,
            robot_config=rc,
            step_callback=capture,
        )
        push_result = push_skill(
            approach_pose=approach_pose,
            push_pose=push_pose,
            clearance_height=0.0,
            lift_height=0.0,
            effort_scale=1.0,
            min_contact_force=0.0,
            open_gripper_after_push=False,
            approach_speed_scale=1.0,
            push_speed_scale=SWEEP_SPEED,
            transit_allowed_contact_links=rc.gripper_link_names,
            approach_allowed_contact_links=rc.gripper_link_names,
            push_abort_on_contact=False,
            push_cutoff_force_threshold=IMPUDENCE_FORCE_CUTOFF,
            push_cutoff_entity_substrings=None,
            free_space_pose_planner_fallback=False,
        )

        # Post-insertion policy: push attempts are always recorded.
        # Only pre-push target reachability is treated as a hard failure.
        stop_reason = "range_limit"
        if push_result.cutoff_reason == "force_cutoff":
            stop_reason = "impudence_cutoff"
        elif push_result.cutoff_reason:
            stop_reason = str(push_result.cutoff_reason)
        executed_distance = float(push_result.executed_push_distance)
        shelf_summary = get_robot_contact_summary(
            env,
            entity_filter=lambda name: "shelf" in name.lower(),
        )

        hold_current_pose(
            env,
            sweep_planner,
            rc.gripper_closed,
            steps=2,
            step_callback=capture,
        )

        info = raw.evaluate()
        output_frames = frames
        if prefix_frames:
            output_frames = [frame.copy() for frame in prefix_frames] + frames[1:]
        return {
            "success": True,
            "env": env,
            "cfg": cfg,
            "frames": output_frames,
            "recorder": recorder,
            "seed": int(seed),
            "candidate": candidate,
            "push_sign": float(push_sign),
            "executed_push_distance": float(executed_distance),
            "stop_reason": stop_reason,
            "max_displacement": float(info["max_displacement"].item()),
            "moved_bottles": int(info["moved_bottles"].item()),
            "shelf_contact_force_peak": float(shelf_summary["peak_force"]),
            "shelf_contact_entities": sorted(shelf_summary["other_entities"]),
            "selected_push_distance": float(selected_distance),
            "push_skill_success": bool(push_result.success),
            "push_skill_failure_reason": push_result.failure_reason or "",
        }
    except Exception:
        env.close()
        raise


def main() -> None:
    args = parse_args()
    if args.demo_mode == "paired" and int(args.num_demos) % 2 != 0:
        raise SystemExit("--num-demos must be even so each scene yields left/right demos.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(int(args.random_seed))

    search_env = _make_env(args, render_mode="rgb_array")
    try:
        demos_written = 0
        seed = int(args.seed_start)
        seed_stop = int(args.seed_start + args.max_seeds)
        summary: list[dict] = []
        while demos_written < int(args.num_demos) and seed < seed_stop:
            insertion_choice = _find_demo_insertion(search_env, seed, args)
            if not insertion_choice["success"]:
                summary.append(
                    {
                        "seed": int(seed),
                        "success": False,
                        "failure_reason": insertion_choice["failure_reason"],
                    }
                )
                print(f"[seed {seed}] {insertion_choice['failure_reason']}")
                seed += 1
                continue
            candidate = insertion_choice["candidate"]
            candidate_idx = insertion_choice["candidate_index"]
            inserted_scene_spec = insertion_choice["inserted_scene_spec"]
            insertion_frames = insertion_choice["insertion_frames"]

            attempts = []
            valid_scene = True
            if args.demo_mode == "paired":
                push_signs = (-1.0, 1.0)
            else:
                push_signs = (float(rng.choice([-1.0, 1.0])),)
            for push_sign in push_signs:
                attempt = _run_demo(
                    seed,
                    candidate,
                    inserted_scene_spec,
                    push_sign,
                    args,
                    rng,
                    prefix_frames=insertion_frames,
                )
                attempts.append(attempt)
                summary.append(
                    {
                        "seed": int(seed),
                        "candidate": asdict(candidate),
                        "candidate_index": candidate_idx,
                        "push_sign": float(push_sign),
                        "success": bool(attempt.get("success", False)),
                        "failure_reason": attempt.get("failure_reason", ""),
                        "selected_push_distance": float(attempt.get("selected_push_distance", 0.0)),
                        "stop_reason": attempt.get("stop_reason", ""),
                    }
                )
                if not attempt["success"]:
                    valid_scene = False
                    break

            if not valid_scene:
                for attempt in attempts:
                    if attempt.get("success") and "env" in attempt:
                        attempt["env"].close()
                print(f"[seed {seed}] execution_failed")
                seed += 1
                continue

            for attempt in attempts[: int(args.num_demos) - demos_written]:
                direction = "left" if attempt["push_sign"] < 0 else "right"
                mode_tag = "random" if args.demo_mode == "random-targets" else "paired"
                stem = (
                    f"shelf_standard_protocol_seed{seed:04d}_{direction}_"
                    f"{mode_tag}_demo{demos_written + 1:02d}"
                )
                mp4_path = args.output_dir / f"{stem}.mp4"
                h5_path = args.output_dir / f"{stem}.hdf5"
                json_path = args.output_dir / f"{stem}.json"
                png_path = args.output_dir / f"{stem}_start.png"

                _write_video(mp4_path, attempt["frames"], args.fps)
                Image.fromarray(attempt["frames"][0]).save(png_path)
                attempt["recorder"].save(
                    str(h5_path),
                    metadata={
                        "seed": seed,
                        "solver": "shelf_standard_protocol_demo",
                        "success": True,
                        "failure_reason": "",
                    },
                    hydra_cfg=attempt["cfg"],
                )
                with json_path.open("w", encoding="utf-8") as f:
                    json.dump(
                        {
                            "seed": int(seed),
                            "direction": direction,
                            "mode": args.demo_mode,
                            "candidate": asdict(candidate),
                            "selected_push_distance": float(attempt["selected_push_distance"]),
                            "executed_push_distance": float(attempt["executed_push_distance"]),
                            "stop_reason": attempt["stop_reason"],
                            "max_displacement": float(attempt["max_displacement"]),
                            "moved_bottles": int(attempt["moved_bottles"]),
                            "shelf_contact_force_peak": float(attempt["shelf_contact_force_peak"]),
                            "shelf_contact_entities": attempt["shelf_contact_entities"],
                            "push_skill_success": bool(attempt["push_skill_success"]),
                            "push_skill_failure_reason": attempt["push_skill_failure_reason"],
                        },
                        f,
                        indent=2,
                    )
                attempt["env"].close()
                demos_written += 1
                print(
                    f"[demo {demos_written}/{args.num_demos}] seed={seed} {direction} "
                    f"push={attempt['executed_push_distance']:.3f} stop={attempt['stop_reason']} -> {mp4_path}"
                )
            seed += 1

        with (args.output_dir / "search_summary.json").open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        if demos_written < int(args.num_demos):
            raise SystemExit(
                f"Only generated {demos_written} demos before hitting seed limit {seed_stop}."
            )
    finally:
        search_env.close()


if __name__ == "__main__":
    main()
