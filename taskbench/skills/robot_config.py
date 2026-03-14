"""Robot-specific constants for motion planning and skill execution.

Each supported robot has a ``RobotConfig`` entry in ``ROBOT_CONFIGS``.
Use ``get_robot_config(env)`` to look up the config for the current robot.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class RobotConfig:
    """Hardware-specific constants that skills and motion planning need."""

    move_group: str  # mplib move group name (from SRDF)
    finger_length: float  # depth of gripper fingers (meters)
    gripper_link_names: frozenset[str]  # links for contact detection
    gripper_open: float = 1.0  # action value for open
    gripper_closed: float = -1.0  # action value for closed


ROBOT_CONFIGS: dict[str, RobotConfig] = {
    "panda": RobotConfig(
        move_group="panda_hand_tcp",
        finger_length=0.025,
        gripper_link_names=frozenset(
            {"panda_hand", "panda_leftfinger", "panda_rightfinger"}
        ),
    ),
    "panda_wristcam": RobotConfig(
        move_group="panda_hand_tcp",
        finger_length=0.025,
        gripper_link_names=frozenset(
            {"panda_hand", "panda_leftfinger", "panda_rightfinger"}
        ),
    ),
    "ur5_robotiq": RobotConfig(
        move_group="eef",
        finger_length=0.035,
        gripper_link_names=frozenset(
            {
                "robotiq_arg2f_base_link",
                "left_inner_finger_pad",
                "right_inner_finger_pad",
                "left_inner_finger",
                "right_inner_finger",
                "left_outer_finger",
                "right_outer_finger",
            }
        ),
        gripper_open=0.0,
        gripper_closed=0.81,
    ),
    "ur5e_robotiq": RobotConfig(
        move_group="eef",
        finger_length=0.035,
        gripper_link_names=frozenset(
            {
                "robotiq_base_mount",
                "robotiq_base",
                "left_driver",
                "left_coupler",
                "left_spring_link",
                "left_follower",
                "left_pad",
                "left_silicone_pad",
                "right_driver",
                "right_coupler",
                "right_spring_link",
                "right_follower",
                "right_pad",
                "right_silicone_pad",
            }
        ),
        gripper_open=0.0,
        gripper_closed=-0.8,
    ),
}


def get_robot_config(env) -> RobotConfig:
    """Look up the RobotConfig for the env's robot.

    Reads ``env.unwrapped.agent.uid`` and looks it up in ``ROBOT_CONFIGS``.
    """
    uid = env.unwrapped.agent.uid
    if uid not in ROBOT_CONFIGS:
        available = ", ".join(sorted(ROBOT_CONFIGS))
        raise KeyError(
            f"No RobotConfig for robot {uid!r}. Available: {available}"
        )
    return ROBOT_CONFIGS[uid]
