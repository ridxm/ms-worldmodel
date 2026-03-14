import taskbench.agents  # noqa: F401 — registers custom agents (UR5, etc.)
import taskbench.envs.bin_with_objects  # noqa: F401 — triggers env registration
import taskbench.envs.shelf_env  # noqa: F401 — triggers env registration
import taskbench.envs.stack_cube_distractor  # noqa: F401 — triggers env registration
import taskbench.envs.stack_n_cube  # noqa: F401 — triggers env registration


def get_objects(env) -> dict[str, object]:
    """Get the name→actor mapping from any supported env.

    Prefers the env's own ``get_objects()`` method. Falls back to
    a convention-based lookup for built-in ManiSkill envs that don't
    implement it (e.g. StackCube-v1).
    """
    raw = env.unwrapped
    if hasattr(raw, "get_objects"):
        return raw.get_objects()

    # Built-in StackCube-v1: cubeB is base (green), cubeA stacks on top (red)
    if hasattr(raw, "cubeA") and hasattr(raw, "cubeB"):
        return {"cube_0": raw.cubeB, "cube_1": raw.cubeA}

    raise NotImplementedError(
        f"No get_objects() on {type(raw).__name__} and no known fallback"
    )
