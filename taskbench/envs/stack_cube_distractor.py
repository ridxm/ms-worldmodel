"""StackCube with an extra blue distractor cube on the table."""

import sapien
import torch

from mani_skill.envs.tasks.tabletop.stack_cube import StackCubeEnv
from mani_skill.envs.utils import randomization
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.pose import Pose


@register_env("StackCubeDistractor-v1", max_episode_steps=50)
class StackCubeDistractorEnv(StackCubeEnv):
    """StackCube-v1 with an additional blue cube as a visual distractor.

    The task and success criteria are identical to StackCube-v1 — stack the
    red cube on the green cube.  The blue cube simply sits on the table.

    Set ``force_close_distractor=True`` to place the blue cube right next
    to cubeA (for testing planner robustness).
    """

    force_close_distractor: bool = False

    def _load_scene(self, options: dict):
        super()._load_scene(options)
        self.cubeC = actors.build_cube(
            self.scene,
            half_size=0.02,
            color=[0, 0, 1, 1],  # blue
            name="cubeC",
            initial_pose=sapien.Pose(p=[0.5, 0, 0.1]),
        )

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)
            self._after_table_scene_init(env_idx)

            xyz = torch.zeros((b, 3))
            xyz[:, 2] = 0.02

            xy = torch.rand((b, 2)) * 0.2 - 0.1
            region = [[-0.1, -0.2], [0.1, 0.2]]
            sampler = randomization.UniformPlacementSampler(
                bounds=region, batch_size=b, device=self.device
            )
            radius = torch.linalg.norm(torch.tensor([0.02, 0.02])) + 0.001

            cubeA_xy = xy + sampler.sample(radius, 100)
            cubeB_xy = xy + sampler.sample(radius, 100, verbose=False)

            # cubeA
            xyz[:, :2] = cubeA_xy
            qs = randomization.random_quaternions(
                b, lock_x=True, lock_y=True, lock_z=False
            )
            self.cubeA.set_pose(Pose.create_from_pq(p=xyz.clone(), q=qs))

            # cubeB
            xyz[:, :2] = cubeB_xy
            qs = randomization.random_quaternions(
                b, lock_x=True, lock_y=True, lock_z=False
            )
            self.cubeB.set_pose(Pose.create_from_pq(p=xyz.clone(), q=qs))

            # cubeC (blue distractor)
            if self.force_close_distractor:
                # Place right next to cubeA (4.5cm offset — nearly touching)
                cubeC_xy = cubeA_xy + torch.tensor([0.045, 0.0])
            else:
                cubeC_xy = xy + sampler.sample(radius, 100, verbose=False)
            xyz[:, :2] = cubeC_xy
            qs = randomization.random_quaternions(
                b, lock_x=True, lock_y=True, lock_z=False
            )
            self.cubeC.set_pose(Pose.create_from_pq(p=xyz, q=qs))

    def get_objects(self) -> dict[str, object]:
        """Return a name→actor mapping for all manipulable objects."""
        return {
            "cube_0": self.cubeB,   # green (base)
            "cube_1": self.cubeA,   # red (stack on top)
            "cube_2": self.cubeC,   # blue (distractor)
        }

    def _get_obs_extra(self, info: dict):
        obs = super()._get_obs_extra(info)
        if "state" in self.obs_mode:
            obs["cubeC_pose"] = self.cubeC.pose.raw_pose
        return obs
