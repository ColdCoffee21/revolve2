"""
Visualize and run a modular robot using Isaac Gym.
"""

import math
from random import Random

from pyrr import Quaternion, Vector3

from revolve2.actor_controller import ActorController
from revolve2.core.modular_robot import ActiveHinge, Body, Brick, ModularRobot
from revolve2.core.modular_robot.brains import BrainCpgNetworkNeighbourRandom
from revolve2.core.physics.running import ActorControl, Batch, Environment, PosedActor
from revolve2.runners.isaacgym import LocalRunner
from revolve2.core.modular_robot.render.render import Render

class Simulator:
    _controller: ActorController

    async def simulate(self, robot: ModularRobot, control_frequency: float) -> None:
        batch = Batch(
            simulation_time=1000000,
            sampling_frequency=0.0001,
            control_frequency=control_frequency,
            control=self._control,
        )

        actor, self._controller = robot.make_actor_and_controller()

        env = Environment()
        env.actors.append(
            PosedActor(
                actor,
                Vector3([0.0, 0.0, 0.1]),
                Quaternion(),
                [0.0 for _ in self._controller.get_dof_targets()],
            )
        )
        batch.environments.append(env)

        runner = LocalRunner(LocalRunner.SimParams())
        await runner.run_batch(batch)

    def _control(self, dt: float, control: ActorControl) -> None:
        self._controller.step(dt)
        control.set_dof_targets(0, 0, self._controller.get_dof_targets())


async def main() -> None:
    rng = Random()
    rng.seed(5)

    # set id is deactivated...will break...
    body = Body()
    body.core.back = ActiveHinge(0)
    # body.core.front = ActiveHinge(math.pi / 2.0)
    body.finalize()

    brain = BrainCpgNetworkNeighbourRandom(rng)
    robot = ModularRobot(body, brain)

    render = Render()
    img_path = f'data/manualrobot.png'
    render.render_robot(robot.body.core, img_path)

    sim = Simulator()
    await sim.simulate(robot, 10)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
