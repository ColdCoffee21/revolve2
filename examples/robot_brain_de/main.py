"""Optimize a neural network for solving XOR."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
import revolve2.standard_resources.modular_robots as standard_robots
from pyrr import Quaternion, Vector3
from revolve2.actor_controllers.cpg import CpgNetworkStructure
from revolve2.core.database import (
    SerializableFrozenSingleton,
    SerializableIncrementableStruct,
    open_async_database_sqlite,
)
from revolve2.core.database.std import Rng
from revolve2.core.modular_robot import Body, ModularRobot
from revolve2.core.modular_robot.brains import (
    BrainCpgNetworkStatic,
    make_cpg_network_structure_neighbour,
)
from revolve2.core.optimization.ea.algorithms import bounce_parameters, de_offspring
from revolve2.core.optimization.ea.population import (
    Individual,
    Parameters,
    SerializableMeasures,
)
from revolve2.core.optimization.ea.population.pop_list import PopList, replace_if_better
from revolve2.core.physics.environment_actor_controller import (
    EnvironmentActorController,
)
from revolve2.core.physics.running import ActorState, Batch
from revolve2.core.physics.running import Environment as PhysicsEnv
from revolve2.core.physics.running import PosedActor, Runner
from revolve2.runners.mujoco import LocalRunner
from sqlalchemy.ext.asyncio import AsyncEngine
from sqlalchemy.ext.asyncio.session import AsyncSession

Genotype = Parameters


@dataclass
class Measures(SerializableMeasures, table_name="measures"):
    """Measures of a genotype/phenotype."""

    fitness: Optional[float] = None


class Population(PopList[Genotype, Measures], table_name="population"):
    """A population of individuals consisting of the above Genotype and Measures."""

    pass


@dataclass
class ProgramState(
    SerializableIncrementableStruct,
    table_name="program_state",
):
    """State of the program."""

    rng: Rng
    population: Population
    generation_index: int


@dataclass
class ProgramRoot(SerializableFrozenSingleton, table_name="program_root"):
    """
    Root object containing program data.

    In the database this is a single row in a single table.
    """

    program_state: ProgramState


class Program:
    """Program that optimizes the neural network parameters."""

    RNG_SEED = 0
    NUM_GENERATIONS: int = 100
    POPULATION_SIZE: int = 100
    CROSSOVER_PROBABILITY: float = 0.9
    DIFFERENTIAL_WEIGHT: float = 0.2

    SIMULATION_TIME = 30
    SAMPLING_FREQUENCY = 5
    CONTROL_FREQUENCY = 60

    NUM_SIMULATORS = 4

    BODY: Body = standard_robots.gecko()
    CPG_NETWORK_STRUCTURE: CpgNetworkStructure = make_cpg_network_structure_neighbour(
        BODY.find_active_hinges()
    )

    db: AsyncEngine

    state: ProgramState

    runner: Runner

    async def run(
        self,
    ) -> None:
        """Run the program."""
        logging.basicConfig(
            level=logging.INFO,
            format="[%(asctime)s] [%(levelname)s] [%(module)s] %(message)s",
        )

        logging.info("Program start.")

        logging.info("Opening database..")
        self.database = open_async_database_sqlite(
            db_root_directory="database", create=True
        )
        logging.info("Opening database done.")

        logging.info("Creating database structure..")
        async with self.database.begin() as conn:
            await ProgramRoot.prepare_db(conn)
        logging.info("Creating database structure done.")

        self._runner = LocalRunner(headless=True, num_simulators=self.NUM_SIMULATORS)

        logging.info("Trying to load program root from database..")
        if await self.load_root():
            logging.info("Root loaded successfully.")
        else:
            logging.info("Unable to load root. Initializing root..")
            await self.init_root()
            logging.info("Initializing state done.")

        logging.info(
            f"Entering optimization loop. Continuing until around {self.NUM_GENERATIONS} generations."
        )
        while self.root.program_state.generation_index < self.NUM_GENERATIONS:
            logging.info(
                f"Current generation: {self.root.program_state.generation_index}"
            )
            logging.info("Evolving..")
            await self.evolve()
            logging.info("Evolving done.")

            logging.info("Saving state..")
            await self.save_state()
            logging.info("Saving state done.")
        logging.info("Optimization loop done. Exiting program.")

    async def save_state(self) -> None:
        """Save the state of the program."""
        async with AsyncSession(self.database) as ses:
            async with ses.begin():
                await self.root.program_state.to_db(ses)

    async def init_root(
        self,
    ) -> None:
        """Initialize the program root, saving it the database as well."""
        initial_rng = Rng(np.random.Generator(np.random.PCG64(self.RNG_SEED)))

        initial_population = Population(
            [
                Individual(
                    Genotype(
                        [
                            float(v)
                            for v in initial_rng.rng.random(
                                size=self.CPG_NETWORK_STRUCTURE.num_connections
                            )
                        ]
                    ),
                    Measures(),
                )
                for _ in range(self.POPULATION_SIZE)
            ]
        )
        logging.info("Measuring initial population..")
        await self.measure(initial_population)
        logging.info("Measuring initial population done.")

        logging.info("Saving root..")
        state = ProgramState(
            rng=initial_rng,
            population=initial_population,
            generation_index=0,
        )
        self.root = ProgramRoot(state)
        async with AsyncSession(self.database) as ses:
            async with ses.begin():
                await self.root.to_db(ses)
        logging.info("Saving root done.")

    async def load_root(self) -> bool:
        """
        Load the state of the program.

        :returns: True if could be loaded from database. False if no data available.
        """
        async with AsyncSession(self.database) as ses:
            async with ses.begin():
                maybe_root = await ProgramRoot.from_db(ses, 1)
                if maybe_root is None:
                    return False
                else:
                    self.root = maybe_root
                    return True

    async def evolve(self) -> None:
        """Iterate one generation further."""
        self.root.program_state.generation_index += 1

        offspring = Population(
            [
                Individual(bounce_parameters(genotype), Measures())
                for genotype in de_offspring(
                    self.root.program_state.population,
                    self.root.program_state.rng,
                    self.DIFFERENTIAL_WEIGHT,
                    self.CROSSOVER_PROBABILITY,
                )
            ]
        )

        await self.measure(offspring)

        original_selection, offspring_selection = replace_if_better(
            self.root.program_state.population, offspring, measure="fitness"
        )

        self.root.program_state.population = Population.from_existing_populations(  # type: ignore # TODO
            [self.root.program_state.population, offspring],
            [original_selection, offspring_selection],
            [
                "fitness",
            ],
        )

    async def measure(self, population: Population) -> None:
        """
        Measure all individuals in a population.

        :param population: The population.
        """
        batch = Batch(
            simulation_time=self.SIMULATION_TIME,
            sampling_frequency=self.SAMPLING_FREQUENCY,
            control_frequency=self.CONTROL_FREQUENCY,
        )

        for individual in population:
            weight_matrix = (
                self.CPG_NETWORK_STRUCTURE.make_connection_weights_matrix_from_params(
                    np.clip(individual.genotype, 0.0, 1.0) * 4.0 - 2.0
                )
            )
            initial_state = self.CPG_NETWORK_STRUCTURE.make_uniform_state(
                math.sqrt(2) / 2.0
            )
            dof_ranges = self.CPG_NETWORK_STRUCTURE.make_uniform_dof_ranges(1.0)

            brain = BrainCpgNetworkStatic(
                initial_state,
                self.CPG_NETWORK_STRUCTURE.num_cpgs,
                weight_matrix,
                dof_ranges,
            )
            actor, controller = ModularRobot(
                self.BODY, brain
            ).make_actor_and_controller()
            bounding_box = actor.calc_aabb()
            env = PhysicsEnv(EnvironmentActorController(controller))
            env.actors.append(
                PosedActor(
                    actor,
                    Vector3(
                        [
                            0.0,
                            0.0,
                            bounding_box.size.z / 2.0 - bounding_box.offset.z,
                        ]
                    ),
                    Quaternion(),
                    [0.0 for _ in controller.get_dof_targets()],
                )
            )
            batch.environments.append(env)

        batch_results = await self._runner.run_batch(batch)

        fitnesses = [
            self.calculate_fitness(
                environment_result.environment_states[0].actor_states[0],
                environment_result.environment_states[-1].actor_states[0],
            )
            for environment_result in batch_results.environment_results
        ]

        for individual, fitness in zip(population, fitnesses):
            individual.measures.fitness = float(fitness)

    @staticmethod
    def calculate_fitness(begin_state: ActorState, end_state: ActorState) -> float:
        """
        Calculate the fitness corresponding to a simulation result.

        :param begin_state: Initial state of the robot. (begin of simulation)
        :param end_state: Final state of the robot. (end of simulation)
        :returns: The calculated fitness. Euclidian distance between initial and final position.
        """
        # distance traveled on the xy plane
        return math.sqrt(
            (begin_state.position[0] - end_state.position[0]) ** 2
            + ((begin_state.position[1] - end_state.position[1]) ** 2)
        )


async def main() -> None:
    """Run the program."""
    await Program().run()


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
