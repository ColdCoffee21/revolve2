import os

import math
import pickle
from random import Random
from typing import List, Tuple

import multineat
import sqlalchemy
from genotype import Genotype, GenotypeSerializer, crossover, develop, mutate
from pyrr import Quaternion, Vector3
from sqlalchemy.ext.asyncio import AsyncEngine
from sqlalchemy.ext.asyncio.session import AsyncSession
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.future import select

import revolve2.core.optimization.ea.generic_ea.population_management as population_management
import revolve2.core.optimization.ea.generic_ea.selection as selection
from revolve2.actor_controller import ActorController
from revolve2.core.database import IncompatibleError
from revolve2.core.database.serializers import FloatSerializer, StatesSerializer
from revolve2.core.optimization import ProcessIdGen
from revolve2.core.modular_robot import Measure

from revolve2.core.optimization.ea.generic_ea import EAOptimizer
import numpy as np
import pprint
import logging



from revolve2.core.physics.running import (
    ActorControl,
    ActorState,
    Batch,
    BatchResults,
    Environment,
    EnvironmentResults,
    EnvironmentState,
    Runner,
    PosedActor,

)

from revolve2.runners.mujoco import LocalRunner as LocalRunnerM
import mujoco
import mujoco_viewer
from dm_control import mjcf
import tempfile
from revolve2.core.physics.actor.urdf import to_urdf as physbot_to_urdf


# isaac import will probably break on mac, so u can comment it out
from revolve2.runners.isaacgym import LocalRunner as LocalRunnerI


class OldOptimizer(EAOptimizer[Genotype, float]):
    _process_id: int

    _runner: Runner

    _controllers: List[ActorController]

    _innov_db_body: multineat.InnovationDatabase
    _innov_db_brain: multineat.InnovationDatabase

    _rng: Random

    _simulation_time: int
    _sampling_frequency: float
    _control_frequency: float

    _num_generations: int
    _offspring_size: int
    _fitness_measure: str
    _experiment_name: str
    _max_modules: int
    _crossover_prob: float
    _mutation_prob: float
    _substrate_radius: str
    _run_simulation: bool
    _env_conditions: List
    _plastic_body: int
    _plastic_brain: int
    _simulator: str

    async def ainit_new(  # type: ignore # TODO for now ignoring mypy complaint about LSP problem, override parent's ainit
        self,
        database: AsyncEngine,
        session: AsyncSession,
        process_id: int,
        process_id_gen: ProcessIdGen,
        initial_population: List[Genotype],
        rng: Random,
        innov_db_body: multineat.InnovationDatabase,
        innov_db_brain: multineat.InnovationDatabase,
        simulation_time: int,
        sampling_frequency: float,
        control_frequency: float,
        num_generations: int,
        offspring_size: int,
        fitness_measure: str,
        experiment_name: str,
        max_modules: int,
        crossover_prob: float,
        mutation_prob: float,
        substrate_radius: str,
        run_simulation: bool,
        env_conditions: List,
        plastic_body: int,
        plastic_brain: int,
        simulator: str
    ) -> None:
        await super().ainit_new(
            database=database,
            session=session,
            process_id=process_id,
            process_id_gen=process_id_gen,
            genotype_type=Genotype,
            genotype_serializer=GenotypeSerializer,
            states_serializer=StatesSerializer,
            measures_type=float,
            measures_serializer=FloatSerializer,
            initial_population=initial_population,
            fitness_measure=fitness_measure,
            offspring_size=offspring_size,
            experiment_name=experiment_name,
            max_modules=max_modules,
            crossover_prob=crossover_prob,
            mutation_prob=mutation_prob,
            substrate_radius=substrate_radius,
            run_simulation=run_simulation,
            env_conditions=env_conditions,
            plastic_body=plastic_body,
            plastic_brain=plastic_brain
        )

        self._process_id = process_id
        self._env_conditions = env_conditions
        self._simulator = simulator
        self._init_runner()
        self._innov_db_body = innov_db_body
        self._innov_db_brain = innov_db_brain
        self._rng = rng
        self._simulation_time = simulation_time
        self._sampling_frequency = sampling_frequency
        self._control_frequency = control_frequency
        self._num_generations = num_generations
        self._fitness_measure = fitness_measure
        self._offspring_size = offspring_size
        self._experiment_name = experiment_name
        self._max_modules = max_modules
        self._crossover_prob = crossover_prob
        self._mutation_prob = mutation_prob
        self._substrate_radius = substrate_radius
        self._plastic_body = plastic_body,
        self._plastic_brain = plastic_brain
        self._run_simulation = run_simulation

        # create database structure if not exists
        # TODO this works but there is probably a better way
        await (await session.connection()).run_sync(DbBase.metadata.create_all)

        # save to database
        self._on_generation_checkpoint(session)

    async def ainit_from_database(  # type: ignore # see comment at ainit_new
        self,
        database: AsyncEngine,
        session: AsyncSession,
        process_id: int,
        process_id_gen: ProcessIdGen,
        rng: Random,
        innov_db_body: multineat.InnovationDatabase,
        innov_db_brain: multineat.InnovationDatabase,
        run_simulation: int,
        num_generations: int,
        simulator: str
    ) -> bool:
        if not await super().ainit_from_database(
            database=database,
            session=session,
            process_id=process_id,
            process_id_gen=process_id_gen,
            genotype_type=Genotype,
            genotype_serializer=GenotypeSerializer,
            states_serializer=StatesSerializer,
            measures_type=float,
            measures_serializer=FloatSerializer,
            run_simulation=run_simulation,
        ):
            return False

        self._process_id = process_id
        self._simulator = simulator
        self._init_runner()

        opt_row = (
            (
                await session.execute(
                    select(DbOptimizerState)
                    .filter(DbOptimizerState.process_id == process_id)
                    .order_by(DbOptimizerState.generation_index.desc())
                )
            )
            .scalars()
            .first()
        )

        # if this happens something is wrong with the database
        if opt_row is None:
            raise IncompatibleError
            raise IncompatibleError

        self._simulation_time = opt_row.simulation_time
        self._sampling_frequency = opt_row.sampling_frequency
        self._control_frequency = opt_row.control_frequency
        self._num_generations = num_generations

        self._rng = rng
        self._rng.setstate(pickle.loads(opt_row.rng))

        self._innov_db_body = innov_db_body
        self._innov_db_body.Deserialize(opt_row.innov_db_body)
        self._innov_db_brain = innov_db_brain
        self._innov_db_brain.Deserialize(opt_row.innov_db_brain)
        self._run_simulation = run_simulation


        return True

    def _init_runner(self) -> None:
        self._runner = {}

        for env in self.env_conditions:
            if self._simulator == 'isaac':
                self._runner[env] = (LocalRunnerI(LocalRunnerI.SimParams(),
                                                  headless=True,
                                                  env_conditions=self.env_conditions[env]))
            elif self._simulator == 'mujoco':
                self._runner[env] = (LocalRunnerM(headless=True))

    def _select_parents(
        self,
        population: List[Genotype],
        fitnesses: List[float],
        num_parent_groups: int,
    ) -> List[List[int]]:

        # TODO: allow variable number
        #  and adapt the to_database to take the crossover probabilistic choice into consideration
        if self.crossover_prob == 0:
            number_of_parents = 1
        else:
            number_of_parents = 2

        return [
            selection.multiple_unique(
                number_of_parents,
                population,
                fitnesses,
                lambda _, fitnesses: selection.tournament(self._rng, fitnesses, k=2),
            )
            for _ in range(num_parent_groups)
        ]

    def _select_survivors(
        self,
        old_individuals: List[Genotype],
        old_fitnesses: List[float],
        new_individuals: List[Genotype],
        new_fitnesses: List[float],
        num_survivors: int,
    ) -> Tuple[List[int], List[int]]:

        assert len(old_individuals) == num_survivors

        return population_management.steady_state(
            old_individuals,
            old_fitnesses,
            new_individuals,
            new_fitnesses,
            lambda _, fitnesses: selection.tournament(self._rng, fitnesses, k=2),
        )

    def _must_do_next_gen(self) -> bool:
        return self.generation_index != self._num_generations

    def _crossover(self, parents: List[Genotype]) -> Genotype:
        if self._rng.uniform(0, 1) > self.crossover_prob:
            return parents[0]
        else:
            return crossover(parents[0], parents[1], self._rng)

    def _mutate(self, genotype: Genotype) -> Genotype:
        if self._rng.uniform(0, 1) > self.mutation_prob:
            return genotype
        else:
            return mutate(genotype, self._innov_db_body, self._innov_db_brain, self._rng)

    async def _evaluate_generation(
            self,
            genotypes: List[Genotype],
            database: AsyncEngine,
            process_id: int,
            process_id_gen: ProcessIdGen,
    ) -> List[float]:

        envs_measures_genotypes = {}
        envs_states_genotypes = {}
        envs_queried_substrates = {}

        for cond in self.env_conditions:

            if self._simulator == 'isaac':
                control_function = self._control_isaac
            elif self._simulator == 'mujoco':
                control_function = self._control_mujoco
                
            batch = Batch(
                simulation_time=self._simulation_time,
                sampling_frequency=self._sampling_frequency,
                control_frequency=self._control_frequency,
                control=control_function,
            )

            self._controllers = []
            phenotypes = []
            queried_substrates = []

            for genotype in genotypes:
                phenotype, queried_substrate = develop(genotype, genotype.mapping_seed, self.max_modules,
                                                       self.substrate_radius,
                                                       self.env_conditions[cond], len(self.env_conditions),
                                                       self.plastic_body, self.plastic_brain)
                phenotypes.append(phenotype)
                queried_substrates.append(queried_substrate)

                actor, controller = phenotype.make_actor_and_controller()
                bounding_box = actor.calc_aabb()
                self._controllers.append(controller)
                env = Environment()

                x_rotation_degrees = float(self.env_conditions[cond][2])
                robot_rotation = x_rotation_degrees * np.pi / 180
                env.actors.append(
                    PosedActor(
                        actor,
                        Vector3(
                            [
                                0.0,
                                0.0,
                                (bounding_box.size.z / 2.0 - bounding_box.offset.z),
                            ]
                        ),
                        Quaternion.from_eulers([robot_rotation, 0, 0]),
                        [0.0 for _ in controller.get_dof_targets()],
                    )
                )
                batch.environments.append(env)

            envs_queried_substrates[cond] = queried_substrates

            if self._run_simulation:
                states = await self._runner[cond].run_batch(batch)
            else:
                states = None

            measures_genotypes = []
            for i, phenotype in enumerate(phenotypes):
                m = Measure(states=states, genotype_idx=i, phenotype=phenotype, \
                            generation=self.generation_index, simulation_time=self._simulation_time)
                measures_genotypes.append(m.measure_all_non_relative())
            envs_measures_genotypes[cond] = measures_genotypes

            states_genotypes = []
            if states is not None:
                for idx_genotype in range(0, len(states.environment_results)):
                    states_genotypes.append({})
                    for idx_state in range(0, len(states.environment_results[idx_genotype].environment_states)):
                        states_genotypes[-1][idx_state] = \
                            states.environment_results[idx_genotype].environment_states[idx_state].actor_states[
                                0].serialize()
            envs_states_genotypes[cond] = states_genotypes

        envs_measures_genotypes = self.measure_plasticity(envs_queried_substrates, envs_measures_genotypes)

        return envs_measures_genotypes, envs_states_genotypes

    def measure_plasticity(self, envs_queried_substrates, envs_measures_genotypes):

        if len(self.env_conditions) > 1:
            # TODO: this works only for two seasons
            first_cond = list(self.env_conditions.keys())[0]
            second_cond = list(self.env_conditions.keys())[1]
            for idg in range(0, len(envs_queried_substrates[first_cond])):

                keys_first = set(envs_queried_substrates[first_cond][idg].keys())
                keys_second = set(envs_queried_substrates[second_cond][idg].keys())
                intersection = keys_first & keys_second
                disjunct_first = [a for a in keys_first if a not in intersection]
                disjunct_second = [b for b in keys_second if b not in intersection]
                body_changes = len(disjunct_first) + len(disjunct_second)

                for i in intersection:
                    if type(envs_queried_substrates[first_cond][idg][i]) != type(envs_queried_substrates[second_cond][idg][i]):
                        body_changes += 1

                envs_measures_genotypes[first_cond][idg]['body_changes'] = body_changes
                envs_measures_genotypes[second_cond][idg]['body_changes'] = body_changes
        else:
            any_cond = list(self.env_conditions.keys())[0]
            for idg in range(0, len(envs_queried_substrates[any_cond])):
                envs_measures_genotypes[any_cond][idg]['body_changes'] = 0

        return envs_measures_genotypes

    def _control_isaac(self, dt: float, control: ActorControl) -> None:
        for control_i, controller in enumerate(self._controllers):
            controller.step(dt)
            control.set_dof_targets(control_i, 0, controller.get_dof_targets())
            
    def _control_mujoco(
            self, environment_index: int, dt: float, control: ActorControl
    ) -> None:
        controller = self._controllers[environment_index]
        controller.step(dt)
        control.set_dof_targets(environment_index, 0, controller.get_dof_targets())
            
    def _on_generation_checkpoint(self, session: AsyncSession) -> None:
        session.add(
            DbOptimizerState(
                process_id=self._process_id,
                generation_index=self.generation_index,
                rng=pickle.dumps(self._rng.getstate()),
                innov_db_body=self._innov_db_body.Serialize(),
                innov_db_brain=self._innov_db_brain.Serialize(),
                simulation_time=self._simulation_time,
                sampling_frequency=self._sampling_frequency,
                control_frequency=self._control_frequency,
            )
        )


class Optimizer(EAOptimizer[Genotype, float]):
    _process_id: int

    _runner: Runner

    _controllers: List[ActorController]

    _innov_db_body: multineat.InnovationDatabase
    _innov_db_brain: multineat.InnovationDatabase

    _rng: Random

    _simulation_time: int
    _sampling_frequency: float
    _control_frequency: float

    _num_generations: int
    _offspring_size: int
    _fitness_measure: str
    _experiment_name: str
    _max_modules: int
    _crossover_prob: float
    _mutation_prob: float
    _substrate_radius: str
    _run_simulation: bool
    _env_conditions: List
    _plastic_body: int
    _plastic_brain: int
    _simulator: str

    async def ainit_new(
            # type: ignore # TODO for now ignoring mypy complaint about LSP problem, override parent's ainit
            self,
            database: AsyncEngine,
            session: AsyncSession,
            process_id: int,
            process_id_gen: ProcessIdGen,
            initial_population: List[Genotype],
            rng: Random,
            innov_db_body: multineat.InnovationDatabase,
            innov_db_brain: multineat.InnovationDatabase,
            simulation_time: int,
            sampling_frequency: float,
            control_frequency: float,
            num_generations: int,
            offspring_size: int,
            fitness_measure: str,
            experiment_name: str,
            max_modules: int,
            crossover_prob: float,
            mutation_prob: float,
            substrate_radius: str,
            run_simulation: bool,
            env_conditions: List,
            plastic_body: int,
            plastic_brain: int,
            simulator: str
    ) -> None:
        await super().ainit_new(
            database=database,
            session=session,
            process_id=process_id,
            process_id_gen=process_id_gen,
            genotype_type=Genotype,
            genotype_serializer=GenotypeSerializer,
            states_serializer=StatesSerializer,
            measures_type=float,
            measures_serializer=FloatSerializer,
            initial_population=initial_population,
            fitness_measure=fitness_measure,
            offspring_size=offspring_size,
            experiment_name=experiment_name,
            max_modules=max_modules,
            crossover_prob=crossover_prob,
            mutation_prob=mutation_prob,
            substrate_radius=substrate_radius,
            run_simulation=run_simulation,
            env_conditions=env_conditions,
            plastic_body=plastic_body,
            plastic_brain=plastic_brain
        )

        self._process_id = process_id
        self._env_conditions = env_conditions
        self._simulator = simulator
        self._init_runner()
        self._innov_db_body = innov_db_body
        self._innov_db_brain = innov_db_brain
        self._rng = rng
        self._simulation_time = simulation_time
        self._sampling_frequency = sampling_frequency
        self._control_frequency = control_frequency
        self._num_generations = num_generations
        self._fitness_measure = fitness_measure
        self._offspring_size = offspring_size
        self._experiment_name = experiment_name
        self._max_modules = max_modules
        self._crossover_prob = crossover_prob
        self._mutation_prob = mutation_prob
        self._substrate_radius = substrate_radius
        self._plastic_body = plastic_body,
        self._plastic_brain = plastic_brain
        self._run_simulation = run_simulation

        # create database structure if not exists
        # TODO this works but there is probably a better way
        await (await session.connection()).run_sync(DbBase.metadata.create_all)

        # save to database
        self._on_generation_checkpoint(session)

    async def ainit_from_database(  # type: ignore # see comment at ainit_new
            self,
            database: AsyncEngine,
            session: AsyncSession,
            process_id: int,
            process_id_gen: ProcessIdGen,
            rng: Random,
            innov_db_body: multineat.InnovationDatabase,
            innov_db_brain: multineat.InnovationDatabase,
            run_simulation: int,
            num_generations: int,
            simulator: str
    ) -> bool:
        if not await super().ainit_from_database(
                database=database,
                session=session,
                process_id=process_id,
                process_id_gen=process_id_gen,
                genotype_type=Genotype,
                genotype_serializer=GenotypeSerializer,
                states_serializer=StatesSerializer,
                measures_type=float,
                measures_serializer=FloatSerializer,
                run_simulation=run_simulation,
        ):
            return False

        self._process_id = process_id
        self._simulator = simulator
        self._init_runner()

        opt_row = (
            (
                await session.execute(
                    select(DbOptimizerState)
                    .filter(DbOptimizerState.process_id == process_id)
                    .order_by(DbOptimizerState.generation_index.desc())
                )
            )
            .scalars()
            .first()
        )

        # if this happens something is wrong with the database
        if opt_row is None:
            raise IncompatibleError
            raise IncompatibleError

        self._simulation_time = opt_row.simulation_time
        self._sampling_frequency = opt_row.sampling_frequency
        self._control_frequency = opt_row.control_frequency
        self._num_generations = num_generations

        self._rng = rng
        self._rng.setstate(pickle.loads(opt_row.rng))

        self._innov_db_body = innov_db_body
        self._innov_db_body.Deserialize(opt_row.innov_db_body)
        self._innov_db_brain = innov_db_brain
        self._innov_db_brain.Deserialize(opt_row.innov_db_brain)
        self._run_simulation = run_simulation

        return True

    def _init_runner(self) -> None:
        self._runner = {}

        for env in self.env_conditions:
            if self._simulator == 'isaac':
                self._runner[env] = (LocalRunnerI(LocalRunnerI.SimParams(),
                                                  headless=True,
                                                  env_conditions=self.env_conditions[env]))
            elif self._simulator == 'mujoco':
                self._runner[env] = (SlantedEnvRunnerMujoco(headless=False))

    def _select_parents(
            self,
            population: List[Genotype],
            fitnesses: List[float],
            num_parent_groups: int,
    ) -> List[List[int]]:

        # TODO: allow variable number
        #  and adapt the to_database to take the crossover probabilistic choice into consideration
        if self.crossover_prob == 0:
            number_of_parents = 1
        else:
            number_of_parents = 2

        return [
            selection.multiple_unique(
                number_of_parents,
                population,
                fitnesses,
                lambda _, fitnesses: selection.tournament(self._rng, fitnesses, k=2),
            )
            for _ in range(num_parent_groups)
        ]

    def _select_survivors(
            self,
            old_individuals: List[Genotype],
            old_fitnesses: List[float],
            new_individuals: List[Genotype],
            new_fitnesses: List[float],
            num_survivors: int,
    ) -> Tuple[List[int], List[int]]:

        assert len(old_individuals) == num_survivors

        return population_management.steady_state(
            old_individuals,
            old_fitnesses,
            new_individuals,
            new_fitnesses,
            lambda _, fitnesses: selection.tournament(self._rng, fitnesses, k=2),
        )

    def _must_do_next_gen(self) -> bool:
        return self.generation_index != self._num_generations

    def _crossover(self, parents: List[Genotype]) -> Genotype:
        if self._rng.uniform(0, 1) > self.crossover_prob:
            return parents[0]
        else:
            return crossover(parents[0], parents[1], self._rng)

    def _mutate(self, genotype: Genotype) -> Genotype:
        if self._rng.uniform(0, 1) > self.mutation_prob:
            return genotype
        else:
            return mutate(genotype, self._innov_db_body, self._innov_db_brain, self._rng)

    async def _evaluate_generation(
            self,
            genotypes: List[Genotype],
            database: AsyncEngine,
            process_id: int,
            process_id_gen: ProcessIdGen,
    ) -> List[float]:

        envs_measures_genotypes = {}
        envs_states_genotypes = {}
        envs_queried_substrates = {}

        for cond in self.env_conditions:

            if self._simulator == 'isaac':
                control_function = self._control_isaac
            elif self._simulator == 'mujoco':
                control_function = self._control_mujoco

            batch = Batch(
                simulation_time=self._simulation_time,
                sampling_frequency=self._sampling_frequency,
                control_frequency=self._control_frequency,
                control=control_function,
            )

            self._controllers = []
            phenotypes = []
            queried_substrates = []

            for genotype in genotypes:
                phenotype, queried_substrate = develop(genotype, genotype.mapping_seed, self.max_modules,
                                                       self.substrate_radius,
                                                       self.env_conditions[cond], len(self.env_conditions),
                                                       self.plastic_body, self.plastic_brain)
                phenotypes.append(phenotype)
                queried_substrates.append(queried_substrate)

                actor, controller = phenotype.make_actor_and_controller()
                bounding_box = actor.calc_aabb()
                self._controllers.append(controller)
                env = Environment()

                x_rotation_degrees = float(self.env_conditions[cond][2])
                robot_rotation = x_rotation_degrees * np.pi / 180
                env.actors.append(
                    PosedActor(
                        actor,
                        Vector3(
                            [
                                0.0,
                                0.0,
                                (bounding_box.size.z / 2.0 - bounding_box.offset.z),
                            ]
                        ),
                        Quaternion.from_eulers([robot_rotation, 0, 0]),
                        [0.0 for _ in controller.get_dof_targets()],
                    )
                )
                batch.environments.append(env)

            envs_queried_substrates[cond] = queried_substrates

            if self._run_simulation:
                states = await self._runner[cond].run_batch(batch)
            else:
                states = None

            measures_genotypes = []
            for i, phenotype in enumerate(phenotypes):
                m = Measure(states=states, genotype_idx=i, phenotype=phenotype, \
                            generation=self.generation_index, simulation_time=self._simulation_time)
                measures_genotypes.append(m.measure_all_non_relative())
            envs_measures_genotypes[cond] = measures_genotypes

            states_genotypes = []
            if states is not None:
                for idx_genotype in range(0, len(states.environment_results)):
                    states_genotypes.append({})
                    for idx_state in range(0, len(states.environment_results[idx_genotype].environment_states)):
                        states_genotypes[-1][idx_state] = \
                            states.environment_results[idx_genotype].environment_states[idx_state].actor_states[
                                0].serialize()
            envs_states_genotypes[cond] = states_genotypes

        envs_measures_genotypes = self.measure_plasticity(envs_queried_substrates, envs_measures_genotypes)

        return envs_measures_genotypes, envs_states_genotypes

    def measure_plasticity(self, envs_queried_substrates, envs_measures_genotypes):

        if len(self.env_conditions) > 1:
            # TODO: this works only for two seasons
            first_cond = list(self.env_conditions.keys())[0]
            second_cond = list(self.env_conditions.keys())[1]
            for idg in range(0, len(envs_queried_substrates[first_cond])):

                keys_first = set(envs_queried_substrates[first_cond][idg].keys())
                keys_second = set(envs_queried_substrates[second_cond][idg].keys())
                intersection = keys_first & keys_second
                disjunct_first = [a for a in keys_first if a not in intersection]
                disjunct_second = [b for b in keys_second if b not in intersection]
                body_changes = len(disjunct_first) + len(disjunct_second)

                for i in intersection:
                    if type(envs_queried_substrates[first_cond][idg][i]) != type(
                            envs_queried_substrates[second_cond][idg][i]):
                        body_changes += 1

                envs_measures_genotypes[first_cond][idg]['body_changes'] = body_changes
                envs_measures_genotypes[second_cond][idg]['body_changes'] = body_changes
        else:
            any_cond = list(self.env_conditions.keys())[0]
            for idg in range(0, len(envs_queried_substrates[any_cond])):
                envs_measures_genotypes[any_cond][idg]['body_changes'] = 0

        return envs_measures_genotypes

    def _control_isaac(self, dt: float, control: ActorControl) -> None:
        for control_i, controller in enumerate(self._controllers):
            controller.step(dt)
            control.set_dof_targets(control_i, 0, controller.get_dof_targets())

    def _control_mujoco(
            self, environment_index: int, dt: float, control: ActorControl
    ) -> None:
        controller = self._controllers[environment_index]
        controller.step(dt)
        control.set_dof_targets(environment_index, 0, controller.get_dof_targets())

    def _on_generation_checkpoint(self, session: AsyncSession) -> None:
        session.add(
            DbOptimizerState(
                process_id=self._process_id,
                generation_index=self.generation_index,
                rng=pickle.dumps(self._rng.getstate()),
                innov_db_body=self._innov_db_body.Serialize(),
                innov_db_brain=self._innov_db_brain.Serialize(),
                simulation_time=self._simulation_time,
                sampling_frequency=self._sampling_frequency,
                control_frequency=self._control_frequency,
            )
        )


DbBase = declarative_base()


class DbOptimizerState(DbBase):
    __tablename__ = "optimizer"

    process_id = sqlalchemy.Column(
        sqlalchemy.Integer,
        nullable=False,
        primary_key=True,
    )
    generation_index = sqlalchemy.Column(
        sqlalchemy.Integer, nullable=False, primary_key=True
    )
    rng = sqlalchemy.Column(sqlalchemy.PickleType, nullable=False)
    innov_db_body = sqlalchemy.Column(sqlalchemy.String, nullable=False)
    innov_db_brain = sqlalchemy.Column(sqlalchemy.String, nullable=False)
    simulation_time = sqlalchemy.Column(sqlalchemy.Integer, nullable=False)
    sampling_frequency = sqlalchemy.Column(sqlalchemy.Float, nullable=False)
    control_frequency = sqlalchemy.Column(sqlalchemy.Float, nullable=False)



class SlantedEnvRunnerMujoco(LocalRunnerM):
    """Runner for simulating using Mujoco."""

    _headless: bool

    def __init__(self, headless: bool = False):
        """
        Initialize this object.

        :param headless: If True, the simulation will not be rendered. This drastically improves performance.
        """
        self._headless = headless

    async def run_batch(self, batch: Batch) -> BatchResults:
        """
        Run the provided batch by simulating each contained environment.

        :param batch: The batch to run.
        :returns: List of simulation states in ascending order of time.
        """
        logging.info("Starting simulation batch with mujoco.")

        control_step = 1 / batch.control_frequency
        sample_step = 1 / batch.sampling_frequency

        results = BatchResults([EnvironmentResults([]) for _ in batch.environments])

        for env_index, env_descr in enumerate(batch.environments):
            logging.info(f"Environment {env_index}")

            model = mujoco.MjModel.from_xml_string(self._make_mjcf(env_descr))

            # TODO initial dof state
            data = mujoco.MjData(model)

            initial_targets = [
                dof_state
                for posed_actor in env_descr.actors
                for dof_state in posed_actor.dof_states
            ]
            self._set_dof_targets(data, initial_targets)

            for posed_actor in env_descr.actors:
                posed_actor.dof_states

            if not self._headless:
                viewer = mujoco_viewer.MujocoViewer(
                    model,
                    data,
                )

            last_control_time = 0.0
            last_sample_time = 0.0

            # sample initial state
            results.environment_results[env_index].environment_states.append(
                EnvironmentState(0.0, self._get_actor_states(env_descr, data, model))
            )

            while (time := data.time) < batch.simulation_time:
                # do control if it is time
                if time >= last_control_time + control_step:
                    last_control_time = math.floor(time / control_step) * control_step
                    control = ActorControl()
                    batch.control(env_index, control_step, control)
                    actor_targets = control._dof_targets
                    actor_targets.sort(key=lambda t: t[0])
                    targets = [
                        target
                        for actor_target in actor_targets
                        for target in actor_target[2]
                    ]
                    self._set_dof_targets(data, targets)

                # sample state if it is time
                if time >= last_sample_time + sample_step:
                    last_sample_time = int(time / sample_step) * sample_step
                    results.environment_results[env_index].environment_states.append(
                        EnvironmentState(
                            time, self._get_actor_states(env_descr, data, model)
                        )
                    )

                # step simulation
                mujoco.mj_step(model, data)

                if not self._headless:
                    viewer.render()

            if not self._headless:
                viewer.close()

            # sample one final time
            results.environment_results[env_index].environment_states.append(
                EnvironmentState(time, self._get_actor_states(env_descr, data, model))
            )

        logging.info("Finished batch.")

        return results

    @staticmethod
    def _make_mjcf(env_descr: Environment) -> str:
        env_mjcf = mjcf.RootElement(model="environment")

        env_mjcf.compiler.angle = "radian"

        env_mjcf.option.timestep = 0.0005
        env_mjcf.option.integrator = "RK4"

        env_mjcf.option.gravity = [0, 0, -9.81]

        """
        <asset>
            <texture builtin="gradient" height="100" rgb1="1 1 1" rgb2="0 0 0" type="skybox" width="100" />
            <texture builtin="flat" height="1278" mark="cross" markrgb="1 1 1" name="texgeom" random="0.01" rgb1="0.8 0.6 0.4" rgb2="0.8 0.6 0.4" type="cube" width="127" />
            <texture builtin="checker" height="100" name="texplane" rgb1="0.2 0.4 0.6" rgb2="0.8 0.8 0.8" type="2d" width="100" />
            <material name="MatPlane" reflectance="0.5" shininess="1" specular="1" texrepeat="60 60" texture="texplane" />
            <material name="geom" texture="texgeom" texuniform="true" />
            <hfield file="terrain1.png" name="mytilted" ncol="0" nrow="0" size="50 50 1 0.1" />
        </asset>
        """

        """
        env_mjcf.worldbody.add(
            "geom",
            name="ground",
            type="plane",
            size=[10, 10, 1],
            rgba=[0.2, 0.2, 0.2, 1],
        )
        """

        """    <geom conaffinity="1" condim="3" hfield="mytilted" material="MatPlane" name="floor" pos="0 0 0" rgba="0.8 0.9 0.8 1" type="hfield" /> """

        env_mjcf.asset.add(
            "texture",
            builtin = "gradient",
            height = 100,
            rgb1 = [1, 1, 1],
            rgb2 = [0, 0, 0],
            type = "skybox",
            width = 100
        )

        env_mjcf.asset.add(
            "texture",
            builtin="flat",
            height = 1278,
            mark = "cross",
            markrgb = [1, 1, 1],
            name = "texgeom",
            random = 0.01,
            rgb1 = [0.8, 0.6, 0.4],
            rgb2 = [0.8, 0.6, 0.4],
            type = "cube",
            width = 127
        )

        env_mjcf.asset.add(
            "texture",
            builtin = "checker",
            height = 100,
            name = "texplane",
            rgb1 = [0.2, 0.4, 0.6],
            rgb2 = [0.8, 0.8, 0.8],
            type = "2d",
            width = 100
        )


        env_mjcf.asset.add(
            "material",
            name = "MatPlane",
            reflectance = 0.5,
            shininess = 1,
            specular = 1,
            texrepeat = [60, 60],
            texture = "texplane"
        )

        env_mjcf.asset.add(
            "material",
            name = "geom1",
            texture = "texgeom",
            texuniform = "true"
        )

        env_mjcf.asset.add(
            "hfield",
            file = 'terrain1.png',
            name = "mytilted",
            ncol = 0,
            nrow = 0,
            size = [50, 50, 20, 0.1]
        )

        """    <geom conaffinity="1" condim="3" hfield="mytilted" material="MatPlane" name="floor" pos="0 0 0" rgba="0.8 0.9 0.8 1" type="hfield" /> """

        env_mjcf.worldbody.add(
            "geom",
            name="ground",
            conaffinity =1,
            condim=3,
            hfield="mytilted",
            material="MatPlane",
            type="hfield",
            pos=[0, 0, -10.5],
            rgba=[0.8, 0.9, 0.8, 1],
        )
        env_mjcf.worldbody.add(
            "light",
            pos=[0, 0, 100],
            ambient=[0.5, 0.5, 0.5],
            directional=True,
            castshadow=False,
        )
        env_mjcf.visual.headlight.active = 0

        for actor_index, posed_actor in enumerate(env_descr.actors):
            urdf = physbot_to_urdf(
                posed_actor.actor,
                f"robot_{actor_index}",
                Vector3(),
                Quaternion(),
            )

            model = mujoco.MjModel.from_xml_string(urdf)

            # mujoco can only save to a file, not directly to string,
            # so we create a temporary file.
            botfile = tempfile.NamedTemporaryFile(
                mode="r+", delete=False, suffix=".urdf"
            )
            mujoco.mj_saveLastXML(botfile.name, model)
            robot = mjcf.from_file(botfile)
            botfile.close()

            for joint in posed_actor.actor.joints:
                robot.actuator.add(
                    "position",
                    kp=5.0,
                    joint=robot.find(
                        namespace="joint",
                        identifier=joint.name,
                    ),
                )
                robot.actuator.add(
                    "velocity",
                    kv=0.05,
                    joint=robot.find(namespace="joint", identifier=joint.name),
                )

            attachment_frame = env_mjcf.attach(robot)
            attachment_frame.add("freejoint")
            attachment_frame.pos = [
                posed_actor.position.x,
                posed_actor.position.y,
                posed_actor.position.z,
            ]

            attachment_frame.quat = [
                posed_actor.orientation.x,
                posed_actor.orientation.y,
                posed_actor.orientation.z,
                posed_actor.orientation.w,
            ]

        xml = env_mjcf.to_xml_string()

        print(xml)


        print(os.getcwd())
        import xmltodict

        xml2_dict  =  xmltodict.parse(xml)

        xml2_dict["mujoco"]["asset"]["hfield"]["@file"] = "terrain1.png"
        xml2=xmltodict.unparse(xml2_dict)

        if not isinstance(xml, str):
            raise RuntimeError("Error generating mjcf xml.")

        return xml2

    @classmethod
    def _get_actor_states(
            cls, env_descr: Environment, data: mujoco.MjData, model: mujoco.MjModel
    ) -> List[ActorState]:
        return [
            cls._get_actor_state(i, data, model) for i in range(len(env_descr.actors))
        ]

    @staticmethod
    def _get_actor_state(
            robot_index: int, data: mujoco.MjData, model: mujoco.MjModel
    ) -> ActorState:
        bodyid = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_BODY,
            f"robot_{robot_index}/",  # the slash is added by dm_control. ugly but deal with it
        )
        assert bodyid >= 0

        qindex = model.body_jntadr[bodyid]

        # explicitly copy because the Vector3 and Quaternion classes don't copy the underlying structure
        position = Vector3([n for n in data.qpos[qindex: qindex + 3]])
        orientation = Quaternion([n for n in data.qpos[qindex + 3: qindex + 3 + 4]])

        return ActorState(position, orientation)

    @staticmethod
    def _set_dof_targets(data: mujoco.MjData, targets: List[float]) -> None:
        if len(targets) * 2 != len(data.ctrl):
            raise RuntimeError("Need to set a target for every dof")
        for i, target in enumerate(targets):
            data.ctrl[2 * i] = target
            data.ctrl[2 * i + 1] = 0

