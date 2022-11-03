from __future__ import annotations

import logging
from abc import abstractmethod
from dataclasses import dataclass
from typing import Generic, List, Optional, Tuple, Type, TypeVar, Dict
import math

from sqlalchemy.exc import OperationalError
from sqlalchemy.ext.asyncio import AsyncEngine
from sqlalchemy.ext.asyncio.session import AsyncSession
from sqlalchemy.future import select
from sqlalchemy.orm.exc import MultipleResultsFound, NoResultFound

from revolve2.core.database import IncompatibleError, Serializer
from revolve2.core.modular_robot import MeasureRelative
from revolve2.core.optimization import Process, ProcessIdGen
from ast import literal_eval
import sys

from ._database import (
    DbBase,
    DbEnvconditions,
    DbEAOptimizer,
    DbEAOptimizerGeneration,
    DbEAOptimizerIndividual,
    DbEAOptimizerParent,
    DbEAOptimizerState,
)

import pprint

Genotype = TypeVar("Genotype")
Measure = TypeVar("Measure")


class EAOptimizer(Process, Generic[Genotype, Measure]):

    @abstractmethod
    async def _evaluate_generation(
        self,
        genotypes: List[Genotype],
        database: AsyncEngine,
        process_id: int,
        process_id_gen: ProcessIdGen,
    ) -> List[Measure]:
        """
        Evaluate a genotype.

        :param genotypes: The genotypes to evaluate. Must not be altered.
        :return: The Measure result.
        """

    @abstractmethod
    def _select_parents(
        self,
        population: List[Genotype],
        measures: List[Measure],
        num_parent_groups: int,
    ) -> List[List[int]]:
        """
        Select groups of parents that will create offspring.

        :param population: The generation to select sets of parents from. Must not be altered.
        :return: The selected sets of parents, each integer representing a population index.
        """

    @abstractmethod
    def _select_survivors(
        self,
        old_individuals: List[Genotype],
        old_measures: List[Measure],
        new_individuals: List[Genotype],
        new_measures: List[Measure],
        num_survivors: int,
    ) -> Tuple[List[int], List[int]]:
        """
        Select survivors from a group of individuals. These will form the next generation.

        :param individuals: The individuals to choose from.
        :param num_survivors: How many individuals should be selected.
        :return: Indices of the old survivors and indices of the new survivors.
        """

    @abstractmethod
    def _crossover(self, parents: List[Genotype]) -> Genotype:
        """
        Combine a set of genotypes into a new genotype.

        :param parents: The set of genotypes to combine. Must not be altered.
        :return: The new genotype.
        """

    @abstractmethod
    def _mutate(self, genotype: Genotype) -> Genotype:
        """
        Apply mutation to an genotype to create a new genotype.

        :param genotype: The original genotype. Must not be altered.
        :return: The new genotype.
        """

    @abstractmethod
    def _must_do_next_gen(self) -> bool:
        """
        Decide if the optimizer must do another generation.
        :return: True if it must.
        """

    @abstractmethod
    def _on_generation_checkpoint(self, session: AsyncSession) -> None:
        """
        This function is called after a generation is finished and results and state are saved to the database.
        Use it to store state and results of the optimizer.
        The session must not be committed, but it may be flushed.
        """

    __database: AsyncEngine

    __ea_optimizer_id: int

    __genotype_type: Type[Genotype]
    __genotype_serializer: Type[Serializer[Genotype]]
    __states_serializer: List[Tuple[float, State]]
    __measures_type: Type[Measure]
    __measures_serializer: Type[Serializer[Measure]]

    __offspring_size: int

    __process_id_gen: ProcessIdGen

    __next_individual_id: int

    __latest_population: List[_Individual[Genotype]]
    __latest_measures: Optional[List[Measure]]  # None only for the initial population
    __latest_states: List[Tuple[float, State]]
    __generation_index: int
    __fitness_measure: str
    __experiment_name: str
    __max_modules: int
    __substrate_radius: str
    __crossover_prob: float
    __mutation_prob: float
    __run_simulation: bool
    __env_conditions: List
    __plastic_body: int
    __plastic_brain: int

    async def ainit_new(
        self,
        database: AsyncEngine,
        session: AsyncSession,
        process_id: int,
        process_id_gen: ProcessIdGen,
        genotype_type: Type[Genotype],
        genotype_serializer: Type[Serializer[Genotype]],
        states_serializer: List[Tuple[float, State]],
        measures_type: Type[Measure],
        measures_serializer: Type[Serializer[Measure]],
        initial_population: List[Genotype],
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
        plastic_brain: int
    ) -> None:
        """
        :id: Unique id between all EAOptimizers in this database.
        :offspring_size: Number of offspring made by the population each generation.
        """
        self.__database = database
        self.__genotype_type = genotype_type
        self.__genotype_serializer = genotype_serializer
        self.__measures_type = measures_type
        self.__measures_serializer = measures_serializer
        self.__states_serializer = states_serializer
        self.__process_id_gen = process_id_gen
        self.__next_individual_id = 1
        self.__latest_measures = None
        self.__latest_states = None
        self.__generation_index = 0
        self.__offspring_size = offspring_size
        self.__fitness_measure = fitness_measure
        self.__experiment_name = experiment_name
        self.__max_modules = max_modules
        self.__crossover_prob = crossover_prob
        self.__mutation_prob = mutation_prob
        self.__substrate_radius = substrate_radius
        self.__run_simulation = run_simulation
        self.__env_conditions = env_conditions
        self.__plastic_body = plastic_body
        self.__plastic_brain = plastic_brain

        self.__latest_population = [
            _Individual(self.__gen_next_individual_id(), g, [])
            for g in initial_population
        ]

        await (await session.connection()).run_sync(DbBase.metadata.create_all)
        await self.__genotype_serializer.create_tables(session)
        await self.__measures_serializer.create_tables(session)
        await self.__states_serializer.create_tables(session)

        new_opt = DbEAOptimizer(
            process_id=process_id,
            genotype_table=self.__genotype_serializer.identifying_table(),
            measures_table=self.__measures_serializer.identifying_table(),
            states_table=self.__states_serializer.identifying_table(),
            fitness_measure=self.__fitness_measure,
            offspring_size=self.__offspring_size,
            experiment_name=self.__experiment_name,
            max_modules=self.__max_modules,
            crossover_prob=self.__crossover_prob,
            mutation_prob=self.__mutation_prob,
            substrate_radius=self.__substrate_radius,
            plastic_body=self.__plastic_body,
            plastic_brain=self.__plastic_brain,
        )
        session.add(new_opt)
        await session.flush()
        assert new_opt.id is not None  # this is impossible because it's not nullable
        self.__ea_optimizer_id = new_opt.id

        conditions = [
            DbEnvconditions(
                ea_optimizer_id=self.__ea_optimizer_id,
                conditions=str(cond), )
            for cond in self.__env_conditions
        ]
        session.add_all(conditions)
        await session.flush()
        self.__env_conditions = {}
        for c in conditions:
            self.__env_conditions[c.id] = literal_eval(c.conditions)

        await self.__save_generation_using_session(
            session, None, None, None, None, self.__latest_population, None, None, None
        )

    @property
    def max_modules(self):
        return self.__max_modules

    @property
    def substrate_radius(self):
        return self.__substrate_radius

    @property
    def crossover_prob(self):
        return self.__crossover_prob

    @property
    def mutation_prob(self):
        return self.__mutation_prob

    @property
    def env_conditions(self):
        return self.__env_conditions

    @property
    def plastic_body(self):
        return self.__plastic_body

    @property
    def plastic_brain(self):
        return self.__plastic_brain

    async def ainit_from_database(
        self,
        database: AsyncEngine,
        session: AsyncSession,
        process_id: int,
        process_id_gen: ProcessIdGen,
        genotype_type: Type[Genotype],
        genotype_serializer: Type[Serializer[Genotype]],
        states_serializer: List[Tuple[float, State]],
        measures_type: Type[Measure],
        measures_serializer: Type[Serializer[Measure]],
        run_simulation: int
    ) -> bool:
        self.__database = database
        self.__genotype_type = genotype_type
        self.__genotype_serializer = genotype_serializer
        self.__states_serializer = states_serializer
        self.__measures_type = measures_type
        self.__measures_serializer = measures_serializer
        self.__run_simulation = run_simulation

        try:
            eo_row = (
                (
                    await session.execute(
                        select(DbEAOptimizer).filter(
                            DbEAOptimizer.process_id == process_id
                        )
                    )
                )
                .scalars()
                .one()
            )
        except MultipleResultsFound as err:
            raise IncompatibleError() from err
        except (NoResultFound, OperationalError):
            return False

        self.__ea_optimizer_id = eo_row.id
        self.__fitness_measure = eo_row.fitness_measure
        self.__offspring_size = eo_row.offspring_size
        self.__experiment_name = eo_row.experiment_name
        self.__max_modules = eo_row.max_modules
        self.__crossover_prob = eo_row.crossover_prob
        self.__mutation_prob = eo_row.mutation_prob
        self.__substrate_radius = eo_row.substrate_radius
        self.__plastic_body = eo_row.plastic_body
        self.__plastic_brain = eo_row.plastic_brain

        c_rows = ((await session.execute(
                    select(DbEnvconditions).filter(
                        DbEnvconditions.ea_optimizer_id == self.__ea_optimizer_id
                    ))).all())
        self.__env_conditions = {}
        for c_row in c_rows:
            self.__env_conditions[c_row[0].id] = literal_eval(c_row[0].conditions)

        # TODO: this name 'state' conflicts a bit with the table of states (positions)...
        # the one below is more like 'status'
        state_row = (
            (
                await session.execute(
                    select(DbEAOptimizerState)
                    .filter(
                        DbEAOptimizerState.ea_optimizer_id == self.__ea_optimizer_id
                    )
                    .order_by(DbEAOptimizerState.generation_index.desc())
                )
            )
            .scalars()
            .first()
        )

        if state_row is None:
            raise IncompatibleError()  # not possible that there is no saved state but DbEAOptimizer row exists

        self.__generation_index = state_row.generation_index
        self.__process_id_gen = process_id_gen
        self.__process_id_gen.set_state(state_row.processid_state)

        any_cond = list(self.__env_conditions.keys())[0]
        gen_rows = (
            (
                await session.execute(
                    select(DbEAOptimizerGeneration)
                    .filter(
                        (
                            DbEAOptimizerGeneration.ea_optimizer_id
                            == self.__ea_optimizer_id
                        )
                        & (
                            DbEAOptimizerGeneration.generation_index
                            == self.__generation_index
                        )
                        & (DbEAOptimizerGeneration.env_conditions_id == any_cond)
                    )
                    .order_by(DbEAOptimizerGeneration.individual_index)
                )
            )
            .scalars()
            .all()
        )

        individual_ids = [row.individual_id for row in gen_rows]

        individual_rows = (
            (
                await session.execute(
                    select(DbEAOptimizerIndividual).filter(
                        (
                            DbEAOptimizerIndividual.ea_optimizer_id
                            == self.__ea_optimizer_id
                        )
                        & (DbEAOptimizerIndividual.individual_id.in_(individual_ids))
                    )
                )
            )
            .scalars()
            .all()
        )
        individual_map = {(i.individual_id, i.env_conditions_id): i for i in individual_rows}

        all_individual_rows = (
            (
                await session.execute(
                    select(DbEAOptimizerIndividual).filter(
                        (DbEAOptimizerIndividual.ea_optimizer_id == self.__ea_optimizer_id)))
            )
            .scalars()
            .all()
        )
        all_individual_ids = {i.individual_id for i in all_individual_rows}

        # the highest individual id ever is the highest id overall.
        self.__next_individual_id = max(all_individual_ids) + 1

        if not len(individual_ids) == (len(individual_rows)/len(self.__env_conditions)):
            raise IncompatibleError()

        genotype_ids = [individual_map[(id, any_cond)].genotype_id for id in individual_ids]
        genotypes = await self.__genotype_serializer.from_database(
            session, genotype_ids
        )

        assert len(genotypes) == len(genotype_ids)
        self.__latest_population = [
            _Individual(g_id, g, None) for g_id, g in zip(individual_ids, genotypes)
        ]

        self.__latest_measures = {}
        self.__latest_states = {}
        for cond in self.__env_conditions:
            if self.__generation_index == 0:
                self.__latest_measures[cond] = None
                self.__latest_states[cond] = None
            else:
                measures_ids = [individual_map[(id, cond)].float_id for id in individual_ids]
                measures = await self.__measures_serializer.from_database(
                    session, measures_ids
                )
                assert len(measures) == len(measures_ids)
                self.__latest_measures[cond] = measures

                # TODO: do we really need to recover states?
                states_ids = [individual_map[(id, cond)].states_id for id in individual_ids]
                states = await self.__states_serializer.from_database(
                    session, states_ids
                )
                # assert len(states) == len(states_ids)
                self.__latest_states[cond] = states

        return True

    def collect_key_value(self, dictionaries, key):
        list = []
        for d in dictionaries:
            list.append(d[key])
        return list

    async def run(self) -> None:
        # evaluate initial population if required
        if self.__latest_measures is None:
            self.__latest_measures, self.__latest_states = await self.__safe_evaluate_generation(
                [i.genotype for i in self.__latest_population],
                self.__database,
                self.__process_id_gen.gen(),
                self.__process_id_gen,
            )
            initial_population = self.__latest_population

            initial_measures = {}
            initial_states = {}
            initial_relative_measures = {}
            for cond in self.__env_conditions:
                initial_measures[cond] = self.__latest_measures[cond]
                initial_states[cond] = self.__latest_states[cond]
                self._pool_and_time_relative_measures(self.__latest_population, self.__latest_measures[cond])

            self._pool_seasonal_relative_measures(self.__latest_population, self.__latest_measures)
            self._pop_relative_measures()

            for cond in self.__env_conditions:
                relative_measures = []
                for i in range(len(self.__latest_population)):
                    relative_measures.append(MeasureRelative(
                        genotype_measures=self.__latest_measures[cond][i])._return_only_relative())

                initial_relative_measures[cond] = relative_measures
        else:
            initial_population = None
            initial_measures = None
            initial_states = None
            initial_relative_measures = None

        while self.__safe_must_do_next_gen():

            self.__generation_index += 1

            any_cond = list(self.__env_conditions.keys())[0]
            # relative measures for pool parents
            for cond in self.__env_conditions:
                self._pool_and_time_relative_measures(self.__latest_population, self.__latest_measures[cond])
            self._pool_seasonal_relative_measures(self.__latest_population, self.__latest_measures)

            # let user select parents
            latest_fitnesses = self.collect_key_value(self.__latest_measures[any_cond],
                                                      self.__fitness_measure)

            parent_selections = self.__safe_select_parents(
                [i.genotype for i in self.__latest_population],
                latest_fitnesses,
                self.__offspring_size,
            )

            # let user create offspring
            offspring = [
                self.__safe_mutate(
                    self.__safe_crossover(
                        [self.__latest_population[i].genotype for i in s]
                    )
                )
                for s in parent_selections
            ]

            # let user evaluate offspring
            new_measures, new_states = await self.__safe_evaluate_generation(
                offspring,
                self.__database,
                self.__process_id_gen.gen(),
                self.__process_id_gen,
            )

            # combine to create list of individuals
            new_individuals = [
                _Individual(
                    -1,  # placeholder until later
                    genotype,
                    [self.__latest_population[i].id for i in parent_indices],
                )
                for parent_indices, genotype in zip(parent_selections, offspring)
            ]

            # set ids for new individuals
            for individual in new_individuals:
                individual.id = self.__gen_next_individual_id()

            pool_individuals = self.__latest_population + new_individuals

            pool_measures = {}
            # relative measures for pool parents + offspring
            for cond in self.__env_conditions:
                pool_measures[cond] = self.__latest_measures[cond] + new_measures[cond]
                self._pool_and_time_relative_measures(pool_individuals, pool_measures[cond])
            self._pool_seasonal_relative_measures(pool_individuals, pool_measures)

            # let user select survivors between old and new individuals
            new_fitnesses = self.collect_key_value(new_measures[any_cond], self.__fitness_measure)
            old_survivors, new_survivors = self.__safe_select_survivors(
                [i.genotype for i in self.__latest_population],
                latest_fitnesses,
                [i.genotype for i in new_individuals],
                new_fitnesses,
                len(self.__latest_population),
            )

            survived_new_individuals = [new_individuals[i] for i in new_survivors]
            survived_new_measures = {}
            survived_new_states = {}
            for cond in self.__env_conditions:
                survived_new_measures[cond] = [new_measures[cond][i] for i in new_survivors]
                if self.__run_simulation:
                    survived_new_states[cond] = [new_states[cond][i] for i in new_survivors]

            # combine old and new and store as the new generation
            self.__latest_population = [
                self.__latest_population[i] for i in old_survivors
            ] + survived_new_individuals

            for cond in self.__env_conditions:
                self.__latest_measures[cond] = [
                    self.__latest_measures[cond][i] for i in old_survivors
                ] + survived_new_measures[cond]

                if self.__run_simulation:
                    self.__latest_states[cond] = [
                        self.__latest_states[cond][i] for i in old_survivors
                    ] + survived_new_states[cond]

            self._pop_relative_measures()

            latest_relative_measures = {}
            for cond in self.__env_conditions:
                latest_relative_measures[cond] = []
                for i in range(len(self.__latest_population)):
                    latest_relative_measures[cond].append(MeasureRelative(
                        genotype_measures=self.__latest_measures[cond][i])._return_only_relative())

            # save generation and possibly measures of initial population
            # and let user save their state
            async with AsyncSession(self.__database) as session:
                async with session.begin():
                    # provide also initial diversity and update old gen table!
                    await self.__save_generation_using_session(
                        session,
                        initial_population,
                        initial_measures,
                        initial_states,
                        initial_relative_measures,
                        new_individuals,
                        new_measures,
                        new_states,
                        latest_relative_measures,
                    )
                    self._on_generation_checkpoint(session)
            # in any case they should be none after saving once
            initial_population = None
            initial_measures = None
            initial_states = None

            logging.info(f"Finished generation {self.__generation_index}")

        assert (
            self.__generation_index > 0
        ), "Must create at least one generation beyond initial population. This behaviour is not supported."  # would break database structure

    # calculates measures relative to pop
    def _pop_relative_measures(self):
        # interdependent measures must be calculated sequentially (for after for)
        for i in range(len(self.__latest_population)):
            for cond in self.__env_conditions:
                self.__latest_measures[cond][i] = MeasureRelative(genotype_measures=self.__latest_measures[cond][i],
                                                            neighbours_measures=self.__latest_measures[cond])._diversity('pop')

    def _pool_and_time_relative_measures(self, pool_individuals, pool_measures):

        # populational-interdependent measures must be calculated sequentially (for after for)
        for i in range(len(pool_individuals)):
            pool_measures[i] = MeasureRelative(genotype_measures=pool_measures[i],
                                               neighbours_measures=pool_measures).\
                                                        _age(self.__generation_index)

            pool_measures[i] = MeasureRelative(genotype_measures=pool_measures[i],
                                               neighbours_measures=pool_measures)._diversity('pool')

        for i in range(len(pool_individuals)):
            pool_measures[i] = MeasureRelative(genotype_measures=pool_measures[i],
                                               neighbours_measures=pool_measures)._pool_dominated_individuals()

    # consolidates dominance among seasons/tasks
    def _pool_seasonal_relative_measures(self, pool_individuals, pool_measures):

        for i in range(len(pool_individuals)):
            pool_measures_conds = {}
            for cond in pool_measures:
                pool_measures_conds[cond] = pool_measures[cond][i]

            seasonal_dominated, seasonal_fullydominated = MeasureRelative(genotype_measures=pool_measures_conds,
                                               neighbours_measures=pool_measures)._pool_seasonal_dominated_individuals()
            backforth_dominated = MeasureRelative(genotype_measures=pool_measures_conds,
                                               neighbours_measures=pool_measures)._pool_backforth_dominated_individuals()
            forthright_dominated = MeasureRelative(genotype_measures=pool_measures_conds,
                                               neighbours_measures=pool_measures)._pool_forthright_dominated_individuals()

            for cond in pool_measures:
                pool_measures[cond][i]['seasonal_dominated'] = seasonal_dominated
                pool_measures[cond][i]['seasonal_fullydominated'] = seasonal_fullydominated
                pool_measures[cond][i]['backforth_dominated'] = backforth_dominated
                pool_measures[cond][i]['forthright_dominated'] = forthright_dominated
    @property
    def generation_index(self) -> Optional[int]:
        """
        Get the current generation.
        The initial generation is numbered 0.
        """

        return self.__generation_index

    def __gen_next_individual_id(self) -> int:
        next_id = self.__next_individual_id
        self.__next_individual_id += 1
        return next_id

    async def __safe_evaluate_generation(
        self,
        genotypes: List[Genotype],
        database: AsyncEngine,
        process_id: int,
        process_id_gen: ProcessIdGen,
    ) -> List[Measure]:
        measures, states = await self._evaluate_generation(
            genotypes=genotypes,
            database=database,
            process_id=process_id,
            process_id_gen=process_id_gen,
        )
        assert type(measures) == dict
        for m in measures:
            assert len(measures[m]) == len(genotypes)
        # TODO : adapt to new types
        # assert all(type(e) == self.__measures_type for e in measures)
        return measures, states

    def __safe_select_parents(
        self,
        population: List[Genotype],
        fitnesses: List[Measure],
        num_parent_groups: int,
    ) -> List[List[int]]:
        parent_selections = self._select_parents(
            population, fitnesses, num_parent_groups
        )

        assert type(parent_selections) == list
        assert len(parent_selections) == num_parent_groups
        assert all(type(s) == list for s in parent_selections)
        assert all(
            [
                all(type(p) == int and p >= 0 and p < len(population) for p in s)
                for s in parent_selections
            ]
        )
        return parent_selections

    def __safe_crossover(self, parents: List[Genotype]) -> Genotype:
        genotype = self._crossover(parents)
        assert type(genotype) == self.__genotype_type
        return genotype

    def __safe_mutate(self, genotype: Genotype) -> Genotype:
        genotype = self._mutate(genotype)
        assert type(genotype) == self.__genotype_type
        return genotype

    def __safe_select_survivors(
        self,
        old_individuals: List[Genotype],
        old_measures: List[float],
        new_individuals: List[Genotype],
        new_measures: List[float],
        num_survivors: int,
    ) -> Tuple[List[int], List[int]]:

        old_survivors, new_survivors = self._select_survivors(
            old_individuals,
            old_measures,
            new_individuals,
            new_measures,
            num_survivors,
        )

        assert type(old_survivors) == list
        assert type(new_survivors) == list
        assert len(old_survivors) + len(new_survivors) == len(self.__latest_population)
        assert all(type(s) == int for s in old_survivors)
        assert all(type(s) == int for s in new_survivors)
        return (old_survivors, new_survivors)

    def __safe_must_do_next_gen(self) -> bool:
        must_do = self._must_do_next_gen()
        assert type(must_do) == bool
        return must_do

    async def __save_generation_using_session(
        self,
        session: AsyncSession,
        initial_population: Optional[List[_Individual[Genotype]]],
        initial_measures: Optional[List[Measure]],
        initial_states: List[Tuple[float, State]],
        initial_relative_measures:  Optional[List[Measure]],
        new_individuals: List[_Individual[Genotype]],
        new_measures: Optional[List[Measure]],
        new_states: List[Tuple[float, State]],
        latest_relative_measures: Dict
    ) -> None:
        # TODO this function can probably be simplified as well as optimized.
        # but it works so I'll leave it for now.
        # update measures/states of initial population if provided

        # does this if make any sense? :/
        if initial_measures is not None:
            assert initial_population is not None

            for cond in self.__env_conditions:
                measures_ids = await self.__measures_serializer.to_database(
                    session, initial_measures[cond]
                )
                assert len(measures_ids) == len(initial_measures[cond])

                states_ids = await self.__states_serializer.to_database(
                    session, initial_states[cond]
                )
                assert len(states_ids) == len(initial_states[cond])

                rows = (
                    (
                        await session.execute(
                            select(DbEAOptimizerIndividual)
                                .filter(
                                (
                                        DbEAOptimizerIndividual.ea_optimizer_id
                                        == self.__ea_optimizer_id
                                )
                                & (
                                    DbEAOptimizerIndividual.individual_id.in_(
                                        [i.id for i in initial_population]
                                    )
                                )
                                & ( DbEAOptimizerIndividual.env_conditions_id == cond)
                            )
                                .order_by(DbEAOptimizerIndividual.individual_id)
                        )
                    )
                        .scalars()
                        .all()
                )
                if len(rows) != len(initial_population):
                    raise IncompatibleError()

                for i, row in enumerate(rows):
                    row.float_id = measures_ids[i]
                    if self.__run_simulation:
                        row.states_id = states_ids[i]

                rows = (
                    (
                        await session.execute(
                            select(DbEAOptimizerGeneration)
                                .filter(
                                (
                                        DbEAOptimizerGeneration.ea_optimizer_id
                                        == self.__ea_optimizer_id
                                )
                                & (
                                    DbEAOptimizerGeneration.individual_id.in_(
                                        [i.id for i in initial_population]
                                    )
                                )
                                & (DbEAOptimizerGeneration.env_conditions_id == cond)
                            )
                                .order_by(DbEAOptimizerGeneration.individual_id)
                        )
                    )
                        .scalars()
                        .all()
                )
                if len(rows) != len(initial_population):
                    raise IncompatibleError()

                for i, row in enumerate(rows):
                    row.pop_diversity = initial_relative_measures[cond][i]['pop_diversity']
                    row.pool_diversity = initial_relative_measures[cond][i]['pool_diversity']
                    row.dominated_quality_youth = initial_relative_measures[cond][i]['dominated_quality_youth']
                    row.fullydominated_quality_youth = initial_relative_measures[cond][i]['fullydominated_quality_youth']
                    row.age = initial_relative_measures[cond][i]['age']
                    row.inverse_age = initial_relative_measures[cond][i]['inverse_age']
                    row.seasonal_dominated = initial_relative_measures[cond][i]['seasonal_dominated']
                    row.seasonal_fullydominated = initial_relative_measures[cond][i]['seasonal_fullydominated']
                    row.backforth_dominated = initial_relative_measures[cond][i]['backforth_dominated']
                    row.forthright_dominated = initial_relative_measures[cond][i]['forthright_dominated']

        # save current optimizer state
        session.add(
            DbEAOptimizerState(
                ea_optimizer_id=self.__ea_optimizer_id,
                generation_index=self.__generation_index,
                processid_state=self.__process_id_gen.get_state(),
            )
        )

        # save new individuals
        genotype_ids = await self.__genotype_serializer.to_database(
            session, [g.genotype for g in new_individuals]
        )
        assert len(genotype_ids) == len(new_individuals)

        conds_measures_ids = {}
        for cond in self.__env_conditions:
            if new_measures is not None:
                measures_ids2 = [
                    m
                    for m in await self.__measures_serializer.to_database(
                        session, new_measures[cond]
                    )
                ]  # this extra comprehension is useless but it stops mypy from complaining
                assert len(measures_ids2) == len(new_measures[cond])
            else:
                measures_ids2 = [None for _ in range(len(new_individuals))]
            conds_measures_ids[cond] = measures_ids2

        conds_states_ids = {}
        for cond in self.__env_conditions:
            if new_states is not None:
                if new_states[cond] is not None:
                    if len(new_states[cond]) > 0:
                        states_ids2 = [
                            s
                            for s in await self.__states_serializer.to_database(
                                session, new_states[cond]
                            )
                        ]  # this extra comprehension is useless but it stops mypy from complaining
                        assert len(states_ids2) == len(new_states[cond])
                    # TODO: remove ugly triple redundancy
                    else:
                        states_ids2 = [None for _ in range(len(new_individuals))]
                else:
                    states_ids2 = [None for _ in range(len(new_individuals))]
            else:
                states_ids2 = [None for _ in range(len(new_individuals))]
            conds_states_ids[cond] = states_ids2

        for cond in self.__env_conditions:
            session.add_all(
                [
                    DbEAOptimizerIndividual(
                        ea_optimizer_id=self.__ea_optimizer_id,
                        env_conditions_id=cond,
                        individual_id=i.id,
                        genotype_id=g_id,
                        float_id=m_id,
                        states_id=s_id,
                    )
                    for i, g_id, m_id, s_id in zip(new_individuals, genotype_ids,
                                                   conds_measures_ids[cond],
                                                   conds_states_ids[cond])
                ]
            )

        # save parents of new individuals
        parents: List[DbEAOptimizerParent] = []
        for individual in new_individuals:
            assert (
                individual.parent_ids is not None
            )  # Cannot be None. They are only None after recovery and then they are already saved.
            for p_id in individual.parent_ids:
                parents.append(
                    DbEAOptimizerParent(
                        ea_optimizer_id=self.__ea_optimizer_id,
                        child_individual_id=individual.id,
                        parent_individual_id=p_id,
                    )
                )
        session.add_all(parents)

        # save current generation
        for index, individual in enumerate(self.__latest_population):
            # TODO: this could be better, but it has to adapt to
            #  the bizarre fact that the initial pop gets saved before evaluated

            for cond in self.__env_conditions:
                if latest_relative_measures is None:
                    pop_diversity = None
                    pool_diversity = None
                    dominated_quality_youth = None
                    fullydominated_quality_youth = None
                    age = None
                    inverse_age = None
                    seasonal_dominated = None
                    seasonal_fullydominated = None
                    backforth_dominated = None
                    forthright_dominated = None
                else:
                    pop_diversity = latest_relative_measures[cond][index]['pop_diversity']
                    pool_diversity = latest_relative_measures[cond][index]['pool_diversity']
                    dominated_quality_youth = latest_relative_measures[cond][index]['dominated_quality_youth']
                    fullydominated_quality_youth = latest_relative_measures[cond][index]['fullydominated_quality_youth']
                    age = latest_relative_measures[cond][index]['age']
                    inverse_age = latest_relative_measures[cond][index]['inverse_age']
                    seasonal_dominated = latest_relative_measures[cond][index]['seasonal_dominated']
                    seasonal_fullydominated = latest_relative_measures[cond][index]['seasonal_fullydominated']
                    backforth_dominated = latest_relative_measures[cond][index]['backforth_dominated']
                    forthright_dominated = latest_relative_measures[cond][index]['forthright_dominated']

                session.add(
                        DbEAOptimizerGeneration(
                            ea_optimizer_id=self.__ea_optimizer_id,
                            generation_index=self.__generation_index,
                            env_conditions_id=cond,
                            individual_index=index,
                            individual_id=individual.id,
                            pop_diversity=pop_diversity,
                            pool_diversity=pool_diversity,
                            dominated_quality_youth=dominated_quality_youth,
                            fullydominated_quality_youth=fullydominated_quality_youth,
                            age=age,
                            inverse_age=inverse_age,
                            seasonal_dominated=seasonal_dominated,
                            seasonal_fullydominated=seasonal_fullydominated,
                            backforth_dominated=backforth_dominated,
                            forthright_dominated=forthright_dominated,
                        )
                )


@dataclass
class _Individual(Generic[Genotype]):
    id: int
    genotype: Genotype
    # Empty list of parents means this is from the initial population
    # None means we did not bother loading the parents during recovery because they are not needed.
    parent_ids: Optional[List[int]]
