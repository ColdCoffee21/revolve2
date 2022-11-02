import math
from sklearn.neighbors import KDTree
from typing import List, Optional, Tuple
import copy
import pprint

# TODO: because of the bizarre logic that ties gen0 to gen1, individuals from gen0 have incorrect relative measures
# it seems as if the measures of gen1 are correct, and get copied to gen0

# relative measures (which depend on the rest on the pop (or gens) to be calculated), and thus can change at every gen
# 'pool' measures depend on the pool of competitors and while 'pop' measures depends only on the survivors
# time dependent measures are also considered relative, e.g., age is relative to gens
class MeasureRelative:

    def __init__(self, genotype_measures=None, neighbours_measures=None):
        self._genotype_measures = genotype_measures
        self._neighbours_measures = neighbours_measures

    def _return_only_relative(self):
        # pool measures used for parent selection are not being saved.
        # they are overwritten, and only the values of survival selection are persisted.
        # persist it in the future?
        relative_measures = ['pop_diversity',
                             'pool_diversity',
                             'dominated_quality_youth',
                             'fullydominated_quality_youth',
                             'seasonal_dominated',
                             'seasonal_fullydominated',
                             'backforth_dominated',
                             'forthright_dominated',
                             'age',
                             'inverse_age']

        copy_genotype_measures = copy.deepcopy(self._genotype_measures)

        for measure in self._genotype_measures:
            if measure not in relative_measures:
                del copy_genotype_measures[measure]

        return copy_genotype_measures

    def _diversity(self, type='pop'):

        # TODO: when type is pool (for novelty), lacking an archive
        # TODO: make which_measures a param
        which_measures = ['symmetry',
                          'proportion',
                          'coverage',
                          'extremities_prop',
                          'hinge_prop',
                          'hinge_ratio',
                          'branching_prop']

        genotype_measures = []
        for key in which_measures:
            genotype_measures.append(self._genotype_measures[key])

        neighbours_measures = []
        for neighbour_measures in self._neighbours_measures:

            neighbours_measures.append([])
            for key in which_measures:
                neighbours_measures[-1].append(neighbour_measures[key])

        kdt = KDTree(neighbours_measures, leaf_size=30, metric='euclidean')

        if type == 'pop':
            k = len(self._neighbours_measures)
        else:
            # TODO: make it a param
            k = 10+1

        # distances from neighbors
        distances, indexes = kdt.query([genotype_measures], k=k)
        diversity = sum(distances[0])/len(distances[0])

        self._genotype_measures[f'{type}_diversity'] = diversity

        # if type == 'pool':
        #     if self._genotype_measures['speed_y'] > 0:
        #         self._genotype_measures['speed_diversity'] = self._genotype_measures['speed_y'] * diversity
        #     else:
        #         self._genotype_measures['speed_diversity'] = self._genotype_measures['speed_y'] / diversity

        return self._genotype_measures

    # counts how many individuals of the current pool this individual dominates
    # an individual a dominates an individual b if a is better in at least one measure and not worse in any measure
    # better=higher > maximization
    def _pool_dominated_individuals(self):
        self._pareto_dominance(['speed_y', 'inverse_age'], 'quality_youth')
        return self._genotype_measures

    def _pareto_dominance(self, which_measures, type):
        pool_dominated_individuals = 0
        pool_fulldominated_individuals = 0
        for neighbour_measures in self._neighbours_measures:
            better = 0
            worse = 0
            for key in which_measures:
                if self._genotype_measures[key] > neighbour_measures[key]:
                    better += 1
                if self._genotype_measures[key] < neighbour_measures[key]:
                    worse += 1
            if better > 0 and worse == 0:
                pool_dominated_individuals += 1
            if better == len(which_measures):
                pool_fulldominated_individuals += 1

        self._genotype_measures[f'dominated_{type}'] = pool_dominated_individuals
        self._genotype_measures[f'fullydominated_{type}'] = pool_fulldominated_individuals

    def _pool_seasonal_dominated_individuals(self):
        which_measure = "speed_y"
        pool_dominated_individuals = 0
        pool_fulldominated_individuals = 0
        for i in range(0, len(self._neighbours_measures[1])):
            better = 0
            worse = 0
            for cond in self._genotype_measures:
                if self._genotype_measures[cond][which_measure] > self._neighbours_measures[cond][i][which_measure]:
                    better += 1
                if self._genotype_measures[cond][which_measure] < self._neighbours_measures[cond][i][which_measure]:
                    worse += 1
            if better > 0 and worse == 0:
                pool_dominated_individuals += 1
            if better == len(self._genotype_measures):
                pool_fulldominated_individuals += 1
        return pool_dominated_individuals, pool_fulldominated_individuals

    def _pool_backforth_dominated_individuals(self):
        which_measure = "speed_y"
        pool_dominated_individuals = 0
        for i in range(0, len(self._neighbours_measures[1])):
            better = 0
            worse = 0
            for cond in self._genotype_measures:
                value_individual = self._genotype_measures[cond][which_measure]
                value_neighbour = self._neighbours_measures[cond][i][which_measure]

                # this is (sadly) hardcoded. cond=2 means the 'back' setup, when target direction gets inverted
                if cond == 2:
                    value_individual = value_individual * -1
                    value_neighbour = value_neighbour * -1

                if value_individual > value_neighbour:
                    better += 1
                if value_individual < value_neighbour:
                    worse += 1
            if better > 0 and worse == 0:
                pool_dominated_individuals += 1
        return pool_dominated_individuals

    def _pool_forthright_dominated_individuals(self):
        pool_dominated_individuals = 0
        for i in range(0, len(self._neighbours_measures[1])):
            better = 0
            worse = 0
            for cond in self._genotype_measures:

                # this is (sadly) hardcoded.
                if cond == 1:
                    which_measure = "speed_y"
                if cond == 2:
                    which_measure = "speed_x"

                value_individual = self._genotype_measures[cond][which_measure]
                value_neighbour = self._neighbours_measures[cond][i][which_measure]

                if value_individual > value_neighbour:
                    better += 1
                if value_individual < value_neighbour:
                    worse += 1
            if better > 0 and worse == 0:
                pool_dominated_individuals += 1
        return pool_dominated_individuals

    def _age(self, generation_index):

        age = generation_index - self._genotype_measures['birth'] + 1
        inverse_age = 1/age
        self._genotype_measures['age'] = age
        self._genotype_measures['inverse_age'] = inverse_age

        return self._genotype_measures
