import math
from typing import List, Optional, Tuple
from squaternion import Quaternion
import pprint

from ._module import Module
from .render.render import Render
from revolve2.core.physics.running import (
    ActorState
)

from revolve2.core.modular_robot import Core, ActiveHinge, Brick


class Measure:

    _states: List[Tuple[float, ActorState]]

    def __init__(self, states=None, genotype_idx=-1, phenotype=None, generation=None,
                 simulation_time=None, env_conditions=None):
        self._states = states
        # TODO: _genotype_idx: obsolete variable
        self._genotype_idx = genotype_idx
        self._phenotype_body = phenotype.body
        self._phenotype_brain = phenotype.brain
        self._generation = generation
        self._simulation_time = simulation_time
        self._env_conditions = env_conditions
        self._orientations = []

    def measure_all_non_relative(self):

        self._measures = {}

        self._measures['birth'] = self._generation
        self._displacement()
        self._head_balance()

        self._calculate_counts()
        self._modules_count()
        self._props()
        self._branching_count()
        self._branching_prop()
        self._extremities_extensiveness(None, False, True)
        self._extremities_extensiveness(None, True, False)
        self._extremities_prop()
        self._extensiveness_prop()
        self._width_height()
        self._coverage()
        self._proportion()
        self._symmetry()

        self._relative_speed_y()

        return self._measures

    def _props(self):
        max_hinges = max(self._measures['hinge_horizontal'], self._measures['hinge_vertical'])
        min_hinges = min(self._measures['hinge_horizontal'], self._measures['hinge_vertical'])
        if max_hinges > 0 and min_hinges > 0:
            self._measures['hinge_ratio'] = float(min_hinges/max_hinges)
        else:
            self._measures['hinge_ratio'] = 0

        self._measures['hinge_prop'] = self._measures['hinge_count'] / self._measures['modules_count']
        self._measures['brick_prop'] = self._measures['brick_count'] / self._measures['modules_count']

    # behavioral measures
    # TODO simulation can continue slightly passed the defined sim time.
    def _displacement(self):

        if self._states is None:
            self._measures['speed_y'] = -math.inf
            self._measures['speed_x'] = -math.inf
            self._measures['average_z'] = -math.inf
            self._measures['displacement'] = -math.inf
            return

        begin_state = self._states.environment_results[self._genotype_idx].environment_states[0].actor_states[0]
        end_state = self._states.environment_results[self._genotype_idx].environment_states[-1].actor_states[0]

        displacement = float(
            math.sqrt(
                (begin_state.position[0] - end_state.position[0]) ** 2
                + ((begin_state.position[1] - end_state.position[1]) ** 2)
            ))

        self._measures['displacement'] = displacement

        # TODO: check if outlier from pop avg
        if displacement >= 10:
            print('suspicious displacement gets minus inf')
            self._measures['speed_y'] = -math.inf
            self._measures['speed_x'] = -math.inf
        else:
            # speed on the y-axis (to the right [uphill] is higher/better)
            displacement_y = float((end_state.position[1]-begin_state.position[1]))
            displacement_x = float((end_state.position[0]-begin_state.position[0]))

            # if there is a platform, truncates displacement
            # if int(self._env_conditions[3]) == 1:
            #     if displacement_y > 1:
            #         displacement_y = 1
            #     if displacement_y < -1:
            #         displacement_y = -1
            #
            #     if displacement_x > 1:
            #         displacement_x = 1
            #     if displacement_x < -1:
            #         displacement_x = -1

            self._measures['speed_y'] = float((displacement_y/self._simulation_time)*100)
            self._measures['speed_x'] = float((displacement_x/self._simulation_time)*100)

        # average z
        z = 0
        for s in self._states.environment_results[self._genotype_idx].environment_states:
            z += s.actor_states[0].position[2]
        z /= len(self._states.environment_results[self._genotype_idx].environment_states)
        self._measures['average_z'] = float(z)

    def _relative_speed_y(self):
        self._measures['relative_speed_y'] = self._measures['speed_y']/self._measures['modules_count']

    def _get_orientations(self):
        for idx_state in range(0, len(self._states.environment_results[self._genotype_idx].environment_states)):
            _orientations = self._states.environment_results[self._genotype_idx].\
                environment_states[idx_state].actor_states[0].serialize()['orientation']
            # w, x, y, z
            qua = Quaternion(_orientations[0], _orientations[1], _orientations[2], _orientations[3])
            euler = qua.to_euler()
            eulers = [euler[0], euler[1], euler[2]]  # roll / pitch / yaw
            self._orientations.append(eulers)

    def _head_balance(self):
        """
        Returns the inverse of the average rotation of the head in the roll and pitch dimensions.
        The closest to 1 the most balanced.
        :return:
        """
        if self._states is None:
            self._measures['head_balance'] = -math.inf
            return

        roll = 0
        pitch = 0
        instants = len(self._states.environment_results[self._genotype_idx].environment_states)
        self._get_orientations()

        for o in self._orientations:
            roll = roll + abs(o[0]) * 180 / math.pi
            pitch = pitch + abs(o[1]) * 180 / math.pi

        #  accumulated angles for each type of rotation
        #  divided by iterations * maximum angle * each type of rotation
        if instants == 0:
            balance = None
        else:
            balance = (roll + pitch) / (instants * 180 * 2)
            # turns imbalance to balance
            balance = 1 - balance

        self._measures['head_balance'] = balance

    # morphological measures

    def _calculate_counts(self, module=None, init=True):
        """
        Count amount of modules for each distinct type
        """

        if init:
            self._measures['hinge_count'] = 0
            self._measures['brick_count'] = 0
            self._measures['hinge_ratio'] = 0
            self._measures['hinge_horizontal'] = 0
            self._measures['hinge_vertical'] = 0

        if module is None:
            module = self._phenotype_body.core
        elif isinstance(module, ActiveHinge):
            self._measures['hinge_count'] += 1
            if module._absolute_rotation == 0:
                self._measures['hinge_horizontal'] += 1
            else:
                self._measures['hinge_vertical'] += 1
        elif isinstance(module, Brick):
            self._measures['brick_count'] += 1

        if module.has_children():
            for core_slot, child_module in enumerate(module.children):
                if child_module is None:
                    continue
                self._calculate_counts(child_module, False)

    def _modules_count(self):
        """
        Count total amount of modules in body excluding sensors
        """
        self._measures['modules_count'] = self._measures['hinge_count'] + self._measures['brick_count'] + 1

    def _branching_count(self, module=None, init=True):
        """
        Count amount of fully branching modules in body
        """
        if init:
            self._measures['branching_count'] = 0
        if module is None:
            module = self._phenotype_body.core

        if module.has_children():
            children_count = 0
            for core_slot, child_module in enumerate(module.children):
                if child_module is None:
                    continue
                children_count += 1
                self._branching_count(child_module, False)
            if (isinstance(module, Brick) and children_count == 3) or (isinstance(module, Core) and children_count == 4):
                self._measures['branching_count'] += 1

    def _branching_prop(self):
        """
        Measure branching by dividing fully branching by possible branching modules
        """
        if self._measures['branching_count'] == 0 or self._measures['modules_count'] < 5:
            self._measures['branching_prop'] = 0
        else:
            practical_limit_branching_bricks = math.floor((self._measures['modules_count']-2)/3)
            self._measures['branching_prop'] = self._measures['branching_count'] / practical_limit_branching_bricks

    def _extremities_extensiveness(self, module=None, extremities=False, extensiveness=False, init=True):
        """
        Calculate extremities or extensiveness in body
        @param extremities: calculate extremities in body if true
        @param extensiveness: calculate extensiveness in body if true
        """
        if module is None:
            module = self._phenotype_body.core
        if init and extremities:
            self._measures['extremities'] = 0
        if init and extensiveness:
            self._measures['extensiveness'] = 0

        children_count = 0
        for core_slot, child_module in enumerate(module.children):
            if child_module is None:
                continue
            children_count += 1
            self._extremities_extensiveness(child_module, extremities, extensiveness, False)
        if children_count == 0 and not (
                isinstance(module, Core)) and extremities:
            self._measures['extremities'] += 1
        if children_count == 1 and not (
                isinstance(module, Core)) and extensiveness:
            self._measures['extensiveness'] += 1

    def _extremities_prop(self):
        if self._measures['extremities'] == 0:
            self._measures['extremities_prop'] = 0
        else:
            if self._measures['modules_count'] < 6:
                practical_limit_limbs = self._measures['modules_count'] - 1
            else:
                practical_limit_limbs = 2 * math.floor((self._measures['modules_count'] - 6) / 3) + (
                            (self._measures['modules_count'] - 6) % 3) + 4
            self._measures['extremities_prop'] = self._measures['extremities'] / practical_limit_limbs

    def _extensiveness_prop(self):
        if self._measures['modules_count'] < 3:
            self._measures['extensiveness_prop'] = 0
        else:
            practical_limit_extensiveness = self._measures['modules_count'] - 2
            self._measures['extensiveness_prop'] = self._measures['extensiveness'] / practical_limit_extensiveness

    def _width_height(self):
        """
        Measure width and height of body, excluding sensors
        """
        render = Render()
        render.traverse_path_of_robot(self._phenotype_body.core, 0, False)
        render.grid.calculate_grid_dimensions()
        self._measures['width'] = render.grid.width
        self._measures['height'] = render.grid.height


    def _coverage(self):
        """
        Measure the coverage of the robot, specified by the amount of modules
        divided by the spanning surface of the robot (excluding sensors)
        :return:
        """
        self._measures['coverage'] = self._measures['modules_count'] / (self._measures['width']*self._measures['height'])

    def _proportion(self):
        """
        Meaure proportion, specified by the 2d ratio of the body
        :return:
        """
        if self._measures['width'] < self._measures['height']:
            self._measures['proportion'] = self._measures['width'] / self._measures['height']
        else:
            self._measures['proportion'] = self._measures['height'] / self._measures['width']

    def _symmetry(self):
        """
        Measure maximum between vertical and horizontal reflective symmetry
        """
        render = Render()
        render.traverse_path_of_robot(self._phenotype_body.core, 0, False)
        coordinates = render.grid.visited_coordinates

        horizontal_mirrored = 0
        horizontal_total = 0
        vertical_mirrored = 0
        vertical_total = 0
        # Calculate symmetry in body
        for position in coordinates:
            if position[0] != 0:
                horizontal_total += 1
                if [-position[0], position[1]] in coordinates:
                    horizontal_mirrored += 1
            if position[1] != 0:
                vertical_total += 1
                if [position[0], -position[1]] in coordinates:
                    vertical_mirrored += 1

        horizontal_symmetry = horizontal_mirrored / horizontal_total if horizontal_mirrored > 0 else 0
        vertical_symmetry = vertical_mirrored / vertical_total if vertical_mirrored > 0 else 0

        self._measures['symmetry'] = max(horizontal_symmetry, vertical_symmetry)

    # controller measures

    # TODO:  old controller measures dont apply, so we gotta think of new ones
