from typing import List, Set, Tuple, cast

import multineat

from revolve2.core.modular_robot import ActiveHinge, Body
from revolve2.core.modular_robot.brains import (
    BrainCpgNetworkNeighbour as ModularRobotBrainCpgNetworkNeighbour,
)


class BrainCpgNetworkNeighbourV1(ModularRobotBrainCpgNetworkNeighbour):
    _genotype: multineat.Genome

    def __init__(self, genotype: multineat.Genome, env_condition: list, n_env_conditions: int, plastic_brain: int):
        self._genotype = genotype
        self._env_condition = env_condition
        self._n_env_conditions = n_env_conditions
        self._plastic_brain = plastic_brain

    def _make_weights(
        self,
        active_hinges: List[ActiveHinge],
        connections: List[Tuple[ActiveHinge, ActiveHinge]],
        body: Body,
    ) -> Tuple[List[float], List[float]]:
        brain_net = multineat.NeuralNetwork()
        self._genotype.BuildPhenotype(brain_net)

        if self._plastic_brain == 0:
            internal_weights = [
                self._evaluate_network(
                    brain_net,
                    [
                        #1.0,
                        float(pos.x),
                        float(pos.y),
                        float(pos.z),
                        float(pos.x),
                        float(pos.y),
                        float(pos.z),
                    ],
                )
                for pos in [
                    body.grid_position(active_hinge) for active_hinge in active_hinges
                ]
            ]

            external_weights = [
                self._evaluate_network(
                    brain_net,
                    [
                        #1.0,
                        float(pos1.x),
                        float(pos1.y),
                        float(pos1.z),
                        float(pos2.x),
                        float(pos2.y),
                        float(pos2.z),
                    ],
                )
                for (pos1, pos2) in [
                    (body.grid_position(active_hinge1), body.grid_position(active_hinge2))
                    for (active_hinge1, active_hinge2) in connections
                ]
            ]

        else:

            staticfriction, dynamicfriction, yrotationdegrees, platform, toxic = \
                float(self._env_condition[0]), \
                float(self._env_condition[1]), \
                float(self._env_condition[2]), \
                float(self._env_condition[3]), \
                float(self._env_condition[4])

            # TODO: make conditions-checking dynamic
            # if inclined
            # if yrotationdegrees > 0:
            #     inclined = -1
            # else:
            #     inclined = 1

            # obsolete name: toxic here means just a change in task
            if toxic > 0:
                toxicenv = 1
            else:
                toxicenv = -1

            internal_weights = [
                self._evaluate_network(
                    brain_net,
                    [
                        # 1.0,
                        float(pos.x),
                        float(pos.y),
                        float(pos.z),
                        float(pos.x),
                        float(pos.y),
                        float(pos.z),
                        #inclined,
                        toxicenv,
                    ],
                )
                for pos in [
                    body.grid_position(active_hinge) for active_hinge in active_hinges
                ]
            ]

            external_weights = [
                self._evaluate_network(
                    brain_net,
                    [
                        # 1.0,
                        float(pos1.x),
                        float(pos1.y),
                        float(pos1.z),
                        float(pos2.x),
                        float(pos2.y),
                        float(pos2.z),
                        #inclined,
                        toxicenv,
                    ],
                )
                for (pos1, pos2) in [
                    (body.grid_position(active_hinge1), body.grid_position(active_hinge2))
                    for (active_hinge1, active_hinge2) in connections
                ]
            ]

        return (internal_weights, external_weights)

    @staticmethod
    def _evaluate_network(
        network: multineat.NeuralNetwork, inputs: List[float]
    ) -> float:
        network.Input(inputs)
        network.ActivateAllLayers()
        return cast(float, network.Output()[0])  # TODO missing multineat typing
