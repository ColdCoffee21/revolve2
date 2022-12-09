import multineat

from revolve2.core.modular_robot import Body

from .._genotype import Genotype
from .._random_v1 import random_v1 as base_random_v1
from ._brain_cpg_network_neighbour_v1 import BrainCpgNetworkNeighbourV1


def random_v1(
    innov_db: multineat.InnovationDatabase,
    rng: multineat.RNG,
    multineat_params: multineat.Parameters,
    output_activation_func: multineat.ActivationFunction,
    num_initial_mutations: int,
    n_env_conditions: int,
    plastic_brain: int,
) -> Genotype:
    assert multineat_params.MutateOutputActivationFunction == False
    # other activation functions could work too, but this has been tested.
    # if you want another one, make sure it's output is between -1 and 1.
    assert output_activation_func == multineat.ActivationFunction.SIGNED_SINE

    if plastic_brain == 0:
        return base_random_v1(
            innov_db,
            rng,
            multineat_params,
            output_activation_func,
            # 7,  # bias(always 1), x1, y1, z1, x2, y2, z2
            6, # x1, y1, z1, x2, y2, z2
            1,  # weight
            num_initial_mutations,
        )
    else:
        return base_random_v1(
            innov_db,
            rng,
            multineat_params,
            output_activation_func,
            # 8,  # bias(always 1), x1, y1, z1, x2, y2, z2, inclined
            7, # x1, y1, z1, x2, y2, z2, inclined
            1,  # weight
            num_initial_mutations,
        )


def develop_v1(genotype: Genotype, body: Body, env_condition: int,
               n_env_conditions: int, plastic_brain: int) -> BrainCpgNetworkNeighbourV1:
    return BrainCpgNetworkNeighbourV1(genotype.genotype, env_condition, n_env_conditions, plastic_brain)
