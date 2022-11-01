from typing import List, Tuple, TypeVar, Union

import numpy as np
from revolve2.core.database import Serializable
from typing_extensions import TypeGuard

from .._serializable_measures import SerializableMeasures
from ._pop_list import PopList

TIndividual = TypeVar("TIndividual", bound=Serializable)
TMeasures = TypeVar("TMeasures", bound=SerializableMeasures)


def _is_number_list(
    xs: List[Union[int, float, str, None]]
) -> TypeGuard[List[Tuple[int, float]]]:
    return all(isinstance(x, int) or isinstance(x, float) for x in xs)


def topn(
    original_population: PopList[TIndividual, TMeasures],
    offspring_population: PopList[TIndividual, TMeasures],
    measure: str,
    n: int,
) -> Tuple[List[int], List[int]]:
    """
    Select the top n individuals from two combined populations based on one of their measures.

    :param original_population: The first population to consider.
    :param offspring_population: The second population to consider.
    :param measure: The measure to rank by.
    :param n: The number of individual to select.
    :returns: Indices of the selected individuals in their respective populations. Original, offspring.
    """
    measures = [i.measures[measure] for i in original_population] + [
        i.measures[measure] for i in offspring_population
    ]
    assert _is_number_list(measures)

    indices = np.argsort(measures)[: -1 - n : -1]
    return [i for i in indices if i < len(original_population)], [
        i - len(original_population) for i in indices if i >= len(original_population)
    ]
