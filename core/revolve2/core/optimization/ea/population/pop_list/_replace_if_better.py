from typing import List, Tuple, TypeVar, Union

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


def replace_if_better(
    original_population: PopList[TIndividual, TMeasures],
    offspring_population: PopList[TIndividual, TMeasures],
    measure: str,
) -> List[int]:
    """
    Compare each individual is offspring population with original population index-wise and replaces if better.

    Populations must be of the same size.

    :param original_population: The original population to replace individuals in. Will not be altered.
    :param offspring_population: The offspring population to take individuals from. Will not be unaltered. Individuals will be copied.
    :param measure: The measure to use for selection.
    :returns: For each index in the population, the 0 to take the individual from the original population, or 1 to take it from the offspring.
    """
    return [
        0 if orig.measures[measure] >= off.measures[measure] else 1
        for orig, off in zip(original_population, offspring_population)
    ]
