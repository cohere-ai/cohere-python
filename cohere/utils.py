from typing import Callable, List, Tuple, TypeVar

T = TypeVar('T')
K = TypeVar('K')


def sort_with_indices(values: List[T], key: Callable[[T], K]) -> Tuple[List[T], List[int]]:
    """Sorts values and returns positions in original list

    Args:
        values (List[T]): Values to sort
        key (Callable[[T], K]): Key used for ordering values

    Returns:
        Tuple[List[T], List[int]]: Sorted values and positions in original list
    """

    values_with_indices = list(enumerate(values))
    values_with_indices = sorted(values_with_indices, key=lambda x: key(x[1]))

    values_sorted: List[T] = []
    indices_sorted: List[int] = []
    for i, v in values_with_indices:
        values_sorted.append(v)
        indices_sorted.append(i)

    return values_sorted, indices_sorted


def restore_order(values: List[T], indices: List[int]) -> List[T]:
    """Restores original order of the values according to positions

    Args:
        values (List[T]): Values to reorder
        indices (List[int]): Positions to use for reordering

    Returns:
        List[T]: Reordered values
    """

    assert len(values) == len(indices)

    values_with_indices = list(zip(indices, values))
    values_with_indices = sorted(values_with_indices, key=lambda x: x[0])

    values_reordered: List[T] = []
    for _, v in values_with_indices:
        values_reordered.append(v)

    return values_reordered
