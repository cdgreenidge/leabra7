"""Utilities."""
import heapq
from typing import Any
from typing import Callable
from typing import List
from typing import Tuple
from typing import TypeVar

T = TypeVar('T')


def partition(n: int, iterable: List[T], key: Callable[[T], Any] = lambda x: x
              ) -> Tuple[List[T], List[T]]:
    """Partitions an iterable using heap-based partial sort.

    Right now this is dead code, but it could eventually be useful for
    the averaging version of kWTA inhibition, so I'm leaving it in.

    Args:
        * n: The number of elements in the first partition.
        * iterable: The iterable to partition.
        * key: A function of one argument that extracts a comparison key
            from each element in the iterable. The keys must be comparable
            with each other, unfortunately MyPy can't check for this at
            the moment.

    Returns:
        A tuple of two lists. The first list contains the n
            largest items in the iterable. The second list contains the rest
            of the items.

    Raises:
        ValueError: if n is <= len(iterable).

    """
    if n > len(iterable):
        raise ValueError("n must be <= len(iterable)")

    nlargest = []  # type: List[Tuple[Any, T]]
    rest = []  # type: List[Tuple[Any, T]]
    for i in range(n):
        x = iterable[i]
        heapq.heappush(nlargest, (key(x), x))

    for i in range(n, len(iterable)):
        x = iterable[i]
        rest.append(heapq.heappushpop(nlargest, (key(x), x)))

    return [x[1] for x in nlargest], [x[1] for x in rest]
