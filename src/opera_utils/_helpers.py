from __future__ import annotations

from itertools import chain, combinations
from typing import Any, Iterable, Mapping


def flatten(list_of_lists: Iterable[Iterable[Any]]) -> chain[Any]:
    """Flatten one level of nesting."""
    return chain.from_iterable(list_of_lists)


def sorted_deduped_values(in_mapping: Mapping[Any, list]):
    """Sort, dedupe, and concatenate all items in the lists of `in_mapping`'s values."""
    all_values = flatten(in_mapping.values())
    return sorted(set(all_values))


def powerset(iterable: Iterable[Any]) -> chain[tuple[Any, ...]]:
    """Generate the powerset of an iterable.

    Examples
    --------
    >>> list(powerset([1,2,3]))
    [(), (1,), (2,), (3,), (1, 2), (1, 3), (2, 3), (1, 2, 3)]
    """
    s = list(iterable)
    return flatten(combinations(s, r) for r in range(len(s) + 1))
