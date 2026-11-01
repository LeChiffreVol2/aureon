"""Explicit cleanup policy shared by context-managed owners.

The caller supplies the side effect. An ordinary cleanup error must not replace a
body/input error; explicit close still reports cleanup failures. Process-level
interrupts are deliberately not swallowed.
"""

from collections.abc import Callable
from typing import TypeVar

_T = TypeVar("_T")


def close_preserving_error(
    close: Callable[[], _T],
    primary_error: BaseException | None,
    *,
    context: str,
) -> _T | None:
    try:
        return close()
    except Exception as cleanup_error:
        if primary_error is None:
            raise
        primary_error.add_note(f"{context}: {cleanup_error}")
        return None
