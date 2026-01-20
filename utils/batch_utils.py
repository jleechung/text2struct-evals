"""Batching utilities for processing large datasets."""

from __future__ import annotations

from typing import List, Sequence, TypeVar

T = TypeVar("T")


def create_batches(items: Sequence[T], batch_size: int) -> List[List[T]]:
    """Split items into contiguous batches.

    Args:
        items: Sequence to batch.
        batch_size: Size of each batch (> 0).

    Returns:
        List of batches.
    """
    if batch_size is None:
        raise ValueError("batch_size must not be None")
    if not isinstance(batch_size, int):
        raise TypeError(f"batch_size must be int, got {type(batch_size).__name__}")
    if batch_size <= 0:
        raise ValueError(f"batch_size must be > 0, got {batch_size}")

    return [list(items[i : i + batch_size]) for i in range(0, len(items), batch_size)]
