"""Microbenchmarks for the slotted page engine.

Run with:
    python -m benchmarks.bench_page

These measure the pure-Python reference implementation. The numbers are not
meant to be competitive with a C storage engine; they exist to catch
regressions and to show where the cost actually sits (struct pack/unpack and
bytearray slicing dominate).
"""

from __future__ import annotations

import statistics
import time
from collections.abc import Callable

from lakepg.page import SlottedPage

_ROW = b"the quick brown fox jumps over the lazy dog"


def _time(label: str, fn: Callable[[], int], repeats: int = 5) -> None:
    """Run ``fn`` several times and report throughput in operations/second."""
    rates: list[float] = []
    for _ in range(repeats):
        start = time.perf_counter()
        operations = fn()
        elapsed = time.perf_counter() - start
        rates.append(operations / elapsed)

    best = max(rates)
    median = statistics.median(rates)
    print(f"{label:<34} {median:>12,.0f} ops/s (median)  {best:>12,.0f} ops/s (best)")


def bench_insert() -> int:
    """Fill pages with rows until each is full, counting insertions."""
    inserted = 0
    for _ in range(200):
        page = SlottedPage.new()
        while page.can_fit(len(_ROW)):
            page.add_item(_ROW)
            inserted += 1
    return inserted


def bench_read() -> int:
    """Read every item on a full page, repeatedly."""
    page = SlottedPage.new()
    while page.can_fit(len(_ROW)):
        page.add_item(_ROW)

    reads = 0
    offsets = list(page.live_offset_numbers())
    for _ in range(400):
        for offset_number in offsets:
            page.get_item(offset_number)
            reads += 1
    return reads


def bench_read_view() -> int:
    """Same as :func:`bench_read` but via zero-copy memoryviews."""
    page = SlottedPage.new()
    while page.can_fit(len(_ROW)):
        page.add_item(_ROW)

    reads = 0
    offsets = list(page.live_offset_numbers())
    for _ in range(400):
        for offset_number in offsets:
            page.get_item_view(offset_number)
            reads += 1
    return reads


def bench_defragment() -> int:
    """Mark half the rows dead, then compact the page."""
    runs = 200
    for _ in range(runs):
        page = SlottedPage.new()
        while page.can_fit(len(_ROW)):
            page.add_item(_ROW)
        for offset_number in list(page.live_offset_numbers())[::2]:
            page.mark_dead(offset_number)
        page.defragment()
    return runs


def bench_checksum() -> int:
    """Compute the page checksum over a full page."""
    page = SlottedPage.new()
    while page.can_fit(len(_ROW)):
        page.add_item(_ROW)

    runs = 300
    for _ in range(runs):
        page.finalize_checksum()
    return runs


def main() -> None:
    """Run the full benchmark suite."""
    page = SlottedPage.new()
    while page.can_fit(len(_ROW)):
        page.add_item(_ROW)

    print(f"LakePG page benchmarks  (row = {len(_ROW)} bytes)")
    print(f"rows per 8 KiB page: {len(page)}")
    print("-" * 80)

    _time("add_item", bench_insert)
    _time("get_item (copy)", bench_read)
    _time("get_item_view (zero-copy)", bench_read_view)
    _time("defragment (full page, 50% dead)", bench_defragment)
    _time("finalize_checksum (8 KiB)", bench_checksum)


if __name__ == "__main__":
    main()
