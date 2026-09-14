"""PostgreSQL data page checksums.

This is a faithful Python port of ``pg_checksum_page`` from
``src/include/storage/checksum_impl.h``. Producing bit-identical checksums is
what allows a LakePG-written block to be validated by an unmodified PostgreSQL
server (and vice versa), so this module is deliberately a literal translation
rather than an idiomatic rewrite.

The algorithm is an FNV-1a variant with two properties worth calling out:

1. It maintains ``N_SUMS`` (32) independent partial sums that are advanced in
   lockstep. In C this shape is what lets the compiler auto-vectorise the inner
   loop into SIMD; in Python it buys nothing, but we keep the structure because
   the output must match byte for byte.
2. Each round mixes in ``hash >> 17`` on top of the standard FNV step. Plain
   FNV-1a never propagates changes into the low bits, which would leave the
   final ``% 65535`` fold blind to a large class of corruptions.
"""

from __future__ import annotations

import struct
from typing import Final

from lakepg.constants import BLCKSZ, PD_CHECKSUM_OFFSET

__all__ = [
    "FNV_PRIME",
    "N_SUMS",
    "checksum_block",
    "checksum_page",
    "verify_page_checksum",
]

#: Number of partial checksums advanced in parallel (``N_SUMS``).
N_SUMS: Final[int] = 32

#: The 32-bit FNV prime (``FNV_PRIME``).
FNV_PRIME: Final[int] = 16777619

#: Mask used to emulate C's 32-bit unsigned wraparound.
_UINT32_MASK: Final[int] = 0xFFFFFFFF

#: Per-sum initialisation vector (``checksumBaseOffsets``). These are arbitrary
#: but fixed values; they exist so that the 32 partial sums do not all start
#: from the same state.
CHECKSUM_BASE_OFFSETS: Final[tuple[int, ...]] = (
    0x5B1F36E9,
    0xB8525960,
    0x02AB50AA,
    0x1DE66D2A,
    0x79FF467A,
    0x9BB9F8A3,
    0x217E7CD2,
    0x83E13D2C,
    0xF8D4474F,
    0xE39EB970,
    0x42C6AE16,
    0x993216FA,
    0x7B093B5D,
    0x98DAFF3C,
    0xF718902A,
    0x0B1C9CDB,
    0xE58F764B,
    0x187636BC,
    0x5D7B3BB1,
    0xE73DE7DE,
    0x92BEC979,
    0xCCA6C0B2,
    0x304A0979,
    0x85AA43D4,
    0x783125BB,
    0x6CA8EAA2,
    0xE407EAC6,
    0x4B5CFC3E,
    0x9FBF8C76,
    0x15CA20BE,
    0xF2CA9FD3,
    0x959BD756,
)

#: Number of ``uint32`` words in a block (2048 for an 8 KiB page).
_WORDS_PER_BLOCK: Final[int] = BLCKSZ // 4

#: Number of rows in the conceptual ``uint32 data[rows][N_SUMS]`` view (64).
_ROWS_PER_BLOCK: Final[int] = _WORDS_PER_BLOCK // N_SUMS

#: Pre-compiled unpacker for the whole block, as little-endian ``uint32``.
_BLOCK_UNPACKER: Final[struct.Struct] = struct.Struct(f"<{_WORDS_PER_BLOCK}I")

assert len(CHECKSUM_BASE_OFFSETS) == N_SUMS
assert _ROWS_PER_BLOCK * N_SUMS == _WORDS_PER_BLOCK


def checksum_block(page: bytes | bytearray | memoryview) -> int:
    """Compute the raw 32-bit block checksum (``pg_checksum_block``).

    This operates on the page exactly as given; callers that want the
    on-disk-comparable value should use :func:`checksum_page`, which first
    zeroes the ``pd_checksum`` field and mixes in the block number.

    Args:
        page: A buffer of exactly :data:`~lakepg.constants.BLCKSZ` bytes.

    Returns:
        The folded 32-bit checksum.

    Raises:
        ValueError: If ``page`` is not exactly one block long.
    """
    if len(page) != BLCKSZ:
        raise ValueError(f"expected a {BLCKSZ}-byte block, got {len(page)} bytes")

    sums = list(CHECKSUM_BASE_OFFSETS)
    words = _BLOCK_UNPACKER.unpack(bytes(page))

    # Main pass: one CHECKSUM_COMP round per word, striped across the 32 sums.
    for row in range(_ROWS_PER_BLOCK):
        base = row * N_SUMS
        for j in range(N_SUMS):
            tmp = (sums[j] ^ words[base + j]) & _UINT32_MASK
            sums[j] = ((tmp * FNV_PRIME) ^ (tmp >> 17)) & _UINT32_MASK

    # Two extra rounds of zeroes, purely for additional avalanche mixing.
    for _ in range(2):
        for j in range(N_SUMS):
            tmp = sums[j]
            sums[j] = ((tmp * FNV_PRIME) ^ (tmp >> 17)) & _UINT32_MASK

    result = 0
    for partial in sums:
        result ^= partial
    return result & _UINT32_MASK


def checksum_page(page: bytes | bytearray | memoryview, block_number: int) -> int:
    """Compute the 16-bit value destined for ``pd_checksum``.

    Mirrors ``pg_checksum_page``. The stored checksum is excluded from its own
    computation, and the block number is mixed in so that a block relocated to
    the wrong offset in a relation fails validation even though its bytes are
    individually intact.

    Args:
        page: A buffer of exactly :data:`~lakepg.constants.BLCKSZ` bytes.
        block_number: The block's position within its relation fork.

    Returns:
        A checksum in the range ``[1, 65535]``. Zero is never returned, which
        lets an all-zero page be distinguished from a checksummed one.
    """
    scratch = bytearray(page)
    if len(scratch) != BLCKSZ:
        raise ValueError(f"expected a {BLCKSZ}-byte block, got {len(scratch)} bytes")

    # Exclude the stored checksum from its own input.
    struct.pack_into("<H", scratch, PD_CHECKSUM_OFFSET, 0)

    checksum = checksum_block(scratch)
    checksum = (checksum ^ (block_number & _UINT32_MASK)) & _UINT32_MASK

    # Fold to 16 bits with a +1 bias so the result is never zero.
    return (checksum % 65535) + 1


def verify_page_checksum(
    page: bytes | bytearray | memoryview,
    block_number: int,
) -> bool:
    """Return whether the checksum stored in ``page`` matches its contents."""
    stored: int = struct.unpack_from("<H", bytes(page), PD_CHECKSUM_OFFSET)[0]
    return stored == checksum_page(page, block_number)
