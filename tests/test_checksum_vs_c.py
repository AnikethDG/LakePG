"""Differential test: Python checksum port vs. compiled C.

The Python implementation in :mod:`lakepg.checksum` emulates C semantics that
Python does not have natively -- 32-bit unsigned multiplication with
wraparound, logical right shift on a fixed-width word, and little-endian word
ordering across a byte buffer. Unit tests that only compare Python against
itself cannot catch a systematic error in any of those.

This module compiles ``tests/oracle/pg_checksum_oracle.c`` -- a standalone
transcription of upstream ``pg_checksum_page`` -- and asserts the two
implementations agree on every generated block.

The whole module is skipped when no C compiler is available.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from lakepg.checksum import checksum_page
from lakepg.constants import BLCKSZ

_ORACLE_SOURCE = Path(__file__).parent / "oracle" / "pg_checksum_oracle.c"

_COMPILER = shutil.which("cc") or shutil.which("gcc") or shutil.which("clang")

pytestmark = pytest.mark.skipif(
    _COMPILER is None, reason="no C compiler available to build the oracle"
)


def _xorshift32_block(seed: int) -> bytes:
    """Reproduce the oracle's ``fill_block`` PRNG exactly.

    Mirrors the xorshift32 generator in the C source so both sides derive
    byte-identical blocks from a seed, with no fixture files to keep in sync.
    """
    state = seed if seed else 1
    mask = 0xFFFFFFFF
    out = bytearray(BLCKSZ)
    for i in range(BLCKSZ):
        state ^= (state << 13) & mask
        state &= mask
        state ^= state >> 17
        state ^= (state << 5) & mask
        state &= mask
        out[i] = state & 0xFF
    return bytes(out)


@pytest.fixture(scope="module")
def oracle_output(tmp_path_factory: pytest.TempPathFactory) -> list[tuple[int, int]]:
    """Compile and run the C oracle, returning ``(block_number, checksum)``."""
    assert _COMPILER is not None
    binary = tmp_path_factory.mktemp("oracle") / "pg_checksum_oracle"

    compile_result = subprocess.run(
        [_COMPILER, "-O2", "-std=c99", "-o", str(binary), str(_ORACLE_SOURCE)],
        capture_output=True,
        text=True,
        check=False,
    )
    if compile_result.returncode != 0:
        pytest.fail(f"failed to compile the C oracle:\n{compile_result.stderr}")

    run_result = subprocess.run([str(binary)], capture_output=True, text=True, check=True)
    parsed: list[tuple[int, int]] = []
    for line in run_result.stdout.strip().splitlines():
        block_number, checksum = line.split()
        parsed.append((int(block_number), int(checksum)))
    return parsed


def _expected_blocks() -> list[bytes]:
    """Build the same sequence of blocks the oracle emits, in the same order."""
    blocks = [bytes(BLCKSZ), b"\xff" * BLCKSZ]
    blocks.extend(_xorshift32_block(seed) for seed in range(1, 17))
    fixed = _xorshift32_block(99)
    blocks.extend([fixed, fixed, fixed, fixed])
    return blocks


def test_oracle_emits_the_expected_number_of_cases(
    oracle_output: list[tuple[int, int]],
) -> None:
    assert len(oracle_output) == len(_expected_blocks()) == 22


def test_python_matches_c_on_every_block(
    oracle_output: list[tuple[int, int]],
) -> None:
    """Every checksum must agree bit for bit with the compiled C oracle."""
    blocks = _expected_blocks()

    mismatches: list[str] = []
    for index, (block, (block_number, expected)) in enumerate(
        zip(blocks, oracle_output, strict=True)
    ):
        actual = checksum_page(block, block_number)
        if actual != expected:
            mismatches.append(
                f"case {index} (blkno={block_number}): C={expected} Python={actual}"
            )

    assert not mismatches, "checksum divergence from C:\n" + "\n".join(mismatches)


def test_checksums_are_well_distributed(
    oracle_output: list[tuple[int, int]],
) -> None:
    """Sanity check that the oracle is not returning a constant."""
    checksums = {checksum for _, checksum in oracle_output}

    assert len(checksums) > 15
    assert all(1 <= checksum <= 0xFFFF for checksum in checksums)
