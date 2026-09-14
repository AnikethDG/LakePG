"""Tests for the PostgreSQL data page checksum implementation."""

from __future__ import annotations

import struct

import pytest

from lakepg.checksum import (
    CHECKSUM_BASE_OFFSETS,
    FNV_PRIME,
    N_SUMS,
    checksum_block,
    checksum_page,
    verify_page_checksum,
)
from lakepg.constants import BLCKSZ, PD_CHECKSUM_OFFSET
from lakepg.page import SlottedPage


class TestAlgorithmConstants:
    """Guard the constants transcribed from ``checksum_impl.h``."""

    def test_fnv_prime(self) -> None:
        assert FNV_PRIME == 16777619

    def test_base_offset_count_matches_n_sums(self) -> None:
        assert N_SUMS == 32
        assert len(CHECKSUM_BASE_OFFSETS) == N_SUMS

    def test_base_offsets_are_distinct_uint32s(self) -> None:
        assert len(set(CHECKSUM_BASE_OFFSETS)) == N_SUMS
        assert all(0 <= value <= 0xFFFFFFFF for value in CHECKSUM_BASE_OFFSETS)

    def test_first_and_last_base_offsets(self) -> None:
        assert CHECKSUM_BASE_OFFSETS[0] == 0x5B1F36E9
        assert CHECKSUM_BASE_OFFSETS[-1] == 0x959BD756


class TestChecksumBlock:
    def test_result_is_a_uint32(self) -> None:
        assert 0 <= checksum_block(bytes(BLCKSZ)) <= 0xFFFFFFFF

    def test_is_deterministic(self) -> None:
        block = bytes(range(256)) * 32
        assert checksum_block(block) == checksum_block(block)

    def test_rejects_short_blocks(self) -> None:
        with pytest.raises(ValueError, match="8192-byte block"):
            checksum_block(bytes(1024))

    def test_differs_for_differing_blocks(self) -> None:
        a = bytearray(BLCKSZ)
        b = bytearray(BLCKSZ)
        b[0] = 1
        assert checksum_block(a) != checksum_block(b)


class TestChecksumPage:
    def test_is_never_zero(self) -> None:
        for block_number in range(64):
            assert checksum_page(bytes(BLCKSZ), block_number) != 0

    def test_fits_in_uint16(self) -> None:
        page = SlottedPage.new()
        page.add_item(b"payload")
        assert 1 <= checksum_page(page.buffer, 0) <= 0xFFFF

    def test_ignores_the_stored_checksum_field(self) -> None:
        page = SlottedPage.new()
        page.add_item(b"payload")

        first = bytearray(page.to_bytes(with_checksum=False))
        second = bytearray(first)
        struct.pack_into("<H", second, PD_CHECKSUM_OFFSET, 0xABCD)

        assert checksum_page(first, 0) == checksum_page(second, 0)

    def test_mixes_in_the_block_number(self) -> None:
        page = SlottedPage.new()
        page.add_item(b"payload")
        image = page.to_bytes(with_checksum=False)

        checksums = {checksum_page(image, blkno) for blkno in range(16)}
        assert len(checksums) > 1

    @pytest.mark.parametrize("position", [0, 24, 100, 4096, 8191])
    def test_detects_a_flipped_bit_anywhere_in_the_block(self, position: int) -> None:
        page = SlottedPage.new()
        page.add_item(b"x" * 64)
        clean = bytearray(page.to_bytes(with_checksum=False))
        dirty = bytearray(clean)
        dirty[position] ^= 0x80

        assert checksum_page(clean, 0) != checksum_page(dirty, 0)

    def test_detects_byte_transposition(self) -> None:
        """A pure FNV-1a without the >> 17 mix would be far weaker here."""
        clean = bytearray(BLCKSZ)
        clean[100:104] = b"\x01\x02\x03\x04"
        swapped = bytearray(BLCKSZ)
        swapped[100:104] = b"\x04\x03\x02\x01"

        assert checksum_page(clean, 0) != checksum_page(swapped, 0)


class TestVerifyPageChecksum:
    def test_accepts_a_freshly_finalised_page(self) -> None:
        page = SlottedPage.new(block_number=9)
        page.add_item(b"verified")
        page.finalize_checksum()

        assert verify_page_checksum(page.buffer, 9)

    def test_rejects_a_page_modified_after_finalisation(self) -> None:
        page = SlottedPage.new(block_number=9)
        page.add_item(b"verified")
        page.finalize_checksum()
        page.add_item(b"added later")

        assert not verify_page_checksum(page.buffer, 9)

    def test_rejects_a_page_that_was_never_checksummed(self) -> None:
        page = SlottedPage.new(block_number=9)
        page.add_item(b"unchecksummed")

        assert page.pd_checksum == 0
        assert not verify_page_checksum(page.buffer, 9)
