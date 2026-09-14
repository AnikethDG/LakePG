"""Structural invariant tests for the slotted page engine.

These tests are deliberately written against the *binary layout* rather than
the Python API surface: they assert byte offsets, bit packing, and alignment,
because those are the properties that determine whether an unmodified
PostgreSQL server can read a page LakePG produced.
"""

from __future__ import annotations

import itertools
import struct

import pytest

from lakepg.constants import (
    BLCKSZ,
    LP_DEAD,
    LP_NORMAL,
    LP_REDIRECT,
    LP_UNUSED,
    MAX_ITEM_SIZE,
    PD_HAS_FREE_LINES,
    SIZE_OF_ITEM_ID_DATA,
    SIZE_OF_PAGE_HEADER_DATA,
    maxalign,
)
from lakepg.page import (
    InvalidOffsetError,
    ItemId,
    ItemPointer,
    PageFormatError,
    PageFullError,
    SlottedPage,
)

# ---------------------------------------------------------------------------
# MAXALIGN
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "expected"),
    [(0, 0), (1, 8), (7, 8), (8, 8), (9, 16), (15, 16), (16, 16), (100, 104)],
)
def test_maxalign_rounds_up_to_eight(value: int, expected: int) -> None:
    assert maxalign(value) == expected


# ---------------------------------------------------------------------------
# ItemId bitfield
# ---------------------------------------------------------------------------


class TestItemId:
    """The packed ``lp_off:15, lp_flags:2, lp_len:15`` bitfield."""

    def test_pack_places_fields_at_documented_bit_positions(self) -> None:
        item_id = ItemId(lp_off=8160, lp_flags=LP_NORMAL, lp_len=32)
        raw = item_id.pack()

        assert raw & 0x7FFF == 8160  # bits 0-14
        assert (raw >> 15) & 0x3 == LP_NORMAL  # bits 15-16
        assert (raw >> 17) & 0x7FFF == 32  # bits 17-31

    def test_pack_unpack_round_trips(self) -> None:
        original = ItemId(lp_off=4096, lp_flags=LP_DEAD, lp_len=127)
        assert ItemId.unpack(original.pack()) == original

    def test_packs_into_exactly_four_bytes(self) -> None:
        raw = ItemId(lp_off=0x7FFF, lp_flags=0x3, lp_len=0x7FFF).pack()
        assert len(struct.pack("<I", raw)) == SIZE_OF_ITEM_ID_DATA

    @pytest.mark.parametrize(
        ("kwargs", "field"),
        [
            ({"lp_off": 1 << 15, "lp_flags": 0, "lp_len": 0}, "lp_off"),
            ({"lp_off": 0, "lp_flags": 4, "lp_len": 0}, "lp_flags"),
            ({"lp_off": 0, "lp_flags": 0, "lp_len": 1 << 15}, "lp_len"),
        ],
    )
    def test_rejects_values_wider_than_their_bitfield(
        self, kwargs: dict[str, int], field: str
    ) -> None:
        with pytest.raises(ValueError, match=field):
            ItemId(**kwargs)

    def test_flag_predicates(self) -> None:
        assert ItemId(0, LP_NORMAL, 8).is_normal
        assert ItemId(0, LP_DEAD, 0).is_dead
        assert ItemId(5, LP_REDIRECT, 0).is_redirect
        assert not ItemId.unused().is_used

    def test_aligned_len_reports_padded_footprint(self) -> None:
        assert ItemId(0, LP_NORMAL, 11).aligned_len == 16


# ---------------------------------------------------------------------------
# PageInit
# ---------------------------------------------------------------------------


class TestPageInitialisation:
    def test_new_page_matches_page_init(self) -> None:
        page = SlottedPage.new(block_number=7)

        assert page.pd_lower == SIZE_OF_PAGE_HEADER_DATA
        assert page.pd_upper == BLCKSZ
        assert page.pd_special == BLCKSZ
        assert page.pd_flags == 0
        assert page.pd_lsn == 0
        assert page.pd_prune_xid == 0
        assert page.block_number == 7

    def test_pagesize_version_encodes_to_0x2004(self) -> None:
        page = SlottedPage.new()
        (encoded,) = struct.unpack_from("<H", page.buffer, 18)

        assert encoded == 0x2004
        assert page.page_size == BLCKSZ
        assert page.layout_version == 4

    def test_new_page_is_empty(self) -> None:
        page = SlottedPage.new()

        assert len(page) == 0
        assert page.max_offset_number == 0
        assert page.free_space == BLCKSZ - SIZE_OF_PAGE_HEADER_DATA - 4

    def test_special_space_is_maxaligned_and_reserved(self) -> None:
        page = SlottedPage.new(special_size=12)

        assert page.pd_special == BLCKSZ - 16  # MAXALIGN(12) == 16
        assert page.pd_upper == page.pd_special

    def test_rejects_special_space_that_consumes_the_page(self) -> None:
        with pytest.raises(ValueError, match="no room"):
            SlottedPage.new(special_size=BLCKSZ)

    def test_rejects_wrongly_sized_buffer(self) -> None:
        with pytest.raises(PageFormatError, match="exactly 8192 bytes"):
            SlottedPage(bytearray(4096))


# ---------------------------------------------------------------------------
# PageAddItem
# ---------------------------------------------------------------------------


class TestAddItem:
    def test_returns_first_offset_number(self) -> None:
        page = SlottedPage.new(block_number=3)
        tid = page.add_item(b"hello")

        assert tid == ItemPointer(block_number=3, offset_number=1)

    def test_round_trips_payload(self) -> None:
        page = SlottedPage.new()
        tid = page.add_item(b"hello world")

        assert page.get_item(tid.offset_number) == b"hello world"

    def test_payloads_grow_downward_from_the_page_end(self) -> None:
        page = SlottedPage.new()
        page.add_item(b"A" * 16)
        first_upper = page.pd_upper
        page.add_item(b"B" * 16)

        assert page.pd_upper == first_upper - 16
        assert page.item_id(2).lp_off < page.item_id(1).lp_off

    def test_line_pointers_grow_upward_from_the_header(self) -> None:
        page = SlottedPage.new()
        for i in range(3):
            page.add_item(bytes([i]) * 8)

        assert page.pd_lower == SIZE_OF_PAGE_HEADER_DATA + 3 * SIZE_OF_ITEM_ID_DATA
        assert page.max_offset_number == 3

    def test_lp_len_is_unpadded_while_storage_is_aligned(self) -> None:
        page = SlottedPage.new()
        page.add_item(b"x" * 11)
        item_id = page.item_id(1)

        assert item_id.lp_len == 11
        assert item_id.aligned_len == 16
        assert page.pd_upper == BLCKSZ - 16

    def test_alignment_padding_is_zeroed(self) -> None:
        page = SlottedPage.new()
        page.add_item(b"x" * 11)
        item_id = page.item_id(1)
        padding = bytes(page.buffer[item_id.lp_off + 11 : item_id.lp_off + 16])

        assert padding == b"\x00" * 5

    def test_payloads_never_overlap(self) -> None:
        page = SlottedPage.new()
        payloads = [bytes([i]) * (i + 1) for i in range(40)]
        for payload in payloads:
            page.add_item(payload)

        spans = sorted(
            (page.item_id(o).lp_off, page.item_id(o).aligned_len)
            for o in page.live_offset_numbers()
        )
        for (off_a, len_a), (off_b, _) in itertools.pairwise(spans):
            assert off_a + len_a <= off_b

    def test_all_payloads_readable_after_bulk_insert(self) -> None:
        page = SlottedPage.new()
        payloads = [f"row-{i:04d}".encode() for i in range(100)]
        for payload in payloads:
            page.add_item(payload)

        assert [payload for _, payload in page.items()] == payloads

    def test_rejects_empty_payload(self) -> None:
        page = SlottedPage.new()
        with pytest.raises(ValueError, match="zero-length"):
            page.add_item(b"")

    def test_rejects_payload_larger_than_max_item_size(self) -> None:
        page = SlottedPage.new()
        with pytest.raises(ValueError, match="exceeds the maximum"):
            page.add_item(b"x" * (MAX_ITEM_SIZE + 1))

    def test_raises_page_full_when_space_is_exhausted(self) -> None:
        page = SlottedPage.new()
        with pytest.raises(PageFullError):
            while True:
                page.add_item(b"x" * 512)

    def test_page_remains_valid_after_filling_to_capacity(self) -> None:
        page = SlottedPage.new()
        with pytest.raises(PageFullError):
            while True:
                page.add_item(b"x" * 200)

        page.validate()
        assert page.pd_lower <= page.pd_upper

    def test_free_space_accounts_for_the_new_line_pointer(self) -> None:
        page = SlottedPage.new()
        before = page.free_space
        page.add_item(b"x" * 64)

        assert page.free_space == before - 64 - SIZE_OF_ITEM_ID_DATA

    def test_can_fit_agrees_with_add_item(self) -> None:
        page = SlottedPage.new()
        while page.can_fit(128):
            page.add_item(b"x" * 128)

        with pytest.raises(PageFullError):
            page.add_item(b"x" * 128)


# ---------------------------------------------------------------------------
# Deletion and slot reuse
# ---------------------------------------------------------------------------


class TestDeletionAndReuse:
    def test_mark_dead_retains_the_slot(self) -> None:
        page = SlottedPage.new()
        page.add_item(b"doomed")
        page.mark_dead(1)

        assert page.item_id(1).is_dead
        assert page.max_offset_number == 1
        assert len(page) == 0

    def test_reading_a_dead_slot_raises(self) -> None:
        page = SlottedPage.new()
        page.add_item(b"doomed")
        page.mark_dead(1)

        with pytest.raises(InvalidOffsetError, match="not LP_NORMAL"):
            page.get_item(1)

    def test_set_unused_sets_the_free_lines_hint(self) -> None:
        page = SlottedPage.new()
        page.add_item(b"gone")
        page.set_unused(1)

        assert page.item_id(1).lp_flags == LP_UNUSED
        assert page.pd_flags & PD_HAS_FREE_LINES

    def test_unused_slot_is_reused_before_appending(self) -> None:
        page = SlottedPage.new()
        page.add_item(b"one")
        page.add_item(b"two")
        page.set_unused(1)

        tid = page.add_item(b"recycled")

        assert tid.offset_number == 1
        assert page.max_offset_number == 2
        assert page.get_item(1) == b"recycled"

    def test_explicit_offset_placement(self) -> None:
        page = SlottedPage.new()
        page.add_item(b"one")
        page.add_item(b"two")
        page.set_unused(2)

        tid = page.add_item(b"placed", offset_number=2)

        assert tid.offset_number == 2

    def test_explicit_offset_rejects_an_occupied_slot(self) -> None:
        page = SlottedPage.new()
        page.add_item(b"one")

        with pytest.raises(InvalidOffsetError, match="already in use"):
            page.add_item(b"two", offset_number=1)

    def test_explicit_offset_rejects_a_gap(self) -> None:
        page = SlottedPage.new()
        page.add_item(b"one")

        with pytest.raises(InvalidOffsetError, match="gap"):
            page.add_item(b"two", offset_number=5)

    def test_redirect_points_at_another_offset_number(self) -> None:
        page = SlottedPage.new()
        page.add_item(b"old")
        page.add_item(b"new")
        page.set_redirect(1, 2)

        assert page.item_id(1).is_redirect
        assert page.item_id(1).lp_off == 2

    @pytest.mark.parametrize("offset_number", [0, 1, 99])
    def test_out_of_range_offsets_raise(self, offset_number: int) -> None:
        page = SlottedPage.new()
        with pytest.raises(InvalidOffsetError):
            page.item_id(offset_number)


# ---------------------------------------------------------------------------
# PageRepairFragmentation
# ---------------------------------------------------------------------------


class TestDefragmentation:
    def test_reclaims_space_from_dead_tuples(self) -> None:
        page = SlottedPage.new()
        for i in range(10):
            page.add_item(bytes([i]) * 64)
        for offset_number in (2, 4, 6):
            page.mark_dead(offset_number)

        before = page.exact_free_space
        reclaimed = page.defragment()

        assert reclaimed == 3 * 64
        assert page.exact_free_space == before + 3 * 64

    def test_offset_numbers_are_stable_across_compaction(self) -> None:
        page = SlottedPage.new()
        for i in range(10):
            page.add_item(f"row-{i}".encode())
        page.mark_dead(3)
        page.mark_dead(7)

        survivors = dict(page.items())
        page.defragment()

        assert dict(page.items()) == survivors

    def test_payloads_survive_intact(self) -> None:
        page = SlottedPage.new()
        for i in range(20):
            page.add_item(f"payload-{i:03d}".encode() * (i % 4 + 1))
        for offset_number in (1, 5, 9, 13, 17):
            page.mark_dead(offset_number)

        expected = dict(page.items())
        page.defragment()

        assert dict(page.items()) == expected
        page.validate()

    def test_live_tuples_become_contiguous(self) -> None:
        page = SlottedPage.new()
        for i in range(8):
            page.add_item(bytes([i]) * 32)
        page.mark_dead(2)
        page.mark_dead(5)
        page.defragment()

        spans = sorted(
            (page.item_id(o).lp_off, page.item_id(o).aligned_len)
            for o in page.live_offset_numbers()
        )
        assert spans[0][0] == page.pd_upper
        for (off_a, len_a), (off_b, _) in itertools.pairwise(spans):
            assert off_a + len_a == off_b
        assert spans[-1][0] + spans[-1][1] == page.pd_special

    def test_trailing_unused_slots_are_released(self) -> None:
        page = SlottedPage.new()
        for i in range(5):
            page.add_item(bytes([i]) * 16)
        page.set_unused(4)
        page.set_unused(5)
        page.defragment()

        assert page.max_offset_number == 3

    def test_interior_unused_slots_are_retained(self) -> None:
        page = SlottedPage.new()
        for i in range(5):
            page.add_item(bytes([i]) * 16)
        page.set_unused(2)
        page.defragment()

        assert page.max_offset_number == 5
        assert page.has_free_line_pointers

    def test_reclaimed_space_is_zeroed(self) -> None:
        page = SlottedPage.new()
        for _ in range(10):
            page.add_item(b"\xff" * 64)
        for offset_number in range(1, 6):
            page.mark_dead(offset_number)
        page.defragment()

        hole = bytes(page.buffer[page.pd_lower : page.pd_upper])
        assert hole == bytes(len(hole))

    def test_space_becomes_reusable(self) -> None:
        page = SlottedPage.new()
        with pytest.raises(PageFullError):
            while True:
                page.add_item(b"x" * 256)
        for offset_number in list(page.live_offset_numbers())[:5]:
            page.mark_dead(offset_number)
        page.defragment()

        page.add_item(b"y" * 256)  # must not raise

    def test_defragmenting_a_clean_page_is_a_no_op(self) -> None:
        page = SlottedPage.new()
        for i in range(5):
            page.add_item(f"row-{i}".encode())
        snapshot = page.to_bytes(with_checksum=False)

        assert page.defragment() == 0
        assert page.to_bytes(with_checksum=False) == snapshot

    def test_defragmenting_an_empty_page_is_safe(self) -> None:
        page = SlottedPage.new()
        page.defragment()

        assert page.max_offset_number == 0
        assert page.pd_upper == page.pd_special
        page.validate()


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class TestValidation:
    def test_accepts_a_well_formed_page(self) -> None:
        page = SlottedPage.new()
        for i in range(10):
            page.add_item(f"row-{i}".encode())
        page.validate()

    def test_rejects_lower_above_upper(self) -> None:
        page = SlottedPage.new()
        page.add_item(b"x" * 32)
        struct.pack_into("<H", page.buffer, 12, BLCKSZ - 8)  # pd_lower

        with pytest.raises(PageFormatError, match="pd_lower <= pd_upper"):
            page.validate()

    def test_rejects_misaligned_lower(self) -> None:
        page = SlottedPage.new()
        struct.pack_into("<H", page.buffer, 12, SIZE_OF_PAGE_HEADER_DATA + 2)

        with pytest.raises(PageFormatError, match="4-byte line pointer boundary"):
            page.validate()

    def test_rejects_unknown_layout_version(self) -> None:
        page = SlottedPage.new()
        struct.pack_into("<H", page.buffer, 18, 0x2003)

        with pytest.raises(PageFormatError, match="layout version 3"):
            page.validate()

    def test_rejects_unknown_flag_bits(self) -> None:
        page = SlottedPage.new()
        page.pd_flags = 0x0080

        with pytest.raises(PageFormatError, match="unknown bits"):
            page.validate()

    def test_rejects_line_pointer_outside_tuple_storage(self) -> None:
        page = SlottedPage.new()
        page.add_item(b"x" * 32)
        struct.pack_into(
            "<I", page.buffer, 24, ItemId(lp_off=64, lp_flags=LP_NORMAL, lp_len=32).pack()
        )

        with pytest.raises(PageFormatError, match="outside tuple storage"):
            page.validate()

    def test_accepts_an_all_zero_page_as_new(self) -> None:
        page = SlottedPage(bytearray(BLCKSZ))

        assert page.is_new
        page.validate()

    def test_rejects_a_partially_zeroed_page(self) -> None:
        buf = bytearray(BLCKSZ)
        buf[100] = 1
        page = SlottedPage(buf)

        with pytest.raises(PageFormatError, match="not all-zero"):
            page.validate()


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------


class TestSerialisation:
    def test_to_bytes_emits_exactly_one_block(self) -> None:
        page = SlottedPage.new()
        page.add_item(b"hello")

        assert len(page.to_bytes()) == BLCKSZ

    def test_round_trips_through_bytes(self) -> None:
        original = SlottedPage.new(block_number=42)
        for i in range(25):
            original.add_item(f"row-{i:03d}".encode())
        original.pd_lsn = 0xDEADBEEFCAFE

        restored = SlottedPage.from_bytes(
            original.to_bytes(), block_number=42, verify_checksum=True
        )

        assert restored.pd_lsn == 0xDEADBEEFCAFE
        assert dict(restored.items()) == dict(original.items())

    def test_checksum_is_stored_and_verifies(self) -> None:
        page = SlottedPage.new(block_number=5)
        page.add_item(b"checksummed")
        page.finalize_checksum()

        assert page.pd_checksum != 0
        assert page.verify_checksum()

    def test_checksum_detects_a_single_flipped_bit(self) -> None:
        page = SlottedPage.new(block_number=5)
        page.add_item(b"tamper-evident")
        image = bytearray(page.to_bytes())
        image[4000] ^= 0x01

        assert not SlottedPage(image, block_number=5).verify_checksum()

    def test_checksum_detects_a_transposed_block(self) -> None:
        page = SlottedPage.new(block_number=5)
        page.add_item(b"located")
        image = page.to_bytes()

        assert SlottedPage(bytearray(image), block_number=5).verify_checksum()
        assert not SlottedPage(bytearray(image), block_number=6).verify_checksum()

    def test_lsn_survives_a_round_trip(self) -> None:
        page = SlottedPage.new()
        page.pd_lsn = 0xFFFFFFFFFFFFFFFF

        assert SlottedPage.from_bytes(page.to_bytes()).pd_lsn == 0xFFFFFFFFFFFFFFFF

    def test_identical_logical_content_yields_identical_bytes(self) -> None:
        def build() -> bytes:
            page = SlottedPage.new(block_number=1)
            for i in range(12):
                page.add_item(f"deterministic-{i}".encode())
            return page.to_bytes()

        assert build() == build()

    def test_defragmented_page_matches_a_freshly_built_one(self) -> None:
        churned = SlottedPage.new(block_number=1)
        for i in range(6):
            churned.add_item(f"keep-{i}".encode())
        churned.add_item(b"scratch" * 4)
        churned.mark_dead(churned.max_offset_number)
        churned.set_unused(churned.max_offset_number)
        churned.defragment()

        pristine = SlottedPage.new(block_number=1)
        for i in range(6):
            pristine.add_item(f"keep-{i}".encode())

        assert churned.to_bytes() == pristine.to_bytes()
