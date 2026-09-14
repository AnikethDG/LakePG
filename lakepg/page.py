"""The 8 KiB slotted page -- LakePG's fundamental unit of storage.

A PostgreSQL heap page is a fixed 8192-byte block using the classic *slotted
page* design. Two regions grow toward each other from opposite ends of the
block, and the gap between them is the free space:

    byte 0      +--------------------------------------+
                |  PageHeaderData (24 bytes, fixed)    |
    byte 24     +--------------------------------------+
                |  pd_linp[]: ItemIdData line pointers |  grows down
                |  4 bytes each, 1-based indexing      |     |
    pd_lower    +--------------------------------------+     v
                |                                      |
                |            FREE SPACE                |
                |                                      |
    pd_upper    +--------------------------------------+     ^
                |  Tuple payloads, MAXALIGN'd          |     |
                |  stored back to front                |  grows up
    pd_special  +--------------------------------------+
                |  Special space (0 bytes for heaps)   |
    byte 8192   +--------------------------------------+

The indirection through the line pointer array is the whole point of the
design. A tuple is addressed by ``(block_number, offset_number)`` -- its
*TID* -- and ``offset_number`` indexes ``pd_linp[]`` rather than naming a byte
position. Payloads can therefore be shuffled around inside the block during
defragmentation without invalidating a single index entry or HOT chain
pointer, because only the line pointer's ``lp_off`` needs rewriting.

Divergence from PostgreSQL
--------------------------
:meth:`SlottedPage.defragment` zeroes the reclaimed free space. PostgreSQL
leaves whatever bytes happened to be there, since the region between
``pd_lower`` and ``pd_upper`` is unaddressable by definition. LakePG zeroes it
so that a page's bytes are a pure function of its logical contents, which
matters once pages are content-hashed into immutable layer files.

Reference: ``src/backend/storage/page/bufpage.c``
"""

from __future__ import annotations

import struct
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Final

from lakepg.checksum import checksum_page, verify_page_checksum
from lakepg.constants import (
    BLCKSZ,
    FIRST_OFFSET_NUMBER,
    LP_DEAD,
    LP_FLAGS_MASK,
    LP_FLAGS_SHIFT,
    LP_LEN_MASK,
    LP_LEN_SHIFT,
    LP_NORMAL,
    LP_OFF_MASK,
    LP_OFF_SHIFT,
    LP_REDIRECT,
    LP_UNUSED,
    MAX_ITEM_SIZE,
    PD_ALL_VISIBLE,
    PD_CHECKSUM_OFFSET,
    PD_FLAGS_OFFSET,
    PD_HAS_FREE_LINES,
    PD_LINP_OFFSET,
    PD_LOWER_OFFSET,
    PD_LSN_OFFSET,
    PD_PAGESIZE_VERSION_OFFSET,
    PD_PRUNE_XID_OFFSET,
    PD_SPECIAL_OFFSET,
    PD_UPPER_OFFSET,
    SIZE_OF_ITEM_ID_DATA,
    SIZE_OF_PAGE_HEADER_DATA,
    decode_pagesize_version,
    encode_pagesize_version,
    maxalign,
)

__all__ = [
    "ChecksumMismatchError",
    "InvalidOffsetError",
    "ItemId",
    "ItemPointer",
    "PageError",
    "PageFormatError",
    "PageFullError",
    "SlottedPage",
]


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class PageError(Exception):
    """Base class for every error raised by the page engine."""


class PageFullError(PageError):
    """Raised when an item cannot fit in the page's remaining free space."""


class InvalidOffsetError(PageError):
    """Raised when an offset number does not address a usable line pointer."""


class PageFormatError(PageError):
    """Raised when a buffer violates a structural page invariant."""


class ChecksumMismatchError(PageError):
    """Raised when a page's stored checksum disagrees with its contents."""


# ---------------------------------------------------------------------------
# Line pointers and tuple identifiers
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ItemId:
    """A decoded ``ItemIdData`` line pointer.

    Packed into 32 bits as ``lp_off:15, lp_flags:2, lp_len:15``.

    Attributes:
        lp_off: Byte offset of the payload from the start of the page. For an
            ``LP_REDIRECT`` pointer this is an *offset number*, not a byte
            offset.
        lp_flags: One of ``LP_UNUSED``, ``LP_NORMAL``, ``LP_REDIRECT``,
            ``LP_DEAD``.
        lp_len: Payload length in bytes, excluding alignment padding.
    """

    lp_off: int
    lp_flags: int
    lp_len: int

    def __post_init__(self) -> None:
        """Reject field values that would silently overflow the packed bitfield.

        Raises:
            ValueError: If any field does not fit in its allotted bit width.
        """
        if not 0 <= self.lp_off <= LP_OFF_MASK:
            raise ValueError(f"lp_off {self.lp_off} exceeds 15 bits")
        if not 0 <= self.lp_flags <= LP_FLAGS_MASK:
            raise ValueError(f"lp_flags {self.lp_flags} exceeds 2 bits")
        if not 0 <= self.lp_len <= LP_LEN_MASK:
            raise ValueError(f"lp_len {self.lp_len} exceeds 15 bits")

    @property
    def is_used(self) -> bool:
        """Whether this slot is anything other than ``LP_UNUSED``."""
        return self.lp_flags != LP_UNUSED

    @property
    def is_normal(self) -> bool:
        """Whether this slot points at real tuple storage."""
        return self.lp_flags == LP_NORMAL

    @property
    def is_dead(self) -> bool:
        """Whether this slot is dead and its storage already reclaimed."""
        return self.lp_flags == LP_DEAD

    @property
    def is_redirect(self) -> bool:
        """Whether this slot is a HOT redirect to another offset number."""
        return self.lp_flags == LP_REDIRECT

    @property
    def aligned_len(self) -> int:
        """Bytes actually consumed by the payload, including MAXALIGN padding."""
        return maxalign(self.lp_len)

    def pack(self) -> int:
        """Encode this line pointer into its packed 32-bit representation."""
        return (
            (self.lp_off << LP_OFF_SHIFT)
            | (self.lp_flags << LP_FLAGS_SHIFT)
            | (self.lp_len << LP_LEN_SHIFT)
        )

    @classmethod
    def unpack(cls, raw: int) -> ItemId:
        """Decode a packed 32-bit line pointer."""
        return cls(
            lp_off=(raw >> LP_OFF_SHIFT) & LP_OFF_MASK,
            lp_flags=(raw >> LP_FLAGS_SHIFT) & LP_FLAGS_MASK,
            lp_len=(raw >> LP_LEN_SHIFT) & LP_LEN_MASK,
        )

    @classmethod
    def unused(cls) -> ItemId:
        """Return a cleared, reusable line pointer."""
        return cls(lp_off=0, lp_flags=LP_UNUSED, lp_len=0)


@dataclass(frozen=True, slots=True, order=True)
class ItemPointer:
    """A TID: the physical address of a tuple within a relation fork.

    Attributes:
        block_number: Zero-based block index within the fork.
        offset_number: One-based index into the block's line pointer array.
    """

    block_number: int
    offset_number: int

    def __str__(self) -> str:
        """Render as PostgreSQL renders a ``tid``: ``(block,offset)``."""
        return f"({self.block_number},{self.offset_number})"


# ---------------------------------------------------------------------------
# The page itself
# ---------------------------------------------------------------------------

_U16: Final[struct.Struct] = struct.Struct("<H")
_U32: Final[struct.Struct] = struct.Struct("<I")
_U64: Final[struct.Struct] = struct.Struct("<Q")


class SlottedPage:
    """A mutable, binary-compatible PostgreSQL heap page.

    The page owns a ``bytearray`` of exactly :data:`~lakepg.constants.BLCKSZ`
    bytes and mutates it in place via :mod:`struct`, so no intermediate copies
    of the block are made during normal operation.

    Example:
        >>> page = SlottedPage.new(block_number=0)
        >>> tid = page.add_item(b"hello world")
        >>> str(tid)
        '(0,1)'
        >>> page.get_item(tid.offset_number)
        b'hello world'
    """

    __slots__ = ("_block_number", "_buf")

    def __init__(self, buf: bytearray, block_number: int = 0) -> None:
        """Wrap an existing buffer without interpreting or validating it.

        This is the low-level constructor. Prefer :meth:`new` to initialise a
        fresh page or :meth:`from_bytes` to adopt an on-disk image.

        Args:
            buf: A mutable buffer of exactly :data:`~lakepg.constants.BLCKSZ`
                bytes. It is adopted by reference, not copied.
            block_number: The block number this page occupies in its relation.
                Used for checksum salting and for minting item pointers.

        Raises:
            PageFormatError: If ``buf`` is not exactly ``BLCKSZ`` bytes.
        """
        if len(buf) != BLCKSZ:
            raise PageFormatError(
                f"page buffer must be exactly {BLCKSZ} bytes, got {len(buf)}"
            )
        self._buf = buf
        self._block_number = block_number

    # -- construction -------------------------------------------------------

    @classmethod
    def new(cls, block_number: int = 0, special_size: int = 0) -> SlottedPage:
        """Initialise a brand new, empty page (``PageInit``).

        Args:
            block_number: The block's position within its relation fork. Mixed
                into the checksum, so it must be accurate.
            special_size: Bytes reserved at the end of the page for
                access-method private data. Zero for heap pages; index AMs use
                it for sibling links and similar metadata.

        Returns:
            A zeroed page with its header initialised.

        Raises:
            ValueError: If ``special_size`` leaves no room for tuples.
        """
        aligned_special = maxalign(special_size)
        if aligned_special + SIZE_OF_PAGE_HEADER_DATA >= BLCKSZ:
            raise ValueError(
                f"special_size {special_size} leaves no room for tuple storage"
            )

        page = cls(bytearray(BLCKSZ), block_number)
        page._set_u16(PD_FLAGS_OFFSET, 0)
        page._set_u16(PD_LOWER_OFFSET, SIZE_OF_PAGE_HEADER_DATA)
        page._set_u16(PD_UPPER_OFFSET, BLCKSZ - aligned_special)
        page._set_u16(PD_SPECIAL_OFFSET, BLCKSZ - aligned_special)
        page._set_u16(PD_PAGESIZE_VERSION_OFFSET, encode_pagesize_version())
        return page

    @classmethod
    def from_bytes(
        cls,
        raw: bytes | bytearray | memoryview,
        block_number: int = 0,
        *,
        verify_checksum: bool = False,
    ) -> SlottedPage:
        """Load an existing page image and validate its structure.

        Args:
            raw: Exactly one block of page bytes.
            block_number: The block's position within its fork.
            verify_checksum: If true, also require that the stored
                ``pd_checksum`` matches the page contents.

        Raises:
            PageFormatError: If any structural invariant is violated.
            ChecksumMismatchError: If ``verify_checksum`` is set and the stored
                checksum does not match.
        """
        buf = bytearray(raw)
        page = cls(buf, block_number)
        page.validate()
        if verify_checksum and not verify_page_checksum(buf, block_number):
            raise ChecksumMismatchError(
                f"checksum mismatch on block {block_number}: "
                f"stored={page.pd_checksum:#06x} "
                f"computed={checksum_page(buf, block_number):#06x}"
            )
        return page

    # -- raw field access ---------------------------------------------------

    def _get_u16(self, offset: int) -> int:
        value: int = _U16.unpack_from(self._buf, offset)[0]
        return value

    def _set_u16(self, offset: int, value: int) -> None:
        _U16.pack_into(self._buf, offset, value)

    def _get_u32(self, offset: int) -> int:
        value: int = _U32.unpack_from(self._buf, offset)[0]
        return value

    def _set_u32(self, offset: int, value: int) -> None:
        _U32.pack_into(self._buf, offset, value)

    # -- header properties --------------------------------------------------

    @property
    def block_number(self) -> int:
        """This block's position within its relation fork."""
        return self._block_number

    @property
    def pd_lsn(self) -> int:
        """LSN of the last WAL record that modified this page."""
        value: int = _U64.unpack_from(self._buf, PD_LSN_OFFSET)[0]
        return value

    @pd_lsn.setter
    def pd_lsn(self, value: int) -> None:
        _U64.pack_into(self._buf, PD_LSN_OFFSET, value)

    @property
    def pd_checksum(self) -> int:
        """The checksum currently stored in the header (0 if never written)."""
        return self._get_u16(PD_CHECKSUM_OFFSET)

    @property
    def pd_flags(self) -> int:
        """Page-level status bits."""
        return self._get_u16(PD_FLAGS_OFFSET)

    @pd_flags.setter
    def pd_flags(self, value: int) -> None:
        self._set_u16(PD_FLAGS_OFFSET, value)

    @property
    def pd_lower(self) -> int:
        """Byte offset to the start of free space (end of ``pd_linp[]``)."""
        return self._get_u16(PD_LOWER_OFFSET)

    @property
    def pd_upper(self) -> int:
        """Byte offset to the end of free space (start of tuple storage)."""
        return self._get_u16(PD_UPPER_OFFSET)

    @property
    def pd_special(self) -> int:
        """Byte offset to the special space at the end of the page."""
        return self._get_u16(PD_SPECIAL_OFFSET)

    @property
    def pd_prune_xid(self) -> int:
        """Oldest XID that may benefit from pruning, or 0 if none."""
        return self._get_u32(PD_PRUNE_XID_OFFSET)

    @pd_prune_xid.setter
    def pd_prune_xid(self, value: int) -> None:
        self._set_u32(PD_PRUNE_XID_OFFSET, value)

    @property
    def page_size(self) -> int:
        """Page size recorded in ``pd_pagesize_version``."""
        return decode_pagesize_version(self._get_u16(PD_PAGESIZE_VERSION_OFFSET))[0]

    @property
    def layout_version(self) -> int:
        """Layout version recorded in ``pd_pagesize_version``."""
        return decode_pagesize_version(self._get_u16(PD_PAGESIZE_VERSION_OFFSET))[1]

    @property
    def is_new(self) -> bool:
        """Whether the page has never been initialised (``PageIsNew``)."""
        return self.pd_upper == 0

    @property
    def all_visible(self) -> bool:
        """Whether every tuple on the page is visible to all transactions."""
        return bool(self.pd_flags & PD_ALL_VISIBLE)

    @property
    def has_free_line_pointers(self) -> bool:
        """Whether at least one ``LP_UNUSED`` slot is available for reuse."""
        return bool(self.pd_flags & PD_HAS_FREE_LINES)

    # -- capacity -----------------------------------------------------------

    @property
    def max_offset_number(self) -> int:
        """Highest allocated offset number, or 0 when no slots exist."""
        lower = self.pd_lower
        if lower <= PD_LINP_OFFSET:
            return 0
        return (lower - PD_LINP_OFFSET) // SIZE_OF_ITEM_ID_DATA

    @property
    def free_space(self) -> int:
        """Payload bytes available for a new item (``PageGetFreeSpace``).

        The cost of the new line pointer is already deducted, so this is the
        largest ``len(data)`` that :meth:`add_item` would accept given that a
        fresh slot must be allocated.
        """
        space = self.pd_upper - self.pd_lower
        if space < SIZE_OF_ITEM_ID_DATA:
            return 0
        return space - SIZE_OF_ITEM_ID_DATA

    @property
    def exact_free_space(self) -> int:
        """Raw gap between ``pd_lower`` and ``pd_upper``, ignoring slot cost."""
        return max(0, self.pd_upper - self.pd_lower)

    def can_fit(self, size: int) -> bool:
        """Whether an item of ``size`` bytes can currently be added."""
        needs_new_slot = self._find_free_slot() is None
        required = maxalign(size) + (SIZE_OF_ITEM_ID_DATA if needs_new_slot else 0)
        return required <= self.exact_free_space

    # -- line pointer access ------------------------------------------------

    def _linp_offset(self, offset_number: int) -> int:
        """Byte offset of a line pointer, validating the 1-based index."""
        if not FIRST_OFFSET_NUMBER <= offset_number <= self.max_offset_number:
            raise InvalidOffsetError(
                f"offset number {offset_number} out of range "
                f"[1, {self.max_offset_number}]"
            )
        return PD_LINP_OFFSET + (offset_number - 1) * SIZE_OF_ITEM_ID_DATA

    def item_id(self, offset_number: int) -> ItemId:
        """Return the decoded line pointer at ``offset_number``."""
        return ItemId.unpack(self._get_u32(self._linp_offset(offset_number)))

    def _set_item_id(self, offset_number: int, item_id: ItemId) -> None:
        self._set_u32(self._linp_offset(offset_number), item_id.pack())

    def _find_free_slot(self) -> int | None:
        """Return a reusable ``LP_UNUSED`` offset number, or ``None``."""
        if not self.has_free_line_pointers:
            return None
        for offset_number in range(FIRST_OFFSET_NUMBER, self.max_offset_number + 1):
            if not self.item_id(offset_number).is_used:
                return offset_number
        # The hint bit was stale; clear it so we stop rescanning.
        self.pd_flags &= ~PD_HAS_FREE_LINES
        return None

    # -- item operations ----------------------------------------------------

    def add_item(
        self,
        data: bytes | bytearray | memoryview,
        offset_number: int | None = None,
    ) -> ItemPointer:
        """Insert an item and return its TID (``PageAddItem``).

        Payloads are written back-to-front from ``pd_upper`` and padded to a
        ``MAXALIGN`` boundary, while ``lp_len`` records the true unpadded
        length.

        Args:
            data: The payload bytes. For a heap page this is a serialised
                tuple; the page engine itself is agnostic to the contents.
            offset_number: Optionally place the item in a specific slot, which
                must currently be ``LP_UNUSED``. Defaults to reusing a free
                slot if one exists, otherwise appending a new one.

        Returns:
            The :class:`ItemPointer` addressing the stored item.

        Raises:
            ValueError: If ``data`` is empty or exceeds the per-item maximum.
            PageFullError: If the page lacks room for the payload.
            InvalidOffsetError: If an explicit ``offset_number`` is unusable.
        """
        size = len(data)
        if size == 0:
            raise ValueError("cannot store a zero-length item")
        if size > MAX_ITEM_SIZE:
            raise ValueError(
                f"item of {size} bytes exceeds the maximum of {MAX_ITEM_SIZE}"
            )

        if offset_number is None:
            target = self._find_free_slot()
            needs_new_slot = target is None
            if target is None:
                target = self.max_offset_number + 1
        else:
            target = offset_number
            if target < FIRST_OFFSET_NUMBER:
                raise InvalidOffsetError(f"offset number {target} must be >= 1")
            if target <= self.max_offset_number:
                if self.item_id(target).is_used:
                    raise InvalidOffsetError(f"offset number {target} is already in use")
                needs_new_slot = False
            elif target == self.max_offset_number + 1:
                needs_new_slot = True
            else:
                raise InvalidOffsetError(
                    f"offset number {target} would leave a gap in pd_linp[]"
                )

        aligned_size = maxalign(size)
        new_lower = self.pd_lower + (SIZE_OF_ITEM_ID_DATA if needs_new_slot else 0)
        new_upper = self.pd_upper - aligned_size

        if new_lower > new_upper:
            raise PageFullError(
                f"cannot fit {size} bytes ({aligned_size} aligned"
                f"{' + 4 for a new slot' if needs_new_slot else ''}): "
                f"only {self.exact_free_space} bytes free"
            )

        # Write the payload, then publish it by updating the header pointers.
        self._buf[new_upper : new_upper + size] = data
        if aligned_size > size:
            # Zero the alignment padding to keep the image deterministic.
            self._buf[new_upper + size : new_upper + aligned_size] = bytes(
                aligned_size - size
            )

        self._set_u16(PD_LOWER_OFFSET, new_lower)
        self._set_u16(PD_UPPER_OFFSET, new_upper)
        self._set_item_id(
            target, ItemId(lp_off=new_upper, lp_flags=LP_NORMAL, lp_len=size)
        )

        return ItemPointer(self._block_number, target)

    def get_item(self, offset_number: int) -> bytes:
        """Return the payload stored at ``offset_number``.

        Raises:
            InvalidOffsetError: If the slot is not ``LP_NORMAL``.
        """
        item_id = self.item_id(offset_number)
        if not item_id.is_normal:
            raise InvalidOffsetError(
                f"offset number {offset_number} is not LP_NORMAL "
                f"(lp_flags={item_id.lp_flags})"
            )
        return bytes(self._buf[item_id.lp_off : item_id.lp_off + item_id.lp_len])

    def get_item_view(self, offset_number: int) -> memoryview:
        """Return a zero-copy view of the payload at ``offset_number``.

        The view aliases the page buffer, so it is invalidated by any
        subsequent mutation of the page.
        """
        item_id = self.item_id(offset_number)
        if not item_id.is_normal:
            raise InvalidOffsetError(
                f"offset number {offset_number} is not LP_NORMAL "
                f"(lp_flags={item_id.lp_flags})"
            )
        return memoryview(self._buf)[item_id.lp_off : item_id.lp_off + item_id.lp_len]

    def mark_dead(self, offset_number: int) -> None:
        """Mark a slot ``LP_DEAD`` (``ItemIdMarkDead``).

        The line pointer is retained so that existing index entries still
        resolve, but its storage becomes reclaimable by :meth:`defragment`.
        """
        item_id = self.item_id(offset_number)
        if not item_id.is_used:
            raise InvalidOffsetError(f"offset number {offset_number} is already unused")
        self._set_item_id(offset_number, ItemId(lp_off=0, lp_flags=LP_DEAD, lp_len=0))

    def set_unused(self, offset_number: int) -> None:
        """Release a slot entirely (``ItemIdSetUnused``).

        Only safe once no index entry can still reference this TID.
        """
        self._set_item_id(offset_number, ItemId.unused())
        self.pd_flags |= PD_HAS_FREE_LINES

    def set_redirect(self, offset_number: int, target_offset: int) -> None:
        """Turn a slot into a HOT redirect pointing at ``target_offset``."""
        if not FIRST_OFFSET_NUMBER <= target_offset <= self.max_offset_number:
            raise InvalidOffsetError(f"redirect target {target_offset} out of range")
        self._set_item_id(
            offset_number,
            ItemId(lp_off=target_offset, lp_flags=LP_REDIRECT, lp_len=0),
        )

    # -- iteration ----------------------------------------------------------

    def offset_numbers(self) -> Iterator[int]:
        """Yield every allocated offset number, including unused slots."""
        yield from range(FIRST_OFFSET_NUMBER, self.max_offset_number + 1)

    def live_offset_numbers(self) -> Iterator[int]:
        """Yield only the offset numbers whose slots are ``LP_NORMAL``."""
        for offset_number in self.offset_numbers():
            if self.item_id(offset_number).is_normal:
                yield offset_number

    def items(self) -> Iterator[tuple[int, bytes]]:
        """Yield ``(offset_number, payload)`` for every live item."""
        for offset_number in self.live_offset_numbers():
            yield offset_number, self.get_item(offset_number)

    def __len__(self) -> int:
        """Number of live (``LP_NORMAL``) items on the page."""
        return sum(1 for _ in self.live_offset_numbers())

    def __repr__(self) -> str:
        """Summarise occupancy and header state for interactive debugging."""
        return (
            f"SlottedPage(block={self._block_number}, "
            f"items={len(self)}/{self.max_offset_number}, "
            f"lower={self.pd_lower}, upper={self.pd_upper}, "
            f"free={self.free_space})"
        )

    # -- maintenance --------------------------------------------------------

    def defragment(self) -> int:
        """Compact live tuples toward the end of the page.

        Mirrors ``PageRepairFragmentation``. Storage belonging to ``LP_DEAD``
        and ``LP_UNUSED`` slots is reclaimed by sliding the surviving payloads
        up against ``pd_special``; trailing unused slots are then released by
        pulling ``pd_lower`` back.

        Crucially, offset numbers are stable across this operation -- only each
        slot's ``lp_off`` changes -- so index entries and HOT chains remain
        valid.

        Returns:
            The number of bytes reclaimed.
        """
        before = self.exact_free_space

        # Sort by current position, highest first, so that copying a payload to
        # its new home can never overwrite one not yet moved.
        live: list[tuple[int, ItemId]] = [
            (offset_number, self.item_id(offset_number))
            for offset_number in self.live_offset_numbers()
        ]
        live.sort(key=lambda entry: entry[1].lp_off, reverse=True)

        upper = self.pd_special
        for offset_number, item_id in live:
            aligned = item_id.aligned_len
            new_off = upper - aligned
            if new_off != item_id.lp_off:
                payload = bytes(
                    self._buf[item_id.lp_off : item_id.lp_off + item_id.lp_len]
                )
                self._buf[new_off : new_off + item_id.lp_len] = payload
                if aligned > item_id.lp_len:
                    self._buf[new_off + item_id.lp_len : new_off + aligned] = bytes(
                        aligned - item_id.lp_len
                    )
                self._set_item_id(
                    offset_number,
                    ItemId(lp_off=new_off, lp_flags=LP_NORMAL, lp_len=item_id.lp_len),
                )
            upper = new_off

        self._set_u16(PD_UPPER_OFFSET, upper)

        # Release trailing unused slots, as PageRepairFragmentation does.
        last = self.max_offset_number
        while last >= FIRST_OFFSET_NUMBER and not self.item_id(last).is_used:
            last -= 1
        self._set_u16(PD_LOWER_OFFSET, PD_LINP_OFFSET + last * SIZE_OF_ITEM_ID_DATA)

        # Refresh the free-slot hint over whatever slots remain.
        if any(not self.item_id(o).is_used for o in self.offset_numbers()):
            self.pd_flags |= PD_HAS_FREE_LINES
        else:
            self.pd_flags &= ~PD_HAS_FREE_LINES

        # Zero the reclaimed hole. See the module docstring for why LakePG does
        # this and PostgreSQL does not.
        self._buf[self.pd_lower : self.pd_upper] = bytes(self.exact_free_space)

        return self.exact_free_space - before

    def validate(self) -> None:
        """Assert every structural invariant, raising on the first violation.

        Raises:
            PageFormatError: If the page is malformed.
        """
        lower, upper, special = self.pd_lower, self.pd_upper, self.pd_special

        if self.is_new:
            if any(self._buf):
                raise PageFormatError("pd_upper is 0 but the page is not all-zero")
            return

        if not SIZE_OF_PAGE_HEADER_DATA <= lower <= upper <= special <= BLCKSZ:
            raise PageFormatError(
                "page pointers violate "
                "24 <= pd_lower <= pd_upper <= pd_special <= 8192 "
                f"(lower={lower}, upper={upper}, special={special})"
            )

        if (lower - PD_LINP_OFFSET) % SIZE_OF_ITEM_ID_DATA != 0:
            raise PageFormatError(
                f"pd_lower={lower} does not land on a 4-byte line pointer boundary"
            )

        page_size, version = decode_pagesize_version(
            self._get_u16(PD_PAGESIZE_VERSION_OFFSET)
        )
        if page_size != BLCKSZ:
            raise PageFormatError(f"page reports size {page_size}, expected {BLCKSZ}")
        if version != 4:
            raise PageFormatError(f"unsupported page layout version {version}")

        if self.pd_flags & ~0x0007:
            raise PageFormatError(f"pd_flags={self.pd_flags:#06x} has unknown bits set")

        for offset_number in self.offset_numbers():
            item_id = self.item_id(offset_number)
            if not item_id.is_normal:
                continue
            if item_id.lp_off < upper or item_id.lp_off + item_id.lp_len > special:
                raise PageFormatError(
                    f"line pointer {offset_number} points outside tuple storage "
                    f"(lp_off={item_id.lp_off}, lp_len={item_id.lp_len}, "
                    f"upper={upper}, special={special})"
                )

    # -- serialisation ------------------------------------------------------

    def finalize_checksum(self) -> int:
        """Compute and store ``pd_checksum``. Call immediately before writing."""
        value = checksum_page(self._buf, self._block_number)
        self._set_u16(PD_CHECKSUM_OFFSET, value)
        return value

    def verify_checksum(self) -> bool:
        """Whether the stored checksum matches the current page contents."""
        return verify_page_checksum(self._buf, self._block_number)

    def to_bytes(self, *, with_checksum: bool = True) -> bytes:
        """Serialise the page to an immutable 8192-byte image.

        Args:
            with_checksum: Compute and embed ``pd_checksum`` before copying.
                Disable only when producing a page for comparison against an
                image whose checksums are not enabled.
        """
        if with_checksum:
            self.finalize_checksum()
        return bytes(self._buf)

    @property
    def buffer(self) -> memoryview:
        """A mutable, zero-copy view of the underlying 8 KiB block."""
        return memoryview(self._buf)
