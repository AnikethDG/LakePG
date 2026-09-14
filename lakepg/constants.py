"""Binary layout constants for the PostgreSQL on-disk page format.

Every constant here is a direct transcription of a value defined in the
PostgreSQL source tree. Magic numbers are deliberately avoided in the rest of
the codebase: if a byte offset or bit width appears in ``lakepg``, it is
defined and documented here first.

Canonical references
--------------------
* ``src/include/storage/bufpage.h``    -- PageHeaderData, page layout version
* ``src/include/storage/itemid.h``     -- ItemIdData bitfield and lp_flags
* ``src/include/storage/itemptr.h``    -- ItemPointerData
* ``src/include/storage/block.h``      -- BlockNumber
* ``src/include/pg_config.h``          -- BLCKSZ, MAXIMUM_ALIGNOF
"""

from __future__ import annotations

from typing import Final

# ---------------------------------------------------------------------------
# Block geometry
# ---------------------------------------------------------------------------

#: Size of a single relation block, in bytes. This is the ``BLCKSZ`` compile
#: time constant. 8192 is the PostgreSQL default and the only value LakePG
#: currently supports.
BLCKSZ: Final[int] = 8192

#: Size of the fixed portion of ``PageHeaderData``, i.e. everything preceding
#: the flexible ``pd_linp[]`` line pointer array.
SIZE_OF_PAGE_HEADER_DATA: Final[int] = 24

#: Size of one ``ItemIdData`` line pointer, in bytes.
SIZE_OF_ITEM_ID_DATA: Final[int] = 4

#: ``PG_PAGE_LAYOUT_VERSION``. Bumped by PostgreSQL whenever the physical page
#: layout changes; version 4 has been stable since PostgreSQL 8.3.
PG_PAGE_LAYOUT_VERSION: Final[int] = 4

#: ``MAXIMUM_ALIGNOF`` -- the alignment requirement for tuple storage on the
#: overwhelmingly common 64-bit builds.
MAXIMUM_ALIGNOF: Final[int] = 8

#: ``InvalidBlockNumber`` sentinel.
INVALID_BLOCK_NUMBER: Final[int] = 0xFFFFFFFF

#: ``InvalidOffsetNumber``. Offset numbers are 1-based; 0 means "no item".
INVALID_OFFSET_NUMBER: Final[int] = 0

#: ``FirstOffsetNumber``.
FIRST_OFFSET_NUMBER: Final[int] = 1

#: ``MaxOffsetNumber`` -- the largest representable line pointer index.
MAX_OFFSET_NUMBER: Final[int] = BLCKSZ // SIZE_OF_ITEM_ID_DATA

# ---------------------------------------------------------------------------
# PageHeaderData field offsets
#
#   typedef struct PageHeaderData
#   {
#       PageXLogRecPtr  pd_lsn;               /*  8 bytes, offset  0 */
#       uint16          pd_checksum;          /*  2 bytes, offset  8 */
#       uint16          pd_flags;             /*  2 bytes, offset 10 */
#       LocationIndex   pd_lower;             /*  2 bytes, offset 12 */
#       LocationIndex   pd_upper;             /*  2 bytes, offset 14 */
#       LocationIndex   pd_special;           /*  2 bytes, offset 16 */
#       uint16          pd_pagesize_version;  /*  2 bytes, offset 18 */
#       TransactionId   pd_prune_xid;         /*  4 bytes, offset 20 */
#       ItemIdData      pd_linp[];            /*  flexible, offset 24 */
#   } PageHeaderData;
# ---------------------------------------------------------------------------

PD_LSN_OFFSET: Final[int] = 0
PD_CHECKSUM_OFFSET: Final[int] = 8
PD_FLAGS_OFFSET: Final[int] = 10
PD_LOWER_OFFSET: Final[int] = 12
PD_UPPER_OFFSET: Final[int] = 14
PD_SPECIAL_OFFSET: Final[int] = 16
PD_PAGESIZE_VERSION_OFFSET: Final[int] = 18
PD_PRUNE_XID_OFFSET: Final[int] = 20

#: Byte offset at which the ``pd_linp[]`` line pointer array begins.
PD_LINP_OFFSET: Final[int] = SIZE_OF_PAGE_HEADER_DATA

# ---------------------------------------------------------------------------
# pd_flags bits (bufpage.h)
# ---------------------------------------------------------------------------

#: At least one line pointer is LP_UNUSED and may be reclaimed.
PD_HAS_FREE_LINES: Final[int] = 0x0001

#: The page is considered too full for any further insertions.
PD_PAGE_FULL: Final[int] = 0x0002

#: Every tuple on the page is visible to all transactions.
PD_ALL_VISIBLE: Final[int] = 0x0004

#: Mask of all currently defined ``pd_flags`` bits.
PD_VALID_FLAG_BITS: Final[int] = 0x0007

# ---------------------------------------------------------------------------
# ItemIdData -- a packed 32-bit bitfield (itemid.h)
#
#   typedef struct ItemIdData
#   {
#       unsigned    lp_off:15,    /* offset to tuple from page start */
#                   lp_flags:2,   /* state of line pointer */
#                   lp_len:15;    /* byte length of tuple */
#   } ItemIdData;
#
# On the little-endian targets PostgreSQL is built for, GCC and Clang allocate
# bitfields starting from the least significant bit, giving the layout below.
# ---------------------------------------------------------------------------

LP_OFF_SHIFT: Final[int] = 0
LP_OFF_BITS: Final[int] = 15
LP_OFF_MASK: Final[int] = (1 << LP_OFF_BITS) - 1

LP_FLAGS_SHIFT: Final[int] = 15
LP_FLAGS_BITS: Final[int] = 2
LP_FLAGS_MASK: Final[int] = (1 << LP_FLAGS_BITS) - 1

LP_LEN_SHIFT: Final[int] = 17
LP_LEN_BITS: Final[int] = 15
LP_LEN_MASK: Final[int] = (1 << LP_LEN_BITS) - 1

# ---------------------------------------------------------------------------
# lp_flags values (itemid.h)
# ---------------------------------------------------------------------------

#: Unused slot; ``lp_len`` must be 0 and the slot is reusable.
LP_UNUSED: Final[int] = 0

#: Used slot pointing at a live or recently-dead heap tuple.
LP_NORMAL: Final[int] = 1

#: HOT redirect; ``lp_off`` holds an offset number rather than a byte offset.
LP_REDIRECT: Final[int] = 2

#: Dead slot whose storage has already been reclaimed.
LP_DEAD: Final[int] = 3

# ---------------------------------------------------------------------------
# Derived limits
# ---------------------------------------------------------------------------

#: Largest item payload that can possibly be stored on an empty heap page.
MAX_ITEM_SIZE: Final[int] = (
    BLCKSZ - SIZE_OF_PAGE_HEADER_DATA - SIZE_OF_ITEM_ID_DATA
) & ~(MAXIMUM_ALIGNOF - 1)


def maxalign(value: int) -> int:
    """Round ``value`` up to the next ``MAXIMUM_ALIGNOF`` boundary.

    Mirrors the ``MAXALIGN`` macro from ``src/include/c.h``.

    >>> maxalign(0), maxalign(1), maxalign(8), maxalign(9)
    (0, 8, 8, 16)
    """
    return (value + (MAXIMUM_ALIGNOF - 1)) & ~(MAXIMUM_ALIGNOF - 1)


def encode_pagesize_version(
    pagesize: int = BLCKSZ,
    version: int = PG_PAGE_LAYOUT_VERSION,
) -> int:
    """Pack ``pd_pagesize_version`` the way ``PageSetPageSizeAndVersion`` does.

    The high byte carries the page size and the low byte the layout version,
    which is why the default 8192/v4 combination encodes to ``0x2004``.

    >>> hex(encode_pagesize_version())
    '0x2004'
    """
    return (pagesize & 0xFF00) | (version & 0x00FF)


def decode_pagesize_version(encoded: int) -> tuple[int, int]:
    """Split ``pd_pagesize_version`` back into ``(pagesize, version)``.

    >>> decode_pagesize_version(0x2004)
    (8192, 4)
    """
    return encoded & 0xFF00, encoded & 0x00FF
