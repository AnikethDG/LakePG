"""LakePG -- an open-format, cloud-native database engine.

LakePG persists transactional state in the standard PostgreSQL 8 KiB page
format on commodity object storage, so that the same bytes serve both
sub-millisecond OLTP and vectorised analytical scans without a proprietary
storage layer in between.

Implemented so far (Phase 2): the slotted page engine.
"""

from __future__ import annotations

from lakepg.checksum import checksum_block, checksum_page, verify_page_checksum
from lakepg.constants import BLCKSZ, MAX_ITEM_SIZE, maxalign
from lakepg.page import (
    ChecksumMismatchError,
    InvalidOffsetError,
    ItemId,
    ItemPointer,
    PageError,
    PageFormatError,
    PageFullError,
    SlottedPage,
)

__version__ = "0.1.0"

__all__ = [
    "BLCKSZ",
    "MAX_ITEM_SIZE",
    "ChecksumMismatchError",
    "InvalidOffsetError",
    "ItemId",
    "ItemPointer",
    "PageError",
    "PageFormatError",
    "PageFullError",
    "SlottedPage",
    "__version__",
    "checksum_block",
    "checksum_page",
    "maxalign",
    "verify_page_checksum",
]
