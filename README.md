# LakePG

**An open-format, cloud-native database engine.**

LakePG is a from-scratch implementation of a third-generation database storage
engine: one that persists transactional state in the **standard PostgreSQL 8 KiB
page format** on commodity object storage, so the same bytes can serve
sub-millisecond OLTP *and* vectorised analytical scans with no proprietary
storage layer in between.

It is a learning project, built to production engineering standards.

---

## Why

Cloud databases have gone through three architectural generations:

| | Generation 1 | Generation 2 | Generation 3 |
|---|---|---|---|
| **Examples** | PostgreSQL, MySQL, Oracle | Aurora, AlloyDB, Socrates | Neon, Databricks Lakebase, **LakePG** |
| **Compute / storage** | Coupled on one node | Separated | Separated |
| **Storage substrate** | Local disk / SAN | Proprietary storage fleet | Commodity object storage |
| **Page format** | Open (PostgreSQL pages) | **Closed** | **Open (PostgreSQL pages)** |
| **Analytics access** | Through the SQL port | Through the SQL port | **Direct from storage** |

Generation 2 solved elasticity but closed the storage layer. To run Spark or
DuckDB over your operational data you must pull it back out through a single
SQL endpoint, or ETL it somewhere else entirely.

Generation 3's insight: if state is persisted in an **open, documented page
format** on object storage, any engine can read it in parallel, directly, with
zero impact on the transactional primary. The bytes themselves become the API.

LakePG implements that idea from first principles.

---

## Status

Built in phases. Each phase ships with tests before the next begins.

| Phase | Component | Status |
|---|---|---|
| 1 | Architecture specification | Done |
| **2** | **8 KiB slotted page engine** | **Done** |
| 3 | Heap tuples and MVCC visibility | Next |
| 4 | Write-ahead log and redo machine | Planned |
| 5 | 2D-LSM pageserver (Key x LSN) | Planned |
| 6 | Vectorised Direct Access reader | Planned |
| 7 | Object storage and zero-copy branching | Planned |

---

## Install

Requires Python 3.11+. No runtime dependencies.

```bash
git clone https://github.com/AnikethDG/LakePG.git
cd LakePG

python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Quick start

```python
from lakepg import SlottedPage

page = SlottedPage.new(block_number=0)

tid = page.add_item(b"the quick brown fox")
print(tid)  # (0,1)
print(page.get_item(tid.offset_number))  # b'the quick brown fox'

for _ in range(50):
    page.add_item(b"another row")

print(page)  # SlottedPage(block=0, items=51, ...)
print(page.free_space, "bytes free")

# Delete some rows, then reclaim their space.
page.mark_dead(2)
page.mark_dead(5)
print(page.defragment(), "bytes reclaimed")

# Offset numbers survive compaction -- this is the point of the design.
print(page.get_item(1))  # b'the quick brown fox'

image = page.to_bytes()  # 8192 bytes, checksum embedded
assert len(image) == 8192
assert SlottedPage.from_bytes(image, 0, verify_checksum=True)
```

---

## What Phase 2 implements

### The slotted page

A PostgreSQL heap page is a fixed 8192-byte block. Two regions grow toward each
other from opposite ends, and the gap between them is the free space:

```
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
```

**Why the indirection matters.** A tuple is addressed by its TID --
`(block_number, offset_number)` -- where `offset_number` indexes the line
pointer array rather than naming a byte position. Payloads can therefore be
moved around inside the block during compaction without invalidating a single
index entry or HOT chain pointer: only the line pointer's `lp_off` is rewritten.

This is the property that makes `VACUUM` possible without rebuilding indexes,
and it is verified directly:
[`test_offset_numbers_are_stable_across_compaction`](tests/test_page_layout.py).

### Binary compatibility

| Structure | Detail |
|---|---|
| `PageHeaderData` | 24 bytes: `pd_lsn`, `pd_checksum`, `pd_flags`, `pd_lower`, `pd_upper`, `pd_special`, `pd_pagesize_version`, `pd_prune_xid` |
| `ItemIdData` | 32-bit bitfield: `lp_off:15, lp_flags:2, lp_len:15` |
| `lp_flags` | `LP_UNUSED`, `LP_NORMAL`, `LP_REDIRECT`, `LP_DEAD` |
| Alignment | `MAXALIGN` to 8 bytes; `lp_len` stores the true unpadded length |
| Version | `pd_pagesize_version` encodes to `0x2004` (8192, layout v4) |
| Checksums | Full `pg_checksum_page` port -- FNV-1a variant, 32 striped sums |

### Checksums, verified against C

The checksum implementation is the one piece where a Python port can silently
diverge: it depends on 32-bit unsigned multiplication with wraparound, logical
right shift on a fixed-width word, and little-endian word ordering. None of
those are native Python semantics.

So the test suite does not just check Python against itself. It
[compiles a standalone C transcription](tests/oracle/pg_checksum_oracle.c) of
upstream `pg_checksum_page` with `cc`, runs both implementations over 22
generated blocks, and asserts they agree bit for bit.

```
tests/test_checksum_vs_c.py ...                                    [100%]
```

The algorithm mixes in `hash >> 17` on top of standard FNV-1a, because plain
FNV-1a never propagates changes into the low bits -- which would leave the
final `% 65535` fold blind to a large class of corruptions. The block number is
also mixed in, so a page relocated to the wrong offset fails validation even
though its bytes are individually intact.

### One deliberate divergence

`defragment()` zeroes the reclaimed free space; PostgreSQL leaves whatever
bytes were there, since the region between `pd_lower` and `pd_upper` is
unaddressable by definition.

LakePG zeroes it so a page's bytes are a pure function of its logical contents.
This matters from Phase 5 onward, when pages are content-hashed into immutable
layer files, and it is what makes
[`test_defragmented_page_matches_a_freshly_built_one`](tests/test_page_layout.py)
possible.

---

## Testing

```bash
pytest                                  # full suite
pytest --cov=lakepg --cov-report=term   # with coverage
ruff check . && mypy lakepg             # lint and type check
```

The suite asserts binary layout rather than API behaviour -- byte offsets, bit
packing, alignment, growth directions -- because those are the properties that
determine whether an unmodified PostgreSQL server could read what LakePG wrote.

Covered: bitfield packing at documented bit positions; `MAXALIGN` padding and
zero-fill; inverted growth of the two regions; non-overlap of payloads;
free-slot reuse; HOT redirects; compaction with stable TIDs; trailing-slot
release; structural validation of malformed pages; and checksum detection of
bit flips, byte transposition, block transposition, and post-write mutation.

## Benchmarks

```bash
python -m benchmarks.bench_page
```

Pure-Python reference implementation, so the absolute numbers are modest by
design; they exist to catch regressions and show where cost actually sits.

```
LakePG page benchmarks  (row = 43 bytes)
rows per 8 KiB page: 157
--------------------------------------------------------------------------------
add_item                                252,482 ops/s (median)
get_item (copy)                         587,219 ops/s (median)
get_item_view (zero-copy)               626,164 ops/s (median)
defragment (full page, 50% dead)            572 ops/s (median)
finalize_checksum (8 KiB)                 2,073 ops/s (median)
```

157 rows per page is exactly `(8192 - 24) / (MAXALIGN(43) + 4)` = `8168 / 52`.

---

## Layout

```
lakepg/
  constants.py   Byte offsets, bitfield shifts, lp_flags, MAXALIGN
  checksum.py    pg_checksum_page port
  page.py        SlottedPage, ItemId, ItemPointer
tests/
  test_page_layout.py    Binary layout invariants
  test_checksum.py       Checksum behaviour
  test_checksum_vs_c.py  Differential test against compiled C
  oracle/                Standalone C transcription
benchmarks/
  bench_page.py
```

Every byte offset and bit width lives in `constants.py`. If a magic number
appears anywhere else in the codebase, that is a bug.

---

## References

- Page layout: [`src/include/storage/bufpage.h`](https://github.com/postgres/postgres/blob/master/src/include/storage/bufpage.h)
- Line pointers: [`src/include/storage/itemid.h`](https://github.com/postgres/postgres/blob/master/src/include/storage/itemid.h)
- Page operations: [`src/backend/storage/page/bufpage.c`](https://github.com/postgres/postgres/blob/master/src/backend/storage/page/bufpage.c)
- Checksums: [`src/include/storage/checksum_impl.h`](https://github.com/postgres/postgres/blob/master/src/include/storage/checksum_impl.h)
- *Lakebase: A Serverless Postgres for the Lakehouse Era*, PVLDB 2026
- *Amazon Aurora: Design Considerations for High Throughput Cloud-Native Relational Databases*, SIGMOD 2017

## License

Apache 2.0. See [LICENSE](LICENSE).

PostgreSQL data structures are reimplemented from the publicly documented
on-disk format; no PostgreSQL source code is included.
