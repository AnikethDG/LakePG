/*
 * Differential test oracle for LakePG's checksum port.
 *
 * This is a standalone, dependency-free transcription of PostgreSQL's
 * pg_checksum_page (src/include/storage/checksum_impl.h and
 * checksum_block_internal.h). It exists so that the Python implementation in
 * lakepg/checksum.py can be validated against real compiled C arithmetic --
 * 32-bit unsigned wraparound, logical right shift, and little-endian word
 * ordering are exactly the places where a Python port silently diverges.
 *
 * Build and run via tests/test_checksum_vs_c.py, or manually:
 *     cc -O2 -o pgchecksum pg_checksum_oracle.c && ./pgchecksum
 *
 * It emits one "<block_number> <checksum>" line per generated test block.
 */

#include <stdio.h>
#include <stdint.h>
#include <string.h>

#define BLCKSZ 8192
#define N_SUMS 32
#define FNV_PRIME 16777619

/* Byte offset of pd_checksum within PageHeaderData. */
#define PD_CHECKSUM_OFFSET 8

static const uint32_t checksumBaseOffsets[N_SUMS] = {
	0x5B1F36E9, 0xB8525960, 0x02AB50AA, 0x1DE66D2A,
	0x79FF467A, 0x9BB9F8A3, 0x217E7CD2, 0x83E13D2C,
	0xF8D4474F, 0xE39EB970, 0x42C6AE16, 0x993216FA,
	0x7B093B5D, 0x98DAFF3C, 0xF718902A, 0x0B1C9CDB,
	0xE58F764B, 0x187636BC, 0x5D7B3BB1, 0xE73DE7DE,
	0x92BEC979, 0xCCA6C0B2, 0x304A0979, 0x85AA43D4,
	0x783125BB, 0x6CA8EAA2, 0xE407EAC6, 0x4B5CFC3E,
	0x9FBF8C76, 0x15CA20BE, 0xF2CA9FD3, 0x959BD756
};

#define CHECKSUM_COMP(checksum, value) \
do { \
	uint32_t __tmp = (checksum) ^ (value); \
	(checksum) = __tmp * FNV_PRIME ^ (__tmp >> 17); \
} while (0)

typedef union
{
	unsigned char raw[BLCKSZ];
	uint32_t	data[BLCKSZ / (sizeof(uint32_t) * N_SUMS)][N_SUMS];
} PGChecksummablePage;

static uint32_t
pg_checksum_block(const PGChecksummablePage *page)
{
	uint32_t	sums[N_SUMS];
	uint32_t	result = 0;
	uint32_t	i,
				j;

	memcpy(sums, checksumBaseOffsets, sizeof(checksumBaseOffsets));

	for (i = 0; i < (uint32_t) (BLCKSZ / (sizeof(uint32_t) * N_SUMS)); i++)
		for (j = 0; j < N_SUMS; j++)
			CHECKSUM_COMP(sums[j], page->data[i][j]);

	for (i = 0; i < 2; i++)
		for (j = 0; j < N_SUMS; j++)
			CHECKSUM_COMP(sums[j], 0);

	for (i = 0; i < N_SUMS; i++)
		result ^= sums[i];

	return result;
}

static uint16_t
pg_checksum_page(unsigned char *page, uint32_t blkno)
{
	PGChecksummablePage *cpage = (PGChecksummablePage *) page;
	unsigned char save0,
				save1;
	uint32_t	checksum;

	/* Transiently zero pd_checksum, byte-wise to stay endianness-agnostic. */
	save0 = cpage->raw[PD_CHECKSUM_OFFSET];
	save1 = cpage->raw[PD_CHECKSUM_OFFSET + 1];
	cpage->raw[PD_CHECKSUM_OFFSET] = 0;
	cpage->raw[PD_CHECKSUM_OFFSET + 1] = 0;

	checksum = pg_checksum_block(cpage);

	cpage->raw[PD_CHECKSUM_OFFSET] = save0;
	cpage->raw[PD_CHECKSUM_OFFSET + 1] = save1;

	checksum ^= blkno;

	return (uint16_t) ((checksum % 65535) + 1);
}

/*
 * Deterministic pseudo-random block filler.
 *
 * A xorshift32 PRNG is used rather than rand() so that the Python side can
 * reproduce byte-identical blocks without shipping fixtures around.
 */
static void
fill_block(unsigned char *page, uint32_t seed)
{
	uint32_t	state = seed ? seed : 1;
	int			i;

	for (i = 0; i < BLCKSZ; i++)
	{
		state ^= state << 13;
		state ^= state >> 17;
		state ^= state << 5;
		page[i] = (unsigned char) (state & 0xFF);
	}
}

int
main(void)
{
	unsigned char page[BLCKSZ];
	uint32_t	seed;

	/* Case 1: an all-zero block. */
	memset(page, 0, BLCKSZ);
	printf("0 %u\n", pg_checksum_page(page, 0));

	/* Case 2: an all-0xFF block. */
	memset(page, 0xFF, BLCKSZ);
	printf("0 %u\n", pg_checksum_page(page, 0));

	/* Case 3: pseudo-random blocks, each checksummed at its own blkno. */
	for (seed = 1; seed <= 16; seed++)
	{
		fill_block(page, seed);
		printf("%u %u\n", seed, pg_checksum_page(page, seed));
	}

	/* Case 4: one fixed block checksummed at a spread of block numbers. */
	fill_block(page, 99);
	printf("0 %u\n", pg_checksum_page(page, 0));
	printf("1 %u\n", pg_checksum_page(page, 1));
	printf("65535 %u\n", pg_checksum_page(page, 65535));
	printf("4294967295 %u\n", pg_checksum_page(page, 4294967295u));

	return 0;
}
