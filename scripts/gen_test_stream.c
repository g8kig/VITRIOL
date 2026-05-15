/*
 * gen_test_stream.c — Generate minimal test stream for P2P DMA verification
 *
 * Produces a 3-drop stream:
 *   1. CLAIM vessel 0x0001 (GPU_MAIN)
 *   2. FLOW 0 → 0 (4096 bytes from GGUF offset 0 to GPU VA offset 0)
 *   3. FENCE metapage==1
 *
 * Output: test_p2p.alkas (binary drop array)
 */

#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <stddef.h>

struct vitriol_drop {
    uint8_t  op_code;
    uint8_t  flags;
    uint16_t vessel_id;
    uint64_t src_addr;
    uint64_t dst_addr;
    uint32_t size;
    uint32_t reserved;
    uint32_t crc;
} __attribute__((packed));

#define OP_CLAIM  0x01
#define OP_FLOW   0x03
#define OP_FENCE  0x05

static uint32_t compute_crc(const struct vitriol_drop *drop)
{
    uint32_t crc = 0;
    const uint8_t *bytes = (const uint8_t *)drop;
    size_t crc_offset = offsetof(struct vitriol_drop, crc);

    for (size_t i = 0; i < crc_offset; i++) {
        crc = (crc << 1) | (crc >> 31);
        crc ^= bytes[i];
    }
    return crc;
}

int main(void)
{
    struct vitriol_drop drops[3];
    memset(drops, 0, sizeof(drops));

    /* Drop 0: CLAIM vessel 0x0001 */
    drops[0].op_code = OP_CLAIM;
    drops[0].vessel_id = 0x0001;
    drops[0].crc = compute_crc(&drops[0]);

    /* Drop 1: FLOW 4096 bytes from GGUF offset 0 to GPU VA offset 0 */
    drops[1].op_code = OP_FLOW;
    drops[1].vessel_id = 0x0001;
    drops[1].src_addr = 0;        /* file offset in GGUF */
    drops[1].dst_addr = 0;        /* offset within GPU buffer */
    drops[1].size = 4096;
    drops[1].crc = compute_crc(&drops[1]);

    /* Drop 2: FENCE metapage==1 */
    drops[2].op_code = OP_FENCE;
    drops[2].vessel_id = 0x0001;
    drops[2].dst_addr = 1;        /* expected metapage value */
    drops[2].crc = compute_crc(&drops[2]);

    FILE *f = fopen("test_p2p.alkas", "wb");
    if (!f) {
        perror("fopen");
        return 1;
    }

    size_t written = fwrite(drops, sizeof(struct vitriol_drop), 3, f);
    fclose(f);

    if (written != 3) {
        fprintf(stderr, "Failed to write all drops\n");
        return 1;
    }

    printf("Generated test_p2p.alkas (%zu bytes, 3 drops)\n",
           3 * sizeof(struct vitriol_drop));

    for (int i = 0; i < 3; i++) {
        const char *names[] = { "", "CLAIM", "", "FLOW", "", "FENCE" };
        const char *name = (drops[i].op_code < 6) ? names[drops[i].op_code] : "???";
        printf("  Drop %d: %s vessel=0x%04x src=0x%lx dst=0x%lx size=%u crc=0x%08x\n",
               i, name, drops[i].vessel_id,
               (unsigned long)drops[i].src_addr,
               (unsigned long)drops[i].dst_addr,
               drops[i].size, drops[i].crc);
    }

    return 0;
}
