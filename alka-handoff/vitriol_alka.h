/*
 * vitriol_alka.h — VITRIOL IOCTL Header for Alka Compiler
 *
 * Copyright 2026 Randy Smits-Schreuder Goedheijt
 * Licensed under Apache 2.0 with Runtime Exception.
 *
 * This header defines the C ABI contract between the Alka compiler
 * and the VITRIOL kernel executor. All structures are 32-byte aligned.
 *
 * Generated from alka-lang/src/codegen/codegen.zig
 * Drop layout: packed struct, 32 bytes exactly.
 */

#ifndef VITRIOL_ALKA_H
#define VITRIOL_ALKA_H

#include <stdint.h>

/* ── Drop Packet (32 bytes) ────────────────────────────────────── */

struct vitriol_drop {
    uint8_t  op_code;      /* Instruction opcode (0x01–0x3F)    */
    uint8_t  flags;        /* Execution flags                   */
    uint16_t vessel_id;    /* Target Vessel identifier           */
    uint64_t src_addr;     /* Source physical address            */
    uint64_t dst_addr;     /* Destination physical address       */
    uint32_t size;         /* Transfer size in bytes             */
    uint32_t reserved;     /* Reserved (must be zero)            */
    uint32_t crc;          /* Packet CRC checksum                */
} __attribute__((packed));

_Static_assert(sizeof(struct vitriol_drop) == 32, "Drop must be 32 bytes");

/* ── Opcodes ───────────────────────────────────────────────────── */

#define OP_CLAIM      0x01  /* Stake hardware node              */
#define OP_STAKE      0x02  /* Claim memory region              */
#define OP_FLOW       0x03  /* DMA transfer (SPARK-verified)    */
#define OP_SHIFT      0x04  /* Remap BAR window (SPARK)         */
#define OP_FENCE      0x05  /* Wait for metapage (SPARK)        */
#define OP_SYNC       0x06  /* Memory barrier                   */
#define OP_SENSE      0x07  /* Read sensor                      */
#define OP_PULSE      0x08  /* Timing signal                    */
#define OP_SIGNAL     0x09  /* GPU compute trigger (SPARK)      */
#define OP_YIELD      0x0A  /* Cooperative yield                */
#define OP_RECAST     0x0B  /* FPGA reconfigure                 */
#define OP_SNAP       0x0C  /* Serialize state                  */
#define OP_REVERT     0x0D  /* Restore state                    */
#define OP_LIMIT      0x0E  /* Thermal limit                    */
#define OP_MOLT       0x14  /* Full state dump                  */
#define OP_ECHO       0x17  /* Non-intrusive introspection      */
#define OP_STASIS     0x18  /* Bus-level locking                */
#define OP_FLUX       0x2A  /* Cache invalidation               */
#define OP_AUDIT      0x2B  /* Post-instruction residue check   */
#define OP_DRY_RUN    0x2C  /* Simulate without executing       */
#define OP_MOCK       0x2D  /* Use mock hardware                */
#define OP_PROVE      0x2E  /* Formal verification              */
#define OP_WATCH      0x2F  /* Real-time monitoring             */
#define OP_TRACE      0x30  /* Execution trace                  */
#define OP_GUARD      0x31  /* Runtime safety sentinel          */
#define OP_ISOLATE    0x32  /* Complete hardware isolation      */
#define OP_VERIFY     0x33  /* Cryptographic state verification */
#define OP_OSSIFY     0x34  /* Pin CPU core to Alka             */
#define OP_BOND       0x35  /* RAM-to-GPU direct tunnel         */
#define OP_STILL      0x36  /* Manual DRAM refresh control      */
#define OP_RESONATE   0x37  /* Coordinate reset for pure window */
#define OP_OSCILLATE  0x38  /* Dual-bank refresh coordination   */
#define OP_IMC_HIJACK 0x39  /* Direct memory controller access  */
#define OP_OCCUPY     0x3A  /* Seize PCIe device                */
#define OP_REFRACT    0x3B  /* Sub-tensor slicer (SPARK)        */
#define OP_PIPE       0x3C  /* Continuous DMA ring buffer       */

/* ── Vial Constraints (passed via IOCTL) ───────────────────────── */

struct vitriol_vial {
    uint64_t aperture_size;   /* BAR window size in bytes        */
    uint64_t aperture_max;    /* Maximum transfer window         */
    uint8_t  thermal_halt;    /* Temperature to halt at (°C)     */
    uint8_t  thermal_throttle;/* Temperature to throttle at (°C) */
    uint8_t  dma_capable;     /* 1 if DMA is available           */
    uint8_t  _pad[5];         /* Alignment padding               */
} __attribute__((packed));

_Static_assert(sizeof(struct vitriol_vial) == 24, "Vial must be 24 bytes");

/* ── Execution Result ──────────────────────────────────────────── */

struct vitriol_result {
    int      success;          /* 1 = success, 0 = failure       */
    uint64_t cycles_spent;     /* CPU cycles consumed            */
    uint64_t bytes_transferred;/* Bytes moved by DMA             */
    char     error_message[256];/* Null-terminated error string   */
};

/* ── IOCTL Commands ────────────────────────────────────────────── */

#define VITRIOL_IOC_MAGIC     0xA1
#define VITRIOL_IOC_EXECUTE   _IOW(VITRIOL_IOC_MAGIC, 1, struct vitriol_drop)
#define VITRIOL_IOC_VALIDATE  _IOWR(VITRIOL_IOC_MAGIC, 2, struct vitriol_drop)
#define VITRIOL_IOC_SET_VIAL  _IOW(VITRIOL_IOC_MAGIC, 3, struct vitriol_vial)
#define VITRIOL_IOC_GET_RESULT _IOR(VITRIOL_IOC_MAGIC, 4, struct vitriol_result)

#endif /* VITRIOL_ALKA_H */
