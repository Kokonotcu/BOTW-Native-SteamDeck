// Native sead::SZSDecompressor::streamDecomp (0x04215778): the streaming Yaz0 decompressor behind BotW's file
// streaming. Drop frames at 8 W ran it on DecompThread (113 of 116 samples there, spikes.txt) next to the worker
// threads; the recompiled version copies back-references byte by byte through guest memory accessors.
// Same state machine as the game (open-ead/sead seadSZSDecompressor.cpp; field offsets checked against the
// recompiled code): non-overlapping back-references are copied 16/8 bytes at a time, overlapping ones byte by byte
// in the same order, and the dcbz cache-line zeroing of the original is reproduced (it can zero up to 31 bytes past
// the copy, which later output overwrites). tools/recompile.py renames the recompiled function to f_04215778_guest.
// BOTW_SZS_CHECK=1: every call also runs the recompiled version from the same state and logs any difference.
// BOTW_NATIVE(0x04215778)
#include <stdlib.h>
#include <string.h>

#include <vector>

#include "../log.h"
#include "ppc.h"

extern "C" void f_04215688(PPCContext* c);        // SZSDecompressor::readHeader_ (recompiled)
extern "C" void f_04215778_guest(PPCContext* c);  // the recompiled streamDecomp

namespace {
// DecompContext (big-endian, guest memory)
enum : uint32_t { kDestp = 0, kDestCount = 4, kForceDestCount = 8, kFlagMask = 12, kFlags = 13, kPackHigh = 14,
                  kStep = 16, kLzOffset = 20, kHeaderSize = 22, kCtxSize = 24 };
enum : uint32_t { cStepNormal = 0, cStepShort = 1, cStepLong = 2 };

inline void dcbz(uint8_t* M, uint32_t ea) { memset(M + (ea & ~31u), 0, 32); }

// n bytes from dst - dist to dst, front to back (overlap repeats the pattern, as the byte loop does)
inline void lz_copy(uint8_t* M, uint32_t dst, uint32_t dist, uint32_t n) {
    uint8_t* d = M + dst;
    const uint8_t* s = d - dist;
    if (dist >= 16) {
        while (n >= 16) { memcpy(d, s, 16); d += 16; s += 16; n -= 16; }
    } else if (dist >= 8) {
        while (n >= 8) { memcpy(d, s, 8); d += 8; s += 8; n -= 8; }
    }
    while (n--) *d++ = *s++;
}

void stream_decomp(PPCContext* c) {
    uint8_t* const M = g_mem;
    const uint32_t ctx = c->r[3];
    uint32_t src = c->r[4];
    uint32_t srcSize = c->r[5];

    if (LD8(ctx + kHeaderSize) != 0) {
        c->r[3] = ctx;
        c->r[4] = src;
        c->r[5] = srcSize;
        f_04215688(c);
        const uint32_t read = c->r[3];
        if ((int32_t)read < 0) return;  // r3 = error
        srcSize -= read;
        src += read;
        if (srcSize == 0) {
            c->r[3] = LD8(ctx + kHeaderSize) != 0 ? (uint32_t)-1 : LD32(ctx + kDestCount);
            return;
        }
    }

    uint32_t destCount = LD32(ctx + kDestCount);
    uint32_t packHigh = LD8(ctx + kPackHigh);
    uint32_t destp = LD32(ctx + kDestp);
    uint32_t flags = LD8(ctx + kFlags);
    uint32_t flagMask = LD8(ctx + kFlagMask);
    uint32_t step = LD32(ctx + kStep);
    uint32_t lzOffset = LD16(ctx + kLzOffset);
    uint32_t sinceCheck = 0;

    // Fast path: with > 25 source bytes (one flag byte + 8 three-byte tokens) and > 2184 output bytes (8 x 273) left,
    // no end-of-input/output check can fire inside a token, so whole tokens decode without the state machine. Same
    // writes, dcbz points and final state as the step loop below, which finishes the tail.
    if (step == cStepNormal) {
        const uint8_t* s = M + src;
        const uint8_t* const s0 = s;
        uint32_t left = srcSize;
        while (left > 25 && (int32_t)destCount > 2184) {
            if (flagMask == 0) {
                flags = *s++;
                left--;
                flagMask = 0x80;
            }
            if (flags & flagMask) {
                if ((destp & 31) == 0) dcbz(M, destp);
                M[destp++] = *s++;
                left--;
                destCount--;
            } else {
                const uint32_t b1 = s[0], pair = (b1 << 8) | s[1];
                s += 2;
                left -= 2;
                packHigh = b1;
                lzOffset = (pair & 0xfff) + 1;
                uint32_t n;
                if ((pair >> 12) == 0) {
                    n = *s++ + 18u;
                    left--;
                    for (uint32_t k = 32; k < n; k += 32) dcbz(M, destp + k);
                } else {
                    n = (pair >> 12) + 2;
                    if ((destp & 31) + n > 32) dcbz(M, destp + n);
                }
                destCount -= n;
                lz_copy(M, destp, lzOffset, n);
                destp += n;
            }
            flagMask >>= 1;
            if (++sinceCheck == 1024) {
                sinceCheck = 0;
                PREEMPT_POINT();
            }
        }
        src += (uint32_t)(s - s0);
        srcSize = left;
    }

    if ((int32_t)destCount > 0) {
        for (;;) {
            if (step == cStepLong) {
                uint32_t n = LD8(src) + 18;
                src++;
                srcSize--;
                if (n > destCount) {
                    if (LD32(ctx + kForceDestCount) == 0) {
                        c->r[3] = (uint32_t)-2;
                        return;
                    }
                    n = destCount & 0xffff;
                }
                destCount -= n;
                for (uint32_t k = 32; k < n; k += 32) dcbz(M, destp + k);
                lz_copy(M, destp, lzOffset, n);
                destp += n;
                step = cStepNormal;
                if (srcSize == 0) break;
            } else if (step == cStepShort) {
                const uint32_t pair = (packHigh << 8) | LD8(src);
                src++;
                srcSize--;
                lzOffset = (pair & 0xfff) + 1;
                const uint32_t len = pair >> 12;
                if (len == 0) {
                    step = cStepLong;
                    if (srcSize == 0) break;
                } else {
                    uint32_t n = len + 2;
                    if (n > destCount) {
                        if (LD32(ctx + kForceDestCount) == 0) {
                            c->r[3] = (uint32_t)-2;
                            return;
                        }
                        n = destCount & 0xffff;
                    }
                    destCount -= n;
                    if ((destp & 31) + n > 32) dcbz(M, destp + n);
                    lz_copy(M, destp, lzOffset, n);
                    destp += n;
                    step = cStepNormal;
                    if (srcSize == 0) break;
                }
            } else {
                if (flagMask == 0) {
                    srcSize--;
                    flagMask = 0x80;
                    flags = LD8(src);
                    src++;
                    if (srcSize == 0) break;
                }
                if (flags & flagMask) {
                    if ((destp & 31) == 0) dcbz(M, destp);
                    const uint8_t b = LD8(src);
                    src++;
                    destCount--;
                    srcSize--;
                    ST8(destp, b);
                    destp++;
                    flagMask >>= 1;
                } else {
                    packHigh = LD8(src);
                    flagMask >>= 1;
                    srcSize--;
                    step = cStepShort;
                    src++;
                }
                if (srcSize == 0) break;
            }
            if ((int32_t)destCount <= 0) break;
            if (++sinceCheck == 1024) {  // the recompiled loop checks for preemption every iteration
                sinceCheck = 0;
                PREEMPT_POINT();
            }
        }
    }

    ST32(ctx + kDestp, destp);
    ST8(ctx + kFlags, flags);
    ST8(ctx + kFlagMask, flagMask);
    ST32(ctx + kStep, step);
    ST8(ctx + kPackHigh, packHigh);
    ST32(ctx + kDestCount, destCount);
    ST16(ctx + kLzOffset, lzOffset);
    if (destCount == 0 && LD32(ctx + kForceDestCount) == 0 && srcSize > 32)
        c->r[3] = (uint32_t)-1;
    else
        c->r[3] = destCount;
}

// BOTW_SZS_CHECK=1: run both versions from the same state, compare return value, context and every byte either
// may have written (the output window plus the dcbz tail), keep the recompiled version's result.
void checked(PPCContext* c) {
    uint8_t* const M = g_mem;
    const uint32_t ctx = c->r[3], srcSize = c->r[5];
    const uint32_t destp = LD32(ctx + kDestp), destCount = LD32(ctx + kDestCount);
    uint64_t maxOut = (uint64_t)srcSize * 273 + 273;
    if ((int32_t)destCount > 0 && destCount < maxOut) maxOut = destCount;
    if ((int32_t)destCount <= 0) maxOut = 0;
    const uint32_t win = (uint32_t)maxOut + 64;
    std::vector<uint8_t> ctx0(M + ctx, M + ctx + kCtxSize), win0(M + destp, M + destp + win);
    const PPCContext c0 = *c;
    stream_decomp(c);
    const uint32_t retN = c->r[3];
    std::vector<uint8_t> ctxN(M + ctx, M + ctx + kCtxSize), winN(M + destp, M + destp + win);
    memcpy(M + ctx, ctx0.data(), kCtxSize);
    memcpy(M + destp, win0.data(), win);
    *c = c0;
    f_04215778_guest(c);
    static uint64_t calls = 0, bad = 0;
    calls++;
    const bool same = retN == c->r[3] && memcmp(ctxN.data(), M + ctx, kCtxSize) == 0 && memcmp(winN.data(), M + destp, win) == 0;
    if (!same && bad++ < 20)
        LOG("SZSCHECK mismatch: call %llu ret native %d guest %d srcSize %u", (unsigned long long)calls, (int)retN, (int)c->r[3], srcSize);
    if ((calls & 255) == 0 || !same)
        LOG("SZSCHECK calls %llu mismatches %llu", (unsigned long long)calls, (unsigned long long)bad);
}
}  // namespace

extern "C" void f_04215778(PPCContext* c) {
    static const bool s_check = [] { const char* e = getenv("BOTW_SZS_CHECK"); return e && e[0] == '1'; }();
    static const bool s_off = [] { const char* e = getenv("BOTW_SZS_NATIVE"); return e && e[0] == '0'; }();
    if (s_off)
        f_04215778_guest(c);
    else if (s_check)
        checked(c);
    else
        stream_decomp(c);
}
