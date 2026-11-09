// Guest CPU model shared by generated code and the runtime.
//
// One 4 GB guest address space is mapped at g_mem; guest data is big-endian. Every recompiled
// function is `void f_XXXXXXXX(PPCContext* __restrict c)` and works on the context directly:
// __restrict lets clang keep registers in host registers between calls, and guest memory goes
// through a separate base pointer so stores to it can't alias the context.
#pragma once

#include <math.h>
#include <stdint.h>
#include <string.h>
#if defined(__x86_64__)
#include <emmintrin.h>
#endif

#define PPC_INLINE static inline __attribute__((always_inline))
#define PPC_NORETURN __attribute__((noreturn))

struct alignas(64) PPCContext {
    uint32_t r[32];
    uint32_t lr, ctr;
    uint8_t cr[32];  // one byte per CR bit, IBM order: cr[4*f+0]=LT, +1 GT, +2 EQ, +3 SO
    uint8_t ca, so, ov;
    uint8_t xer_bc;  // byte count for lswx/stswx
    uint32_t fpscr;
    double f[32];    // ps0 (the scalar FPR)
    double ps1[32];  // paired-single slot 1
    uint32_t gqr[8];
    uint32_t res_addr, res_val;  // lwarx/stwcx. reservation (res_val in guest byte order)
    // runtime bookkeeping
    void* thread;     // host-side guest thread object
    uint32_t core;    // current core id (0..2)
    uint32_t pad_;
};

typedef void (*GuestFn)(PPCContext*);

extern "C" uint8_t* g_mem;  // base of the guest address space

// Generated functions open with PPC_SEPARATE(M, c): guest memory (one mapping) and the PPCContext (a host object)
// never overlap. Once c has been passed to a call, clang otherwise assumes any guest-memory store may change c's
// fields, so it stored registers back each loop iteration and reloaded them after every guest store.
#if defined(__has_builtin)
#if __has_builtin(__builtin_assume_separate_storage)
#define PPC_SEPARATE(m, c) __builtin_assume_separate_storage((m), (c))
#endif
#endif
#ifndef PPC_SEPARATE
#define PPC_SEPARATE(m, c) ((void)0)
#endif

// Compiler-only barrier at the back-edge of loops that store nothing to guest memory (codegen jump): spin-waits
// read the word from memory every iteration. No instruction is emitted.
#define PPC_MEM_BARRIER() __asm__ volatile("" ::: "memory")

// Per emulated core: set while a higher-priority thread waits for a core held by a background thread
// (prio >= 18). Generated code tests it at loop back-edges and hands the core over (thread.cpp).
extern "C" uint8_t g_core_preempt[4];
#define PREEMPT_POINT()                                                                        \
    do {                                                                                       \
        if (__builtin_expect(__atomic_load_n(&g_core_preempt[c->core & 3], __ATOMIC_RELAXED), 0)) \
            rt_preempt(c);                                                                     \
    } while (0)

extern "C" {
void rt_preempt(PPCContext* c);
void rt_call(PPCContext* c, uint32_t target);  // indirect call through a guest code pointer
PPC_NORETURN void rt_trap(PPCContext* c, uint32_t addr, const char* why);
void rt_bad_switch(PPCContext* c, uint32_t addr, uint32_t target);
uint64_t rt_timebase(void);
void rt_probe(PPCContext* c, uint32_t addr);
}

#if defined(BOTW_INLINE_RT_CALL)
// Generated code only (CMake defines it for the guest objects on Linux): each indirect call site keeps its last
// target in one 64-bit word, target | epoch << 32 | (rva / 16) << 36 (dispatch.cpp rt_call_fill). A hit skips the
// 38 MB guest-address -> function table, whose load missed cache on most indirect calls.
extern "C" {
extern char __ehdr_start[] __attribute__((visibility("hidden")));
extern uint32_t botw_ic_epoch;
void rt_call_fill(PPCContext* c, uint32_t target, uint64_t* entry);
}
#define rt_call(c_, t_)                                                                                           \
    do {                                                                                                          \
        static uint64_t ic_;                                                                                      \
        const uint32_t ict_ = (t_);                                                                               \
        const uint64_t ice_ = __atomic_load_n(&ic_, __ATOMIC_RELAXED);                                            \
        if (__builtin_expect((uint32_t)ice_ == ict_ && ((ice_ >> 32) & 15) == __atomic_load_n(&botw_ic_epoch, __ATOMIC_RELAXED), 1)) \
            ((GuestFn)(__ehdr_start + ((ice_ >> 36) << 4)))(c_);                                                  \
        else                                                                                                      \
            rt_call_fill((c_), ict_, &ic_);                                                                       \
    } while (0)
#endif

// ---- guest memory (big-endian) ---------------------------------------------------------
PPC_INLINE uint8_t ld8_(uint8_t* M, uint32_t a) { return M[a]; }
PPC_INLINE uint16_t ld16_(uint8_t* M, uint32_t a) { uint16_t v; memcpy(&v, M + a, 2); return __builtin_bswap16(v); }
PPC_INLINE uint32_t ld32_(uint8_t* M, uint32_t a) { uint32_t v; memcpy(&v, M + a, 4); return __builtin_bswap32(v); }
PPC_INLINE uint64_t ld64_(uint8_t* M, uint32_t a) { uint64_t v; memcpy(&v, M + a, 8); return __builtin_bswap64(v); }
PPC_INLINE void st8_(uint8_t* M, uint32_t a, uint8_t v) { M[a] = v; }
PPC_INLINE void st16_(uint8_t* M, uint32_t a, uint16_t v) { v = __builtin_bswap16(v); memcpy(M + a, &v, 2); }
PPC_INLINE void st32_(uint8_t* M, uint32_t a, uint32_t v) { v = __builtin_bswap32(v); memcpy(M + a, &v, 4); }
PPC_INLINE void st64_(uint8_t* M, uint32_t a, uint64_t v) { v = __builtin_bswap64(v); memcpy(M + a, &v, 8); }
// byte-reversed (i.e. little-endian) forms: lhbrx/lwbrx/sthbrx/stwbrx
PPC_INLINE uint16_t ld16r_(uint8_t* M, uint32_t a) { uint16_t v; memcpy(&v, M + a, 2); return v; }
PPC_INLINE uint32_t ld32r_(uint8_t* M, uint32_t a) { uint32_t v; memcpy(&v, M + a, 4); return v; }
PPC_INLINE void st16r_(uint8_t* M, uint32_t a, uint16_t v) { memcpy(M + a, &v, 2); }
PPC_INLINE void st32r_(uint8_t* M, uint32_t a, uint32_t v) { memcpy(M + a, &v, 4); }

#define LD8(a) ld8_(M, (uint32_t)(a))
#define LD16(a) ld16_(M, (uint32_t)(a))
#define LD32(a) ld32_(M, (uint32_t)(a))
#define LD64(a) ld64_(M, (uint32_t)(a))
#define ST8(a, v) st8_(M, (uint32_t)(a), (uint8_t)(v))
#define ST16(a, v) st16_(M, (uint32_t)(a), (uint16_t)(v))
#define ST32(a, v) st32_(M, (uint32_t)(a), (uint32_t)(v))
#define ST64(a, v) st64_(M, (uint32_t)(a), (uint64_t)(v))

// ---- integer helpers ------------------------------------------------------------------------
PPC_INLINE uint32_t rotl32(uint32_t v, uint32_t n) { n &= 31; return n ? (v << n) | (v >> (32 - n)) : v; }

PPC_INLINE void set_cr_s(PPCContext* c, int f, int32_t a, int32_t b) {
    c->cr[4 * f + 0] = a < b;
    c->cr[4 * f + 1] = a > b;
    c->cr[4 * f + 2] = a == b;
    c->cr[4 * f + 3] = c->so;
}
PPC_INLINE void set_cr_u(PPCContext* c, int f, uint32_t a, uint32_t b) {
    c->cr[4 * f + 0] = a < b;
    c->cr[4 * f + 1] = a > b;
    c->cr[4 * f + 2] = a == b;
    c->cr[4 * f + 3] = c->so;
}
#define CR0(v) set_cr_s(c, 0, (int32_t)(v), 0)
// The same for generated code that keeps CR bits in a local array `cr_` (cafe/codegen.py local_cr).
PPC_INLINE void crl_s(uint8_t* cr, const PPCContext* c, int f, int32_t a, int32_t b) {
    cr[4 * f + 0] = a < b;
    cr[4 * f + 1] = a > b;
    cr[4 * f + 2] = a == b;
    cr[4 * f + 3] = c->so;
}
PPC_INLINE void crl_u(uint8_t* cr, const PPCContext* c, int f, uint32_t a, uint32_t b) {
    cr[4 * f + 0] = a < b;
    cr[4 * f + 1] = a > b;
    cr[4 * f + 2] = a == b;
    cr[4 * f + 3] = c->so;
}
#define CRL0(v) crl_s(cr_, c, 0, (int32_t)(v), 0)

PPC_INLINE uint32_t ppc_divw(int32_t a, int32_t b) {
    if (b == 0) return a < 0 ? 0xFFFFFFFFu : 0;  // Espresso: sign-dependent garbage; Cemu matches this
    if (a == INT32_MIN && b == -1) return 0xFFFFFFFFu;
    return (uint32_t)(a / b);
}
PPC_INLINE uint32_t ppc_divwu(uint32_t a, uint32_t b) { return b ? a / b : 0; }

// overflow-recording add/sub (the 'o' forms)
PPC_INLINE uint32_t add_ov(PPCContext* c, uint32_t a, uint32_t b) {
    int32_t r;
    c->ov = __builtin_add_overflow((int32_t)a, (int32_t)b, &r);
    c->so |= c->ov;
    return (uint32_t)r;
}
PPC_INLINE uint32_t sub_ov(PPCContext* c, uint32_t a, uint32_t b) {  // a - b
    int32_t r;
    c->ov = __builtin_sub_overflow((int32_t)a, (int32_t)b, &r);
    c->so |= c->ov;
    return (uint32_t)r;
}
// a + b + carry with carry out
PPC_INLINE uint32_t adde_(PPCContext* c, uint32_t a, uint32_t b, uint32_t cin) {
    uint64_t s = (uint64_t)a + b + cin;
    c->ca = (uint8_t)(s >> 32);
    return (uint32_t)s;
}

PPC_INLINE uint32_t ppc_mask(uint32_t mb, uint32_t me) {
    uint32_t m1 = 0xFFFFFFFFu >> mb;
    uint32_t m2 = me >= 31 ? 0xFFFFFFFFu : ~(0xFFFFFFFFu >> (me + 1));
    return mb <= me ? (m1 & m2) : (m1 | m2);
}

// ---- floating point ---------------------------------------------------------------------------
PPC_INLINE uint64_t dbits(double d) { uint64_t u; memcpy(&u, &d, 8); return u; }
PPC_INLINE double bitsd(uint64_t u) { double d; memcpy(&d, &u, 8); return d; }

// Bit-exact single <-> double conversions as the FPU does them for lfs/stfs (keeps NaN payloads,
// expands denormals). Same results as Cemu's ConvertToDoubleNoFTZ/ConvertToSingleNoFTZ.
// Every finite value (denormals too, with the default MXCSR) converts exactly in hardware; only inf/NaN
// take the bit path, because cvtss2sd would quiet a signalling NaN. lfs is in most hot loops, so this
// replaces ~15 instructions and three branches per load (checked over all 2^32 inputs).
PPC_INLINE double f32_to_f64(uint32_t v) {
    if (__builtin_expect((v & 0x7F800000u) != 0x7F800000u, 1)) {
        float f;
        memcpy(&f, &v, 4);
        return (double)f;
    }
    uint64_t x = v;
    return bitsd(((x & 0xC0000000) << 32) | (7ull << 59) | ((x & 0x3FFFFFFF) << 29));
}
PPC_INLINE uint32_t f64_to_f32(double d) {
    uint64_t x = dbits(d);
    uint32_t exp = (x >> 52) & 0x7FF;
    // Only exponents 874..896 (results in the single denormal range) need the shift; one range test
    // instead of the old "exp > 896 || x == +-0" pair, same results.
    if (__builtin_expect(exp - 874u <= 22u, 0)) {
        uint32_t t = (uint32_t)(0x80000000 | ((x & 0x000FFFFFFFFFFFFFULL) >> 21));
        t = t >> (905 - exp);
        t |= (uint32_t)((x >> 32) & 0x80000000);
        return t;
    }
    return (uint32_t)(((x >> 32) & 0xC0000000) | ((x >> 29) & 0x3FFFFFFF));
}
PPC_INLINE double round_single(double d) { return (double)(float)d; }
// Paired-single multiplies round frC to 25 bits of mantissa first (Gekko/Espresso quirk).
PPC_INLINE double round25(double d) {
    uint64_t v = dbits(d);
    v = (v & 0xFFFFFFFFF8000000ULL) + (v & 0x8000000ULL);
    return bitsd(v);
}
PPC_INLINE double ftz_single(double d) {
    float f = (float)d;
    uint32_t u;
    memcpy(&u, &f, 4);
    if ((u & 0x7F800000u) == 0) u &= 0x80000000u;  // subnormal -> signed zero (zero stays zero)
    memcpy(&f, &u, 4);
    return f;
}

extern "C" double ppc_fres(double x);
extern "C" double ppc_frsqrte(double x);

// Branch-free: with a NaN all three ordered compares are false and only `un` is set, otherwise exactly
// one of lt/gt/eq is (the old if/else chain mispredicted on data-dependent compares).
PPC_INLINE void set_cr_f(PPCContext* c, int f, double a, double b) {
    uint8_t lt = a < b, gt = a > b, eq = a == b, un = __builtin_isunordered(a, b);
    c->cr[4 * f + 0] = lt;
    c->cr[4 * f + 1] = gt;
    c->cr[4 * f + 2] = eq;
    c->cr[4 * f + 3] = un;
    c->fpscr = (c->fpscr & 0xFFFF0FFFu) | ((uint32_t)(lt << 3 | gt << 2 | eq << 1 | un) << 12);
}
PPC_INLINE void crl_f(uint8_t* cr, PPCContext* c, int f, double a, double b) {
    uint8_t lt = a < b, gt = a > b, eq = a == b, un = __builtin_isunordered(a, b);
    cr[4 * f + 0] = lt;
    cr[4 * f + 1] = gt;
    cr[4 * f + 2] = eq;
    cr[4 * f + 3] = un;
    c->fpscr = (c->fpscr & 0xFFFF0FFFu) | ((uint32_t)(lt << 3 | gt << 2 | eq << 1 | un) << 12);
}

// x86 cvttsd2si: truncation; NaN and out-of-range give 0x80000000 (Cemu's recompiler relies on this)
PPC_INLINE int32_t cvtt_i32(double b) {
#if defined(__x86_64__)
    return _mm_cvttsd_si32(_mm_set_sd(b));
#else
    return (b != b || b >= 2147483648.0 || b < -2147483648.0) ? (int32_t)0x80000000 : (int32_t)b;
#endif
}
// fctiwz as Cemu's recompiler (OP_FPR_FCTIWZ): cvttsd2si into the low word, high word zero
PPC_INLINE uint64_t ppc_fctiwz(double b) { return (uint64_t)(uint32_t)cvtt_i32(b); }
// fctiw is not taken by Cemu's recompiler: its interpreter (PPCInterpreter_FCTIW), rounding loop included
PPC_INLINE uint64_t ppc_fctiw(double b) {
    uint64_t v;
    if (b > (double)0x7FFFFFFF) v = 0x7FFFFFFF;
    else if (b < -(double)0x80000000u) v = 0x80000000u;
    else {
        double t = b + 0.5;
        int32_t i = cvtt_i32(t);
        if (t - i < 0 || (t - i == 0 && b > 0)) i--;
        v = (uint64_t)(uint32_t)i;
    }
    uint64_t r = 0xFFF8000000000000ULL | v;
    if (v == 0 && (dbits(b) >> 63)) r |= 0x100000000ULL;
    return r;
}

// ---- paired-single quantized loads/stores -------------------------------------------------------
extern "C" const float g_dequant_scale[64];
extern "C" const float g_quant_scale[64];

// As Cemu's recompiler (EmitPSQLoadCase / EmitPSQStoreCase): integers are cvtsi2sd'd and scaled in double,
// float stores round (cvtsd2ss), integer stores scale in double, truncate with cvttsd2si and clamp.
PPC_INLINE double psq_ld1(uint8_t* M, uint32_t ea, uint32_t type, uint32_t scale) {
    switch (type) {
        case 4: return (double)LD8(ea) * (double)g_dequant_scale[scale];
        case 5: return (double)LD16(ea) * (double)g_dequant_scale[scale];
        case 6: return (double)(int8_t)LD8(ea) * (double)g_dequant_scale[scale];
        case 7: return (double)(int16_t)LD16(ea) * (double)g_dequant_scale[scale];
        default: return f32_to_f64(LD32(ea));
    }
}
// psq_st float stores: cvtsd2ss rounding; inf/NaN keep their bits (Cemu turns psq_l -> psq_st copies into raw
// word copies, and games copy plain data through paired singles)
PPC_INLINE uint32_t f64_to_f32_round(double v) {
    if (__builtin_expect(((dbits(v) >> 52) & 0x7FF) == 0x7FF, 0)) return f64_to_f32(v);
    float f = (float)v;
    uint32_t u;
    memcpy(&u, &f, 4);
    return u;
}
PPC_INLINE int32_t psq_clamp(int32_t v, int32_t lo, int32_t hi) { return v > hi ? hi : v < lo ? lo : v; }
PPC_INLINE uint32_t psq_size(uint32_t type) { return type == 4 || type == 6 ? 1 : type == 5 || type == 7 ? 2 : 4; }

PPC_INLINE void psq_load(PPCContext* c, uint8_t* M, uint32_t ea, int w, uint32_t gqr, int frd) {
    uint32_t type = (gqr >> 16) & 7, scale = (gqr >> 24) & 0x3F;
    c->f[frd] = psq_ld1(M, ea, type, scale);
    c->ps1[frd] = w ? 1.0 : psq_ld1(M, ea + psq_size(type), type, scale);
}

PPC_INLINE void psq_st1(uint8_t* M, uint32_t ea, double v, uint32_t type, uint32_t scale) {
    if (type == 0 || type > 7 || (type >= 1 && type <= 3)) {
        ST32(ea, f64_to_f32_round(v));
        return;
    }
    const int32_t i = cvtt_i32(v * (double)g_quant_scale[scale]);
    switch (type) {
        case 4: ST8(ea, (uint8_t)psq_clamp(i, 0, 255)); break;
        case 5: ST16(ea, (uint16_t)psq_clamp(i, 0, 65535)); break;
        case 6: ST8(ea, (uint8_t)(int8_t)psq_clamp(i, -128, 127)); break;
        case 7: ST16(ea, (uint16_t)(int16_t)psq_clamp(i, -32768, 32767)); break;
    }
}
PPC_INLINE void psq_store(PPCContext* c, uint8_t* M, uint32_t ea, int w, uint32_t gqr, int frs) {
    uint32_t type = gqr & 7, scale = (gqr >> 8) & 0x3F;
    psq_st1(M, ea, c->f[frs], type, scale);
    if (!w) psq_st1(M, ea + psq_size(type), c->ps1[frs], type, scale);
}
// psq_load / psq_store through GQR0, which stays 0 (type 0: plain floats, no scale): no type switch.
// w is the instruction's W bit, a constant in the generated code.
PPC_INLINE void psq_load_f32(PPCContext* c, uint8_t* M, uint32_t ea, int w, int frd) {
    c->f[frd] = f32_to_f64(LD32(ea));
    c->ps1[frd] = w ? 1.0 : f32_to_f64(LD32(ea + 4));
}
PPC_INLINE void psq_store_f32(PPCContext* c, uint8_t* M, uint32_t ea, int w, int frs) {
    ST32(ea, f64_to_f32_round(c->f[frs]));
    if (!w) ST32(ea + 4, f64_to_f32_round(c->ps1[frs]));
}

// The same with the register's two halves by reference / value (the generator may keep them in locals).
PPC_INLINE void psq_loadr(double& f, double& p, uint8_t* M, uint32_t ea, int w, uint32_t gqr) {
    uint32_t type = (gqr >> 16) & 7, scale = (gqr >> 24) & 0x3F;
    f = psq_ld1(M, ea, type, scale);
    p = w ? 1.0 : psq_ld1(M, ea + psq_size(type), type, scale);
}
PPC_INLINE void psq_storer(double f, double p, uint8_t* M, uint32_t ea, int w, uint32_t gqr) {
    uint32_t type = gqr & 7, scale = (gqr >> 8) & 0x3F;
    psq_st1(M, ea, f, type, scale);
    if (!w) psq_st1(M, ea + psq_size(type), p, type, scale);
}
PPC_INLINE void psq_load_f32r(double& f, double& p, uint8_t* M, uint32_t ea, int w) {
    f = f32_to_f64(LD32(ea));
    p = w ? 1.0 : f32_to_f64(LD32(ea + 4));
}
PPC_INLINE void psq_store_f32r(double f, double p, uint8_t* M, uint32_t ea, int w) {
    ST32(ea, f64_to_f32_round(f));
    if (!w) ST32(ea + 4, f64_to_f32_round(p));
}

// ---- atomics (lwarx/stwcx.) ------------------------------------------------------------------
PPC_INLINE uint32_t ppc_lwarx(PPCContext* c, uint8_t* M, uint32_t ea) {
    uint32_t raw = __atomic_load_n((uint32_t*)(M + ea), __ATOMIC_ACQUIRE);
    c->res_addr = ea;
    c->res_val = raw;
    return __builtin_bswap32(raw);
}
PPC_INLINE void ppc_stwcx(PPCContext* c, uint8_t* M, uint32_t ea, uint32_t v) {
    uint8_t ok = 0;
    if (c->res_addr == ea) {
        uint32_t expect = c->res_val;
        ok = __atomic_compare_exchange_n((uint32_t*)(M + ea), &expect, __builtin_bswap32(v), false,
                                         __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST);
    }
    c->res_addr = 0xFFFFFFFFu;
    c->cr[0] = 0;
    c->cr[1] = 0;
    c->cr[2] = ok;
    c->cr[3] = c->so;
}
// ppc_stwcx with CR0 in the generated code's local CR array
PPC_INLINE void ppc_stwcx_cr(uint8_t* cr, PPCContext* c, uint8_t* M, uint32_t ea, uint32_t v) {
    ppc_stwcx(c, M, ea, v);
    for (int k = 0; k < 4; k++) cr[k] = c->cr[k];
}
