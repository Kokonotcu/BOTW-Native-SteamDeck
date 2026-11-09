// Float helpers vs Cemu's recompiler / interpreter semantics (values worked out from Cemu's code).
// clang++ -std=c++20 -Iruntime/include tests/cemu_fp_check.cpp -o /tmp/cemu_fp_check && /tmp/cemu_fp_check
#include <cassert>
#include <cmath>
#include <cstdio>

#include "ppc.h"
#include <xmmintrin.h>

uint8_t* g_mem = nullptr;
extern "C" const float g_dequant_scale[64] = {1.0f};
extern "C" const float g_quant_scale[64] = {1.0f};
extern "C" double ppc_fres(double) { return 0; }
extern "C" double ppc_frsqrte(double) { return 0; }

int main() {
    // fctiwz = cvttsd2si: truncation, 0x80000000 for NaN and both overflow directions, high word zero
    assert(ppc_fctiwz(-1.7) == 0xFFFFFFFFull);
    assert(ppc_fctiwz(3e9) == 0x80000000ull);
    assert(ppc_fctiwz(-3e9) == 0x80000000ull);
    assert(ppc_fctiwz(NAN) == 0x80000000ull);
    // fctiw = Cemu interpreter loop: 2.5 -> 2, 3.5 -> 3 (rounds halves down for positives), -2.5 -> -2
    assert((uint32_t)ppc_fctiw(2.5) == 2u);
    assert((uint32_t)ppc_fctiw(3.5) == 3u);
    assert((uint32_t)ppc_fctiw(-2.5) == (uint32_t)-2);
    assert((ppc_fctiw(2.5) >> 32) == 0xFFF80000u);
    // psq_st float rounds (cvtsd2ss) where stfs truncates
    const double just_below_2 = 1.99999999999;
    assert(f64_to_f32_round(just_below_2) == 0x40000000u);
    assert(f64_to_f32(just_below_2) == 0x3FFFFFFFu);
    // NaN keeps its bits on the psq_st path (raw copies)
    assert(f64_to_f32_round(f32_to_f64(0x7FA00001u)) == 0x7FA00001u);
    // integer quantize: cvttsd2si then clamp; NaN -> INT_MIN -> clamped to the low end
    assert(psq_clamp(cvtt_i32(NAN), 0, 255) == 0);
    assert(psq_clamp(cvtt_i32(300.7), 0, 255) == 255);
    assert(psq_clamp(cvtt_i32(-1.9), -128, 127) == -1);
    // guest float mode as Cemu (thread.cpp default_fp_mode): flush-to-zero on, denormals-are-zero off
    _mm_setcsr((_mm_getcsr() & ~0x0040u) | 0x8000u);
    volatile double tiny = 1e-40, small = 1e-20;
    assert(f32_to_f64(0x00000001u) == std::ldexp(1.0, -149));  // denormal loads still widen exactly
    assert(round_single(tiny) == 0.0 && !std::signbit(round_single(tiny)));
    assert(f64_to_f32_round(tiny) == 0u && f64_to_f32_round(-tiny) == 0x80000000u);  // psq_st flushes, sign kept
    assert(round_single(small * small) == 0.0);  // 1e-40 product rounds to single zero, as in Cemu
    assert(f64_to_f32(tiny) != 0u);  // stfs keeps the exact denormal (Cemu's fprConvertToSingleBits)
    puts("cemu_fp_check OK");
}
