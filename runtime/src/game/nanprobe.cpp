// TEST ONLY (tower-climb freeze, 2026-11-02): BOTW_NANPROBE=1 logs collision-tree queries (f_040fe074)
// whose box or vector is non-finite or huge, with the guest call chain. Called by a hand-inserted line
// in generated/fn_041f.cpp; not a codegen hook.
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <mutex>

#include "hooks.h"

namespace {
bool on() {
    static const bool v = getenv("BOTW_NANPROBE") && atoi(getenv("BOTW_NANPROBE")) != 0;
    return v;
}
std::atomic<int> g_logs{0};
std::atomic<uint64_t> g_calls{0};

bool bad(float f) { return !std::isfinite(f) || std::fabs(f) > 1.0e5f; }
}  // namespace

// TEST ONLY: BOTW_SKIPPROBE=1 counts the per-actor job-push skip decision (vtable+220 call in f_0378bcf8):
// per target function, skips/calls, logged every ~5 s.
namespace {
struct SkipStat { uint32_t fn; uint64_t calls, skips; };
std::mutex g_skip_lock;
SkipStat g_skip[64];
int g_skip_n = 0;
uint64_t g_skip_last = 0;
bool skip_on() {
    static const bool v = getenv("BOTW_SKIPPROBE") && atoi(getenv("BOTW_SKIPPROBE")) != 0;
    return v;
}
}  // namespace

thread_local uint32_t t_skip_fn;
extern "C" void skipprobe_pre(PPCContext* c) { t_skip_fn = c->ctr; }

extern "C" void skipprobe_0378bd30(PPCContext* c) {
    if (!skip_on()) return;
    std::lock_guard<std::mutex> l(g_skip_lock);
    uint32_t fn = t_skip_fn;
    int i = 0;
    while (i < g_skip_n && g_skip[i].fn != fn) i++;
    if (i == g_skip_n) {
        if (g_skip_n == 64) return;
        g_skip[g_skip_n++] = {fn, 0, 0};
    }
    g_skip[i].calls++;
    if (c->r[3]) g_skip[i].skips++;
    uint64_t now = (uint64_t)time(nullptr);
    if (now - g_skip_last < 5) return;
    g_skip_last = now;
    char line[1024];
    int k = 0;
    for (int j = 0; j < g_skip_n && k < 900; j++)
        k += snprintf(line + k, sizeof line - k, " %08x:%llu/%llu", g_skip[j].fn, (unsigned long long)g_skip[j].skips,
                      (unsigned long long)g_skip[j].calls);
    LOG("skipprobe skips/calls:%s", line);
}

namespace {
std::mutex g_np_lock;
uint64_t g_np_win_start = 0, g_np_win_calls = 0;
float g_np_win_ext = 0;
int g_np_burst_logs = 0;
}  // namespace

extern "C" void nanprobe_040fe074(PPCContext* c) {
    if (!on()) return;
    ++g_calls;
    float v[12];
    for (int i = 0; i < 8; i++) v[i] = mem::rf32(c->r[4] + 4 * i);
    for (int i = 0; i < 4; i++) v[8 + i] = mem::rf32(c->r[5] + 4 * i);
    bool hit = false;
    for (float f : v) hit |= bad(f);
    float ext = std::max(std::max(v[4] - v[0], v[5] - v[1]), v[6] - v[2]);
    bool burst = false;
    {
        std::lock_guard<std::mutex> l(g_np_lock);
        uint64_t now = (uint64_t)time(nullptr);
        if (now - g_np_win_start >= 5) {
            if (g_np_win_start) LOG("nanprobe rate: %llu calls in %llus, max box extent %g", (unsigned long long)g_np_win_calls,
                                    (unsigned long long)(now - g_np_win_start), (double)g_np_win_ext);
            g_np_win_start = now;
            g_np_win_calls = 0;
            g_np_win_ext = 0;
        }
        g_np_win_calls++;
        if (!(ext <= g_np_win_ext)) g_np_win_ext = ext;  // NaN shows up as a NaN maximum
        // a flood: sample every 4096th call of a window past 200k calls, at most 60 lines a run
        burst = g_np_win_calls > 200000 && (g_np_win_calls & 4095) == 0 && g_np_burst_logs < 60 && ++g_np_burst_logs;
    }
    if (!burst && (!hit || g_logs.fetch_add(1) >= 40)) return;
    char chain[256];
    int k = snprintf(chain, sizeof chain, "%08x", c->lr);
    uint32_t sp = c->r[1];
    for (int d = 0; d < 10 && sp && sp < 0xF0000000u; d++) {
        uint32_t prev = mem::r32(sp);
        if (!prev || prev <= sp) break;
        k += snprintf(chain + k, sizeof chain - k, " %08x", mem::r32(prev + 4));
        sp = prev;
    }
    LOG("nanprobe: r3 %08x box %g %g %g %g | %g %g %g %g vec %g %g %g %g chain %s", c->r[3], v[0], v[1], v[2], v[3], v[4],
        v[5], v[6], v[7], v[8], v[9], v[10], v[11], chain);
}
