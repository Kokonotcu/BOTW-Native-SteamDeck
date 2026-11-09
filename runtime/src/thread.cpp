#include "thread.h"

#include "botw_target.h"
#if !defined(_WIN32)
#include <dlfcn.h>
#include <execinfo.h>
#include <sched.h>
#include <signal.h>
#include <unistd.h>
#endif
#if defined(__linux__)
#include <sys/resource.h>
#include <sys/syscall.h>
#endif
#include <stdlib.h>
#include <string.h>
#include <time.h>
#if defined(__x86_64__)
#include <xmmintrin.h>
#endif

#include <algorithm>
#include <map>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "diag.h"
#include "hle.h"
#include "loader.h"
#include "telemetry.h"

extern "C" uint8_t g_core_preempt[4] = {};
static const bool g_preempt_on = [] {  // BOTW_PREEMPT=0: background threads keep their core until they block
    const char* e = getenv("BOTW_PREEMPT");
    return !(e && e[0] == '0');
}();

#if TARGET_OS_IOS  // iOS port: stack size vs. the VA limit without extended VA (ios/src/vm_ios.cpp)
size_t botw_ios_stack_size(size_t want);
#endif

// Core plan for 4 cores x 2 SMT threads (Steam Deck), BOTW_CORES=0 off. Telemetry showed 5 busy threads floating over
// 4 physical cores, the main thread sharing its core with another busy one more than half the time. One physical
// core per busy lane: guest core 1 (main thread) alone on cpus 0-1, guest cores 0 and 2 (one busy worker each) on
// 2-3 and 4-5 (the g_exec lock already runs one guest thread per lane), LatteThread + VkRecord on 6-7; every other
// thread floats over the worker cores' cpus 2-5. BOTW_CORES=force skips the topology check (PC tests).
// Tried and dropped (b66): keeping our threads off core 3 to lower the 8 W CPU power spikes. The spikes still hit
// 4.5 W on three cores, and late frames went 5% -> 16% (session 154015).
static bool core_plan_on() {
#if defined(__linux__)
    static const bool on = [] {
        const char* e = getenv("BOTW_CORES");
        if (e && e[0] == '0') return false;
        if (e && strcmp(e, "force") == 0) return sysconf(_SC_NPROCESSORS_ONLN) >= 8;
        if (sysconf(_SC_NPROCESSORS_ONLN) != 8) return false;
        for (int c = 0; c < 8; c++) {
            char path[80], buf[16] = {}, want[8];
            snprintf(path, sizeof path, "/sys/devices/system/cpu/cpu%d/topology/thread_siblings_list", c);
            snprintf(want, sizeof want, "%d-%d", c & ~1, c | 1);
            FILE* f = fopen(path, "r");
            if (!f) return false;
            (void)!fread(buf, 1, sizeof buf - 1, f);
            fclose(f);
            if (strncmp(buf, want, strlen(want)) != 0) return false;
        }
        return true;
    }();
    return on;
#else
    return false;
#endif
}

static void set_cpus(int first, int last) {
#if defined(__linux__)
    cpu_set_t s;
    CPU_ZERO(&s);
    for (int c = first; c <= last; c++) CPU_SET(c, &s);
    sched_setaffinity(0, sizeof s, &s);
#else
    (void)first, (void)last;
#endif
}

static thread_local int t_placed_lane = -1;  // guest core whose cpus this thread is on, -1 = not a guest lane

// Host priority of the critical threads (the guest lanes: main thread, job workers and their lane-mates, one running
// per lane at a time; LatteThread + VkRecord). Steam starts games with RLIMIT_NICE soft 0 / hard 28, so the process may
// raise its soft limit and use nice down to -8 without privileges. At nice -4 (BOTW_NICE=-4; default off) Steam's own processes
// (steamwebhelper, mangoapp, Xwayland) no longer take the cpus from these threads; at most five run at once over the
// four cores, so gamescope and the system always find a cpu. Linux core plan only (Steam Deck).
static void set_critical_nice() {
#if defined(__linux__)
    static const int nice = [] {
        // opt-in: b100 vs b99 on the Deck (3+3 runs of the repeatable phase) showed no change, 26.2% vs 26.4% late
        const char* e = getenv("BOTW_NICE");
        int n = e ? atoi(e) : 0;
        if (n >= 0) return 0;
        rlimit rl;
        if (getrlimit(RLIMIT_NICE, &rl) != 0) return 0;
        if (rl.rlim_cur < rl.rlim_max) {
            rl.rlim_cur = rl.rlim_max;
            setrlimit(RLIMIT_NICE, &rl);
            getrlimit(RLIMIT_NICE, &rl);
        }
        const int lowest = 20 - (int)std::min<rlim_t>(rl.rlim_cur, 40);  // RLIMIT_NICE r allows nice >= 20 - r
        n = std::max(n, lowest);
        return n < 0 ? n : 0;
    }();
    if (nice) setpriority(PRIO_PROCESS, (id_t)syscall(SYS_gettid), nice);
#endif
}

// Tried and dropped: a worker lane on ONE cpu (2 or 4) with the floaters on 3/5. The workers' interruptions come from
// their own lane-mates (RadarMgr, WorkerSupport, NavMesh, Sleeper), which always sit on the lane's cpus: one cpu packs
// them onto the worker (b80, session 191713: worker forced switches 774 -> 1637/s, main waits on workers 3.0 -> 4.6
// ms/frame, cpus 3/5 idle at 30%). Also moving threads off the lane cpu at every guest-core hand-off: each affinity
// change is a forced switch (b79, session 190823: workers 6000/s, late frames 27% -> 58%).
static void place_lane(int core) {
    if (core == t_placed_lane || !core_plan_on()) return;
    t_placed_lane = core;
    static const int first_cpu[3] = {2, 0, 4};
    set_cpus(first_cpu[core], first_cpu[core] + 1);
    set_critical_nice();
}

// kind 0: any thread that is not a guest lane or the GPU pair (called when a thread names itself); 1: LatteThread/VkRecord
// Tried and dropped (b77, session 183000): VkRecord on cpus 3/5 made LatteThread 22% faster, but the job workers then
// found their lane's cpus taken (worker CPU 91% -> 79%, main waits on them +1.3-1.6 ms/frame), late frames 27% -> 37%.
extern "C" void botw_place_thread(int kind) {
    if (t_placed_lane >= 0 || !core_plan_on()) return;
    if (kind == 1) {
        set_cpus(6, 7);
        set_critical_nice();
    } else {
        set_cpus(2, 5);
    }
}

// threads created before naming themselves inherit this from the process' first thread
static const bool g_core_plan_default = [] {
    botw_place_thread(0);
    return true;
}();

namespace threads {
std::mutex g_klock;
namespace {
thread_local GuestThread* t_current = nullptr;
thread_local int t_exec_depth = 0;
thread_local int t_exec_core = -1;
thread_local uint64_t t_exec_acquire_ns = 0;
thread_local std::condition_variable t_core_cv;  // a thread waits for at most one emulated core at a time

static inline uint64_t sched_now_ns() {
    return (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

// As on Cafe OS, a waiting foreground thread (prio <= 17) takes the core from a lower-priority holder (e.g.
// the prio-16 job workers from the prio-17 WorkerSupport threads sharing their cores). Background threads
// never preempt each other, and the main thread is never preempted this way.
// Background time slice (BOTW_TIMESLICE_US, default 1000, 0 off). b95 session 024856: NavMeshSystemThread (prio 20)
// held core 0 for 125-180 ms bursts while tera::ImageResourceMgr, Resource Control/Memory and ActorCreate (all 20)
// queued behind it; equal priorities never preempted each other, so streaming (and anything waiting on it) stalled.
// b97 used 4 ms; with ~4 such threads queued a job waiting on an event from one of them (no priority inheritance for
// events) still waited ~16 ms: the "worker" class of >25 ms frames, ~20 per 100 s run. Cemu interleaves equal
// priorities far finer (thread quantum of tens of thousands of guest cycles).
static const uint64_t g_slice_ns = [] {
    const char* e = getenv("BOTW_TIMESLICE_US");
    return (uint64_t)(e ? atoll(e) : 1000) * 1000ull;
}();

// a background owner (effective prio >= 18) takes part in the time slice. (b130 also sliced background threads boosted
// by priority inheritance, against NavMeshSystemThread holding core 0 while boosted; free-running background threads,
// b136, took NavMesh off core 0 instead.)
static inline bool sliceable(const GuestThread* t) { return eff_prio(t) >= 18; }

static inline bool preemptible_by(const GuestThread* holder, int waiter_prio) {
    return waiter_prio <= 17 && waiter_prio < eff_prio(holder) && !holder->is_main;
}

struct CoreExec {
    int idx = 0;  // which emulated core, for g_core_preempt
    std::mutex m;
    GuestThread* owner = nullptr;
    uint64_t next_ticket = 0;
    struct Waiter {
        GuestThread* t;
        int prio;
        uint64_t ticket;
        std::condition_variable* cv;  // the waiting thread's own (t_core_cv); unlock wakes only the next owner
    };
    std::vector<Waiter> waiters;
    std::atomic<int> total_waiters{0};
    std::atomic<int> min_waiter_prio{32};
    std::atomic<uint64_t> wake_high_until_ns{0};
    // Time slice (Cemu runs equal priorities round-robin): set when a background owner (prio >= 18) first has a
    // waiter of its own or a better priority; at g_slice_ns it hands the core over at its next PREEMPT_POINT. 0 = none.
    // The first waiter in line sleeps until the slice ends and only then raises g_core_preempt (slice_due). b97-b106
    // raised it at once, so every loop back-edge of the owner called rt_preempt for the whole slice: ~7% of
    // PlacementMgr/AutoPlacementMgr in maybe_preempt + clock reads during the hitches (all-thread perf, b106).
    std::atomic<uint64_t> slice_start{0};

    // under m: a background owner with a waiter it must share with starts (or keeps) its slice
    void arm_slice_locked() {
        if (!g_slice_ns || !owner || owner->is_main) return;
        const int op = eff_prio(owner);
        if (!sliceable(owner) || min_waiter_prio.load(std::memory_order_relaxed) > op) return;
        if (slice_start.load(std::memory_order_relaxed) == 0) slice_start.store(sched_now_ns(), std::memory_order_relaxed);
    }

    // under m: ns until the owner's slice ends for a waiter of priority p that is first in line (0 = ended: the
    // preempt flag is raised), or UINT64_MAX when no slice applies (strict priority / no owner / not first)
    uint64_t slice_due(int p, uint64_t ticket) {
        if (!g_slice_ns || !owner || owner->is_main) return UINT64_MAX;
        const int op = eff_prio(owner);
        if (!sliceable(owner) || p > op) return UINT64_MAX;
        for (const auto& w : waiters)
            if (w.prio < p || (w.prio == p && w.ticket < ticket)) return UINT64_MAX;
        uint64_t s = slice_start.load(std::memory_order_relaxed);
        const uint64_t now = sched_now_ns();
        if (s == 0) slice_start.store(s = now, std::memory_order_relaxed);
        if (now - s < g_slice_ns) return g_slice_ns - (now - s);
        __atomic_store_n(&g_core_preempt[idx], 1, __ATOMIC_RELAXED);
        return 0;
    }

    void recompute_min_locked() {
        int mp = 32;
        for (const auto& w : waiters) {
            if (w.prio < mp) mp = w.prio;
        }
        min_waiter_prio.store(mp, std::memory_order_release);
        total_waiters.store((int)waiters.size(), std::memory_order_release);
    }

    void lock(GuestThread* t) {
        std::unique_lock<std::mutex> lk(m);
        int p = t ? std::clamp(eff_prio(t), 0, 31) : 16;  // read under m: boost() raises it under m too
        if (p <= 17) {
            wake_high_until_ns.store(0, std::memory_order_relaxed);
        }
        auto can_acquire = [&](uint64_t my_ticket) -> bool {
            if (owner != nullptr) return false;
            for (const auto& w : waiters) {
                if (w.t == t) continue;
                if (w.prio < p) return false;
                if (w.prio == p && w.ticket < my_ticket) return false;
            }
            if (p >= 18) {
                uint64_t until = wake_high_until_ns.load(std::memory_order_relaxed);
                if (until != 0 && sched_now_ns() < until) return false;
            }
            return true;
        };
        if (waiters.empty() && can_acquire(0)) {
            owner = t;
            return;
        }
        uint64_t ticket = ++next_ticket;
        std::condition_variable& cv = t_core_cv;
        waiters.push_back({t, p, ticket, &cv});
        recompute_min_locked();
        // A lower-priority holder on this core gives it up at its next loop back-edge (PREEMPT_POINT).
        if (owner && preemptible_by(owner, p) && g_preempt_on) __atomic_store_n(&g_core_preempt[idx], 1, __ATOMIC_RELAXED);
        arm_slice_locked();  // an equal-priority background holder: after its slice
        // telemetry: who waited for this core, behind whom, and how long (slow path only; recorded after unlock)
        const uint64_t wait_t0 = telem::on() ? sched_now_ns() : 0;
        GuestThread* const holder = owner;
        for (;;) {
            for (const auto& w : waiters)  // boost() may have raised this waiter's entry
                if (w.ticket == ticket) p = w.prio;
            if (can_acquire(ticket)) break;
            uint64_t until = (p >= 18) ? wake_high_until_ns.load(std::memory_order_relaxed) : 0;
            if (owner == nullptr && until != 0) {
                uint64_t cur = sched_now_ns();
                if (cur < until) {
                    uint64_t rem = std::min<uint64_t>(until - cur, 150000ull);
                    cv.wait_for(lk, std::chrono::nanoseconds(rem));
                    continue;
                }
            }
            // first in line behind an equal-priority background owner: wake when its slice ends and ask it to
            // yield (slice_due raises the flag then); its unlock hands the core to us
            const uint64_t due = slice_due(p, ticket);
            if (due != 0 && due != UINT64_MAX) {
                cv.wait_for(lk, std::chrono::nanoseconds(due));
                continue;
            }
            cv.wait(lk);
        }
        for (size_t i = 0; i < waiters.size(); i++) {
            if (waiters[i].ticket == ticket) {
                waiters.erase(waiters.begin() + i);
                break;
            }
        }
        recompute_min_locked();
        owner = t;
        arm_slice_locked();  // others still queued at our priority: our slice starts now
        lk.unlock();
        if (wait_t0) telem::note_exec_wait(t, holder, sched_now_ns() - wait_t0, t && t->is_main);
    }

    // priority inheritance: t (queued here or not) now runs at p; a lower-priority owner yields at its next
    // PREEMPT_POINT, and a free core goes to t if it is now first
    void boost(GuestThread* t, int p) {
        std::lock_guard<std::mutex> lk(m);
        for (auto& w : waiters) {
            if (w.t != t || w.prio <= p) continue;
            w.prio = p;
            recompute_min_locked();
            if (owner && preemptible_by(owner, p) && g_preempt_on) __atomic_store_n(&g_core_preempt[idx], 1, __ATOMIC_RELAXED);
            if (!owner) w.cv->notify_one();
        }
    }

    void unlock() {
        std::lock_guard<std::mutex> lk(m);
        owner = nullptr;
        slice_start.store(0, std::memory_order_relaxed);
        __atomic_store_n(&g_core_preempt[idx], 0, __ATOMIC_RELAXED);
        // Only the lowest (prio, ticket) waiter can pass can_acquire, so wake just it: notify_all woke every queued
        // lane-mate on each worker hand-off, each to recheck and sleep again on the workers' cpus. Notified under m:
        // the waiter is still in the list, so its thread (and its thread_local cv) is alive.
        const Waiter* best = nullptr;
        for (const auto& w : waiters) {
            if (!best || w.prio < best->prio || (w.prio == best->prio && w.ticket < best->ticket)) best = &w;
        }
        if (best) best->cv->notify_one();
    }
};

CoreExec g_exec[3];
const bool g_exec_indexed = [] {
    for (int i = 0; i < 3; i++) g_exec[i].idx = i;
    return true;
}();
std::unordered_map<uint32_t, GuestThread*> g_threads;  // OSThread* -> thread
uint16_t g_next_id = 1;
constexpr size_t kHostStack = 64u << 20;  // guest recursion maps onto the host stack

int core_index(int core) { return core >= 0 && core <= 2 ? core : 1; }

// Cemu's ConfigureHostFloatingPointEnvironment (coreinit_Thread.cpp): guest threads run with flush-to-zero on,
// so denormal results (products, quotients, single rounding, psq_st) become signed zero exactly as in Cemu.
// Denormals-are-zero stays off: f32_to_f64 widens denormal inputs exactly (FTZ never touches a widening).
void default_fp_mode() {
#if defined(__x86_64__)
    _mm_setcsr((_mm_getcsr() & ~0x0040u) | 0x8000u);
#elif defined(__aarch64__) && defined(__clang__)
    // FZ=1, DN=0: flush to zero, keep NaN signs and payloads
    __builtin_arm_wsr64("fpcr", (__builtin_arm_rsr64("fpcr") | (1ull << 24)) & ~(1ull << 25));
#endif
}

int pick_core(uint32_t aff, int32_t prio, int fallback_core) {
    aff &= 7u;
    if (aff == 1) return 0;
    if (aff == 2) return 1;
    if (aff == 4) return 2;
    if (aff == 5) return 0;  // Core 0 | Core 2: never place on Core 1
    if (aff == 3) return (prio >= 18) ? 0 : core_index(fallback_core == 2 ? 0 : fallback_core);
    if (aff == 6) return (prio >= 18) ? 2 : core_index(fallback_core == 0 ? 2 : fallback_core);
    // aff == 7 or 0 (any core): keep background threads off Core 1
    if (prio >= 18) return (fallback_core == 2) ? 2 : 0;
    return core_index(fallback_core);
}

void apply_host_priority(GuestThread* t) {
    if (!t) return;
#if defined(_WIN32)
    HANDLE h = (t == t_current) ? GetCurrentThread() : (HANDLE)t->win_handle;
    if (!h) return;
    int win_prio = THREAD_PRIORITY_NORMAL;
    if (t->is_main || t->name == "AX" || t->name == "alarms" || t->name == "GX2 callbacks") {
        win_prio = THREAD_PRIORITY_HIGHEST;
    } else if (t->prio <= 17) {
        win_prio = THREAD_PRIORITY_ABOVE_NORMAL;
    } else if (t->prio >= 19) {
        win_prio = THREAD_PRIORITY_BELOW_NORMAL;
    }
    SetThreadPriority(h, win_prio);
#endif
}

void run_thread(GuestThread* t) {
    t_current = t;
    t->host_tid = pthread_self();
    default_fp_mode();
#if defined(_WIN32)
    HANDLE dup = nullptr;
    if (DuplicateHandle(GetCurrentProcess(), GetCurrentThread(), GetCurrentProcess(), &dup, 0, FALSE,
                        DUPLICATE_SAME_ACCESS)) {
        std::lock_guard<std::mutex> l(g_klock);
        t->win_handle = dup;
    }
    apply_host_priority(t);
#endif
    // The game's main thread and the light timing threads (audio, alarms, GX2 callbacks) get the highest QoS so
    // the scheduler keeps them on performance cores (a phone has two); worker threads one step below.
    const bool critical = t->is_main || t->name == "AX" || t->name == "alarms" || t->name == "GX2 callbacks";
#if defined(__APPLE__)
    pthread_set_qos_class_self_np(critical ? QOS_CLASS_USER_INTERACTIVE : QOS_CLASS_USER_INITIATED, 0);
#else
    (void)critical;
#endif
    botw_set_thread_name(t->name.empty() ? "guest" : t->name.c_str());
    int core = core_index(t->core);
    if (t->free_run) {
        // no emulated core: t_exec_core -1 makes exec_yield / exec_resume / maybe_preempt no-ops for this thread
#if defined(__linux__)
        // cpus 2-5 (the worker cores' second threads); 6-7 and 0-5 were tried (b137) with no clear gain
        if (core_plan_on()) set_cpus(2, 5);
        setpriority(PRIO_PROCESS, (id_t)syscall(SYS_gettid), 5);
#endif
        t_exec_core = -1;
        t_exec_depth = 0;
    } else {
        place_lane(core);
        g_exec[core].lock(t);
        t_exec_core = core;
        t_exec_depth = 1;
    }
    t_exec_acquire_ns = sched_now_ns();
    LOG("thread %s running (entry %08x core %d prio %d)", t->name.empty() ? "guest" : t->name.c_str(), t->entry, t->core,
        t->os ? (int)mem::r32(t->os + T_BASEPRIO) : -1);
    PPCContext* c = &t->ctx;
    c->lr = 0;
    rt_call(c, t->entry);
    uint32_t v = c->r[3];
    bool sound = false;
    {
        std::lock_guard<std::mutex> l(g_klock);
        t->exited = true;
#if defined(_WIN32)
        if (t->win_handle) {
            CloseHandle((HANDLE)t->win_handle);
            t->win_handle = nullptr;
        }
#endif
        t->exit_value = v;
        mem::w32(t->os + T_EXITVAL, v);
        mem::w8(t->os + T_STATE, 8);  // moribund
        t->cv.notify_all();
        LOG("thread %s exited (%08x)", t->name.c_str(), v);
        sound = t->name == "SoundCalcThread";
    }
    if (t_exec_depth > 0) {
        int held = t_exec_core;
        t_exec_depth = 0;
        t_exec_core = -1;
        g_exec[held].unlock();
    }
    // After the core is released, so the dump does not sit in the scheduler lock.
    // The guest function has already returned; this cannot change what it wrote.
    if (sound) diag_dump("SoundCalcThread exit");
}

void launch(GuestThread* t) {
    t->started = true;
    pthread_attr_t attr;
    pthread_attr_init(&attr);
#if TARGET_OS_IOS
    pthread_attr_setstacksize(&attr, botw_ios_stack_size(kHostStack));
#else
    pthread_attr_setstacksize(&attr, kHostStack);
#endif
    pthread_t th;
    pthread_create(&th, &attr, [](void* p) -> void* { run_thread((GuestThread*)p); return nullptr; }, t);
    pthread_attr_destroy(&attr);
    pthread_detach(th);
}
}  // namespace

GuestThread* current() { return t_current; }
const char* current_name() { return t_current ? t_current->name.c_str() : nullptr; }


static constexpr bool kEnablePerfDiag = false;

struct SignalInfo {
    uint32_t obj = 0;
    const char* thread_name = "?";
    int core = -1;
    uint32_t lr = 0;
};
static SignalInfo g_sig_table[256];

// Last signal time per sync object (hashed), for the main thread's wake-up latency (telemetry).
struct SignalStamp {
    std::atomic<uint32_t> obj{0};
    std::atomic<uint64_t> ns{0};
};
static SignalStamp g_sig_stamp[256];

void note_signal(uint32_t obj) {
    if (!kEnablePerfDiag && !telem::on()) return;  // telemetry only: no clock read per signal while it is off
    SignalStamp& st = g_sig_stamp[(obj >> 2) & 255];
    st.ns.store(sched_now_ns(), std::memory_order_relaxed);
    st.obj.store(obj, std::memory_order_relaxed);
    if constexpr (!kEnablePerfDiag) return;
    GuestThread* me = t_current;
    auto& s = g_sig_table[(obj >> 2) & 255];
    s.obj = obj;
    s.thread_name = (me && !me->name.empty()) ? me->name.c_str() : "host";
    s.core = me ? me->core : -1;
    s.lr = me ? me->ctx.lr : 0;
}

struct FrameWaitEvent {
    const char* what;
    uint32_t obj;
    uint64_t ns;
    uint32_t lr;
    const char* sig_name;
    int sig_core;
    uint32_t sig_lr;
};
static std::mutex g_diag_frame_mu;
static std::vector<FrameWaitEvent> g_cur_frame_waits;
static uint64_t g_cur_frame_wait_ns = 0;
static uint64_t g_cur_frame_exec1_wait_ns = 0;
static const char* g_last_core1_other = "none";
static uint32_t g_last_core1_other_lr = 0;

void diag_on_frame(double frame_ms, size_t frame_idx) {
    if constexpr (!kEnablePerfDiag) return;
    std::lock_guard<std::mutex> l(g_diag_frame_mu);
    if (frame_ms >= 13.5 && frame_idx > 0) {
        char buf[1536];
        int k = snprintf(buf, sizeof(buf),
            "SLOW_FRAME[%llu]: total=%.2f ms, waits=%.2f ms, exec1_lock=%.2f ms (holder=%s lr=%08x), cpu=%.2f ms |",
            (unsigned long long)frame_idx, frame_ms, g_cur_frame_wait_ns / 1e6, g_cur_frame_exec1_wait_ns / 1e6,
            g_last_core1_other, g_last_core1_other_lr,
            frame_ms - (g_cur_frame_wait_ns + g_cur_frame_exec1_wait_ns) / 1e6);
        for (auto& e : g_cur_frame_waits) {
            if (e.ns >= 200000 && k < (int)sizeof(buf) - 140) {
                k += snprintf(buf + k, sizeof(buf) - k,
                    " [%s obj=%08x %.2fms lr=%08x <- %s(c%d,lr=%08x)]",
                    e.what, e.obj, e.ns / 1e6, e.lr, e.sig_name, e.sig_core, e.sig_lr);
            }
        }
        LOG("%s", buf);
    }
    g_cur_frame_waits.clear();
    g_cur_frame_wait_ns = 0;
    g_cur_frame_exec1_wait_ns = 0;
}

void diag_dump_threads() {
    if constexpr (!kEnablePerfDiag) return;
    std::lock_guard<std::mutex> l(g_klock);
    LOG("=== GUEST THREAD TABLE (%llu threads) ===", (unsigned long long)g_threads.size());
    for (auto& [os, t] : g_threads) {
        if (!t) continue;
        uint32_t aff = os ? mem::r32(os + T_AFFINITY) : 0;
        uint8_t attr = os ? mem::r8(os + T_ATTR) : 0;
        int32_t prio = os ? (int32_t)mem::r32(os + T_BASEPRIO) : 0;
        LOG("  thr os=%08x name='%s' core=%d aff=%u attr=0x%02x prio=%d entry=%08x lr=%08x started=%d exited=%d",
            os, t->name.c_str(), t->core, aff, attr, prio, t->entry, t->ctx.lr, (int)t->started, (int)t->exited);
    }
}

void reserve_high_wake(int core) {
    auto& cs = g_exec[core_index(core)];
    cs.wake_high_until_ns.store(sched_now_ns() + 1000000ull, std::memory_order_release);
}

bool should_yield_core(GuestThread* me) {
    if (t_exec_depth == 0 || t_exec_core < 0) return false;
    auto& cs = g_exec[t_exec_core];
    int p = me ? std::clamp((int)me->prio, 0, 31) : 16;
    if (cs.min_waiter_prio.load(std::memory_order_acquire) <= p) return true;
    if (p >= 18) {
        uint64_t until = cs.wake_high_until_ns.load(std::memory_order_acquire);
        if (until != 0 && sched_now_ns() < until) return true;
    }
    return false;
}

}  // namespace threads

// PREEMPT_POINT (ppc.h): a higher-priority thread waits for this core. maybe_preempt hands it over unless
// this thread holds a guest mutex (it keeps the core until the unlock, which preempts there).
extern "C" void rt_preempt(PPCContext* c) {
    (void)c;
    threads::maybe_preempt(threads::current());
}

namespace threads {
// A waiting foreground thread (prio <= 17) also takes the core from a holder inside a guest mutex, as Cafe OS does
// (OSMutex never blocks preemption; a preemptor that wants the mutex blocks and hands the core back). The old rule
// kept the core until the unlock: in b89 session 020501 the job worker waited ~1 ms each time behind
// NavMeshSystemThread (prio 20), 290 ms in the second of a 45 fps dip, and the main thread waited on that worker.
// BOTW_PREEMPT_MUTEX=0: old rule. The 'wake high' grace for prio >= 18 holders still waits for the unlock.
static const bool g_preempt_in_mutex = [] {
    const char* e = getenv("BOTW_PREEMPT_MUTEX");
    return !(e && e[0] == '0');
}();

// Priority inheritance (Cafe OS lends a mutex waiter's priority to the owner). b93 session 023555: the job worker
// (16) waited 155-180 ms on a mutex held by tera::ImageResourceMgr (20), which waited as long for core 0 while
// NavMeshSystemThread (20) ran; the main thread waited on that job (the 170-190 ms city hitches). Boosted, the
// owner preempts equal-priority background threads. BOTW_PRIO_INHERIT=0 off.
static const bool g_prio_inherit = [] {
    const char* e = getenv("BOTW_PRIO_INHERIT");
    return !(e && e[0] == '0');
}();

void inherit_prio(GuestThread* holder, int p) {
    if (!g_prio_inherit || !holder) return;
    int cur = holder->boost_prio.load(std::memory_order_relaxed);
    do {
        if (p >= cur) return;
    } while (!holder->boost_prio.compare_exchange_weak(cur, p, std::memory_order_relaxed));
    g_exec[core_index(holder->core)].boost(holder, p);
}

void maybe_preempt_current_fast() {
    GuestThread* me = t_current;
    if (__builtin_expect(!me || eff_prio(me) < 17 || (me->held_mutex_count > 0 && !g_preempt_in_mutex) || t_exec_depth == 0 || t_exec_core < 0, 1)) return;
    auto& cs = g_exec[t_exec_core];
    if (__builtin_expect(preemptible_by(me, cs.min_waiter_prio.load(std::memory_order_relaxed)) ||
                         cs.wake_high_until_ns.load(std::memory_order_relaxed) != 0 ||
                         __atomic_load_n(&g_core_preempt[t_exec_core], __ATOMIC_RELAXED), 0)) {
        maybe_preempt(me);
    }
}

void maybe_preempt(GuestThread* me) {
    if (!me || (eff_prio(me) < 17 && !sliceable(me)) || t_exec_depth == 0 || t_exec_core < 0) return;
    const bool in_mutex = me->held_mutex_count > 0;
    if (in_mutex && !g_preempt_in_mutex) return;
    auto& cs = g_exec[t_exec_core];
    bool demand = preemptible_by(me, cs.min_waiter_prio.load(std::memory_order_acquire));
    if (!demand && eff_prio(me) >= 18 && !in_mutex) {
        uint64_t until = cs.wake_high_until_ns.load(std::memory_order_acquire);
        if (until != 0 && sched_now_ns() < until) demand = true;
    }
    // time slice used up with an equal-or-better background thread queued: round-robin. Only reached once the first
    // waiter raised the flag at the slice end (slice_due), so the clock is read then, not per back-edge.
    if (!demand && sliceable(me) && __atomic_load_n(&g_core_preempt[t_exec_core], __ATOMIC_RELAXED)) {
        const uint64_t s = cs.slice_start.load(std::memory_order_relaxed);
        if (s != 0 && cs.min_waiter_prio.load(std::memory_order_relaxed) <= eff_prio(me) && sched_now_ns() - s >= g_slice_ns)
            demand = true;
    }
    if (demand) {
        telem::add(telem::Preempts, 1);
        exec_yield();
        exec_resume();
    }
}

void maybe_preempt_sync(GuestThread* me) {
    if (!me || eff_prio(me) < 18 || t_exec_depth == 0 || t_exec_core < 0) return;
    auto& cs = g_exec[t_exec_core];
    bool demand = (cs.min_waiter_prio.load(std::memory_order_acquire) <= 17);
    if (!demand) {
        uint64_t until = cs.wake_high_until_ns.load(std::memory_order_acquire);
        if (until != 0 && sched_now_ns() < until) demand = true;
    }
    if (demand) {
        exec_yield();
        exec_resume();
    }
}

bool exec_yield() {
    if (t_exec_depth == 0) return false;
    int core = t_exec_core;
    if constexpr (kEnablePerfDiag) {
        GuestThread* self = t_current;
        if (self && self->prio >= 18 && t_exec_acquire_ns != 0) {
            uint64_t held_ns = sched_now_ns() - t_exec_acquire_ns;
            if (held_ns >= 2500000ull) {
                LOG("LONG_HOLD[core=%d]: %s (prio=%d) held=%.2f ms lr=%08x",
                    core, self->name.c_str(), self->prio, held_ns / 1e6, self->ctx.lr);
            }
        }
        if (core == 1 && self && !self->is_main) {
            g_last_core1_other = self->name.c_str();
            g_last_core1_other_lr = self->ctx.lr;
        }
    }
    t_exec_depth = 0;
    g_exec[core].unlock();
    return true;
}

void exec_resume() {
    if (t_exec_depth > 0 || t_exec_core < 0) return;
    GuestThread* self = t_current;
    int core = t_exec_core;
    if (self) core = core_index(self->core);
    t_exec_core = core;
    place_lane(core);
    // an inherited priority that raced with the unlock (the owner released before inherit_prio): drop it here
    if (self && self->held_mutex_count == 0 && self->boost_prio.load(std::memory_order_relaxed) != 32)
        self->boost_prio.store(32, std::memory_order_relaxed);
    if constexpr (kEnablePerfDiag) {
        bool is_main = (self && self->is_main);
        auto t0 = is_main ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
        g_exec[core].lock(self);
        t_exec_acquire_ns = sched_now_ns();
        if (is_main) {
            uint64_t ns = (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0).count();
            std::lock_guard<std::mutex> l(g_diag_frame_mu);
            g_cur_frame_exec1_wait_ns += ns;
        } else if (core == 1 && self) {
            g_last_core1_other = self->name.c_str();
            g_last_core1_other_lr = self->ctx.lr;
        }
    } else {
        g_exec[core].lock(self);
    }
    t_exec_depth = 1;
}

void set_blocked(GuestThread* t, bool blocked) {
    if (!t || !t->os || t->exited) return;
    mem::w8(t->os + T_STATE, blocked ? 4 : 2);
}

GuestThread* get(uint32_t os) {
    auto it = g_threads.find(os);
    return it == g_threads.end() ? nullptr : it->second;
}

uint32_t sda_base() { return loader::main_module().sda_base; }

namespace {
std::mutex g_wait_stats_lock;
std::vector<std::pair<const char*, uint64_t>> g_wait_stats;  // kind -> ns
auto g_wait_stats_t0 = std::chrono::steady_clock::now();
}  // namespace
void note_main_wait(const char* kind, uint32_t obj, uint32_t lr, uint64_t ns) {  // WaitMark calls it only while telemetry is on
    if constexpr (kEnablePerfDiag) {
        std::lock_guard<std::mutex> dl(g_diag_frame_mu);
        g_cur_frame_wait_ns += ns;
        auto& sig = g_sig_table[(obj >> 2) & 255];
        g_cur_frame_waits.push_back({kind, obj, ns, lr,
            sig.obj == obj ? sig.thread_name : "?",
            sig.obj == obj ? sig.core : -1,
            sig.obj == obj ? sig.lr : 0});
    } else {
        (void)obj;
        (void)lr;
    }
    const bool on_gpu = !strncmp(kind, "gpu", 3) || !strncmp(kind, "flip", 4) || !strcmp(kind, "vsync");
    telem::add(on_gpu ? telem::MainWaitGpuNs : telem::MainWaitOtherNs, ns);
    telem::add(telem::MainWaits, 1);
    {  // woken by a guest signal during this wait: how long until the main thread ran again
        const uint64_t now = sched_now_ns();
        const SignalStamp& st = g_sig_stamp[(obj >> 2) & 255];
        const uint64_t sig = st.ns.load(std::memory_order_relaxed);
        if (st.obj.load(std::memory_order_relaxed) == obj && sig <= now && now - sig <= ns) {
            telem::add(telem::MainWakes, 1);
            telem::add(telem::MainWakeLatNs, now - sig);
        }
    }
    std::lock_guard<std::mutex> l(g_wait_stats_lock);
    for (auto& [k, v] : g_wait_stats)
        if (k == kind) {
            v += ns;
            return;
        }
    g_wait_stats.push_back({kind, ns});
}
std::string main_wait_report() {
    std::lock_guard<std::mutex> l(g_wait_stats_lock);
    auto now = std::chrono::steady_clock::now();
    double secs = std::chrono::duration<double>(now - g_wait_stats_t0).count();
    g_wait_stats_t0 = now;
    std::string s;
    char buf[96];
    for (auto& [k, v] : g_wait_stats) {
        snprintf(buf, sizeof buf, "%s%s %.0f", s.empty() ? "" : ", ", k, v / 1e6 / secs);
        s += buf;
        v = 0;
    }
    return s + " (ms/s)";
}

#if !defined(_WIN32)
namespace {
std::atomic<const char*> g_sample_label{"?"};  // name for non-guest threads (sample_host)

void log_host_frames(const char* label, void* const* frames, int n) {
    char line[4096];
    int k = snprintf(line, sizeof line, "  %s host frames (base 0x%llx):", label ? label : "?",
                     (unsigned long long)(uintptr_t)botw_image_base());
    for (int i = 0; i < n && k < (int)sizeof line - 24; i++)
        k += snprintf(line + k, sizeof line - k, " %p", frames[i]);
    LOG("%s", line);
    for (int i = 0; i < n; i++) {
        Dl_info info{};
        const char* name = nullptr;
        unsigned long long off = (unsigned long long)(uintptr_t)frames[i];
        if (dladdr(frames[i], &info) && info.dli_fbase) {
            name = info.dli_fname;
            if (name) {
                const char* slash = strrchr(name, '/');
                if (slash) name = slash + 1;
            }
            off = (unsigned long long)((uintptr_t)frames[i] - (uintptr_t)info.dli_fbase);
        }
        if (name) LOG("    %s+0x%llx", name, off);
        else LOG("    0x%llx", off);
    }
}

void on_stack_signal(int) {  // runs on the interrupted thread
#if defined(__APPLE__) || defined(__linux__)
    void* frames[64];
    int n = backtrace(frames, 64);
    // backtrace's first frames are the signal trampoline and this handler.
    int skip = n > 2 ? 2 : 0;
    const char* label = current_name() ? current_name() : g_sample_label.load();
    log_host_frames(label, frames + skip, n - skip);
#else
    LOG("  %s (host stack sample unavailable)", current_name() ? current_name() : g_sample_label.load());
#endif
}
}  // namespace

void install_stack_signal() {
    static bool installed = [] {
        struct sigaction sa = {};
        sa.sa_handler = on_stack_signal;
        sigaction(SIGUSR2, &sa, nullptr);
        return true;
    }();
    (void)installed;
}

// The SIGUSR2 handler runs backtrace() and LOG on the interrupted thread. If that thread holds the
// malloc, loader or log lock, it deadlocks and the game freezes, so the slow-frame and watchdog
// samplers only interrupt threads under BOTW_DIAG=1. Windows suspends the thread and reads it from outside.
void sample_host(pthread_t th, const char* label) {
    if (!diag_on()) return;
    install_stack_signal();
    g_sample_label = label;
    pthread_kill(th, SIGUSR2);
}

void host_stack_log(const char* label, void*) {
#if defined(__APPLE__) || defined(__linux__)
    void* frames[64];
    int n = backtrace(frames, 64);
    int skip = n > 1 ? 1 : 0;
    log_host_frames(label, frames + skip, n - skip);
#else
    LOG("  %s (host stack sample unavailable)", label ? label : "?");
#endif
}
#else
namespace {
constexpr int kMaxFrames = 64;

// x64 unwind data, not the frame pointer. MinGW omits RBP, so a chain walk stops at the
// first system frame (the single KERNELBASE address in the wake-up stall).
int capture_frames(CONTEXT ctx, uint64_t* pcs, int max) {
    UNWIND_HISTORY_TABLE history{};
    int n = 0;
    for (; n < max;) {
        DWORD64 pc = ctx.Rip;
        if (!pc) break;
        pcs[n++] = pc;
        DWORD64 imageBase = 0;
        PRUNTIME_FUNCTION entry = RtlLookupFunctionEntry(pc, &imageBase, &history);
        if (!entry) {
            DWORD64 ret = 0;
            SIZE_T got = 0;
            if (!ReadProcessMemory(GetCurrentProcess(), (LPCVOID)ctx.Rsp, &ret, sizeof ret, &got) || got != sizeof ret || !ret)
                break;
            ctx.Rip = ret;
            ctx.Rsp += 8;
        } else {
            PVOID handlerData = nullptr;
            DWORD64 establisher = 0;
            RtlVirtualUnwind(UNW_FLAG_NHANDLER, imageBase, pc, entry, &ctx, &handlerData, &establisher, nullptr);
        }
        if (ctx.Rip == pc) break;
    }
    return n;
}

void format_module(uint64_t pc, char* buf, size_t n) {
    HMODULE mod = nullptr;
    if (GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                           (LPCSTR)(uintptr_t)pc, &mod) &&
        mod) {
        char path[MAX_PATH];
        if (GetModuleFileNameA(mod, path, MAX_PATH)) {
            const char* name = path;
            for (const char* p = path; *p; p++)
                if (*p == '\\' || *p == '/') name = p + 1;
            snprintf(buf, n, "%s+0x%llx", name, (unsigned long long)(pc - (uint64_t)(uintptr_t)mod));
            return;
        }
    }
    snprintf(buf, n, "0x%llx", (unsigned long long)pc);
}

void log_pcs(const char* label, const uint64_t* pcs, int n) {
    char line[4096];
    int k = snprintf(line, sizeof line, "  %s host frames (base 0x%llx):", label ? label : "?",
                     (unsigned long long)(uintptr_t)botw_image_base());
    for (int i = 0; i < n && k < (int)sizeof line - 24; i++)
        k += snprintf(line + k, sizeof line - k, " 0x%llx", (unsigned long long)pcs[i]);
    LOG("%s", line);
    for (int i = 0; i < n; i++) {
        char mod[192];
        format_module(pcs[i], mod, sizeof mod);
        LOG("    %s", mod);
    }
}
}  // namespace

void install_stack_signal() {}

void host_stack_log(const char* label, void* host_context) {
    CONTEXT ctx{};
    if (host_context) ctx = *(CONTEXT*)host_context;
    else RtlCaptureContext(&ctx);
    uint64_t pcs[kMaxFrames];
    int n = capture_frames(ctx, pcs, kMaxFrames);
    log_pcs(label, pcs, n);
}

void sample_host(pthread_t th, const char* label) {
    // winpthreads: the pthread_t is not a HANDLE. pthread_gethandle returns the OS thread.
    HANDLE h = (HANDLE)pthread_gethandle(th);
    if (!h) {
        LOG("  %s (no host thread handle)", label ? label : "?");
        return;
    }
    if (GetThreadId(h) == GetCurrentThreadId()) {
        host_stack_log(label, nullptr);
        return;
    }
    if (SuspendThread(h) == (DWORD)-1) {
        LOG("  %s (SuspendThread failed %lu)", label ? label : "?", GetLastError());
        return;
    }
    CONTEXT ctx{};
    ctx.ContextFlags = CONTEXT_FULL;
    BOOL ok = GetThreadContext(h, &ctx);
    uint64_t pcs[kMaxFrames];
    int n = 0;
    if (ok) n = capture_frames(ctx, pcs, kMaxFrames);
    ResumeThread(h);
    if (!ok) {
        LOG("  %s (GetThreadContext failed %lu)", label ? label : "?", GetLastError());
        return;
    }
    log_pcs(label, pcs, n);
}
#endif

void dump(bool stacks) {
    std::vector<std::pair<pthread_t, std::string>> running;
    {
        std::lock_guard<std::mutex> l(g_klock);
        int suspended = 0;
        for (auto& [os, t] : g_threads) {
            if (t->exited) continue;
            if (!t->started) {
                suspended++;
                continue;
            }
            const char* k = t->wait_kind;
            if (k) LOG("thread %-32s %08x entry %08x core %d: %s %08x (lr %08x)", t->name.c_str(), os, t->entry, t->core, k,
                       t->wait_obj, t->wait_lr);
            else {
                LOG("thread %-32s %08x entry %08x core %d: running (lr %08x)", t->name.c_str(), os, t->entry, t->core, t->ctx.lr);
                if (t->host_tid) running.emplace_back(t->host_tid, t->name);
            }
        }
        LOG("%d threads created but never started", suspended);
    }
    if (!stacks) return;
#if defined(_WIN32)
    for (auto& [th, name] : running) sample_host(th, name.c_str());
#else
    if (!diag_on()) return;  // see sample_host
    install_stack_signal();
    for (auto& [th, name] : running) {
        (void)name;
        pthread_kill(th, SIGUSR2);
        usleep(50000);
    }
#endif
}

std::vector<std::pair<pthread_t, std::string>> snapshot_live() {
    std::vector<std::pair<pthread_t, std::string>> out;
    std::lock_guard<std::mutex> l(g_klock);
    for (auto& [os, t] : g_threads)
        if (t->started && !t->exited && t->host_tid) out.emplace_back(t->host_tid, t->name);
    return out;
}

std::vector<std::pair<void*, std::string>> snapshot_live_handles() {
    std::vector<std::pair<void*, std::string>> out;
    std::lock_guard<std::mutex> l(g_klock);
    for (auto& [os, t] : g_threads)
        if (t->started && !t->exited && t->win_handle) out.emplace_back(t->win_handle, t->name);
    return out;
}

void init_context(PPCContext* c, uint32_t stack_top, int core) {
    memset(c, 0, sizeof *c);
    c->r[1] = (stack_top - 0x20) & ~0xFu;
    mem::w32(c->r[1], 0);  // back chain terminator
    c->r[2] = loader::main_module().sda2_base;
    c->r[13] = loader::main_module().sda_base;
    c->gqr[2] = 0x00040004;
    c->gqr[3] = 0x00050005;
    c->gqr[4] = 0x00060006;
    c->gqr[5] = 0x00070007;
    c->res_addr = 0xFFFFFFFFu;
    c->core = core;
}

static uint32_t new_os_thread(const char* name, uint32_t entry, uint32_t stack_top, uint32_t stack_size, int prio,
                              int core) {
    uint32_t os = mem::sys_alloc(kOSThreadSize, 8);
    mem::w32(os + T_MAGIC, 0x74487244);
    mem::w8(os + T_STATE, 2);
    mem::w16(os + T_ID, g_next_id++);
    mem::w32(os + T_EFFPRIO, prio);
    mem::w32(os + T_BASEPRIO, prio);
    mem::w32(os + T_STACKBASE, stack_top);
    mem::w32(os + T_STACKEND, stack_top - stack_size);
    mem::w32(os + T_ENTRY, entry);
    mem::w32(os + T_AFFINITY, 1u << core);
    mem::w32(os + T_NAME, mem::sys_strdup(name));
    mem::w32(os + 0x1DC, core);  // context.upir
    return os;
}

GuestThread* adopt_host_thread(const char* name, uint32_t stack_size, int core) {
    GuestThread* t = new GuestThread;
    uint32_t stack = mem::sys_alloc(stack_size, 16);
    t->name = name;
    t->core = core;
    t->prio = 0;
    t->started = true;
    t->suspend = 0;
    std::lock_guard<std::mutex> l(g_klock);
    t->os = new_os_thread(name, 0, stack + stack_size, stack_size, 0, core);
    init_context(&t->ctx, stack + stack_size, core);
    t->ctx.thread = t;
    g_threads[t->os] = t;
    t_current = t;
    apply_host_priority(t);
    default_fp_mode();
    return t;
}

void start_main(uint32_t entry, uint32_t argc, uint32_t argv, uint32_t stack_size) {
    GuestThread* t = new GuestThread;
    uint32_t stack = mem::sys_alloc(stack_size, 16);
    t->name = "main";
    t->is_main = true;
    t->core = 1;
    t->entry = entry;
    t->suspend = 0;
    {
        std::lock_guard<std::mutex> l(g_klock);
        t->os = new_os_thread("Default Thread 1", entry, stack + stack_size, stack_size, 16, 1);
        init_context(&t->ctx, stack + stack_size, 1);
        t->ctx.thread = t;
        t->ctx.r[3] = argc;
        t->ctx.r[4] = argv;
        g_threads[t->os] = t;
    }
    launch(t);
}
}  // namespace threads

extern "C" void botw_assert_fail(const char* file, int line, const char* expr) {
    static thread_local int depth = 0;
    if (depth) return;
    depth++;
    const char* show = file ? file : "?";
    if (file) {
        const char* p = strstr(file, "runtime/");
        if (!p) p = strstr(file, "runtime\\");
        if (p) show = p;
    }
    static std::mutex seen_mu;
    struct Site {
        const char* file;
        int line;
        int count;
    };
    static Site seen[48];
    static int nseen = 0;
    bool first = false;
    {
        std::lock_guard<std::mutex> lock(seen_mu);
        int slot = -1;
        for (int i = 0; i < nseen; i++)
            if (seen[i].line == line && seen[i].file == file) {
                slot = i;
                break;
            }
        if (slot >= 0) {
            int count = ++seen[slot].count;
            depth--;
            if (count == 2 || count == 100 || count == 10000)
                LOG("assert: %s:%d repeated (%d)", show, line, count);
            return;
        }
        if (nseen >= 48) {
            depth--;
            LOG("assert: %s:%d: %s", show, line, expr ? expr : "?");
            return;
        }
        seen[nseen++] = {file, line, 1};
        first = true;
    }
    LOG("assert: %s:%d: %s", show, line, expr ? expr : "?");
    threads::host_stack_log("assert", nullptr);
    if (first) LOG("assert: continuing");
    depth--;
}

using namespace threads;

// ---- coreinit thread API ---------------------------------------------------------------------
// BOOL OSCreateThread(OSThread*, entry, argc, argv, stackTop, stackSize, priority, attr)
HLE_EXPORT(coreinit, OSCreateThread) {
    uint32_t os = ARG(0), entry = ARG(1), argc = ARG(2), argv = ARG(3), stack_top = ARG(4), stack_size = ARG(5);
    int32_t prio = (int32_t)ARG(6);
    uint32_t attr = ARG(7);
    GuestThread* me = current();
    int fallback_core = me ? me->core : 1;
    uint32_t aff = attr & 7;
    int core = pick_core(aff, prio, fallback_core);
    GuestThread* t = new GuestThread;
    t->os = os;
    t->entry = entry;
    t->core = core;
    t->prio = prio;
    // BOTW_FREE_BG=0 turns it off. Background threads that may run on any core (affinity 0/7, prio >= 18:
    // RadarMgr, PlacementMgr, AutoPlacementMgr, NavMeshSystemThread, resource / actor-creation threads) run without
    // the emulated-core lock, in parallel with the guest cores, at host nice +5 on cpus 2-5 (the job workers win any
    // contest); OSGetCoreId still reports their nominal core. b131 long Deck run vs b129: frames > 20 ms 3.70 -> 2.17%,
    // presents > 20 ms while walking 99 -> 73/min, CPU power unchanged. (The b132 PC soak crash first blamed on it was
    // the lavapipe done-marker race fixed in b134; b135 PC soaks with it on, seeds 25/26: FATAL 0.)
    static const bool free_bg = !(getenv("BOTW_FREE_BG") && getenv("BOTW_FREE_BG")[0] == '0');
    t->free_run = free_bg && (aff == 7 || aff == 0) && prio >= 18;
    t->name = "thread";
    {
        std::lock_guard<std::mutex> l(g_klock);
        mem::commit(os, kOSThreadSize, true);
        memset(mem::host(os), 0, kOSThreadSize);
        mem::w32(os + T_MAGIC, 0x74487244);
        mem::w8(os + T_STATE, 1);
        mem::w8(os + T_ATTR, (uint8_t)attr);
        mem::w16(os + T_ID, g_next_id++);
        mem::w32(os + T_SUSPEND, 1);
        mem::w32(os + T_EFFPRIO, prio);
        mem::w32(os + T_BASEPRIO, prio);
        mem::w32(os + T_STACKBASE, stack_top);
        mem::w32(os + T_STACKEND, stack_top - stack_size);
        mem::w32(os + T_ENTRY, entry);
        mem::w32(os + T_AFFINITY, aff ? aff : 7);
        mem::w32(os + 0x1DC, core);
        init_context(&t->ctx, stack_top, core);
        t->ctx.thread = t;
        t->ctx.pad_ = (prio >= 18) ? 1u : 0u;
        t->ctx.r[3] = argc;
        t->ctx.r[4] = argv;
        if (GuestThread* old = get(os)) {
            if (!old->exited) LOG("warning: OSCreateThread reuses a live OSThread %08x", os);
        }
        g_threads[os] = t;
    }
    TRACE("OSCreateThread %08x entry %08x stack %08x+%x prio %d attr %x core %d", os, entry, stack_top, stack_size,
          prio, attr, core);
    RET(1);
}

HLE_EXPORT(coreinit, OSResumeThread) {
    uint32_t os = ARG(0);
    std::lock_guard<std::mutex> l(g_klock);
    GuestThread* t = get(os);
    if (!t) {
        LOG("OSResumeThread: unknown thread %08x", os);
        RET(0);
        return;
    }
    int prev = t->suspend;
    if (t->suspend > 0) t->suspend--;
    mem::w32(os + T_SUSPEND, t->suspend);
    if (t->suspend == 0 && !t->started) {
        mem::w8(os + T_STATE, 2);
        // The creator keeps this core. The new thread blocks in run_thread until we
        // sleep, yield, or wait, so stores that follow OSResumeThread are visible
        // before it runs. Switching here races that setup.
        launch(t);
    }
    RET(prev);
}

HLE_EXPORT(coreinit, OSJoinThread) {
    uint32_t os = ARG(0), out = ARG(1);
    threads::CoreHandoff handoff;
    std::unique_lock<std::mutex> l(g_klock);
    GuestThread* t = get(os);
    if (!t) {
        RET(0);
        return;
    }
    if (!t->exited) {
        l.unlock();
        handoff.yield();
        l.lock();
        if (!t->exited) {
            threads::WaitMark w("join", os);
            t->cv.wait(l, [&] { return t->exited; });
        }
    }
    if (out) mem::w32(out, t->exit_value);
    RET(1);
}

HLE_EXPORT(coreinit, OSGetCurrentThread) {
    GuestThread* t = c && c->thread ? (GuestThread*)c->thread : current();
    RET(t ? t->os : 0);
}

HLE_EXPORT(coreinit, OSSetThreadName) {
    uint32_t os = ARG(0), name = ARG(1);
    mem::w32(os + T_NAME, name);
    std::lock_guard<std::mutex> l(g_klock);
    if (GuestThread* t = get(os)) {
        t->name = name ? mem::str(name) : "";
        if (t == current()) botw_set_thread_name(t->name.c_str());
    }
}

HLE_EXPORT(coreinit, OSSetThreadPriority) {
    uint32_t os = ARG(0);
    int32_t new_prio = (int32_t)ARG(1);
    mem::w32(os + T_BASEPRIO, new_prio);
    mem::w32(os + T_EFFPRIO, new_prio);
    {
        std::lock_guard<std::mutex> l(g_klock);
        if (GuestThread* t = get(os)) {
            t->prio = new_prio;
            t->ctx.pad_ = (new_prio >= 18 && !t->is_main) ? 1u : 0u;
            apply_host_priority(t);
        }
    }
    RET(1);
}
HLE_EXPORT(coreinit, OSGetThreadPriority) { RET(mem::r32(ARG(0) + T_BASEPRIO)); }

HLE_EXPORT(coreinit, OSSetThreadAffinity) {
    uint32_t os = ARG(0), aff = ARG(1) & 7;
    mem::w32(os + T_AFFINITY, aff);
    GuestThread* me = current();
    GuestThread* target = nullptr;
    int new_core = 1;
    {
        std::lock_guard<std::mutex> l(g_klock);
        target = get(os);
        if (target) {
            new_core = pick_core(aff, target->prio, target->core);
            if (target != me) {
                target->core = new_core;
                target->ctx.core = new_core;
                mem::w32(os + 0x1DC, new_core);
            }
        }
    }
    if (target && target == me && target->core != new_core) {
        bool was_holding = threads::exec_yield();
        target->core = new_core;
        target->ctx.core = new_core;
        mem::w32(os + 0x1DC, new_core);
        if (was_holding) threads::exec_resume();
    }
    RET(1);
}
HLE_EXPORT(coreinit, OSGetThreadAffinity) { RET(mem::r32(ARG(0) + T_AFFINITY)); }

HLE_EXPORT(coreinit, OSGetThreadSpecific) {
    uint32_t i = ARG(0);
    GuestThread* t = c && c->thread ? (GuestThread*)c->thread : current();
    RET(t && i < 16 ? mem::r32(t->os + T_SPECIFIC + 4 * i) : 0);
}
HLE_EXPORT(coreinit, OSSetThreadSpecific) {
    uint32_t i = ARG(0);
    GuestThread* t = c && c->thread ? (GuestThread*)c->thread : current();
    if (t && i < 16) mem::w32(t->os + T_SPECIFIC + 4 * i, ARG(1));
}

HLE_EXPORT(coreinit, OSGetCoreId) { RET(c ? (uint32_t)c->core : (current() ? current()->core : 1)); }
HLE_EXPORT(coreinit, OSGetStackPointer) { RET(c->r[1]); }
HLE_EXPORT(coreinit, OSYieldThread) {
    diag_sync("yield");
    GuestThread* me = current();
    if (!threads::should_yield_core(me)) return;
    threads::exec_yield();
    threads::exec_resume();
}
HLE_EXPORT(coreinit, OSBlockThreadsOnExit) {}

#if defined(_WIN32)
static void host_sleep_ns(uint64_t ns) {
    if (ns == 0) {
        SwitchToThread();
        return;
    }
#ifndef CREATE_WAITABLE_TIMER_HIGH_RESOLUTION
#define CREATE_WAITABLE_TIMER_HIGH_RESOLUTION 0x00000002
#endif
    thread_local HANDLE timer = [] {
        HANDLE t = CreateWaitableTimerExW(nullptr, nullptr, CREATE_WAITABLE_TIMER_HIGH_RESOLUTION, TIMER_ALL_ACCESS);
        if (!t) t = CreateWaitableTimerExW(nullptr, nullptr, 0, TIMER_ALL_ACCESS);
        return t;
    }();
    if (!timer) {
        Sleep((DWORD)((ns + 999999ull) / 1000000ull));
        return;
    }
    LARGE_INTEGER due;
    due.QuadPart = -(LONGLONG)((ns + 99) / 100);  // 100 ns units, relative
    if (!SetWaitableTimer(timer, &due, 0, nullptr, nullptr, FALSE)) {
        Sleep((DWORD)((ns + 999999ull) / 1000000ull));
        return;
    }
    WaitForSingleObject(timer, INFINITE);
}
#endif

// void OSSleepTicks(OSTime ticks)
HLE_EXPORT(coreinit, OSSleepTicks) {
    int64_t ticks = (int64_t)ARG64(0);
    diag_sync("sleep ticks %lld", (long long)ticks);
    if (ticks <= 0) {
        threads::exec_yield();
#if defined(_WIN32)
        SwitchToThread();
#else
        sched_yield();
#endif
        threads::exec_resume();
        return;
    }
    // 1e9 / 62156250 = 32000 / 1989 exactly; 64-bit is enough below ~53 days (no __udivti3, 1.3% of lane-mate CPU)
    uint64_t ns = (uint64_t)ticks * 32000u / 1989u;
    static const bool s_probe = getenv("BOTW_SLEEPPROBE") && getenv("BOTW_SLEEPPROBE")[0] == '1';  // TEST: PC only
    if (s_probe) {
        static std::mutex mu;
        static std::map<std::string, uint64_t> counts;
        static uint64_t next_ns = 0;
        std::lock_guard<std::mutex> l(mu);
        char key[160];
        const char* who = threads::current_name();
        uint32_t chain[4] = {}, sp = c->r[1];  // guest back chain: word 0 = caller's frame, +4 = saved LR
        for (int d = 0; d < 4 && sp >= 0x0E000000u && sp < 0xFF000000u; d++) {
            const uint32_t prev = mem::r32(sp);
            if (prev <= sp || prev >= sp + 0x100000u) break;
            chain[d] = mem::r32(prev + 4);
            sp = prev;
        }
        snprintf(key, sizeof key, "%s lr=%08x us=%llu up %08x %08x %08x %08x", who ? who : "?", c->lr,
                 (unsigned long long)(ns / 1000), chain[0], chain[1], chain[2], chain[3]);
        counts[key]++;
        const uint64_t now = threads::sched_now_ns();
        if (now > next_ns) {
            next_ns = now + 10000000000ull;
            std::vector<std::pair<uint64_t, std::string>> v;
            for (auto& [k, n] : counts) v.emplace_back(n, k);
            std::sort(v.rbegin(), v.rend());
            for (size_t i = 0; i < v.size() && i < 15; i++) LOG("SLEEPPROBE %llu %s", (unsigned long long)v[i].first, v[i].second.c_str());
        }
    }
    threads::CoreHandoff handoff;
    handoff.yield();
    threads::WaitMark w("sleep", (uint32_t)(ns / 1000));
#if defined(_WIN32)
    host_sleep_ns(ns);
#else
    struct timespec ts = {(time_t)(ns / 1000000000), (long)(ns % 1000000000)};
    nanosleep(&ts, nullptr);
#endif
}

// The game's "Sleeper" thread is a sead thread whose calc() (0x031ff71c, reached only through its vtable) just sleeps
// 250 us; its run loop polls its message queue without blocking and only ever acts on the quit message. On the port
// each poll is a host sleep + wake on the job worker's cpus + a hand-off of Worker2's guest core: ~4000 a second
// (PC probe), about 2/3 of all guest sleeps. 20 ms keeps the loop and the quit path, ~50 wakes a second; the only
// change the game can see is the quit message taking up to 20 ms. BOTW_SLEEPER=0 keeps the original.
extern "C" void botw_sleeper_calc(PPCContext* c) {
    (void)c;
    threads::CoreHandoff handoff;
    handoff.yield();
    threads::WaitMark w("sleep", 20000);
#if defined(_WIN32)
    host_sleep_ns(20000000ull);
#else
    struct timespec ts = {0, 20000000};
    nanosleep(&ts, nullptr);
#endif
}

void install_sleeper_calc() {
    if (getenv("BOTW_SLEEPER") && getenv("BOTW_SLEEPER")[0] == '0') return;
    dispatch::replace_fn_native(0x031ff71cu, botw_sleeper_calc);
}

HLE_EXPORT(coreinit, OSInitThreadQueue) {
    uint32_t q = ARG(0);
    mem::w32(q, 0);
    mem::w32(q + 4, 0);
    mem::w32(q + 8, 0);
    mem::w32(q + 12, 0);
}

// Cemu-compat: guest callback on the calling guest thread.
uint32_t cemu_core_callback(uint32_t fn, std::initializer_list<uint32_t> args) {
    threads::GuestThread* t = threads::current();
    if (!t) {
        LOG("cemu_core_callback(%08x) from a non-guest thread", fn);
        return 0;
    }
    return guest_call(&t->ctx, fn, args);
}


