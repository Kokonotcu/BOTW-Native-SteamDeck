// Performance recorder: see telemetry.h. Everything slow (files, /proc, /sys, symbol lookup) runs on the
// writer thread once a second; the game threads only push one row per frame under an uncontended lock.
#include "telemetry.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <filesystem>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#if defined(__linux__)
#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <signal.h>
#include <sys/resource.h>
#include <sys/syscall.h>
#include <sys/time.h>
#include <sys/utsname.h>
#include <ucontext.h>
#include <unistd.h>
extern "C" char __ehdr_start, _end;  // linker-defined bounds of the program image
#endif

#include "botw_target.h"
#include "log.h"
#include "mem.h"
#include "settings.h"
#include "thread.h"

extern float g_render_scale;

namespace telem {
std::atomic<uint64_t> g_ctr[Count];
std::atomic<bool> g_enabled{false};

#if !defined(__linux__)
void set_root(const char*) {}
void start(const char*) {}
void stop() {}
void note_exec_wait(const void*, const void*, uint64_t, bool) {}
void gpu_pass_stats(std::vector<GpuPassStat>&&) {}
void on_guest_frame() {}
void on_present(uint32_t) {}
#else
namespace {
constexpr int kMainCtrs[] = {MainWaitGpuNs, MainWaitOtherNs, FsReads, FsBytes, FsNs, MainWaitCoreNs, MainWaits, MainWakes, MainWakeLatNs};
constexpr int kGpuCtrs[] = {Submits,   SubmitNs,  FenceWaitNs, AcquireNs,   PresentNs,  PipeNew,    PipeAsync, PipeNs,
                            ShaderNew, ShaderNs,  GpuTimeNs,   GpuPasses,   GpuBarriers, GpuUploads, GpuCopies, GpuClears};
// GPU pass groups: batches from the GPU thread, cumulative totals kept by the writer
std::mutex g_gpu_mu;
std::vector<GpuPassStat> g_gpu_pending;
struct GpuCum {
    uint64_t ns = 0, passes = 0, draws = 0;
};
std::unordered_map<std::string, GpuCum> g_gpu_cum;
constexpr int kProfileUs = 4000;  // one sample per 4 ms of process CPU time

struct Row {
    uint64_t t, dt, cpu, draws;
    uint64_t c[Count];
};
std::atomic<bool> g_on{false};
// The menu switch works live: each start() is a new session (new file prefix) with its own generation; the writer
// and clock sampler of an older generation end at their next tick. g_writer_busy: one writer drains at a time.
std::atomic<uint32_t> g_gen{0};
std::atomic<bool> g_writer_busy{false};
std::string g_root;
uint64_t g_t0 = 0;
std::string g_prefix;  // <user dir>/telemetry/<session>-
std::mutex g_rows_mu;
std::vector<Row> g_frames, g_presents;
// core waits: lock-free ring of (waiter, holder, ns), aggregated by the writer
struct ExecWaitEvent {
    std::atomic<const void*> waiter, holder;
    std::atomic<uint64_t> ns;
};
constexpr size_t kExecEvents = 1 << 16;
ExecWaitEvent g_exec_events[kExecEvents];
std::atomic<uint64_t> g_exec_head{0};
uint64_t g_exec_tail = 0;

uint64_t thread_cpu_ns() {
    timespec ts;
    clock_gettime(CLOCK_THREAD_CPUTIME_ID, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}

template <size_t N>
void record(std::vector<Row>& out, const int (&ids)[N], uint64_t& last_t, uint64_t& last_cpu, uint64_t draws) {
    Row r{};
    r.t = now_ns();
    uint64_t cpu = thread_cpu_ns();
    r.dt = last_t ? r.t - last_t : 0;
    r.cpu = last_cpu ? cpu - last_cpu : 0;
    r.draws = draws;
    last_t = r.t;
    last_cpu = cpu;
    for (size_t i = 0; i < N; i++) r.c[i] = g_ctr[ids[i]].exchange(0, std::memory_order_relaxed);
    std::lock_guard<std::mutex> l(g_rows_mu);
    out.push_back(r);
}

// ---- sampling profiler -------------------------------------------------------------------------------
// ITIMER_PROF signals whichever thread is using CPU; the handler only stores pc and tid.
constexpr size_t kSamples = 1 << 16;
constexpr int kChain = 10;
struct Sample {
    std::atomic<uint64_t> pc;
    std::atomic<uint64_t> t;  // now_ns() when taken: spike frames are matched to their samples (spikes.txt)
    std::atomic<uint32_t> tid;
    std::atomic<uint32_t> lr;  // guest link register: which game code led here (libc, driver, HLE time)
    std::atomic<uint32_t> chain[kChain];  // guest return addresses up the stack back chain (longframes.txt)
};
Sample g_samples[kSamples];
std::atomic<uint64_t> g_head{0};
uint64_t g_tail = 0, g_prof_total = 0, g_prof_lost = 0;
bool g_prof_on = false;
using ProfMap = std::unordered_map<std::string, std::unordered_map<uint64_t, uint32_t>>;  // thread name -> pc -> hits
using LrMap = std::unordered_map<std::string, std::unordered_map<uint32_t, uint32_t>>;    // thread name -> guest lr -> hits
ProfMap g_prof;
LrMap g_prof_lr;
// Spike profile: only the samples taken inside main-thread frames over 25 ms (the drops), across all threads.
struct RecentSample {
    uint64_t t, pc;
    uint32_t lr;
    std::string name;
    std::array<uint32_t, kChain> chain;
};
std::vector<RecentSample> g_recent;  // the last ~2 s of drained samples
ProfMap g_spike_prof;
LrMap g_spike_lr;
uint64_t g_spike_frames = 0;
// Main-thread frames over 500 ms (freezes): the main thread's guest call chains, counted.
std::unordered_map<std::string, uint32_t> g_long_chains;
uint64_t g_long_frames = 0;

void on_prof(int, siginfo_t*, void* ctx) {
    int saved = errno;
    uint64_t pc = 0;
#if defined(__x86_64__)
    pc = (uint64_t)((ucontext_t*)ctx)->uc_mcontext.gregs[REG_RIP];
#else
    (void)ctx;
#endif
    threads::GuestThread* g = threads::current();  // thread_local of the executable: safe in a handler
    uint64_t i = g_head.fetch_add(1, std::memory_order_relaxed);
    Sample& s = g_samples[i & (kSamples - 1)];
    s.pc.store(pc, std::memory_order_relaxed);
    s.t.store(now_ns(), std::memory_order_relaxed);
    s.lr.store(g ? g->ctx.lr : 0, std::memory_order_relaxed);
    // Guest stack back chain from the last r1 the context holds: word 0 = caller's frame, +4 = saved LR.
    // Plain loads of guest memory (all 4 GB is mapped on POSIX); bounded and monotonic, so safe here.
    uint32_t sp = g ? g->ctx.r[1] : 0;
    for (int d = 0; d < kChain; d++) {
        uint32_t v = 0;
        if (sp >= 0x0E000000u && sp < 0xFF000000u) {  // thread stacks live in the runtime area (0x0E100000+) or heaps
            uint32_t prev = mem::r32(sp);
            if (prev > sp && prev < sp + 0x100000u) {
                v = mem::r32(prev + 4);
                sp = prev;
            } else {
                sp = 0;
            }
        }
        s.chain[d].store(v, std::memory_order_relaxed);
    }
    s.tid.store((uint32_t)syscall(SYS_gettid), std::memory_order_relaxed);
    errno = saved;
}

// ---- /proc and /sys ------------------------------------------------------------------------------------
bool read_file(const char* path, char* buf, size_t size) {
    FILE* f = fopen(path, "r");
    if (!f) return false;
    size_t n = fread(buf, 1, size - 1, f);
    fclose(f);
    buf[n] = 0;
    return n > 0;
}
long long read_ll(const std::string& path) {
    char buf[64];
    if (path.empty() || !read_file(path.c_str(), buf, sizeof buf)) return -1;
    return atoll(buf);
}
bool exists(const std::string& p) { return access(p.c_str(), R_OK) == 0; }

struct SysPaths {
    std::vector<std::string> cpufreq;
    std::string gpu, gpu_busy, sclk, fclk, vram, gtt, power, power_cap, temp, bat_power, bat_cap;
} g_sys;

void find_sys_paths() {
    for (int i = 0; i < 64; i++) {
        std::string p = "/sys/devices/system/cpu/cpu" + std::to_string(i) + "/cpufreq/scaling_cur_freq";
        if (exists(p)) g_sys.cpufreq.push_back(p);
    }
    for (int i = 0; i < 8 && g_sys.gpu.empty(); i++) {
        std::string d = "/sys/class/drm/card" + std::to_string(i) + "/device/";
        if (!exists(d + "gpu_busy_percent")) continue;
        g_sys.gpu = d;
        g_sys.gpu_busy = d + "gpu_busy_percent";
        if (exists(d + "pp_dpm_sclk")) g_sys.sclk = d + "pp_dpm_sclk";
        if (exists(d + "pp_dpm_fclk")) g_sys.fclk = d + "pp_dpm_fclk";  // memory fabric clock (game thread is memory-bound)
        if (exists(d + "mem_info_vram_used")) g_sys.vram = d + "mem_info_vram_used";
        if (exists(d + "mem_info_gtt_used")) g_sys.gtt = d + "mem_info_gtt_used";
        if (DIR* h = opendir((d + "hwmon").c_str())) {
            while (dirent* e = readdir(h)) {
                if (strncmp(e->d_name, "hwmon", 5) != 0) continue;
                std::string hw = d + "hwmon/" + e->d_name + "/";
                if (exists(hw + "power1_average")) g_sys.power = hw + "power1_average";
                else if (exists(hw + "power1_input")) g_sys.power = hw + "power1_input";
                if (exists(hw + "power1_cap")) g_sys.power_cap = hw + "power1_cap";
                if (exists(hw + "temp1_input")) g_sys.temp = hw + "temp1_input";
                break;
            }
            closedir(h);
        }
    }
    if (DIR* h = opendir("/sys/class/power_supply")) {
        while (dirent* e = readdir(h)) {
            if (strncmp(e->d_name, "BAT", 3) != 0) continue;
            std::string b = std::string("/sys/class/power_supply/") + e->d_name + "/";
            if (exists(b + "power_now")) g_sys.bat_power = b + "power_now";
            if (exists(b + "capacity")) g_sys.bat_cap = b + "capacity";
            break;
        }
        closedir(h);
    }
}

int dpm_mhz(const std::string& path) {  // pp_dpm_*: "1: 1600Mhz *" marks the current level
    char buf[1024];
    if (path.empty() || !read_file(path.c_str(), buf, sizeof buf)) return -1;
    char* star = strchr(buf, '*');
    if (!star) return -1;
    char* line = star;
    while (line > buf && line[-1] != '\n') line--;
    char* colon = strchr(line, ':');
    return colon && colon < star ? atoi(colon + 1) : -1;
}

long long meminfo_kb(const char* key) {
    char buf[4096];
    if (!read_file("/proc/meminfo", buf, sizeof buf)) return -1;
    const char* p = strstr(buf, key);
    return p ? atoll(p + strlen(key)) : -1;
}

// Per-thread CPU use from /proc/self/task/*/stat; also refreshes the tid -> name map the profiler uses.
std::unordered_map<uint32_t, uint64_t> g_prev_ticks;
std::unordered_map<uint32_t, std::string> g_tid_name;

std::string clean_name(std::string s) {
    for (char& ch : s)
        if (ch == ' ' || ch == ',' || ch == ';' || ch == ':') ch = '_';
    return s;
}

std::unordered_map<uint32_t, long long> g_prev_nvcsw;

long long status_field(uint32_t tid, const char* key) {
    char path[64], buf[2048];
    snprintf(path, sizeof path, "/proc/self/task/%u/status", tid);
    if (!read_file(path, buf, sizeof buf)) return -1;
    const char* p = strstr(buf, key);
    return p ? atoll(p + strlen(key)) : -1;
}

// threads: "name/tid:cpu%" for threads over 2%; detail: "name/tid:cpuN:forced switches this second"
std::string scan_threads(double secs, double* proc_pct, std::string* detail = nullptr) {
    static const double hz = (double)sysconf(_SC_CLK_TCK);
    std::unordered_map<uint32_t, uint64_t> now;
    std::vector<std::pair<double, std::string>> rows;
    std::vector<std::pair<double, std::string>> details;
    double total = 0;
    if (DIR* d = opendir("/proc/self/task")) {
        while (dirent* e = readdir(d)) {
            if (e->d_name[0] < '0' || e->d_name[0] > '9') continue;
            uint32_t tid = (uint32_t)atoi(e->d_name);
            char path[64], buf[1024];
            snprintf(path, sizeof path, "/proc/self/task/%u/stat", tid);
            if (!read_file(path, buf, sizeof buf)) continue;
            char* l = strchr(buf, '(');
            char* r = strrchr(buf, ')');
            if (!l || !r || r < l) continue;
            std::string name = clean_name(std::string(l + 1, r - l - 1));
            // after "pid (comm) ": field 3 (state) is token 0, utime/stime 11/12, processor (field 39) 36
            const char* tok[40];
            int nt = 0;
            char* save = nullptr;
            for (char* p = strtok_r(r + 1, " \n", &save); p && nt < 40; p = strtok_r(nullptr, " \n", &save)) tok[nt++] = p;
            if (nt < 13) continue;
            uint64_t ticks = strtoull(tok[11], nullptr, 10) + strtoull(tok[12], nullptr, 10);
            now[tid] = ticks;
            g_tid_name[tid] = name;
            auto it = g_prev_ticks.find(tid);
            if (it == g_prev_ticks.end()) continue;
            double pct = (double)(ticks - it->second) * 100.0 / (hz * secs);
            total += pct;
            if (pct < 2.0) continue;
            std::string id = name + "/" + std::to_string(tid);
            rows.push_back({pct, id});
            if (detail) {
                long long nv = status_field(tid, "nonvoluntary_ctxt_switches:");
                long long prev = g_prev_nvcsw.count(tid) ? g_prev_nvcsw[tid] : nv;
                g_prev_nvcsw[tid] = nv;
                details.push_back({pct, id + ":cpu" + (nt > 36 ? tok[36] : "?") + ":" + std::to_string(nv - prev)});
            }
        }
        closedir(d);
    }
    g_prev_ticks.swap(now);
    if (detail) {
        std::sort(details.begin(), details.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
        detail->clear();
        for (auto& [pct, s] : details) *detail += (detail->empty() ? "" : ";") + s;
    }
    std::sort(rows.begin(), rows.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
    std::string out;
    char buf[160];
    for (auto& [pct, name] : rows) {
        snprintf(buf, sizeof buf, "%s%s:%.0f", out.empty() ? "" : ";", name.c_str(), pct);
        out += buf;
    }
    *proc_pct = total;
    return out;
}

// Busy % of every logical CPU since the last call, "b0;b1;...", from /proc/stat.
std::string cpu_busy() {
    static std::vector<std::pair<unsigned long long, unsigned long long>> prev;  // busy, total
    char buf[8192];
    if (!read_file("/proc/stat", buf, sizeof buf)) return "";
    std::string out;
    size_t i = 0;
    char* save = nullptr;
    for (char* line = strtok_r(buf, "\n", &save); line; line = strtok_r(nullptr, "\n", &save)) {
        if (strncmp(line, "cpu", 3) != 0 || line[3] < '0' || line[3] > '9') continue;
        unsigned long long v[8] = {};
        if (sscanf(line, "%*s %llu %llu %llu %llu %llu %llu %llu %llu", &v[0], &v[1], &v[2], &v[3], &v[4], &v[5], &v[6], &v[7]) < 8)
            continue;
        unsigned long long total = v[0] + v[1] + v[2] + v[3] + v[4] + v[5] + v[6] + v[7];
        unsigned long long busy = total - v[3] - v[4];  // minus idle and iowait
        if (i >= prev.size()) prev.push_back({busy, total});
        auto& p = prev[i++];
        unsigned long long dt = total - p.second;
        out += (out.empty() ? "" : ";") + std::to_string(dt ? (int)((busy - p.first) * 100 / dt) : 0);
        p = {busy, total};
    }
    return out;
}

void proc_faults(unsigned long long* minflt, unsigned long long* majflt) {
    char buf[1024];
    *minflt = *majflt = 0;
    if (!read_file("/proc/self/stat", buf, sizeof buf)) return;
    char* r = strrchr(buf, ')');
    if (r) sscanf(r + 1, " %*s %*s %*s %*s %*s %*s %*s %llu %*s %llu", minflt, majflt);
}

void drain_samples() {
    uint64_t head = g_head.load(std::memory_order_acquire);
    if (head < 8) return;
    head -= 8;  // the newest slots may still be being written
    if (head <= g_tail) return;
    if (head - g_tail > kSamples) {
        g_prof_lost += head - g_tail - kSamples;
        g_tail = head - kSamples;
    }
    for (; g_tail < head; g_tail++) {
        Sample& s = g_samples[g_tail & (kSamples - 1)];
        uint64_t pc = s.pc.load(std::memory_order_relaxed);
        uint32_t tid = s.tid.load(std::memory_order_relaxed);
        uint32_t lr = s.lr.load(std::memory_order_relaxed);
        auto it = g_tid_name.find(tid);
        const std::string& name = it != g_tid_name.end() ? it->second : "?";
        g_prof[name][pc]++;
        if (lr) g_prof_lr[name][lr]++;
        g_prof_total++;
        RecentSample r{s.t.load(std::memory_order_relaxed), pc, lr, name, {}};
        for (int d = 0; d < kChain; d++) r.chain[d] = s.chain[d].load(std::memory_order_relaxed);
        g_recent.push_back(std::move(r));
    }
}

// Attribute the recent samples that fall inside this second's spike frames, then drop samples older than 2 s.
void attribute_spikes(const std::vector<Row>& frames) {
    std::vector<std::pair<uint64_t, uint64_t>> windows;
    std::vector<std::pair<uint64_t, uint64_t>> longw;
    for (const Row& r : frames) {
        if (r.dt > 25000000ull) windows.push_back({r.t - r.dt, r.t});
        if (r.dt > 500000000ull) longw.push_back({r.t - r.dt, r.t});
    }
    g_spike_frames += windows.size();
    g_long_frames += longw.size();
    for (const RecentSample& s : g_recent) {
        for (auto& [a, b] : windows)
            if (s.t >= a && s.t <= b) {
                g_spike_prof[s.name][s.pc]++;
                if (s.lr) g_spike_lr[s.name][s.lr]++;
                break;
            }
        if (s.name != "main") continue;
        for (auto& [a, b] : longw)
            if (s.t >= a && s.t <= b) {
                char buf[16 * (kChain + 1)];
                int k = snprintf(buf, sizeof buf, "%08x", s.lr);
                for (int d = 0; d < kChain && s.chain[d]; d++) k += snprintf(buf + k, sizeof buf - k, " %08x", s.chain[d]);
                g_long_chains[buf]++;
                break;
            }
    }
    const uint64_t cut = now_ns() - 2000000000ull;
    g_recent.erase(g_recent.begin(), std::find_if(g_recent.begin(), g_recent.end(), [cut](const RecentSample& s) { return s.t >= cut; }));
}

// pc -> "module+offset" with the offset as an ELF virtual address (addr2line -e <module> <offset>).
struct Mapping {
    uint64_t start, end, bias;
    std::string name, path;
};
std::vector<Mapping> read_maps() {
    std::vector<Mapping> out;
    std::unordered_map<std::string, uint64_t> bias;
    std::vector<std::pair<Mapping, std::string>> exec;
    FILE* f = fopen("/proc/self/maps", "r");
    if (!f) return out;
    char line[1024];
    while (fgets(line, sizeof line, f)) {
        unsigned long long s, e, off;
        char perms[8];
        int path_at = 0;
        if (sscanf(line, "%llx-%llx %7s %llx %*s %*s %n", &s, &e, perms, &off, &path_at) < 4 || !path_at) continue;
        std::string path = line + path_at;
        while (!path.empty() && (path.back() == '\n' || path.back() == ' ')) path.pop_back();
        // hugetext.cpp turned the middle of the program's code into an anonymous mapping
        if (path.empty() && strchr(perms, 'x') && s >= (uintptr_t)&__ehdr_start && e <= (uintptr_t)&_end) {
            static std::string self = std::filesystem::read_symlink("/proc/self/exe").string();
            exec.push_back({Mapping{s, e, 0, "", self}, self});
            continue;
        }
        if (path.empty() || path[0] != '/') continue;
        if (off == 0 && !bias.count(path)) bias[path] = s;
        if (strchr(perms, 'x')) exec.push_back({Mapping{s, e, 0, "", path}, path});
    }
    fclose(f);
    for (auto& [m, path] : exec) {
        size_t slash = path.rfind('/');
        Mapping r = m;
        r.name = slash == std::string::npos ? path : path.substr(slash + 1);
        r.bias = bias.count(path) ? bias[path] : m.start;
        out.push_back(r);
    }
    std::sort(out.begin(), out.end(), [](const Mapping& a, const Mapping& b) { return a.start < b.start; });
    return out;
}

void write_profile_file(const ProfMap& prof, const LrMap& prof_lr, const char* file, const std::string& note);

void write_profile() {
    if (!g_prof_on) return;
    write_profile_file(g_prof, g_prof_lr, "profile.txt", "whole session so far");
    write_profile_file(g_spike_prof, g_spike_lr, "spikes.txt",
                       "only samples inside main-thread frames over 25 ms (" + std::to_string(g_spike_frames) + " frames)");
    if (g_long_chains.empty()) return;
    std::string tmp = g_prefix + "longframes.txt.tmp";
    FILE* f = fopen(tmp.c_str(), "w");
    if (!f) return;
    fprintf(f, "# main thread guest call chains (lr, then return addresses up the stack) in %llu frames over 500 ms\n",
            (unsigned long long)g_long_frames);
    std::vector<std::pair<uint32_t, const std::string*>> v;
    for (auto& [chain, n] : g_long_chains) v.push_back({n, &chain});
    std::sort(v.rbegin(), v.rend());
    if (v.size() > 300) v.resize(300);
    for (auto& [n, chain] : v) fprintf(f, "%u %s\n", n, chain->c_str());
    fclose(f);
    rename(tmp.c_str(), (g_prefix + "longframes.txt").c_str());
}

void write_profile_file(const ProfMap& prof, const LrMap& prof_lr, const char* file, const std::string& note) {
    std::vector<Mapping> maps = read_maps();
    auto where = [&](uint64_t pc, char* buf, size_t n) {
        auto it = std::upper_bound(maps.begin(), maps.end(), pc, [](uint64_t v, const Mapping& m) { return v < m.start; });
        if (it != maps.begin() && pc < (--it)->end) snprintf(buf, n, "%s+0x%llx", it->name.c_str(), (unsigned long long)(pc - it->bias));
        else snprintf(buf, n, "?+0x%llx", (unsigned long long)pc);
    };
    std::string tmp = g_prefix + file + ".tmp";
    FILE* f = fopen(tmp.c_str(), "w");
    if (!f) return;
    fprintf(f, "# samples=%llu lost=%llu interval_us=%d (process CPU time); %s\n",
            (unsigned long long)g_prof_total, (unsigned long long)g_prof_lost, kProfileUs, note.c_str());
    fprintf(f, "# offsets are ELF virtual addresses: addr2line -f -e BotWRecomp.debug <offset>\n");
    for (const Mapping& m : maps) fprintf(f, "# module %s %s\n", m.name.c_str(), m.path.c_str());
    std::vector<std::pair<uint64_t, const std::string*>> threads;
    for (auto& [name, pcs] : prof) {
        uint64_t n = 0;
        for (auto& [pc, c] : pcs) n += c;
        threads.push_back({n, &name});
    }
    std::sort(threads.rbegin(), threads.rend());
    char buf[256];
    for (auto& [n, name] : threads) {
        fprintf(f, "\n[thread %s samples=%llu]\n", name->c_str(), (unsigned long long)n);
        std::vector<std::pair<uint32_t, uint64_t>> pcs;
        for (auto& [pc, c] : prof.at(*name)) pcs.push_back({c, pc});
        std::sort(pcs.rbegin(), pcs.rend());
        if (pcs.size() > 20000) pcs.resize(20000);
        for (auto& [c, pc] : pcs) {
            where(pc, buf, sizeof buf);
            fprintf(f, "%u %s\n", c, buf);
        }
    }
    for (auto& [name, lrs] : prof_lr) {
        std::vector<std::pair<uint32_t, uint32_t>> v;
        for (auto& [lr, c] : lrs) v.push_back({c, lr});
        std::sort(v.rbegin(), v.rend());
        if (v.size() > 2000) v.resize(2000);
        fprintf(f, "\n[guest-lr %s]\n", name.c_str());
        for (auto& [c, lr] : v) fprintf(f, "%u %08x\n", c, lr);
    }
    fclose(f);
    rename(tmp.c_str(), (g_prefix + file).c_str());
}

// Session totals per render pass group, most GPU time first.
void write_gpu_passes(uint64_t secs) {
    if (g_gpu_cum.empty()) return;
    std::string tmp = g_prefix + "gpu_passes.txt.tmp";
    FILE* f = fopen(tmp.c_str(), "w");
    if (!f) return;
    std::vector<std::pair<uint64_t, const std::string*>> order;
    uint64_t total = 0;
    for (auto& [name, c] : g_gpu_cum) order.push_back({c.ns, &name}), total += c.ns;
    std::sort(order.rbegin(), order.rend());
    fprintf(f, "# GPU time per render pass group over %llu s with GPU work; total %.1f ms/s\n", (unsigned long long)secs,
            secs ? total / 1e6 / secs : 0.0);
    fprintf(f, "# share  ms/s  passes/s  draws/s  pass (WxH, color targets, depth, first-draw shaders when first seen)\n");
    for (auto& [ns, name] : order) {
        const GpuCum& c = g_gpu_cum[*name];
        double s = secs ? (double)secs : 1.0;
        fprintf(f, "%5.1f%% %7.3f %8.1f %8.1f  %s\n", total ? 100.0 * ns / total : 0.0, ns / 1e6 / s, c.passes / s,
                c.draws / s, name->c_str());
    }
    fclose(f);
    rename(tmp.c_str(), (g_prefix + "gpu_passes.txt").c_str());
}

FILE* open_csv(const char* suffix, const char* header) {
    FILE* f = fopen((g_prefix + suffix).c_str(), "w");
    if (f) fprintf(f, "%s\n", header);
    return f;
}

void write_info() {
    FILE* f = fopen((g_prefix + "info.txt").c_str(), "w");
    if (!f) return;
    Settings s = settings::get();
    fprintf(f, "fps=%d scale=%.3f render_scale=%.3f display=%d filter=%d\n", s.fps, s.scale, g_render_scale, s.display,
            s.filter);
    fprintf(f, "image_base=0x%llx\n", (unsigned long long)botw_image_base());
    utsname u{};
    if (uname(&u) == 0) fprintf(f, "kernel=%s %s %s\n", u.sysname, u.release, u.machine);
    char buf[8192];
    if (read_file("/proc/cpuinfo", buf, sizeof buf)) {
        if (const char* m = strstr(buf, "model name")) {
            const char* e = strchr(m, '\n');
            fprintf(f, "%.*s\n", (int)(e ? e - m : strlen(m)), m);
        }
    }
    fprintf(f, "cpus=%ld\n", sysconf(_SC_NPROCESSORS_ONLN));
    for (const char* k : {"SteamDeck", "SteamAppId", "SteamGameId", "GAMESCOPE_WAYLAND_DISPLAY", "DISPLAY",
                          "WAYLAND_DISPLAY", "SDL_VIDEO_DRIVER", "XDG_SESSION_TYPE", "MESA_VK_WSI_PRESENT_MODE",
                          "RADV_PERFTEST", "ENABLE_GAMESCOPE_WSI"})
        if (const char* v = getenv(k)) fprintf(f, "env %s=%s\n", k, v);
    fprintf(f, "gpu_sysfs=%s\n", g_sys.gpu.c_str());
    fclose(f);
}

// GPU clock / power at 100 Hz from amdgpu's gpu_metrics (v2.x layout, Steam Deck: v2.4, 168 bytes): the per-second
// clock sample cannot show dips of a few hundred ms, and single frames ran at half speed with unchanged work.
// BOTW_GPU_CLOCK_LOG=0 off.
void clock_sampler(std::string path, uint32_t gen) {
    botw_set_thread_name("telemetry-clk");
    setpriority(PRIO_PROCESS, (id_t)syscall(SYS_gettid), 10);
    int fd = open(path.c_str(), O_RDONLY);
    if (fd < 0) return;
    uint8_t b[256];
    ssize_t n = pread(fd, b, sizeof b, 0);
    if (n < 112 || b[2] != 2) {  // format revision 2 (APU layout) only
        close(fd);
        return;
    }
    FILE* f = open_csv("gpu_clock.csv", "t_ms,gfxclk_mhz,gfxclk_avg_mhz,uclk_mhz,socket_mw,cpu_mw,soc_mw,gfx_mw,throttle");
    if (!f) {
        close(fd);
        return;
    }
    auto u16 = [&](int off) { return (unsigned)(b[off] | (b[off + 1] << 8)); };
    for (int i = 1;; i++) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        if (!g_on.load(std::memory_order_acquire) || g_gen.load(std::memory_order_acquire) != gen) {  // stop() / newer session
            fclose(f);
            close(fd);
            return;
        }
        if (pread(fd, b, sizeof b, 0) < 112) continue;
        const unsigned throttle = b[108] | (b[109] << 8) | (b[110] << 16) | ((unsigned)b[111] << 24);
        fprintf(f, "%.1f,%u,%u,%u,%u,%u,%u,%u,%u\n", (now_ns() - g_t0) / 1e6, u16(76), u16(64), u16(80), u16(40), u16(42),
                u16(44), u16(46), throttle);
        if (i % 100 == 0) fflush(f);
    }
}

void writer(uint32_t gen) {
    botw_set_thread_name("telemetry");
    // a writer of the previous session (switched off, then on again within a second) ends at its next tick
    for (bool idle = false; !g_writer_busy.compare_exchange_weak(idle, true); idle = false) {
        if (g_gen.load(std::memory_order_acquire) != gen) return;
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
    {  // a new session: nothing carried over from the previous one
        std::lock_guard<std::mutex> l(g_rows_mu);
        g_frames.clear();
        g_presents.clear();
    }
    {
        std::lock_guard<std::mutex> l(g_gpu_mu);
        g_gpu_pending.clear();
    }
    g_gpu_cum.clear();
    g_prof.clear();
    g_prof_lr.clear();
    g_spike_prof.clear();
    g_spike_lr.clear();
    g_long_chains.clear();
    g_recent.clear();
    g_spike_frames = g_long_frames = g_prof_total = g_prof_lost = 0;
    g_tail = g_head.load(std::memory_order_acquire);
    g_exec_tail = g_exec_head.load(std::memory_order_acquire);
    for (auto& c : g_ctr) c.store(0, std::memory_order_relaxed);
    if (const char* e = getenv("BOTW_GPU_CLOCK_LOG"); !(e && e[0] == '0') && !g_sys.gpu.empty())
        std::thread(clock_sampler, g_sys.gpu + "gpu_metrics", gen).detach();
    setpriority(PRIO_PROCESS, (id_t)syscall(SYS_gettid), 10);
    FILE* ff = open_csv("frames.csv",
                        "t_ms,frame_ms,main_cpu_ms,wait_gpu_ms,wait_other_ms,fs_reads,fs_kb,fs_ms,wait_core_ms,waits,wakes,"
                        "wake_lat_ms");
    FILE* fp = open_csv("present.csv",
                        "t_ms,present_ms,gpu_thread_cpu_ms,draws,submits,submit_ms,fence_wait_ms,acquire_ms,present_call_ms,"
                        "pipe_new,pipe_async,pipe_ms,shader_new,shader_ms,gpu_ms,passes,barriers,uploads,copies,clears");
    FILE* fg = open_csv("gpu_passes.csv", "t_s,gpu_ms,passes,draws,pass");
    uint64_t gpu_secs = 0;
    FILE* fs = open_csv("seconds.csv",
                        "t_s,game_fps,present_fps,proc_cpu_pct,cpu_mhz_avg,cpu_mhz_max,gpu_busy_pct,gpu_mhz,apu_w,apu_cap_w,"
                        "gpu_temp_c,bat_w,bat_pct,rss_mb,mem_avail_mb,swap_free_mb,vram_mb,gtt_mb,minflt,majflt,samples,threads,"
                        "core_waits,preempts,cpu_busy,thread_detail,fclk_mhz");
    std::vector<Row> fr, pr;
    uint64_t last = now_ns();
    unsigned long long last_min = 0, last_maj = 0;
    proc_faults(&last_min, &last_maj);
    double dummy;
    std::string first_detail;
    scan_threads(1.0, &dummy, &first_detail);  // first sample of every thread's CPU time
    cpu_busy();
    const long page_kb = sysconf(_SC_PAGESIZE) / 1024;
    for (int sec = 1;; sec++) {
        std::this_thread::sleep_for(std::chrono::seconds(1));
        if (!g_on.load(std::memory_order_acquire) || g_gen.load(std::memory_order_acquire) != gen) {
            for (FILE* f : {ff, fp, fg, fs})  // stop() or a newer session: close the files and end the thread
                if (f) fclose(f);
            g_writer_busy.store(false, std::memory_order_release);
            return;
        }
        uint64_t t = now_ns();
        double secs = (t - last) / 1e9;
        last = t;
        fr.clear();
        pr.clear();
        {
            std::lock_guard<std::mutex> l(g_rows_mu);
            std::swap(fr, g_frames);
            std::swap(pr, g_presents);
        }
        if (ff)
            for (const Row& r : fr)
                fprintf(ff, "%.1f,%.2f,%.2f,%.2f,%.2f,%llu,%llu,%.2f,%.2f,%llu,%llu,%.3f\n", (r.t - g_t0) / 1e6, r.dt / 1e6,
                        r.cpu / 1e6, r.c[0] / 1e6, r.c[1] / 1e6, (unsigned long long)r.c[2], (unsigned long long)(r.c[3] / 1024),
                        r.c[4] / 1e6, r.c[5] / 1e6, (unsigned long long)r.c[6], (unsigned long long)r.c[7], r.c[8] / 1e6);
        if (fp)
            for (const Row& r : pr)
                fprintf(fp, "%.1f,%.2f,%.2f,%llu,%llu,%.2f,%.2f,%.2f,%.2f,%llu,%llu,%.2f,%llu,%.2f,%.2f,%llu,%llu,%llu,%llu,%llu\n",
                        (r.t - g_t0) / 1e6, r.dt / 1e6, r.cpu / 1e6, (unsigned long long)r.draws, (unsigned long long)r.c[0],
                        r.c[1] / 1e6, r.c[2] / 1e6, r.c[3] / 1e6, r.c[4] / 1e6, (unsigned long long)r.c[5],
                        (unsigned long long)r.c[6], r.c[7] / 1e6, (unsigned long long)r.c[8], r.c[9] / 1e6, r.c[10] / 1e6,
                        (unsigned long long)r.c[11], (unsigned long long)r.c[12], (unsigned long long)r.c[13],
                        (unsigned long long)r.c[14], (unsigned long long)r.c[15]);
        {  // GPU pass groups: this second's top 30 by GPU time, and the session totals
            std::vector<GpuPassStat> batch;
            {
                std::lock_guard<std::mutex> l(g_gpu_mu);
                batch.swap(g_gpu_pending);
            }
            if (!batch.empty()) gpu_secs++;
            std::unordered_map<std::string, GpuCum> secmap;
            for (auto& s : batch) {
                GpuCum& a = secmap[s.name];
                a.ns += s.ns, a.passes += s.passes, a.draws += s.draws;
                GpuCum& c = g_gpu_cum[s.name];
                c.ns += s.ns, c.passes += s.passes, c.draws += s.draws;
            }
            std::vector<std::pair<uint64_t, const std::string*>> order;
            for (auto& [name, a] : secmap) order.push_back({a.ns, &name});
            std::sort(order.rbegin(), order.rend());
            if (fg)
                for (size_t i = 0; i < order.size() && i < 30; i++) {
                    const GpuCum& a = secmap[*order[i].second];
                    fprintf(fg, "%d,%.3f,%llu,%llu,\"%s\"\n", sec, a.ns / 1e6, (unsigned long long)a.passes,
                            (unsigned long long)a.draws, order[i].second->c_str());
                }
        }

        double proc_pct = 0;
        std::string detail;
        std::string threads = scan_threads(secs, &proc_pct, &detail);
        std::string busy = cpu_busy();
        if (g_prof_on) {
            drain_samples();
            attribute_spikes(fr);
        }
        long long mhz_sum = 0, mhz_max = 0;
        int mhz_n = 0;
        for (auto& p : g_sys.cpufreq) {
            long long k = read_ll(p);
            if (k <= 0) continue;
            mhz_sum += k / 1000;
            mhz_max = std::max(mhz_max, k / 1000);
            mhz_n++;
        }
        unsigned long long minflt, majflt;
        proc_faults(&minflt, &majflt);
        char statm[128];
        long long rss_pages = 0;
        if (read_file("/proc/self/statm", statm, sizeof statm)) sscanf(statm, "%*s %lld", &rss_pages);
        long long vram = read_ll(g_sys.vram), gtt = read_ll(g_sys.gtt), pw = read_ll(g_sys.power),
                  cap = read_ll(g_sys.power_cap), temp = read_ll(g_sys.temp), bat = read_ll(g_sys.bat_power);
        std::string core_waits;
        {
            struct Agg {
                uint64_t ns = 0, n = 0;
            };
            std::unordered_map<const void*, std::unordered_map<const void*, Agg>> w;  // waiter -> holder
            uint64_t head = g_exec_head.load(std::memory_order_acquire);
            head = head > 8 ? head - 8 : 0;  // the newest slots may still be being written
            if (head > g_exec_tail + kExecEvents) g_exec_tail = head - kExecEvents;
            for (; g_exec_tail < head; g_exec_tail++) {
                ExecWaitEvent& e = g_exec_events[g_exec_tail & (kExecEvents - 1)];
                Agg& a = w[e.waiter.load(std::memory_order_relaxed)][e.holder.load(std::memory_order_relaxed)];
                a.ns += e.ns.load(std::memory_order_relaxed);
                a.n++;
            }
            std::vector<std::pair<uint64_t, std::string>> top;
            {
                std::lock_guard<std::mutex> l(threads::g_klock);  // thread names change under it (OSSetThreadName)
                auto name = [](const void* p, const char* none) {
                    return p ? clean_name(((const threads::GuestThread*)p)->name) : std::string(none);
                };
                for (auto& [wt, holders] : w)
                    for (auto& [ht, a] : holders) {
                        char ms[32];
                        snprintf(ms, sizeof ms, "%.1f/%llu", a.ns / 1e6, (unsigned long long)a.n);
                        top.push_back({a.ns, name(wt, "host") + "<" + name(ht, "(queue)") + ":" + ms});
                    }
            }
            std::sort(top.rbegin(), top.rend());
            for (size_t i = 0; i < top.size() && i < 8; i++) core_waits += (i ? ";" : "") + top[i].second;
        }
        if (fs) {
            fprintf(fs, "%d,%.1f,%.1f,%.0f,%lld,%lld,%lld,%d,%.2f,%.2f,%.1f,%.2f,%lld,%lld,%lld,%lld,%lld,%lld,%llu,%llu,%llu,%s,%s,%llu,%s,%s,%d\n",
                    sec, fr.size() / secs, pr.size() / secs, proc_pct, mhz_n ? mhz_sum / mhz_n : -1, mhz_n ? mhz_max : -1,
                    read_ll(g_sys.gpu_busy), dpm_mhz(g_sys.sclk), pw >= 0 ? pw / 1e6 : -1.0, cap >= 0 ? cap / 1e6 : -1.0,
                    temp >= 0 ? temp / 1000.0 : -1.0, bat >= 0 ? bat / 1e6 : -1.0, read_ll(g_sys.bat_cap),
                    rss_pages * page_kb / 1024, meminfo_kb("MemAvailable:") / 1024, meminfo_kb("SwapFree:") / 1024,
                    vram >= 0 ? vram >> 20 : -1, gtt >= 0 ? gtt >> 20 : -1, minflt - last_min, majflt - last_maj,
                    (unsigned long long)g_prof_total, threads.c_str(), core_waits.c_str(),
                    (unsigned long long)g_ctr[Preempts].exchange(0, std::memory_order_relaxed), busy.c_str(), detail.c_str(),
                    dpm_mhz(g_sys.fclk));
        }
        last_min = minflt;
        last_maj = majflt;
        if (ff) fflush(ff);
        if (fp) fflush(fp);
        if (fs) fflush(fs);
        if (fg) fflush(fg);
        if (sec % 30 == 0) {
            write_profile();
            write_gpu_passes(gpu_secs);
        }
    }
}
}  // namespace

std::mutex g_start_mu;  // start/stop come from boot and from the menu

void set_root(const char* user_root) {
    std::lock_guard<std::mutex> l(g_start_mu);
    g_root = user_root;
}

// Boot (setting on) and the menu (switched on mid-game): a new session at once, no restart.
void start(const char* user_root) {
    std::lock_guard<std::mutex> l(g_start_mu);
    if (user_root) g_root = user_root;
    const char* e = getenv("BOTW_TELEMETRY");
    if ((e && e[0] == '0') || g_root.empty() || g_on.load(std::memory_order_acquire)) return;
    std::string dir = g_root + "/telemetry";
    std::error_code ec;
    std::filesystem::create_directories(dir, ec);
    char stamp[32];
    time_t now = time(nullptr);
    strftime(stamp, sizeof stamp, "%Y%m%d-%H%M%S", localtime(&now));
    g_prefix = dir + "/" + stamp + "-";
    static bool paths_found = false;
    if (!paths_found) find_sys_paths(), paths_found = true;
    write_info();
    g_frames.reserve(512);
    g_presents.reserve(512);
    g_t0 = now_ns();
    const uint32_t gen = g_gen.fetch_add(1, std::memory_order_acq_rel) + 1;
    g_on.store(true, std::memory_order_release);
    g_enabled.store(true, std::memory_order_release);
    const char* p = getenv("BOTW_PROFILE");
    if (!(p && p[0] == '0')) {
        struct sigaction sa = {};
        sa.sa_sigaction = on_prof;
        sa.sa_flags = SA_SIGINFO | SA_RESTART;
        sigemptyset(&sa.sa_mask);
        sigaction(SIGPROF, &sa, nullptr);
        itimerval it{};
        it.it_interval.tv_usec = kProfileUs;
        it.it_value.tv_usec = kProfileUs;
        g_prof_on = setitimer(ITIMER_PROF, &it, nullptr) == 0;
    }
    std::thread(writer, gen).detach();
    LOG("telemetry: %s* (profiler %s)", g_prefix.c_str(), g_prof_on ? "on" : "off");
}

void stop() {
    std::lock_guard<std::mutex> l(g_start_mu);
    g_enabled.store(false, std::memory_order_release);  // every recording site checks this first (telemetry.h)
    if (!g_on.exchange(false, std::memory_order_acq_rel)) return;
    if (g_prof_on) {
        itimerval off{};
        setitimer(ITIMER_PROF, &off, nullptr);  // no more SIGPROF samples
        g_prof_on = false;
    }
    LOG("telemetry: off (menu)");
}

void gpu_pass_stats(std::vector<GpuPassStat>&& one_second) {
    if (!g_on.load(std::memory_order_relaxed)) return;
    std::lock_guard<std::mutex> l(g_gpu_mu);
    for (auto& s : one_second) g_gpu_pending.push_back(std::move(s));
}

void note_exec_wait(const void* waiter, const void* holder, uint64_t ns, bool is_main) {
    if (!g_on.load(std::memory_order_relaxed)) return;
    if (is_main) add(MainWaitCoreNs, ns);
    ExecWaitEvent& e = g_exec_events[g_exec_head.fetch_add(1, std::memory_order_relaxed) & (kExecEvents - 1)];
    e.waiter.store(waiter, std::memory_order_relaxed);
    e.holder.store(holder, std::memory_order_relaxed);
    e.ns.store(ns, std::memory_order_relaxed);
}

void on_guest_frame() {
    if (!g_on.load(std::memory_order_relaxed)) return;
    static thread_local uint64_t last_t = 0, last_cpu = 0;
    record(g_frames, kMainCtrs, last_t, last_cpu, 0);
}

void on_present(uint32_t draw_counter) {
    if (!g_on.load(std::memory_order_relaxed)) return;
    static thread_local uint64_t last_t = 0, last_cpu = 0;
    static thread_local uint32_t last_draws = 0;
    uint32_t draws = draw_counter - last_draws;
    last_draws = draw_counter;
    record(g_presents, kGpuCtrs, last_t, last_cpu, draws);
}
#endif
}  // namespace telem
