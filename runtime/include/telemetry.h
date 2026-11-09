// Performance recorder for tuning on real hardware (Linux). A background thread writes, once a second,
// under <user dir>/telemetry/<session>-*:
//   frames.csv   one row per game frame (GX2SwapScanBuffers): frame time, main-thread CPU time and waits
//   present.csv  one row per presented image (GPU thread): GPU waits, presents, new shaders/pipelines
//   seconds.csv  per second: CPU% per thread, CPU/GPU clocks, GPU busy, power, temperature, memory
//   profile.txt  sampled hot code per thread (module+offset; BotWRecomp offsets match the .debug file)
//   gpu_passes.csv / gpu_passes.txt  GPU time per render pass group, per second / whole session
// Hot paths only add to relaxed atomics. BOTW_TELEMETRY=0 turns it all off, BOTW_PROFILE=0 the sampler.
#pragma once
#include <stdint.h>
#include <time.h>

#include <atomic>
#include <string>
#include <vector>

namespace telem {
enum Counter : int {
    // main thread (frames.csv)
    MainWaitGpuNs,    // blocked on the GPU, flips or vsync
    MainWaitOtherNs,  // blocked on other guest threads (mutex, event, message queue, ...)
    MainWaitCoreNs,   // waiting for emulated core 1 while another guest thread ran on it (overlaps the waits above)
    MainWaits,        // blocking waits this frame
    MainWakes,        // ... of those ended by a signal from another guest thread
    MainWakeLatNs,    // signal -> main thread running again, summed over those
    GpuTimeNs,        // GPU execution time of the command buffers that finished (GPU thread, timestamps)
    GpuPasses,        // render passes begun
    GpuBarriers,      // pipeline barriers recorded (each can drain the GPU pipeline)
    GpuUploads,       // buffer -> image copies (texture uploads)
    GpuCopies,        // image -> image copies and blits
    GpuClears,        // color / depth image clears
    Preempts,         // background threads that handed their core to a waiting thread (seconds.csv)
    FsReads,
    FsBytes,
    FsNs,
    // GPU thread (present.csv)
    Submits,
    SubmitNs,         // vkQueueSubmit
    FenceWaitNs,      // waiting for the GPU to finish earlier command buffers
    AcquireNs,        // swapchain image acquire
    PresentNs,        // vkQueuePresentKHR (and present wait)
    PipeNew,          // pipelines first used this frame
    PipeAsync,        // ... of those, handed to the compile threads (draw skipped meanwhile)
    PipeNs,           // GPU-thread time creating pipelines
    ShaderNew,        // shaders decompiled on the GPU thread
    ShaderNs,
    Count
};
extern std::atomic<uint64_t> g_ctr[Count];

// The in-game Telemetry switch (Graphics > Telemetry; BOTW_TELEMETRY=0 too). Every recording site in the program
// checks it first: switched off, nothing is timed, counted, sampled, logged or sent to the GPU for measurement, and
// only this one load + branch remains. start() sets it, stop() (menu) clears it.
extern std::atomic<bool> g_enabled;
inline bool on() { return g_enabled.load(std::memory_order_relaxed); }

inline uint64_t now_ns() {
    timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}
inline void add(Counter c, uint64_t v) {
    if (on()) g_ctr[c].fetch_add(v, std::memory_order_relaxed);
}

struct Scope {  // times its scope into a counter; reads no clock while telemetry is off
    Counter c;
    uint64_t t0;
    explicit Scope(Counter k) : c(k), t0(on() ? now_ns() : 0) {}
    ~Scope() {
        if (t0) add(c, now_ns() - t0);
    }
};

// GPU time per render pass group over about one second (GPU thread, VulkanRenderer.cpp) -> gpu_passes.csv/.txt
struct GpuPassStat {
    std::string name;
    uint64_t ns, passes, draws;
};
void gpu_pass_stats(std::vector<GpuPassStat>&& one_second);

void set_root(const char* user_root);  // boot: where sessions go, so the menu can start one later
void start(const char* user_root = nullptr);  // boot (setting on) or the menu switched on: a new session at once
void stop();                        // the menu switched off: everything stops at once (profiler, files, GPU timing)
// A guest thread (threads::GuestThread*) waited `ns` for its emulated core while `holder` ran on it (null:
// queued behind others). Lock-free; names are resolved on the writer thread. thread.cpp, slow path only.
void note_exec_wait(const void* waiter, const void* holder, uint64_t ns, bool is_main);
void on_guest_frame();              // game thread, GX2SwapScanBuffers
void on_present(uint32_t draw_counter);  // GPU thread, after the TV image was presented
}  // namespace telem
