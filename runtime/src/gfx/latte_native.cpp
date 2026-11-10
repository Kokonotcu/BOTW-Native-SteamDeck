// The latte:: interface (runtime/src/latte/latte.h) on the native renderer: the renderer is created on the host
// window, and the display timing the game paces itself by (flips, vsync, GX2 event callbacks) runs on a timer
// thread with Cemu's timing (LatteTiming.cpp), which the port's frame pacing and 60 fps mode were tuned with:
// vsync at 60 * 0.999 Hz; every `swap interval` vsyncs one requested flip executes (BotW: unless the game's swaps
// lag 4 behind), then the FLIP event, then the VSYNC event. The native render thread presents frames on its own,
// as Cemu's GPU thread did, independently of these counts.
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <thread>

#include "botw_target.h"
#include "gfx/renderer.h"
#include "host/window.h"
#include "latte/latte.h"
#include "log.h"
#include "ppc.h"
#include "telemetry.h"

namespace {
std::mutex g_sync_lock;
std::condition_variable g_flip_cv, g_vsync_cv;
uint64_t g_vsync_count = 0;  // under g_sync_lock
std::atomic<uint32_t> g_flip_requests{0}, g_flips_executed{0}, g_swap_interval{1};
std::atomic<uint64_t> g_last_flip_time{0}, g_last_vsync_time{0}, g_frames{0};
std::function<void(latte::Event)> g_sink;

// LatteTiming_signalVsync
void signal_vsync() {
    static uint32_t intervalCounter = 0;
    if (++intervalCounter >= g_swap_interval.load(std::memory_order_relaxed)) {
        const uint32_t requested = g_flip_requests.load(), executed = g_flips_executed.load();
        if (requested >= executed || executed - requested < 4) g_flips_executed.fetch_add(1);
        g_last_flip_time = rt_timebase();
        const uint64_t f = ++g_frames;
        window::on_frame(f);
        if (telem::on() && f % 300 == 0) log_write("frame %llu", (unsigned long long)f);  // progress line: telemetry
        {
            std::lock_guard<std::mutex> l(g_sync_lock);
            g_flip_cv.notify_all();
        }
        if (g_sink) g_sink(latte::Event::FLIP);
        intervalCounter = 0;
    }
    g_last_vsync_time = rt_timebase();
    {
        std::lock_guard<std::mutex> l(g_sync_lock);
        g_vsync_count++;
        g_vsync_cv.notify_all();
    }
    if (g_sink) g_sink(latte::Event::VSYNC);
}

void vsync_thread() {
    botw_set_thread_name("vsync");
    const auto period = std::chrono::nanoseconds(1000000000ll * 1002 / 60000);  // 60 * 0.999 Hz, LatteTime_CalculateTimeBetweenVSync
    auto next = std::chrono::steady_clock::now() + period;
    for (;;) {
        std::this_thread::sleep_until(next);
        signal_vsync();
        next += period;
        const auto now = std::chrono::steady_clock::now();
        if (now - next > period * 4) next = now + period;  // the process was stopped: no burst of catch-up vsyncs
    }
}
}  // namespace

namespace latte {
void select_backend(const char*) {}  // the native renderer is Vulkan only
bool using_vulkan() { return true; }
void set_host_surface(HostSurface, void*, void*) {}  // the renderer takes the SDL window itself (create_renderer)
void create_renderer(void*, int, int) { render::init(window::g_win_ptr()); }
void resize_window(int, int) {}  // the swapchain follows the window (recreated when it is out of date)
bool present_empty_frame() { return true; }
void set_scanout_srgb(bool, bool) {}  // GX2SetTVBuffer passes the scan-out format to the renderer itself
void set_display_mode(int) {}
void set_output_filter(int) {}
void set_gamepad_view(bool) {}
void set_capture(int, const char*) {}
void init() {
    static std::once_flag once;
    std::call_once(once, [] { std::thread(vsync_thread).detach(); });
}
uint32_t flip_requests() { return g_flip_requests.load(); }
uint32_t flips_executed() { return g_flips_executed.load(); }
void add_flip_request() { g_flip_requests.fetch_add(1); }
void wait_for_flip() {
    std::unique_lock<std::mutex> l(g_sync_lock);
    if (flip_requests() == flips_executed()) return;
    g_flip_cv.wait_for(l, std::chrono::milliseconds(100));
}
void wait_for_vsync() {
    std::unique_lock<std::mutex> l(g_sync_lock);
    const uint64_t v = g_vsync_count;
    g_vsync_cv.wait_for(l, std::chrono::milliseconds(100), [&] { return g_vsync_count != v; });
}
uint64_t last_flip_time() { return g_last_flip_time; }
uint64_t last_vsync_time() { return g_last_vsync_time; }
uint32_t swap_interval() { return g_swap_interval.load(); }
void set_swap_interval(uint32_t v) { g_swap_interval = v ? v : 1; }
void set_event_sink(std::function<void(Event)> sink) { g_sink = std::move(sink); }
}  // namespace latte
