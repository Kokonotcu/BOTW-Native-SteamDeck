#include <sys/resource.h>
#include "../bench/bench.h"
#include "../os.h"
#include "../thread.h"
#include "botw_target.h"
#include "latte/latte.h"
#include "telemetry.h"
#include <condition_variable>
#include <deque>
// GX2 core: command execution, display lists, context states, draws, clears,
// copies and presentation.
#include <algorithm>
#include <chrono>
#include <atomic>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <vector>

#include "gx2.h"
#include "gx2_cmd.h"
#include "gx2_regs.h"
#include "register_blocks.h"
#include "register_class.h"
#include "thread_identity.h"
#include "gx2_texture_regs.h"
#ifdef WWHD_HAS_VULKAN
#include "shader_key_dirty.h"
#include "gfx/vulkan/api.h"
#endif
#include "hle.h"
#include "guest_addr.h"
#include "gfx/renderer.h"

using namespace Latte;

namespace gx2 {

// ---------------------------------------------------------------- register file and context states
static uint32 g_regs[kNumRegs];
static uint32* g_shadow = nullptr;  // register copy of the active GX2ContextState
static std::unordered_map<uint32, std::vector<uint32>> g_contexts;
static std::recursive_mutex g_exec_mutex;
static RegisterBlocks<kNumRegs> g_register_blocks;

uint32* regs() { return g_regs; }

// Register writes that can change how shaders are translated bump g_shader_state_gen; the
// renderer reuses its last shader lookup while it is unchanged. Uniforms, uniform/vertex buffer
// addresses and rewrites of an identical value don't count.
extern "C" { uint64_t g_shader_state_gen = 1; }

// shader_irrelevant_reg (register_class.h); WWHD_VK_SHADER_ADDRESS_MEMO=1 (Vulkan) adds the texture
// address words, which bind_stage resolves freshly
static bool shader_address_memo() {
#ifdef WWHD_HAS_VULKAN
    const char* e = getenv("WWHD_VK_SHADER_ADDRESS_MEMO");
    return render::vulkan() && e && !strcmp(e, "1");
#else
    return false;
#endif
}

// ---------------------------------------------------------------- register classes
// register_class.h: per-register tables built on the first register command (the renderer is chosen by
// then), with the switches the apply paths read, so they are not looked up per write. Metal: whether a
// register write can change a shader key (metal_key_changes_ref). Draws rebind textures constantly, and
// every rebind used to make the next draw recompute both shader keys (~9% of the render thread in busy
// scenes); viewport, scissor, blend and stencil changes did the same. WWHD_SHADER_KEY_ALL_TEX=1: every
// shader-relevant write counts; WWHD_SHADER_KEY_ALL_REGS=1: every register outside the texture units.
static RegisterClasses g_reg_classes;
static const uint8* const g_reg_class = g_reg_classes.cls;
static bool g_reg_tables_ready = false;
#ifdef WWHD_HAS_VULKAN
static ShaderKeyBits g_vk_key_bits;   // vulkan_shader_key_mask with WWHD_VK_SHADER_KEY_DIRTY=1
static bool g_fused_small = false;    // Vulkan: the per-word path for writes of up to 16 words
static bool g_key_dirty = false;      // WWHD_VK_SHADER_KEY_DIRTY=1 (Vulkan)
static bool g_collect_stats = false;  // WWHD_VK_STATS (Vulkan)
#else
static constexpr bool g_collect_stats = false;
#endif

static void build_reg_tables() {
    MetalKeyOptions o;
    const char* wide = getenv("WWHD_MTL_WIDE_SHADER_KEYS");
    o.wideKeys = wide && strcmp(wide, "0");
    o.allTex = getenv("WWHD_SHADER_KEY_ALL_TEX") != nullptr;
    o.allRegs = getenv("WWHD_SHADER_KEY_ALL_REGS") != nullptr;
    g_reg_classes.build(shader_address_memo(), o, [](uint32) { return false; });
#ifdef WWHD_HAS_VULKAN
    const char* e = getenv("WWHD_VK_FUSE_SMALL_REGS");
    // Enabled by default; explicit zero retains the original bulk path.
    g_fused_small = render::vulkan() && (!e || strcmp(e, "0") != 0);
    e = getenv("WWHD_VK_SHADER_KEY_DIRTY");
    g_key_dirty = render::vulkan() && e && !strcmp(e, "1");
    g_collect_stats = render::vulkan() && getenv("WWHD_VK_STATS") != nullptr;
    if (g_key_dirty) g_vk_key_bits.build(vulkan_shader_key_mask);
#endif
    g_reg_tables_ready = true;
}
static inline void ensure_reg_tables() {
    if (__builtin_expect(!g_reg_tables_ready, 0)) build_reg_tables();
}
// a write (old != value) can change a shader key
static inline bool key_changes(uint32 reg, uint32 old, uint32 value) {
    if (g_reg_class[reg] & kRcShaderIrrelevant) return false;
#ifdef WWHD_HAS_VULKAN
    if (render::vulkan()) return !g_key_dirty || g_vk_key_bits.changed(reg, old, value);
#endif
    return g_reg_classes.metal_key_changes(reg, old, value, g_regs);
}

// register writes copy a few words mostly (display lists set registers in groups of 1-7): short ones
// are copied inline instead of calling memcpy
static inline void copy_words(uint32* d, const uint32* v, uint32 n) {
    if (n > 8) {
        memcpy(d, v, n * 4);
        return;
    }
    for (uint32 i = 0; i < n; ++i) d[i] = v[i];
}
// debug: WWHD_GEN_STATS=1 logs, every 20000 bumps, which registers bumped the shader generation
static void note_gen_bump(uint32 reg) {
    static const bool on = getenv("WWHD_GEN_STATS") != nullptr;
    if (!on) return;
    static std::unordered_map<uint32, uint32> tally;
    static uint32 total = 0;
    tally[reg]++;
    if (++total % 20000) return;
    std::vector<std::pair<uint32, uint32>> v(tally.begin(), tally.end());
    std::sort(v.begin(), v.end(), [](auto& a, auto& b) { return a.second > b.second; });
    std::string line;
    for (size_t i = 0; i < v.size() && i < 16; i++) {
        char b[32];
        snprintf(b, sizeof b, " %04X=%u", v[i].first, v[i].second);
        line += b;
    }
    LOG("[genstats] %u bumps; by register:%s", total, line.c_str());
    tally.clear();
}
static ShaderKeyDirtyStats shaderKeyDirtyStats;
ShaderKeyDirtyStats shader_key_dirty_stats() { return shaderKeyDirtyStats; }

#ifdef WWHD_HAS_VULKAN
static void apply_small_regs(uint32 first, const uint32* v, uint32 n) {
    const bool keyDirty = g_key_dirty, collectStats = g_collect_stats;
    bool changed = false, baselineBump = false, actualBump = false;
    uint64_t maskedWords = 0;
    for (uint32 i = 0; i < n; ++i) {
        const uint32 reg = first + i, value = v[i], old = g_regs[reg];
        if (old != value) {
            const uint8 cls = g_reg_class[reg];
            changed = true;
            if ((collectStats || !actualBump) && !(cls & kRcShaderIrrelevant)) {
                baselineBump = true;
                if (keyDirty && !g_vk_key_bits.changed(reg, old, value)) {
                    if (collectStats) ++maskedWords;
                } else actualBump = true;
            }
            g_regs[reg] = value;
        }
    }
    // Shadow can differ even when registers already match: draw mutates
    // primitive type directly, and context setup initializes only shadow.
    if (g_shadow && g_shadow != v) copy_words(&g_shadow[first], v, n);
    if (actualBump) ++g_shader_state_gen;
    if (changed && collectStats) {
        ++shaderKeyDirtyStats.changedBatches;
        shaderKeyDirtyStats.baselineWouldBumps += baselineBump;
        shaderKeyDirtyStats.actualBumps += actualBump;
        shaderKeyDirtyStats.avoidedBumps += baselineBump && !actualBump;
        shaderKeyDirtyStats.maskedWords += maskedWords;
    }
}
#endif

static void apply_regs(uint32 first, const uint32* v, uint32 n) {
    if (first >= kNumRegs || n > kNumRegs - first) return;
    ensure_reg_tables();
    g_register_blocks.touch(first, n);
    // ALU constants (uniform registers, most of the register traffic): shader-irrelevant and a
    // buffer/constant word for the draw classifier, nothing to decide per word: compare and copy.
    // (Not with WWHD_VK_STATS, which counts these batches too.)
    if (first >= mmSQ_ALU_CONSTANT0_0 && first + n <= mmSQ_ALU_CONSTANT0_0 + 0x1000 && !g_collect_stats) {
        uint32* d = &g_regs[first];
        if (memcmp(d, v, n * 4) != 0) {
            copy_words(d, v, n);
        }
        if (g_shadow && g_shadow != v) copy_words(&g_shadow[first], v, n);
        return;
    }
#ifdef WWHD_HAS_VULKAN
    // Vulkan renderer only (the Metal renderer keeps the original bulk path)
    if (g_fused_small && n <= 16) {
        apply_small_regs(first, v, n);
        return;
    }
#endif
    if (memcmp(&g_regs[first], v, n * 4) != 0) {
#ifdef WWHD_HAS_VULKAN
        if(g_key_dirty || g_collect_stats) {
            const bool collectStats = g_collect_stats;
            bool baselineBump = false, actualBump = false;
            uint64_t maskedWords = 0;
            for(uint32 i = 0; i < n; ++i) {
                uint32 reg = first + i;
                if(g_regs[reg] == v[i] || (g_reg_class[reg] & kRcShaderIrrelevant)) continue;
                baselineBump = true;
                if(g_key_dirty && !g_vk_key_bits.changed(reg, g_regs[reg], v[i])) {
                    if(collectStats) ++maskedWords;
                } else actualBump = true;
                if(actualBump && !collectStats) break;
            }
            if(actualBump) ++g_shader_state_gen;
            if(collectStats) {
                ++shaderKeyDirtyStats.changedBatches;
                shaderKeyDirtyStats.baselineWouldBumps += baselineBump;
                shaderKeyDirtyStats.actualBumps += actualBump;
                shaderKeyDirtyStats.avoidedBumps += baselineBump && !actualBump;
                shaderKeyDirtyStats.maskedWords += maskedWords;
            }
            memcpy(&g_regs[first], v, n * 4);
        } else
#endif
        {
            // one pass over the changed span: the shader-generation check (by table) and only the
            // changed words copied
            uint32 lo = 0;
            while (g_regs[first + lo] == v[lo]) lo++;  // memcmp found a difference
            uint32 hi = n - 1;
            while (g_regs[first + hi] == v[hi]) hi--;
            for (uint32 i = lo; i <= hi; i++)
                if (g_regs[first + i] != v[i] && key_changes(first + i, g_regs[first + i], v[i])) {
                    g_shader_state_gen++;
                    note_gen_bump(first + i);
                    break;
                }
            copy_words(&g_regs[first + lo], v + lo, hi - lo + 1);
        }
    }
    if (g_shadow && g_shadow != v) copy_words(&g_shadow[first], v, n);
}

// ---------------------------------------------------------------- display list recording
// Display lists being recorded: one slot per recording thread, found by the thread's identity
// (pthread_self) instead of a thread_local, which cost a TLS lookup on every emit of every thread. A
// slot is written only by its owner; an emit while no thread records skips the scan. Ported from
// ZeldaWWHDRecomp bde7fb1 (idea from the GreenNaugahyde Android fork v0.9.5).
struct alignas(64) Recording {
    uint32 start = 0, pos = 0, end = 0;
};
constexpr int kMaxRecorders = 64;
alignas(64) static std::atomic<uintptr_t> g_rec_owner[kMaxRecorders];
static Recording g_recs[kMaxRecorders];
alignas(64) static std::atomic<int> g_recording{0};  // slots in use
static std::atomic<int> g_rec_slots{0};               // slots ever claimed (the scan's end)
static inline Recording* find_rec(uintptr_t id) {
    for (int i = 0, n = g_rec_slots.load(std::memory_order_relaxed); i < n; i++)
        if (g_rec_owner[i].load(std::memory_order_relaxed) == id) return &g_recs[i];
    return nullptr;
}
// the display list this thread is recording, or nullptr
static inline Recording* recording() {
    if (!g_recording.load(std::memory_order_relaxed)) return nullptr;
    return find_rec(thread_identity());
}
static Recording* claim_rec() {
    const uintptr_t id = thread_identity();
    if (Recording* r = find_rec(id)) return r;
    for (int i = 0; i < kMaxRecorders; i++) {
        uintptr_t free = 0;
        if (g_rec_owner[i].compare_exchange_strong(free, id)) {
            int n = g_rec_slots.load();
            while (n < i + 1 && !g_rec_slots.compare_exchange_weak(n, i + 1)) {}
            g_recording++;
            return &g_recs[i];
        }
    }
    LOG("[gx2] too many threads recording display lists");
    return nullptr;
}
static void release_rec(Recording* r) {
    *r = Recording{};
    g_rec_owner[r - g_recs].store(0);
    g_recording--;
}
// a command of `words` words (header included) goes into the recording display list: where it is
// written, or nullptr if it doesn't fit (dropped)
static inline uint32* record_space(Recording* rec, uint32 words) {
    const uint32 bytes = 4 * words;
    if (rec->pos + bytes > rec->end) {
        LOG("[gx2] display list overflow at %08X", rec->start);
        return nullptr;
    }
    uint32* w = (uint32*)mem::host(rec->pos);
    rec->pos += bytes;
    return w;
}

static void execute_one(Op op, const uint32* p, uint32 n);

// ---------------------------------------------------------------- render thread
// Like the real GPU, command execution runs asynchronously to the game: GX2 calls append to a
// queue that a render thread turns into Vulkan work.
static std::mutex g_q_mutex;
static std::condition_variable g_q_cv;
static std::vector<uint32> g_q_pending, g_q_work;
static bool g_q_waiting = false;

static uint64_t g_fence_issued = 0;
// fences the render thread has reached: the game thread waits on it with atomic wait (a __ulock futex); a
// condition variable's broadcast cost the render thread ~0.3 ms in the kernel at every DrawDone (8% of it)
static std::atomic<uint64_t> g_fence_done{0};

static void render_thread_main() {
    botw_set_thread_name("GX2 render");
    botw_place_thread(1);  // the GPU pair's cpus (6-7) and priority, as Latte's GPU thread had
    for (;;) {
        {
            std::unique_lock<std::mutex> lk(g_q_mutex);
            g_q_waiting = true;
            if (g_q_pending.empty()) g_q_cv.wait(lk, [] { return !g_q_pending.empty(); });
            g_q_waiting = false;
            g_q_work.swap(g_q_pending);
        }
        {
            std::lock_guard<std::recursive_mutex> lk(g_exec_mutex);
            execute(g_q_work.data(), (uint32)g_q_work.size());
        }
        g_q_work.clear();
    }
}

static void start_render_thread() {
    static std::once_flag once;
    std::call_once(once, [] {
        std::thread([] {
            try {
                render_thread_main();
            } catch (const std::exception& e) {  // an unsupported GPU state: stop with the reason in the log
                LOG("FATAL: GX2 render thread: %s", e.what());
                abort();
            }
        }).detach();
    });
}
static void enqueue(Op op, const uint32* payload, uint32 n) {
    start_render_thread();
    bool wake;
    {
        std::lock_guard<std::mutex> lk(g_q_mutex);
        g_q_pending.push_back(op | (n << 8));
        g_q_pending.insert(g_q_pending.end(), payload, payload + n);
        wake = g_q_waiting;
    }
    if (wake) g_q_cv.notify_one();  // after unlocking: the woken render thread would block on the mutex at once
}
static void enqueue_regs(uint32 first, const uint32* values, uint32 count) {  // OP_SET_REGS {first, values...}
    start_render_thread();
    bool wake;
    {
        std::lock_guard<std::mutex> lk(g_q_mutex);
        g_q_pending.push_back(OP_SET_REGS | ((count + 1) << 8));
        g_q_pending.push_back(first);
        g_q_pending.insert(g_q_pending.end(), values, values + count);
        wake = g_q_waiting;
    }
    if (wake) g_q_cv.notify_one();
}

// block the game thread until the render thread has executed everything queued so far
static void render_sync() {
    uint64_t id;
    {
        std::lock_guard<std::mutex> lk(g_q_mutex);
        id = ++g_fence_issued;
    }
    uint32 w = (uint32)id;
    enqueue(OP_FENCE, &w, 1);
    for (uint64_t done; (done = g_fence_done.load(std::memory_order_acquire)) < id;) g_fence_done.wait(done, std::memory_order_acquire);
}

static std::atomic<uint64_t> g_draws_executed{0};  // render thread; the frame-rate code tells loading screens apart
void emit(Op op, const uint32* payload, uint32 n) {
    if (Recording* rec = recording(); rec && rec->start) {
        if (uint32* w = record_space(rec, n + 1)) {
            w[0] = op | (n << 8);
            memcpy(w + 1, payload, n * 4);
        }
        return;
    }
    enqueue(op, payload, n);
}

// host-only commands never go into display lists
static void emit_host(Op op, std::initializer_list<uint32> payload) {
    enqueue(op, payload.begin(), (uint32)payload.size());
}

#ifdef WWHD_HAS_VULKAN
void checkpoint_vulkan_caches() {
    // Wait for queued work, then exclude further renderer mutations while the
    // SDL thread writes the final cache checkpoint during orderly shutdown.
    render_sync();
    std::lock_guard<std::recursive_mutex> lk(g_exec_mutex);
    gfxvk::wait_idle();
    gfxvk::save_renderer_caches();
}
#endif

void set_regs(uint32 first, const uint32* values, uint32 count) {
    if (!count) return;
    // the command is the first register and the values: written in place (no staging copy)
    const uint32 n = count + 1;
    if (Recording* rec = recording(); rec && rec->start) {
        if (uint32* w = record_space(rec, n + 1)) {
            w[0] = OP_SET_REGS | (n << 8);
            w[1] = first;
            memcpy(w + 2, values, count * 4);
        }
        return;
    }
    if (g_render_thread) {
        enqueue_regs(first, values, count);
        return;
    }
    std::vector<uint32> buf(n);  // WWHD_NO_RENDER_THREAD (debug)
    buf[0] = first;
    memcpy(&buf[1], values, count * 4);
    emit(OP_SET_REGS, buf.data(), n);
}
void set_reg(uint32 reg, uint32 value) { emit(OP_SET_REGS, {reg, value}); }

void execute(const uint32* words, uint32 count) {
    uint32 i = 0;
    while (i < count) {
        uint32 hdr = words[i];
        Op op = (Op)(hdr & 0xFF);
        uint32 n = hdr >> 8;
        if (op >= OP_COUNT || i + 1 + n > count) {
            LOG("[gx2] corrupt display list command %08X", hdr);
            return;
        }
        execute_one(op, &words[i + 1], n);
        i += 1 + n;
    }
}

static void set_context(uint32 ctx) {
    if (!ctx) {
        g_shadow = nullptr;
        return;
    }
    auto it = g_contexts.find(ctx);
    if (it == g_contexts.end()) {
        g_shadow = nullptr;
        return;
    }
    g_shadow = it->second.data();
    // Draw writes primitive type without updating the context shadow.
    g_register_blocks.touch(uint32(REGADDR::VGT_PRIMITIVE_TYPE), 1);
    ensure_reg_tables();
    if (g_register_blocks.restore(g_regs, g_shadow, key_changes)) ++g_shader_state_gen;
}

constexpr uint32 kColorBufferWords = 0x9C / 4, kDepthBufferWords = 0xAC / 4, kSurfaceWords = 0x74 / 4;
// struct copies carried in a command, placed back in guest memory for the renderer (commands run
// one at a time, so a couple of fixed slots suffice)
static uint32 unpack_struct(const uint32* words, uint32 count, int slot) {
    static uint32 scratch = 0;
    if (!scratch) scratch = mem::host_alloc(2 * 0x100, 0x40);
    uint32 addr = scratch + slot * 0x100;
    memcpy(mem::host(addr), words, count * 4);
    return addr;
}

// GX2DrawDone queues the work instead of waiting for an idle GPU. WWHD kept the full wait on Metal
// (its renderer reads large vertex buffers straight from guest memory); in BotW that wait was 55% of
// the main thread's time. Flips still wait for the GPU frame (frames_completed), so the CPU stays at
// most a frame ahead. WWHD_LAZY_DRAW_DONE=0 (Vulkan: WWHD_VK_LAZY_DRAW_DONE=0) restores the full wait.
static bool lazy_draw_done() {
    static const bool on = [] {
        const char* e = getenv(render::vulkan() ? "WWHD_VK_LAZY_DRAW_DONE" : "WWHD_LAZY_DRAW_DONE");
        return !e || atoi(e) != 0;
    }();
    return on;
}

static void execute_op(Op op, const uint32* p, uint32 n);
static void execute_one(Op op, const uint32* p, uint32 n) { execute_op(op, p, n); }

static void execute_op(Op op, const uint32* p, uint32 n) {
    switch (op) {
    case OP_NOP: break;
    case OP_SET_REGS: apply_regs(p[0], p + 1, n - 1); break;
    case OP_DRAW: g_draws_executed.fetch_add(1, std::memory_order_relaxed); render::draw(g_regs, p[0], p[1], 0, 0, p[2], p[3]); break;
    case OP_DRAW_INDEXED:
        g_draws_executed.fetch_add(1, std::memory_order_relaxed);
        render::g_base_instance = n > 6 ? p[6] : 0;
        render::draw(g_regs, p[0], p[1], p[2], p[3], p[4], p[5]);
        render::g_base_instance = 0;
        break;
    case OP_CLEAR_COLOR: {
        const uint32* q = p + kColorBufferWords;
        float rgba[4] = {bitsf(q[0]), bitsf(q[1]), bitsf(q[2]), bitsf(q[3])};
        render::clear_color(g_regs, unpack_struct(p, kColorBufferWords, 0), rgba);
        break;
    }
    case OP_CLEAR_DEPTH: {
        const uint32* q = p + kDepthBufferWords;
        render::clear_depth_stencil(g_regs, unpack_struct(p, kDepthBufferWords, 0), bitsf(q[0]), q[1], q[2]);
        break;
    }
    case OP_CLEAR_BUFFERS: {
        uint32 cb = unpack_struct(p, kColorBufferWords, 0), db = unpack_struct(p + kColorBufferWords, kDepthBufferWords, 1);
        const uint32* q = p + kColorBufferWords + kDepthBufferWords;
        float rgba[4] = {bitsf(q[0]), bitsf(q[1]), bitsf(q[2]), bitsf(q[3])};
        render::clear_color(g_regs, cb, rgba);
        render::clear_depth_stencil(g_regs, db, bitsf(q[4]), q[5], q[6]);
        break;
    }
    case OP_COPY_SURFACE: {
        // debug: WWHD_GX2_DELAY_COPY=ms stalls the render thread before each surface copy (a slow
        // or busy render thread; reproduced the agl boot crash every time before GX2CopySurface waited)
        static const int delay = getenv("WWHD_GX2_DELAY_COPY") ? atoi(getenv("WWHD_GX2_DELAY_COPY")) : 0;
        if (delay) std::this_thread::sleep_for(std::chrono::milliseconds(delay));
        uint32 src = unpack_struct(p, kSurfaceWords, 0);
        const uint32* q = p + kSurfaceWords;
        uint32 dst = unpack_struct(q + 2, kSurfaceWords, 1);
        const uint32* r = q + 2 + kSurfaceWords;
        render::copy_surface(src, q[0], q[1], dst, r[0], r[1]);
        break;
    }
    case OP_COPY_TO_SCAN: render::copy_to_scan(unpack_struct(p, kColorBufferWords, 0), p[kColorBufferWords]); break;
    case OP_CALL: execute((const uint32*)mem::host(p[0]), p[1] / 4); break;
    case OP_SET_CONTEXT: set_context(p[0]); break;
    case OP_INVALIDATE: render::invalidate(p[0], p[1], p[2]); break;
    case OP_EXPAND_COLOR: case OP_EXPAND_DEPTH: break;  // MSAA/HiZ decompression: nothing to do on the host
    case OP_PEEK_Z: render::peek_z(p, n); break;
    case OP_FLUSH: render::guest_flush(); break;  // Vulkan: asynchronous submission
    case OP_DRAW_DONE:
        // The Vulkan renderer writes GPU results back to guest memory only for linear surfaces (guest data is copied
        // into fenced upload slices when work is recorded), so GX2DrawDone needs this op executed
        // (render_sync in the HLE), not an idle GPU. Lazy DrawDone (lazy_draw_done(), the default)
        // queues the work instead of waiting for the whole device every frame. A payload word of 1
        // (save states) always waits for the idle GPU.
        if (lazy_draw_done() && !(n && p[0])) render::guest_flush();
        else render::wait_idle();
        // except what the CPU reads back: linear render targets (the Picto Box picture, issue #53)
        render::write_back();
        break;
    case OP_SWAP:
        if (n) render::set_frame_aspect(gx2::bitsf(p[0]));  // aspect ratio from the next frame on (aspect.cpp)
        if (n >= 3 && p[2]) render::request_capture();
        render::swap();
        break;
    case OP_SET_PROJ_REGS: {
        uint32 v[16];
        memcpy(v, p + 1, sizeof v);
        float kx, ky;
        if (n == 17 && render::target_aspect_factors(g_regs[mmCB_COLOR0_TILE] & 0xFFFF, g_regs[mmCB_COLOR0_FRAG], kx, ky))
            for (int i = 0; i < 4; i++) {  // rows x and y: the 16:9 layout space centred in the wider picture
                v[i] = fbits(bitsf(v[i]) / kx);
                v[4 + i] = fbits(bitsf(v[4 + i]) / ky);
            }
        apply_regs(p[0], v, std::min<uint32>(n - 1, 16));
        break;
    }
    case OP_WAIT_MEM: {
        // the command processor waits until guest memory meets the condition (BotW: its vsync counter,
        // kept by the vsync callback, >= 2: the game's 30 fps cap; without it menus ran at 60 fps
        // and their animations at double speed). Work recorded so far is submitted first; 100 ms at
        // most, so a counter nobody updates can't hang the GPU.
        render::guest_flush();
        const auto give_up = std::chrono::steady_clock::now() + std::chrono::milliseconds(100);
        auto met = [&] {
            const uint32 v = mem::r32(p[0]) & p[1], ref = p[3];
            switch (p[2] & 7) {
            case 0: return false;
            case 1: return v < ref;
            case 2: return v == ref;
            case 3: return v <= ref;
            case 4: return v > ref;
            case 5: return v != ref;
            case 6: return v >= ref;
            default: return true;
            }
        };
        while (!met() && std::chrono::steady_clock::now() < give_up) std::this_thread::sleep_for(std::chrono::microseconds(250));
        break;
    }
    case OP_QUERY_BEGIN:
        if (n >= 1) render::query_begin(p[0]);
        break;
    case OP_QUERY_END:
        if (n >= 2) render::query_end(p[0], p[1]);
        break;
    case OP_STREAMOUT_UPDATE: {  // the buffer's write offset (bytes), in command order like the GPU's
        uint32 reg = mmVGT_STRMOUT_BUFFER_OFFSET_0 + 4 * (p[0] & 3);
        if (p[1] == 0) g_regs[reg] = 0;
        else if (p[1] == 1) g_regs[reg] = p[2] ? mem::r32(p[2]) : 0;
        else if (p[1] == 3 && p[2]) mem::w32(p[2], g_regs[reg]);
        break;
    }
    case OP_SETUP_CONTEXT:
        g_contexts[p[0]].assign(kNumRegs, 0);
        g_shadow = g_contexts[p[0]].data();
        break;
    case OP_FENCE: {
        uint64_t done = g_fence_done.load(std::memory_order_relaxed);
        while (done < p[0] && !g_fence_done.compare_exchange_weak(done, p[0], std::memory_order_release)) {}
        g_fence_done.notify_all();
        break;
    }
    default: break;
    }
}

// ---------------------------------------------------------------- default state
static void set_default_state() {
    // GX2SetShaderModeEx(UNIFORM_REGISTER, ...)
    LATTE_SQ_CONFIG sq;
    sq.set_DX9_CONSTS(true).set_ALU_INST_PREFER_VECTOR(true).set_PS_PRIO(3).set_VS_PRIO(2).set_GS_PRIO(1).set_ES_PRIO(0);
    set_reg(REGADDR::SQ_CONFIG, sq.getRawValue());
    set_reg(REGADDR::VGT_GS_MODE, 0);
    LATTE_PA_CL_VTE_CNTL vte{};
    vte.set_VPORT_X_OFFSET_ENA(true).set_VPORT_X_SCALE_ENA(true).set_VPORT_Y_OFFSET_ENA(true).set_VPORT_Y_SCALE_ENA(true);
    vte.set_VPORT_Z_OFFSET_ENA(true).set_VPORT_Z_SCALE_ENA(true).set_VTX_W0_FMT(true);
    set_reg(REGADDR::PA_CL_VTE_CNTL, vte.getRawValue());
    set_reg(REGADDR::DB_DEPTH_CONTROL, (1 << 1) | (1 << 2) | (1 << 4));  // z test + write, LESS
    LATTE_SX_ALPHA_TEST_CONTROL at;
    at.set_ALPHA_FUNC(LATTE_SX_ALPHA_TEST_CONTROL::E_ALPHA_FUNC::LESS).set_ALPHA_TEST_ENABLE(false);
    set_reg(REGADDR::SX_ALPHA_TEST_CONTROL, at.getRawValue());
    set_reg(REGADDR::SX_ALPHA_REF, 0);
    LATTE_PA_SU_SC_MODE_CNTL pm{};
    pm.set_FRONT_FACE(LATTE_PA_SU_SC_MODE_CNTL::E_FRONTFACE::CCW);
    pm.set_FRONT_POLY_MODE(LATTE_PA_SU_SC_MODE_CNTL::E_PTYPE::TRIANGLES).set_BACK_POLY_MODE(LATTE_PA_SU_SC_MODE_CNTL::E_PTYPE::TRIANGLES);
    set_reg(REGADDR::PA_SU_SC_MODE_CNTL, pm.getRawValue());
    set_reg(REGADDR::VGT_MULTI_PRIM_IB_RESET_INDX, 0xFFFFFFFF);
    set_reg(REGADDR::CB_TARGET_MASK, 0xFFFFFFFF);
    for (int i = 0; i < 4; i++) set_reg(REGADDR::CB_BLEND_RED + i, 0);
    set_reg(REGADDR::PA_SU_POINT_SIZE, LATTE_PA_SU_POINT_SIZE().set_WIDTH(8).set_HEIGHT(8).getRawValue());
    LATTE_CB_COLOR_CONTROL cc;
    cc.set_SPECIAL_OP(LATTE_CB_COLOR_CONTROL::E_SPECIALOP::NORMAL).set_ROP(LATTE_CB_COLOR_CONTROL::E_LOGICOP::COPY);
    set_reg(REGADDR::CB_COLOR_CONTROL, cc.getRawValue());
    LATTE_PA_CL_CLIP_CNTL clip{};
    clip.set_DX_LINEAR_ATTR_CLIP_ENA(true);
    set_reg(REGADDR::PA_CL_CLIP_CNTL, clip.getRawValue());
    set_reg(mmDB_DEPTH_CLEAR, fbits(1.0f));
}

}  // namespace gx2

using namespace gx2;

// ---------------------------------------------------------------- display timing
// Swaps, flips, vsync and the GX2 event callbacks use the port's display timing (latte::, gfx/latte_native.cpp), so
// its frame pacing and the 60 fps mode (game/fps.cpp) work as they did with the Latte GPU.
thread_local bool g_skip_vsync_wait = false;  // game/hooks.h: a 60 fps hook skips the next GX2WaitForVsync
namespace {
struct EventCallback {
    uint32 func = 0, user = 0;
} g_event_cb[5];
std::mutex g_ev_lock;
std::condition_variable g_ev_cv;
std::deque<uint32> g_ev_queue;

void event_thread() {
    threads::GuestThread* t = threads::adopt_host_thread("GX2 callbacks", 0x10000, 1);
    for (;;) {
        uint32 type;
        EventCallback cbk;
        {
            std::unique_lock<std::mutex> l(g_ev_lock);
            g_ev_cv.wait(l, [] { return !g_ev_queue.empty(); });
            type = g_ev_queue.front();
            g_ev_queue.pop_front();
            cbk = g_event_cb[type];
        }
        if (cbk.func) guest_call(&t->ctx, cbk.func, {type, cbk.user});
    }
}
}  // namespace

HLE_EXPORT(gx2, GX2Init) {
    set_default_state();
    LOG("[gx2] initialized (native GX2 -> %s)", render::api_name(render::active()));
    latte::set_event_sink([](latte::Event e) {
        const uint32 i = (uint32)e;
        if (i >= 5) return;
        std::lock_guard<std::mutex> l(g_ev_lock);
        if (g_event_cb[i].func) {
            g_ev_queue.push_back(i);
            g_ev_cv.notify_one();
        }
    });
    static std::once_flag started;
    std::call_once(started, [] { std::thread(event_thread).detach(); });
    latte::init();
}

HLE_EXPORT(gx2, GX2SetupContextStateEx) {
    uint32 ctx = ARG(0);
    emit_host(OP_SETUP_CONTEXT, {ctx});
    set_default_state();
    // the context's "restore" display list lives inside the (0xA100 byte) context structure
    uint32 dl = ctx + 0x9800;
    uint32* w = (uint32*)mem::host(dl);
    w[0] = OP_SET_CONTEXT | (1u << 8);
    w[1] = ctx;
}
HLE_EXPORT(gx2, GX2SetContextState) { emit(OP_SET_CONTEXT, {ARG(0)}); }
HLE_EXPORT(gx2, GX2GetContextStateDisplayList) {
    if (ARG(1)) mem::w32(ARG(1), ARG(0) + 0x9800);
    if (ARG(2)) mem::w32(ARG(2), 8);
}

// ---------------------------------------------------------------- display lists
HLE_EXPORT(gx2, GX2BeginDisplayListEx) {
    if (Recording* rec = claim_rec()) {
        rec->start = rec->pos = ARG(0);
        rec->end = ARG(0) + ARG(1);
        if (!rec->start) release_rec(rec);  // as before: a null list records nothing
    }
}
HLE_EXPORT(gx2, GX2EndDisplayList) {
    uint32 size = 0;
    if (Recording* rec = recording()) {
        size = rec->pos - rec->start;
        release_rec(rec);
    }
    RET(size);
}
HLE_EXPORT(gx2, GX2GetCurrentDisplayList) {
    const Recording* rec = recording();
    const uint32 start = rec ? rec->start : 0;
    if (ARG(0)) mem::w32(ARG(0), start);
    if (ARG(1)) mem::w32(ARG(1), start ? rec->end - rec->start : 0);
    RET(start != 0);
}
HLE_EXPORT(gx2, GX2CallDisplayList) { emit(OP_CALL, {ARG(0), ARG(1)}); }
// a copy into the command stream: run where it stands (our display lists are op lists, not PM4)
HLE_EXPORT(gx2, GX2CopyDisplayList) { emit(OP_CALL, {ARG(0), ARG(1)}); }
HLE_EXPORT(gx2, GX2SetDefaultState) { set_default_state(); }
HLE_EXPORT(gx2, GX2DirectCallDisplayList) { emit(OP_CALL, {ARG(0), ARG(1)}); }

// ---------------------------------------------------------------- draws
HLE_EXPORT(gx2, GX2DrawEx) { emit(OP_DRAW, {ARG(0), ARG(1), ARG(2), ARG(3)}); }
HLE_EXPORT(gx2, GX2DrawIndexedEx) { emit(OP_DRAW_INDEXED, {ARG(0), ARG(1), ARG(2), ARG(3), ARG(4), ARG(5)}); }
// Stream out (transform feedback, BotW). Registers as Cemu's GX2_Streamout.cpp; the strides come
// from GX2SetVertexShader. GX2StreamOutBuffer: +0 size, +4 dataPtr, +8 vertexStride, +0xC GX2RBuffer
// (+0x10 elementSize, +0x14 elementCount, +0x18 ptr; used when dataPtr is null), +0x1C ctxPtr.
HLE_EXPORT(gx2, GX2SetStreamOutEnable) { set_reg(mmVGT_STRMOUT_EN, ARG(0) & 1); }
HLE_EXPORT(gx2, GX2SetStreamOutBuffer) {
    uint32 i = ARG(0), b = ARG(1);
    if (i > 3 || !b) return;
    uint32 addr = mem::r32(b + 4), size = mem::r32(b);
    if (!addr) { addr = mem::r32(b + 0x18); size = mem::r32(b + 0x10) * mem::r32(b + 0x14); }
    set_reg(mmVGT_STRMOUT_BUFFER_SIZE_0 + i * 4, size >> 2);
    set_reg(mmVGT_STRMOUT_BUFFER_BASE_0 + i * 4, addr >> 8);
}
// (index, buffer, reset): reset != 0 starts writing at 0, otherwise continues from the saved context
HLE_EXPORT(gx2, GX2SetStreamOutContext) {
    uint32 i = ARG(0), b = ARG(1);
    if (i > 3) return;
    emit(OP_STREAMOUT_UPDATE, {i, ARG(2) ? 0u : 1u, b ? mem::r32(b + 0x1C) : 0u});
}
HLE_EXPORT(gx2, GX2SaveStreamOutContext) {
    uint32 i = ARG(0), b = ARG(1);
    if (i > 3 || !b) return;
    emit(OP_STREAMOUT_UPDATE, {i, 3u, mem::r32(b + 0x1C)});
}

// Occlusion queries. BotW reads one result on the CPU (GX2QueryGetOcclusionResult: the sun's lens
// flare, dimmed by how much of the sun is visible) and uses the rest for conditional rendering.
// The renderer counts the samples that pass between begin and end (Metal visibility results) and
// reports them once the GPU finished; until then the query is "not ready" and the game keeps its
// previous value, as on hardware. A query whose draws were skipped (shader still compiling) keeps
// the last count. Conditional rendering always draws (Metal has no predication; nothing is culled
// wrongly, some hidden things are drawn). WWHD_NO_QUERIES=1: every query "fully visible" (before).
static std::atomic<uint64_t> g_queries{0}, g_cond_renders{0};
extern "C" uint64_t wwhd_query_counts(uint64_t* conditional) {
    *conditional = g_cond_renders.exchange(0);
    return g_queries.exchange(0);
}
namespace {
struct QueryState {
    uint32 issued = 0, done = 0;  // generations: begun by the game, reported by the renderer
    uint64_t samples = 0x100000;  // until a real result: fully visible
    uint64_t issuedFrame = 0;     // a result that never comes (recorded in a display list never run):
};                                // ready with the last count after a second
std::mutex g_query_mu;
std::unordered_map<uint32, QueryState> g_query_state;
bool queries_off() {
    static const bool off = getenv("WWHD_NO_QUERIES") && atoi(getenv("WWHD_NO_QUERIES"));
    return off;
}
}  // namespace
extern "C" void wwhd_query_result(uint32_t addr, uint32_t gen, uint64_t samples, bool ok) {
    std::lock_guard<std::mutex> l(g_query_mu);
    QueryState& q = g_query_state[addr];
    if ((int32_t)(gen - q.done) <= 0) return;
    static const bool log = getenv("WWHD_QUERY_LOG") && atoi(getenv("WWHD_QUERY_LOG"));
    if (log) LOG("[query] %08X gen %u: %llu samples%s", addr, gen, (unsigned long long)samples, ok ? "" : " (lost, kept)");
    q.done = gen;
    if (ok) q.samples = samples;
}
HLE_EXPORT(gx2, GX2QueryBegin) {  // (type, GX2Query*)
    memset(mem::host(ARG(1)), 0, 0x40);
    g_queries++;
    if (queries_off()) return;
    emit(OP_QUERY_BEGIN, {ARG(1)});
}
HLE_EXPORT(gx2, GX2QueryEnd) {  // (type, GX2Query*)
    if (queries_off()) return;
    uint32 gen;
    {
        std::lock_guard<std::mutex> l(g_query_mu);
        QueryState& q = g_query_state[ARG(1)];
        gen = ++q.issued;
        q.issuedFrame = render::frame_count();
    }
    emit(OP_QUERY_END, {ARG(1), gen});
}
HLE_EXPORT(gx2, GX2QueryGetOcclusionResult) {  // (GX2Query*, u64* result) -> ready
    uint64_t samples = 0x100000;
    bool ready = true;
    if (!queries_off()) {
        std::lock_guard<std::mutex> l(g_query_mu);
        auto it = g_query_state.find(ARG(0));
        if (it != g_query_state.end()) {
            ready = it->second.done == it->second.issued || render::frame_count() > it->second.issuedFrame + 60;
            samples = it->second.samples;
        }
    }
    static const bool log = getenv("WWHD_QUERY_LOG") && atoi(getenv("WWHD_QUERY_LOG"));
    if (log) LOG("[query] read %08X: %s %llu", ARG(0), ready ? "ready" : "not ready", (unsigned long long)samples);
    if (!ready) { RET(0); return; }
    if (ARG(1)) { mem::w32(ARG(1), (uint32)(samples >> 32)); mem::w32(ARG(1) + 4, (uint32)samples); }
    RET(1);
}
HLE_EXPORT(gx2, GX2QueryBeginConditionalRender) { g_cond_renders++; }
HLE_EXPORT(gx2, GX2QueryEndConditionalRender) {}

HLE_EXPORT(gx2, GX2DrawIndexedEx2) {  // (prim, count, indexType, indices, baseVertex, instances, baseInstance)
    emit(OP_DRAW_INDEXED, {ARG(0), ARG(1), ARG(2), ARG(3), ARG(4), ARG(5), ARG(6)});
}

// ---------------------------------------------------------------- clears and copies
static float farg(PPCContext* c, int i) { return (float)c->f[1 + i].ps0; }
// Struct parameters are copied into the command (as GX2 encodes them into the command buffer):
// games reuse one GX2ColorBuffer/GX2DepthBuffer and change its view between calls, also while
// recording display lists that run later.
static void put_struct(std::vector<uint32>& p, uint32 addr, uint32 words) {
    size_t at = p.size();
    p.resize(at + words);
    if (addr) memcpy(&p[at], mem::host(addr), words * 4);
}
HLE_EXPORT(gx2, GX2ClearColor) {
    std::vector<uint32> p;
    put_struct(p, ARG(0), kColorBufferWords);
    for (int i = 0; i < 4; i++) p.push_back(fbits(farg(c, i)));
    emit(OP_CLEAR_COLOR, p.data(), (uint32)p.size());
}
HLE_EXPORT(gx2, GX2ClearDepthStencilEx) {
    std::vector<uint32> p;
    put_struct(p, ARG(0), kDepthBufferWords);
    p.insert(p.end(), {fbits(farg(c, 0)), ARG(1) & 0xFF, ARG(2)});
    emit(OP_CLEAR_DEPTH, p.data(), (uint32)p.size());
}
HLE_EXPORT(gx2, GX2ClearBuffersEx) {
    std::vector<uint32> p;
    put_struct(p, ARG(0), kColorBufferWords);
    put_struct(p, ARG(1), kDepthBufferWords);
    p.insert(p.end(), {fbits(farg(c, 0)), fbits(farg(c, 1)), fbits(farg(c, 2)), fbits(farg(c, 3)), fbits(farg(c, 4)),
                       ARG(2) & 0xFF, ARG(3)});
    emit(OP_CLEAR_BUFFERS, p.data(), (uint32)p.size());
}
HLE_EXPORT(gx2, GX2SetClearDepthStencil) {
    auto* db = (GX2::GX2DepthBuffer*)mem::host(ARG(0));
    db->clearDepth = farg(c, 0);
    db->clearStencil = ARG(1) & 0xFF;
}
HLE_EXPORT(gx2, GX2CopySurface) {
    // debug: WWHD_COPYDBG=1 logs each copy as issued (thread, caller, source and destination images)
    static const bool dbg = getenv("WWHD_COPYDBG") != nullptr;
    if (dbg)
        LOG("[copydbg] issue t=%.3f thread %08X lr %08X src %08X img %08X dst %08X img %08X size %X", timebase::now() / (double)timebase::kTicksPerSec,
            threads::current_thread(), c->lr, ARG(0), mem::r32(ARG(0) + 0x24), ARG(3), mem::r32(ARG(3) + 0x24), mem::r32(ARG(3) + 0x20));
    std::vector<uint32> p;
    put_struct(p, ARG(0), kSurfaceWords);
    p.insert(p.end(), {ARG(1), ARG(2)});
    put_struct(p, ARG(3), kSurfaceWords);
    p.insert(p.end(), {ARG(4), ARG(5)});
    emit(OP_COPY_SURFACE, p.data(), (uint32)p.size());
    // The copy is complete when GX2CopySurface returns: the game uses the result (and frees the
    // surfaces) right away. agl's tile-mode conversion (027B5EEC) copies into a temporary surface,
    // OSBlockMoves it back and frees it at once; executed later on the render thread, the copy
    // wrote into the freed memory after the heap had reused it (boot crash: agl shader program
    // array 21EFE28C, program 0's +0x7c zeroed). Not for display lists (they run when called).
    if (Recording* rec = recording(); !rec || !rec->start) {
        threads::CoreHandoff handoff;
        handoff.yield();
        render_sync();
    }
}
HLE_EXPORT(gx2, GX2CopyColorBufferToScanBuffer) {
    std::vector<uint32> p;
    put_struct(p, ARG(0), kColorBufferWords);
    p.push_back(ARG(1));
    emit(OP_COPY_TO_SCAN, p.data(), (uint32)p.size());
}
HLE_EXPORT(gx2, GX2ExpandAAColorBuffer) { emit(OP_EXPAND_COLOR, {ARG(0)}); }
HLE_EXPORT(gx2, GX2ExpandDepthBuffer) { emit(OP_EXPAND_DEPTH, {ARG(0)}); }
HLE_EXPORT(gx2, GX2Invalidate) { emit(OP_INVALIDATE, {ARG(0), ARG(1), ARG(2)}); }

// ---------------------------------------------------------------- submission and presentation
HLE_EXPORT(gx2, GX2Flush) { emit_host(OP_FLUSH, {}); }
HLE_EXPORT(gx2, GX2DrawDone) {
    emit_host(OP_DRAW_DONE, {});
    threads::WaitMark w("gpu: GX2DrawDone", 0);
    threads::CoreHandoff handoff;
    handoff.yield();
    render_sync();
    RET(1);
}
HLE_EXPORT(gx2, GX2SwapScanBuffers) {
    emit_host(OP_SWAP, {fbits(16.0f / 9.0f), 0u});
    latte::add_flip_request();
    bench::on_swap();
    if (telem::on()) telem::on_guest_frame();
    while ((int32_t)(latte::flip_requests() - latte::flips_executed()) > 5) {
        threads::WaitMark w("flip (swap backlog)", 0);
        threads::CoreHandoff handoff;
        handoff.yield();
        latte::wait_for_flip();
    }
}
HLE_EXPORT(gx2, GX2GetSwapStatus) {
    if (ARG(0)) mem::w32(ARG(0), latte::flip_requests());
    if (ARG(1)) mem::w32(ARG(1), latte::flips_executed());
    if (ARG(2)) mem::w64(ARG(2), latte::last_flip_time());
    if (ARG(3)) mem::w64(ARG(3), latte::last_vsync_time());
}
HLE_EXPORT(gx2, GX2SetSwapInterval) {
    if (ARG(0) < 20) latte::set_swap_interval(ARG(0));
}
// (address, mask, compare op, value): the GPU waits there until the condition holds (see OP_WAIT_MEM)
HLE_EXPORT(gx2, GX2SetGPUFence) {
    if (ARG(0)) emit(OP_WAIT_MEM, {ARG(0), ARG(1), ARG(2), ARG(3)});
}
HLE_EXPORT(gx2, GX2SetEventCallback) {
    const uint32_t type = ARG(0);
    if (type >= 5) return;
    std::lock_guard<std::mutex> l(g_ev_lock);
    g_event_cb[type] = {ARG(1), ARG(2)};
}
HLE_EXPORT(gx2, GX2GetEventCallback) {
    const uint32_t type = ARG(0);
    if (type >= 5) return;
    EventCallback cbk;
    {
        std::lock_guard<std::mutex> l(g_ev_lock);
        cbk = g_event_cb[type];
    }
    if (ARG(1)) mem::w32(ARG(1), cbk.func);
    if (ARG(2)) mem::w32(ARG(2), cbk.user);
}
HLE_EXPORT(gx2, GX2WaitForVsync) {
    if (g_skip_vsync_wait) {  // a game hook (60 fps) asked to skip this wait
        g_skip_vsync_wait = false;
        return;
    }
    threads::WaitMark w("vsync", 0);
    threads::CoreHandoff handoff;
    handoff.yield();
    latte::wait_for_vsync();
}

// scan buffers: the game renders into its own color buffers and copies to "scan buffers";
// the renderer presents whatever was copied to the TV target.
// (buffer, size, mode, surfaceFormat, bufferingMode): an sRGB format means scan-out applies the encoding
HLE_EXPORT(gx2, GX2SetTVBuffer) {
    LOG("[gx2] TV buffer format %X", ARG(3));
    render::set_tv_format(ARG(3), true);
}
HLE_EXPORT(gx2, GX2SetDRCBuffer) { render::set_tv_format(ARG(3), false); }
HLE_EXPORT(gx2, GX2SetTVScale) {}
HLE_EXPORT(gx2, GX2SetDRCScale) {}
HLE_EXPORT(gx2, GX2SetTVEnable) {}
HLE_EXPORT(gx2, GX2SetDRCEnable) {}
HLE_EXPORT(gx2, GX2CalcTVSize) {
    // (mode, format, bufferingMode, uint32* size, bool* scaleNeeded)
    uint32 mode = ARG(0), buffers = std::max<uint32>(ARG(2), 1);
    uint32 w = mode >= 5 ? 1920 : mode <= 2 ? 854 : 1280, h = mode >= 5 ? 1080 : mode <= 2 ? 480 : 720;
    mem::w32(ARG(3), w * h * 4 * buffers);
    mem::w32(ARG(4), 0);
}
HLE_EXPORT(gx2, GX2CalcDRCSize) {
    mem::w32(ARG(3), 854 * 480 * 4 * std::max<uint32>(ARG(2), 1));
    mem::w32(ARG(4), 0);
}

// ---------------------------------------------------------------- misc queries
HLE_EXPORT(gx2, GX2TempGetGPUVersion) { RET(2); }
HLE_EXPORT(gx2, GX2CalcGeometryShaderInputRingBufferSize) { RET(ARG(0) * 4 * 0x1000); }
HLE_EXPORT(gx2, GX2CalcGeometryShaderOutputRingBufferSize) { RET(ARG(0) * 4 * 0x1000); }
HLE_EXPORT(gx2, GX2CalcFetchShaderSizeEx) {
    uint32 n = ARG(0);
    uint32 cf = ((((n + 15) / 16) + 1) * 8 + 0xF) & ~0xFu;
    RET(std::max<uint32>(cf + n * 16, 16 + n * 16));
}
HLE_EXPORT(gx2, GX2GPUTimeToCPUTime) { ret64(c, arg64(c, 3)); }
HLE_EXPORT(gx2, GX2SampleTopGPUCycle) { if (ARG(0)) mem::w64(ARG(0), timebase::guest_now()); }
HLE_EXPORT(gx2, GX2SampleBottomGPUCycle) { if (ARG(0)) mem::w64(ARG(0), timebase::guest_now()); }
