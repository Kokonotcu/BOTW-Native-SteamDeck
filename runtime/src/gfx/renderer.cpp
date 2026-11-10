// Renderer start-up on the host window (Vulkan; see renderer.h).
#include "renderer.h"

#include <atomic>
#include <cstdlib>
#include <exception>

#include "hle.h"

// decompiler output flavour (gx2/decompiler_glue.cpp)
void select_decompiler_api(render::Api api);

namespace render {

const Backend* g_backend = nullptr;

const char* api_name(Api a) { return a == Api::Vulkan ? "Vulkan" : "Metal"; }
Api active() { return Api::Vulkan; }
uint64_t frame_count() { return g_backend ? g_backend->frame_count() : 0; }

void init(SDL_Window* window) {
    select_decompiler_api(Api::Vulkan);
    g_backend = &vulkan_backend();
    try {
        g_backend->init(window);
    } catch (const std::exception& e) {
        LOG("FATAL: the Vulkan renderer could not start: %s", e.what());
        abort();
    }
    LOG("[gfx] Vulkan renderer started");
}

void shutdown() {
    static std::atomic<bool> shut{false};
    if (shut.exchange(true) || !g_backend || !g_backend->shutdown) return;
    g_backend->shutdown();
}

}  // namespace render
