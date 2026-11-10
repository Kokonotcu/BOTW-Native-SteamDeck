// Definitions the vendored Cemu decompiler expects from its host.
#include "Cafe/HW/Latte/Renderer/Vulkan/VulkanRenderer.h"
#include "gfx/renderer.h"
#include "hle.h"

// The decompiler emits MSL or GLSL depending on g_renderer's type: set once at start-up (and again if
// Vulkan cannot start and the game falls back to Metal), before any shader is translated.
std::unique_ptr<Renderer> g_renderer = std::make_unique<VulkanRenderer>();
void select_decompiler_api(render::Api api) {
    if (api == render::Api::Vulkan) {
        g_renderer = std::make_unique<VulkanRenderer>();
        return;
    }
}
void cemu_shim_log(const std::string& msg) { LOG("[decompiler] %s", msg.c_str()); }
