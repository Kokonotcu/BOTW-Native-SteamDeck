// 60 fps: Cemu's FPS++ graphic pack itself (gfxpacks/FPS++, Cemu's files unchanged), assembled into the game by
// tools/gfxpack.py at recompile time - game speed, pacing (performance fence, vsync off), cutscene limits, physics
// step fixes, controller scroll speed, loading screen. Two FPS++ constructs that jump to computed addresses are
// replaced by their exact one-word equivalents (see tools/gfxpack.py _rewrite). Nothing else is patched here.
//
// The only hook left is a port fix that is not part of FPS++: the boot frame clamp below.
#include "hooks.h"
#include "telemetry.h"

namespace {
constexpr double kTicksPerSecond = 62156250.0;  // OSGetSystemTime
int g_span_logs = 0;
int g_clamp_logs = 0;
}  // namespace

// subfe r0 (0x031FA974): r0:r12 = OSGetSystemTime ticks since the last frame, stored to +120/+124 next and read by
// FPS++ (_calculateGamespeed, bla at 0x031FA97C). A boot frame of several seconds (last timestamp still 0) steps
// the title timeline once, which skips the logo fade-in, the TitleBG scene and the BGM cue and leaves the idle
// "(A) Start" panes; cap a frame at 50 ms (FPS++'s own floor is 20 fps).
BOTW_HOOK(0x031fa974) {
    uint64_t ticks = ((uint64_t)c->r[0] << 32) | c->r[12];
    const uint64_t kMaxStep = (uint64_t)(kTicksPerSecond / 20.0);
    const bool clamp = ticks > kMaxStep;
    if (!telem::on()) {
    } else if (g_span_logs < 4) {  // log lines are telemetry
        g_span_logs++;
        LOG("frame span %.3fs%s", (double)ticks / kTicksPerSecond, clamp ? " (clamped to 50 ms)" : "");
    } else if (clamp && g_clamp_logs < 8) {
        g_clamp_logs++;
        LOG("frame span %.3fs clamped to 50 ms", (double)ticks / kTicksPerSecond);
    }
    if (clamp) {
        c->r[0] = (uint32_t)(kMaxStep >> 32);
        c->r[12] = (uint32_t)kMaxStep;
    }
}
