# Native GPU layer (parked)

This folder holds a native Wii U graphics layer: GX2 implemented directly on Vulkan, with no PM4 command processor and
no Latte emulation. It is **not built and not wired into the port yet**. The port still draws through Cemu's Latte
layer (`runtime/cemu`). The work is parked as of 2026-11-10 so effort goes into the recompiler and the Latte path. This
branch keeps it findable and ready to pick up.

## Where it comes from

- **Source:** the GPU pieces of [vasist15-source/BOTWRecomp](https://github.com/vasist15-source/BOTWRecomp) at commit
  `f9ef36b`, a BotW (Wii U 1.5.0) port for macOS built on the
  [ZeldaWWHDRecomp](https://github.com/ZeldaWWHDRecomp/ZeldaWWHDRecomp) runtime.
- **License:** MPL-2.0 (`LICENSE-MPL-2.0.txt`); the files keep their origin.
- **Not copied:** everything else from that project (its recompiler, runtime, Metal backend, mods, installer).

| Folder | What it is |
|---|---|
| `gx2/` | GX2 API: every call updates a Latte-layout register file and emits a compact command of the layer's own (`gx2_cmd.h`). Display lists recorded by the game are stored in that encoding in guest memory and decoded from guest memory each time they are called, so the game's in-place edits are seen. |
| `gfx/vulkan/` | Vulkan backend that executes the command queue on a render thread. Android-only files and Metal files left out. |
| `gfx/` | Renderer interface and shared helpers (display modes, texture pieces, shader keys, LOD bias). |
| `write_watch.*` | Page write tracking. The Metal backend uses it for texture change detection; the Vulkan backend does not yet (see below). |
| `render_prof.*` | Render thread profiling (`[prof]` lines). |
| `compat/` | Small Cemu-side helpers the layer uses (surface size calculation, texture register layout, fetch shader parsing, guest address helpers). |
| `deck/` | How it was tested on the Steam Deck: Linux/Vulkan build script, launch script, and the patch that made the macOS port build on Linux. |

## Steam Deck test, 2026-11-10

How it was tested: the whole macOS port was built for Linux with the Vulkan backend on the PC and run on the Deck
(RADV VANGOGH). It was started as a separate Steam shortcut, with its own copy of the saves.

| | Result |
|---|---|
| Boot | Loads U-King.rpx v208, all game threads start, DLC mounts, `native GX2 -> Vulkan` comes up on RADV |
| Title screen | 2D layout (the "(A) Start" prompt) correct at 30 fps; the 3D scene behind it black |
| Loading a save | Reaches the world after three generic fixes (below) |
| In the world | About 6 fps: 156-172 ms per frame at about 5,000 draws per frame |

### Fixes made here (generic, any Wii U game)

1. **CPU GX2CopySurface:** the element size now comes from the source's GX2 format (`Latte::GetFormatBits`), as in Cemu
   and the Metal backend. Before, both formats had to map to a Vulkan format of the same size, and BotW aborted at
   startup on formats 1, 7, 1A, 806, 810, 820 and 122.
2. **Cube maps:** square 2D arrays with 6n layers are created cube-compatible, so a target rendered face by face can
   later be sampled as a cube. BotW aborted after the title.
3. **Texture units without a sampler:** these are units the shader only reads with texel fetches (the decompiler's
   SAMPLER_NONE). They get an all-zero sampler, as Cemu's Vulkan renderer does. BotW aborted at game load.
4. **Build fixes:** Wind Waker HD-only leftovers (bloom strength, the climb mod HUD) removed from the Vulkan files, and a
   stub for two Metal frame-timing hooks. Changes to that project's own runtime files are in
   `deck/macos-port-linux-build.patch`.

### Why it is slow (`WWHD_VK_STATS=1`, in the world, render thread per frame)

| Cost | ms per frame |
|---|---|
| Texture handling in draws | ~89 |
| Vertex data preparation (about 200 MB of vertex data copied per frame) | 21-27 |
| Pipeline creation on the render thread (997 new pipelines in 4 s) | 1-42 |
| Invalidations from the game (about 590 per frame) | 17-20 |
| Waiting for the GPU before submits | ~46 |
| **Per draw** | **28-37 µs** |

For comparison, the Latte path in this port spends about 2 µs per draw on its GPU thread at the same scene.

All of the BotW tuning in the source project went into its Metal backend; this Vulkan copy never had it.

## To pick it up again

1. **Performance, in order of size:**
   1. Texture change detection with write tracking (`write_watch`, as the Metal backend).
   2. Keep vertex data on the GPU (the Vulkan buffer cache) instead of snapshotting it every draw.
   3. Compile pipelines off the render thread. Upstream ZeldaWWHDRecomp's Vulkan backend has
      `pipeline_worker`, `pipeline_demand` and `headstart` for exactly this.
2. **Missing GPU features** (done only for Metal in the source project): stream-out, occlusion queries, compute
   shaders, GPU-to-guest readback, the sampler LOD bias. Write each one generically.
3. **Wiring it into this port** (an adapter layer). The layer expects these from its old runtime:
   - an `HLE(lib, name)` registration with `Cpu*` arguments (map it to `CAFE_EXPORT`);
   - `mem::ptr`, `LOG`, `config`, `exception_report`;
   - `savestate`, `game_clock`, `interp`, `aspect`, `platform/host`, `input`, `audio_out`, `perf_metrics`,
     `screenshot`, mods (`fast_forward`, `cemu_pack`), `botw_fps`, imgui.
   
   Keep Latte as the default behind a switch (`BOTW_GPU=native|latte`) until the native path is faster and correct.
   Put its render thread on cpus 6-7 like the Latte GPU thread.
4. **Saves:** the source runtime maps `/vol/save/user/...` to `<save dir>/user/...`; this port keeps the account folder
   (`80000001`).
