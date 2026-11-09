# BotW-Native for Steam Deck — v0.8.0

Breath of the Wild (Wii U) running natively on the **Steam Deck** (SteamOS, Vulkan/RADV) and other Linux PCs.
The PowerPC game code is statically recompiled to C++ and runs on a Cafe OS runtime; Cemu's Latte GPU layer
draws through Vulkan. There is no emulator and no JIT.

This is a fork of [ryderclanky/BotW-Native](https://github.com/ryderclanky/BotW-Native) (Windows/Vulkan), which
itself builds on [BotW Recompiled](https://github.com/IshaanKalra2103/BotWRecomp). This fork adds the Linux /
Steam Deck build and a long performance pass: about 145 Deck builds, each measured on the hardware, covering the
recompiler's generated code, the guest thread scheduler, Cemu's GPU thread, the shaders, file streaming and the
Deck platform itself. Work in progress, not a full-game release.

> [!IMPORTANT]
> This repository contains no game files, no keys, and no decryption tools. You must supply your own legally
> obtained copy of the game (USA, update v208 / 1.5.0), as an already-extracted game folder. This is an
> unofficial fan project and is not affiliated with or endorsed by Nintendo.

## Results on the Steam Deck

LCD Steam Deck, SteamOS, 60 fps mode, 1280x720 render. "15 W" / "8 W" are the Deck's TDP limit.

| | Start of the port (2026-10-31) | Now (v0.8.0) |
| --- | --- | --- |
| Open world, 15 W | 37–48 fps, CPU-bound | locked 60 fps (59.6–60 average) |
| Open world, 8 W | 46–52 fps | 56–58 fps |
| Heavy scene, 15 W, 99th percentile frame | 59 ms | 25 ms |
| Kakariko, frames over 20 ms | ~5 % | ~2–2.5 % |
| Kakariko, late presents per minute (standing / walking) | 70–100 / ~160 | ~30 / ~90 |
| Main thread waiting on the job workers | 2.8 ms per frame | 0.5–0.8 ms per frame |
| World load (frame-rate dip after loading a save) | ~25 s | ~2 s |
| Area-load GPU-thread bursts | ~230 ms | ~110 ms |

What is left: at 8 W the GPU runs near 100 % in heavy scenes, and the main thread is bound by memory latency on
the game's own data (its hot spots are dependent loads, not wasted instructions). The remaining frame drops are
frames where the game's own work runs past 16.7 ms.

## What works

- Boot, title screen, movies (H.264 decoded with FFmpeg on Linux), loading saves, the open world, combat, the map
  and fast travel, villages with NPCs and shops, the DLC (as `aoc/`).
- Gyro on the Deck (motion shrines, aiming), with a "world" turning mode.
- Long sessions: 15–16 minute random-input soaks with no fatal errors; 10-minute Deck runs at a steady 60 fps.
- In-game menu with L3+R3 (frame rate, render scale, gyro, telemetry, ...).

Not checked: a full play-through. See [Known issues](#known-issues).

## Install on the Steam Deck

1. Build the package (below) or take one you built; it contains `BotWRecomp`, `botw.sh`, `lib/` and
   `gamecontrollerdb.txt`. Add your own merged game folder as `game/` (`code/ content/ meta/` from base + v208
   update, `aoc/` for the DLC).
2. Copy the folder to the Deck, e.g. `/home/deck/Games/BotW`, and run `chmod +x botw.sh BotWRecomp` once.
3. Steam (Desktop Mode) > Add a Non-Steam Game > pick `botw.sh`. Start it from Game Mode.

Saves, settings, logs and the shader cache live in `~/.local/share/BotWRecomp`. The first time through an area
stutters while shaders compile; later runs use the cache. `tools/deck/README-DECK.txt` has the details.

## Build

### Linux / Steam Deck

Built on Ubuntu 24.04 (glibc 2.39, older than SteamOS's, so the binary runs there) with clang 18 and lld.

```bash
sh tools/deck/ffmpeg.sh            # once: minimal static FFmpeg (H.264) for the movies, into /root/ffmpeg-min
python3 tools/recompile.py /path/to/game/code/U-King.rpx generated
FLAGS="-march=x86-64-v3 -mtune=znver2 -ffp-contract=off"
CC=clang CXX=clang++ cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DCMAKE_C_FLAGS="$FLAGS" -DCMAKE_CXX_FLAGS="$FLAGS" -DBOTW_USE_LLD=ON -DBOTW_GUEST_OPT=-O2 \
  -DBOTW_SYMBOL_ORDER="$PWD/tools/deck/symbol-order.txt" -DBOTW_FFMPEG_DIR=/root/ffmpeg-min
cmake --build build -j12
sh tools/deck/package.sh build <package dir> <symbols dir>
```

- `-march=x86-64-v3`, not `-march=znver2`: with znver2 clang emits AMD-only SSE4a, which faults on Intel CPUs.
- `-ffp-contract=off` keeps float results identical to the Wii U (no fused multiply-adds the game didn't ask for).
- `tools/deck/symbol-order.txt` puts the 3429 functions that take 90 % of the Deck's samples at the start of the
  222 MB code section.
- A full build of `generated/` takes 11–20 minutes; each compile job needs about 0.5 GB of RAM.

### Windows

Unchanged from upstream (MSYS2 UCRT64, MinGW gcc): see the
[upstream README](https://github.com/ryderclanky/BotW-Native#build-on-windows). The Windows path was not
re-tested in this fork; Linux-only changes are behind `__linux__`.

## How the speedups work

Every change below was measured on the Deck before it stayed. Changes that only move work around (priorities,
thread placement, spins) were kept only if a Deck run showed a gain. Changes that strictly remove work and give
identical results were kept even when one run couldn't show the gain. Most have an environment switch to turn
them off (see [Switches](#switches)).

### Recompiled game code (the main thread and job workers)

- **Float conversion fast path** (`ppc.h f32_to_f64`): the Wii U keeps floats as doubles; the old conversion
  handled denormals with a slow generic path on every load. The hardware path is used whenever the result is
  identical (checked for every 32-bit input). Main thread −17 % CPU per frame (15.9 → 13.3 ms at 15 W).
- **No aliasing between guest memory and the CPU context** (`__builtin_assume_separate_storage` in every
  generated function): clang had assumed a guest store could change a register in the context struct, so it
  stored and reloaded guest registers around every memory access. Code 252 → 237 MB, main thread −5 %.
- **Condition-register stores only where they are visible**: calls and returns no longer store the volatile CR
  fields the PowerPC ABI lets a callee clobber (2.6 M stores removed at 660k call sites and returns). Code → 223 MB.
- **Cached indirect calls**: each indirect call site keeps its last target; `rt_call_fill` resolves misses.
- **Huge pages for code and data**: the 222 MB of code is remapped onto 2 MB pages at startup (the file mapping
  was not 2 MB aligned, so the kernel never used huge pages); guest RAM is `MADV_HUGEPAGE`. Main thread
  13.2 → 11.8 ms per frame. The launcher also puts glibc's heap on huge pages (`glibc.malloc.hugetlb=1`): GPU
  thread −8 % cycles per frame, visible stutter −20 %.
- **Hot code first**: the link order follows Deck profiles (see Build).
- **Native Yaz0 decompressor** (`runtime/src/game/szs.cpp`): the game's streaming decompressor
  (`SZSDecompressor::streamDecomp`) was the top thread in streaming-burst frames, copying byte by byte through
  guest memory accessors. The native version keeps the same state machine, writes and cache-line zeroing, copies
  whole literal runs and back-references, and can check itself against the recompiled original
  (`BOTW_SZS_CHECK=1`: 0 differences over thousands of calls).

### Guest thread scheduling (Cafe OS on four Deck cores)

The Wii U has three cores with cooperative, priority-based threads; the port runs them on the Deck's eight
hardware threads.

- **Preemption points** at loop back-edges in the generated code let a higher-priority guest thread take a core
  from a long-running loop, as Cafe OS would. Heavy scene at 15 W 44 → 56 fps, 99th percentile 59 → 25 ms.
- **Priority inheritance** for guest mutexes, plus a 1 ms time slice between equal-priority background threads:
  removed 150–190 ms hitches where a job worker waited on a mutex held by a thread that couldn't get a core
  (NavMesh holding core 0).
- **Background threads run free**: threads the game lets run on any core (radar, placement, navmesh, resource
  loading) no longer queue for an emulated core; they run beside the guest cores at lower host priority. Frames
  over 20 ms 3.7 → 2.2 %. It also removes the core starvation of the actor-placement threads (RadarMgr waited
  ~700 ms of every second for a core).
- **Short spin before sleeping** on events, semaphores and mutexes (workers 1 ms, main 300 µs, with `MWAITX`):
  the Deck's cores dropped into a deep sleep state (350 µs wake-up) hundreds of times a second between jobs, so
  every hand-off between the main thread and the workers paid that delay. Main thread wait on workers
  2.8 → 0.8 ms per frame; frames over 25 ms 36 → 7 per run.
- **File reads hand the core over**: `FSReadFile` released the emulated core only after the blocking read, so
  streaming threads stalled behind it for up to 900 ms per second. World-load dip ~25 s → ~2 s.
- **Core plan**: main thread, the two worker lanes and the GPU pair each get their own core pair (cpu 0-1, 2-3,
  4-5, 6-7). Many layouts were tried; this one measured best.
- Wake-ups are issued after releasing the kernel lock, and a core hand-off wakes only the one thread that can
  take it (worker forced switches 363 → 77 per second).

### Cemu's GPU thread (Latte command processor, Vulkan)

- **VkRecord thread** (`VulkanDeferred.cpp`): Vulkan command recording and submission run on a second thread fed
  by a lock-free ring, so the GPU thread only parses the game's command buffers. A ring-full test compared a
  never-wrapping counter, which put a memory fence on every record; fixing it cut the GPU thread 8–9 %.
- **Fewer barriers**: transfer barriers are deferred and merged into the next render pass; texture copies, buffer
  uploads and streamout copies are batched under one barrier. Barriers per frame 1006 → ~620 (8 W 46 → 49 fps).
- **Faster caches**: flat hash map for the index cache (−8 % GPU thread at 4500+ draws), page cache for buffer
  node lookups, a 1 bit per KB page bitmap for buffer invalidation, cached texture views and micro tile tables.
- **Texture streaming**: uploads batched per texture, change hashes per array slice, texture decode on three
  helper threads, a pre-allocated staging ring (area-load bursts 230 → ~110 ms). Tiled-to-linear conversion
  copies whole rows for 16-bit formats (only when the tile's own index table proves the row is contiguous) and
  computes one full address per 8 texels for 1D-tiled mips.
- **VRAM spill**: the Deck's VRAM carve-out is 1 GB and RADV reports a fake 2/3-of-RAM heap; texture memory goes
  to GTT before the carve-out fills (read from sysfs).

### Shaders (the GPU at 8 W)

- **`mul_nonIEEE` as `fmulz`**: Cemu's `if (a==0||b==0) return 0; return a*b;` cost ~4 instructions per multiply.
  Written as `(b==0?0:a)*(a==0?0:b)`, Mesa turns it into one `v_mul_legacy_f32`. 74 % of BotW's multiplies are of
  this kind: heaviest pixel shader 1507 → 811 instructions, G-buffer shader 333 → 167, vertex shaders ~−40 %.
- **Plain saturate on Mesa** (`clampFI32`) and vertex attribute byte swaps as rotates (6 → 3 instructions per
  word): top pixel shaders −11 to −23 %.
- **Native 8-bit vertex fetch**: 8-bit attributes are fetched as `USCALED`/`SSCALED`/`SINT` so the fetch unit does
  the integer-to-float conversion (exact for every 8-bit value): −2.7 % instructions in 177 vertex shaders.
- `RADV_PERFTEST=gewave32`, anisotropic filtering capped at 4x, shader compile pool sized to the CPU.
- Tried and rejected: colour compression (DCC). BotW samples its G-buffer while drawing into it, which corrupts
  DCC images on RADV.

### Steam Deck platform

- Gyro: Steam switches the Deck's IMU off in-game; the runtime re-enables it every 2 s through the HID feature
  report. "Gyro turning: World" measures turns around the real vertical.
- Linux fixes: case-insensitive game paths, a 32-bit indirect-call table on Linux, H.264 movies through FFmpeg,
  pipeline-compile threads at lower priority, a timebase without 128-bit division.
- Fixed for all platforms: a DLC boot crash (`AOC_OpenTitle` wiped the next string), a dangling `string_view` in
  the mod loader, a Cemu crash when an async pipeline finished mid draw sequence, a lavapipe query race.

### Telemetry

In-game menu (L3+R3) > Telemetry: per-frame, per-present and per-second CSVs, a sampling profiler, GPU timings and
Deck power/clock readings in `~/.local/share/BotWRecomp/telemetry/`. Off by default; when off it costs nothing (no
recording, no GPU queries, no log file).

## Switches

Environment variables, for comparing or working around a problem. Defaults are the measured best.

| Variable | Default | What it does |
| --- | --- | --- |
| `BOTW_GAME` | next to `botw.sh` | Folder with `code/`, `content/`, `meta/` (and `aoc/`). |
| `BOTW_USER_ROOT` | `~/.local/share/BotWRecomp` | Saves, settings, logs, shader cache. |
| `BOTW_LOG=1` | off | Write `runtime.log` (play builds write nothing). |
| `BOTW_MALLOC_THP=0` | on | Turn off glibc heap huge pages (set in `botw.sh`). |
| `BOTW_HUGETEXT=0` | on | Keep code on 4 KB pages. |
| `BOTW_PREEMPT=0` | on | No preemption points. |
| `BOTW_PRIO_INHERIT=0` / `BOTW_PREEMPT_MUTEX=0` | on | No priority inheritance / no preempting mutex holders. |
| `BOTW_TIMESLICE_US` | 1000 | Background time slice (0 = off). |
| `BOTW_FREE_BG=0` | on | Background threads queue for an emulated core again. |
| `BOTW_WAIT_SPIN_US` / `BOTW_MAIN_SPIN_US` / `BOTW_MUTEX_SPIN=0` | 1000 / 300 / on | Spin before sleeping. |
| `BOTW_CORES=0` | on (Deck) | No core plan. |
| `BOTW_VKTHREAD=0` | on | Record Vulkan commands on the GPU thread. |
| `BOTW_SO_BATCH=0` | on | No streamout copy batching. |
| `BOTW_VRAM_SPILL=0` | on | No GTT spill. |
| `BOTW_SZS_NATIVE=0` | on | Use the recompiled Yaz0 decompressor. |
| `BOTW_EXACT_SATURATE=1` | off | Cemu's exact saturate in shaders. |
| `BOTW_GEWAVE32=0` | on | No RADV `gewave32`. |
| `BOTW_ANISO` | 4 | Anisotropic filtering cap (1–16). |
| `BOTW_DCC=1` | off | Colour compression (corrupts the G-buffer, test only). |

## Known issues

- First time through an area: pipelines compile, short hitches. A build that changes shaders recompiles once.
- A few cutscenes stay at 30 fps on purpose (FPS++ behaviour).
- At 8 W the GPU is the limit in heavy scenes; drops there are frames where the game's work exceeds 16.7 ms.
- Coverage: the opening, Kakariko, Hateno, Wetland, Tarrey Town, Zora's Domain and Rito were tested on the Deck;
  not a full play-through.
- Windows was not re-tested in this fork. macOS/iOS (Metal) are upstream's and untouched.

## Version

v0.8.0: about 145 Deck builds on top of upstream's main from before its v0.1.2 release (upstream v0.1.2's
controller, region and launcher fixes are not merged yet), with the Linux / Steam Deck port and changes to the
generated code, the scheduler, the renderer and the shaders. Not 1.0 yet: no full play-through, and Windows was
not re-tested.

## Credits and origins

- [BotW Recompiled](https://github.com/IshaanKalra2103/BotWRecomp) by Ishaan Kalra: the recompiler, the Cafe OS
  runtime and the Metal backend.
- [ryderclanky/BotW-Native](https://github.com/ryderclanky/BotW-Native): the Windows build and the Vulkan backend
  this fork starts from.
- [Cemu](https://github.com/cemu-project/Cemu): the GPU layer in `runtime/cemu/` (vendored from `a53cf0e`) and
  parts of GX2, snd_core and the motion code. The FPS++ graphic pack (Xalphenos, rajkosto, epigramx, Crementif,
  leoetlino, Pig, Exzap) provides the 60 fps patches.
- [SDL3](https://libsdl.org), [fmt](https://fmt.dev), [zstd](https://facebook.github.io/zstd/),
  [glslang](https://github.com/KhronosGroup/glslang), [FFmpeg](https://ffmpeg.org),
  [Dear ImGui](https://github.com/ocornut/imgui), [metal-cpp](https://developer.apple.com/metal/cpp/).

This project's own code is MIT licensed ([LICENSE](LICENSE)). Code from Cemu stays under the Mozilla Public
License 2.0 ([runtime/cemu/LICENSE.txt](runtime/cemu/LICENSE.txt)).

The Legend of Zelda and Breath of the Wild are trademarks of Nintendo. This is an unofficial fan project, not
affiliated with or endorsed by Nintendo.
