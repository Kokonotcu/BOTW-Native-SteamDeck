// Deck test build: the Metal renderer records per-frame GPU start/done times for gx2_core.cpp's [frame] log
// line (metal_main.mm). The Vulkan backend has no such record yet; 0 means "not known yet" to that log.
#include <cstdint>

extern "C" int64_t wwhd_frame_done_ns(uint64_t) { return 0; }
extern "C" int64_t wwhd_frame_gpu_start_ns(uint64_t) { return 0; }
