// Linux: move the program's code onto 2 MB pages. 250 MB of recompiled code on 4 KB pages misses the
// TLB on nearly every guest function call. The code is copied into an anonymous huge-page region and
// mremap swaps it in place atomically, so every address stays the same. BOTW_HUGETEXT=0 skips it.
#if defined(__linux__)
#include <link.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include <sys/mman.h>

#include "log.h"

namespace {
constexpr uintptr_t kHuge = 2u << 20;

int find_text(struct dl_phdr_info* info, size_t, void* out) {
    auto* range = (uintptr_t*)out;
    for (int i = 0; i < info->dlpi_phnum; i++) {
        const ElfW(Phdr)& ph = info->dlpi_phdr[i];
        if (ph.p_type != PT_LOAD || !(ph.p_flags & PF_X)) continue;
        range[0] = info->dlpi_addr + ph.p_vaddr;
        range[1] = range[0] + ph.p_memsz;
        return 1;  // the first object is the program itself
    }
    return 1;
}
}  // namespace

void hugetext_init() {
    if (const char* e = getenv("BOTW_HUGETEXT"); e && e[0] == '0') return;
    uintptr_t range[2] = {};
    dl_iterate_phdr(find_text, range);
    const uintptr_t a = (range[0] + kHuge - 1) & ~(kHuge - 1), b = range[1] & ~(kHuge - 1);
    if (b <= a) return;
    const size_t size = b - a;
    uint8_t* raw = (uint8_t*)mmap(nullptr, size + kHuge, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (raw == MAP_FAILED) return;
    uint8_t* dst = (uint8_t*)(((uintptr_t)raw + kHuge - 1) & ~(kHuge - 1));
    if (dst > raw) munmap(raw, dst - raw);
    if (dst + size < raw + size + kHuge) munmap(dst + size, raw + size + kHuge - (dst + size));
    madvise(dst, size, MADV_HUGEPAGE);
    memcpy(dst, (const void*)a, size);
    if (mprotect(dst, size, PROT_READ | PROT_EXEC) != 0 ||
        mremap(dst, size, size, MREMAP_MAYMOVE | MREMAP_FIXED, (void*)a) == MAP_FAILED) {
        munmap(dst, size);
        LOG("hugetext: remap failed, code stays on 4 KB pages");
        return;
    }
    LOG("hugetext: %zu MB of code on 2 MB pages", size >> 20);
}
#endif
