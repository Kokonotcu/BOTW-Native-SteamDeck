"""Guest image of an RPX at its link addresses, relocations applied.

Imports keep their link-time stub addresses in the .fimport_/.dimport_ sections (0xC000xxxx);
the runtime maps those addresses to HLE functions / variables.
"""
import bisect
import struct

from . import rpx

DATA_IMPORT_BASE = 0x0DF00000


class Image:
    def __init__(self, path: str):
        self.rpx = r = rpx.load(path)
        # Data imports get fixed 4 KB slots: recompiled code bakes their addresses in, and the
        # runtime places the exported variables at the same addresses (see guest_table.h).
        self.data_import_addr = {}
        k = 0
        for imp in sorted(r.imports, key=lambda i: i.addr):
            if imp.is_data and not imp.name.startswith("."):
                self.data_import_addr[imp.sym_index] = DATA_IMPORT_BASE + 0x1000 * k
                k += 1

        def resolve(i, s):
            return self.data_import_addr.get(i, s.value)

        self.secs = rpx.build_image(r, resolve)
        self.starts = sorted(self.secs)
        self.text = [s for s in r.text_sections() if s.name == ".text"][0]
        self.text_lo, self.text_hi = self.text.addr, self.text.addr + self.text.size
        self.import_at = {i.addr: i for i in r.imports}
        # relocation index: address being patched -> Reloc
        self.reloc_at = {x.offset: x for x in r.relocs}
        # graphic pack (tools/gfxpack.py): code cave, its data ranges and its import calls
        self.cave_lo = self.cave_hi = 0
        self.data_ranges: list[tuple[int, int]] = []
        self.extra_entries: set[int] = set()
        self.cave_imports: dict[int, rpx.Import] = {}

    def apply_pack(self, pack) -> None:
        """Patch the game words and add the pack's code cave as code right after .text."""
        cave = bytearray(pack.cave_hi - pack.cave_lo)
        for a, w in pack.words.items():
            if pack.cave_lo <= a < pack.cave_hi:
                struct.pack_into(">I", cave, a - pack.cave_lo, w)
                continue
            buf, o = self._loc(a)
            if buf is None:
                raise ValueError(f"pack patch outside the image: {a:08x}")
            struct.pack_into(">I", buf, o, w)
            self.reloc_at.pop(a, None)  # the patched word no longer carries the original relocation
        if pack.cave_hi > self.text_lo and pack.cave_lo < self.text_hi:
            raise ValueError("pack cave overlaps .text")
        self.secs[pack.cave_lo] = cave
        self.starts = sorted(self.secs)
        self.cave_lo, self.cave_hi = pack.cave_lo, pack.cave_hi
        self.data_ranges = sorted(pack.data_ranges)
        self.extra_entries = set(pack.entries)
        by_name = {(i.lib, i.name): i for i in self.rpx.imports if not i.is_data}
        for a, (lib, name) in pack.import_calls.items():
            imp = by_name.get((lib, name)) or rpx.Import(lib, name, 0, False, -1)
            self.cave_imports[a] = imp
            # a REL24 marker, so analysis treats the bl like any other import call
            self.reloc_at[a] = rpx.Reloc(a, 10, -1, 0, 0)

    def is_data(self, addr: int) -> bool:
        # called once per instruction in the analysis scans: everything below the cave is code
        if not self.data_ranges or addr < self.data_ranges[0][0] or addr >= self.cave_hi:
            return False
        return any(lo <= addr < hi for lo, hi in self.data_ranges)

    def _loc(self, addr: int):
        i = bisect.bisect_right(self.starts, addr) - 1
        if i < 0:
            return None, 0
        b = self.starts[i]
        buf = self.secs[b]
        if addr - b >= len(buf):
            return None, 0
        return buf, addr - b

    def u32(self, addr: int) -> int | None:
        buf, o = self._loc(addr)
        if buf is None or o + 4 > len(buf):
            return None
        return struct.unpack_from(">I", buf, o)[0]

    def is_text(self, addr: int) -> bool:
        return self.text_lo <= addr < self.text_hi or self.cave_lo <= addr < self.cave_hi

    def code_ranges(self) -> list[tuple[int, int]]:
        """.text, plus a graphic pack's code cave (Cemu's cave area, below .text) when one is applied."""
        r = [(self.text_lo, self.text_hi)]
        if self.cave_hi > self.cave_lo:
            r.insert(0, (self.cave_lo, self.cave_hi))
        return r

    def reloc_target(self, at: int) -> int | None:
        """S + A of the relocation patching address `at` (any type)."""
        r = self.reloc_at.get(at)
        if r is None:
            return None
        sym = self.rpx.symbols[r.sym]
        return (self.data_import_addr.get(r.sym, sym.value) + r.addend) & 0xFFFFFFFF

    def reloc_import(self, at: int):
        if at in self.cave_imports:
            return self.cave_imports[at]
        r = self.reloc_at.get(at)
        if r is None:
            return None
        return self.import_at.get(self.rpx.symbols[r.sym].value)
