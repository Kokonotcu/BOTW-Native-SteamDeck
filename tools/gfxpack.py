"""Cemu graphic pack patches (patch_*.asm + rules.txt), assembled for the static recompiler.

Cemu applies a pack's patches at load: "0xADDR = insn" overwrites a game word, everything else goes into a code
cave that Cemu allocates. Here the same files are assembled ahead of time: game words are patched in the image
before analysis, and the code cave becomes extra guest code right after .text (CAVE_BASE), compiled like the rest.
The runtime writes the same words into guest memory at load (generated/gfxpack_data.cpp), so data patches and the
cave's variables exist at run time too.

Presets come from rules.txt [Default]; overrides as NAME=VALUE (e.g. fpsLimitNormal=60).

Two FPS++ constructs jump to computed addresses inside game functions, which a static recompiler cannot follow
(it compiles calls and returns, not "return to somewhere else"). They are replaced by their exact one-word
equivalents, documented next to each rewrite below; everything else is assembled as written.
"""
import os
import re
import struct

# Cemu's code cave area (MMU.h MEMORY_CODECAVEAREA_ADDR, 4 MB): packs rely on it being low enough for absolute
# branches (ba/bla reach 0-0x01FFFFFC), which is how FPS++ hooks jump into its cave.
CAVE_BASE = 0x01800000

_REG = re.compile(r"^[rf](\d+)$", re.I)


class PackError(Exception):
    pass


def _ha(v):
    return ((v + 0x8000) >> 16) & 0xFFFF


class Pack:
    def __init__(self):
        self.words: dict[int, int] = {}  # guest address -> big-endian word (patches and cave contents)
        self.cave_lo = CAVE_BASE
        self.cave_hi = CAVE_BASE
        self.data_ranges: list[tuple[int, int]] = []  # cave ranges holding data, not code
        self.entries: set[int] = set()  # cave addresses entered from outside (bla/ba/bl targets)
        self.import_calls: dict[int, tuple[str, str]] = {}  # cave address of a "bl import.lib.f" -> (lib, f)
        self.symbols: dict[str, int] = {}
        self.source: dict[int, str] = {}  # address -> assembly text (instructions only), for listings


def _read_rules(path):
    vars_ = {}
    section = None
    for raw in open(path, encoding="utf-8", errors="replace"):
        line = raw.split("#", 1)[0].strip()
        m = re.match(r"^\[(.+)\]$", line)
        if m:
            section = m.group(1)
            continue
        if section != "Default":
            continue
        m = re.match(r"^\$(\w+)(?::\w+)?\s*=\s*(.+)$", line)
        if m:
            v = m.group(2).strip()
            vars_[m.group(1)] = float(v) if re.match(r"^-?\d+\.\d*$", v) else int(v, 0)
    return vars_


def _strip(line):
    line = line.split(";", 1)[0]
    if line.lstrip().startswith("#"):
        return ""
    return line.strip()


def _rewrite(fname, lines, vars_):
    """The documented equivalences (see module docstring)."""
    if fname == "patch_VSync.asm":
        # FPS++ "bla conditionalVsyncWait" at 0x031FACCC (the stwu at waitForVBlank's entry) returns past the
        # bl GX2WaitForVsync at 0x031FACD0 when keepVsync is 0, and "bla conditionalSwapStatus" at 0x031FACF0
        # makes the following blt (0x031FACF4) never loop. The file itself carries the one-word forms (its lines
        # 80-81, commented out): nop those two words when keepVsync is 0, keep them otherwise. Same instructions
        # executed, same registers and memory afterwards.
        out = []
        for ln in lines:
            s = ln.strip()
            if s.startswith("0x031FACCC") or s.startswith("0x031FACF0"):
                continue
            if s.startswith("# 0x031FACD0") or s.startswith("# 0x031FACF4"):
                ln = s[2:]
            out.append(ln)
        return out
    if fname == "patch_FenceMethod.asm":
        # FPS++ "b conditionalFence" at 0x031FAAFC jumps into the cave and back into the middle of the frame
        # function (bctrl to 0x031FAB00 / 0x031FAB04). With $fenceMethod 1 (performanceFence, the default) the cave
        # computes r6 = r12 + (1 - r10) and continues at the bl GX2SetGPUFence (0x031FAB00). r0 is only read by the
        # subf at 0x031FAAF4 between its load at 0x031FAAE8 (lwz r0, 904(r31)) and 0x031FAAFC, so loading 1 there
        # gives the same r0, r6 and fence call.
        if vars_.get("fenceMethod") != 1:
            raise PackError("FenceMethod: only $fenceMethod 1 (performanceFence) is supported")
        return ["0x031FAAE8 = li r0, 1"]
    return lines


class _Asm:
    def __init__(self, pack, vars_, imports):
        self.p = pack
        self.vars = vars_
        self.imports = imports  # (lib, name) -> True if known

    # ---- expressions -------------------------------------------------------------------------------
    _TOK = re.compile(r"0[xX][0-9a-fA-F]+|\d+\.\d*|\d+|\$\w+|[A-Za-z_][A-Za-z0-9_.]*(?:@ha|@hi|@h|@lo|@l)?")

    def value(self, expr, pc=None, allow_float=False):
        expr = expr.strip()
        if expr.startswith(".") and pc is not None:  # .+0x08
            return pc + int(eval(expr[1:] or "0", {"__builtins__": {}}))

        def sub(m):
            t = m.group(0)
            if t[0].isdigit():
                return t
            if t[0] == "$":
                if t[1:] not in self.vars:
                    raise PackError(f"unknown preset variable {t}")
                return repr(self.vars[t[1:]])
            mod = ""
            for suf in ("@ha", "@hi", "@h", "@lo", "@l"):
                if t.endswith(suf):
                    t, mod = t[: -len(suf)], {"@hi": "@h", "@lo": "@l"}.get(suf, suf)
                    break
            if t not in self.p.symbols:
                raise PackError(f"unknown symbol {t}")
            v = self.p.symbols[t]
            if mod == "@ha":
                v = _ha(v)
            elif mod == "@h":
                v = (v >> 16) & 0xFFFF
            elif mod == "@l":
                v = v & 0xFFFF
                v = v - 0x10000 if v & 0x8000 else v  # signed displacement
            return str(v)

        py = self._TOK.sub(sub, expr)
        v = eval(py, {"__builtins__": {}})
        if isinstance(v, bool):
            v = int(v)
        if not allow_float:
            v = int(v)
        return v

    # ---- operands ----------------------------------------------------------------------------------
    @staticmethod
    def reg(s):
        s = s.strip().lower()
        m = _REG.match(s)
        if not m:
            raise PackError(f"bad register {s}")
        return int(m.group(1))

    @staticmethod
    def crf(s):
        s = s.strip().lower()
        if not s.startswith("cr"):
            raise PackError(f"bad cr field {s}")
        return int(s[2:])

    def mem(self, s, pc):
        m = re.match(r"^(.*)\((r\d+)\)$", s.strip(), re.I)
        if not m:
            raise PackError(f"bad memory operand {s}")
        d = self.value(m.group(1) or "0", pc)
        return d & 0xFFFF, self.reg(m.group(2))

    # ---- encoding ----------------------------------------------------------------------------------
    D_OPS = {"addi": 14, "addis": 15, "mulli": 7}
    D_LOGIC = {"ori": 24, "oris": 25, "xoris": 27, "andi.": 28}
    D_MEM = {"lwz": 32, "lwzu": 33, "lbz": 34, "lbzu": 35, "stw": 36, "stwu": 37, "stb": 38, "stbu": 39,
             "lhz": 40, "lha": 42, "sth": 44, "lfs": 48, "lfd": 50, "stfs": 52, "stfd": 54}
    X_ARITH = {"add": 266, "subf": 40, "mullw": 235, "divwu": 459}
    X_LOGIC = {"or": 444, "andc": 60, "slw": 24, "srw": 536, "and": 28, "xor": 316}
    FP_A = {"fadds": (59, 21), "fsubs": (59, 20), "fdivs": (59, 18), "fadd": (63, 21), "fsub": (63, 20),
            "fdiv": (63, 18)}
    FP_MUL = {"fmuls": (59, 25), "fmul": (63, 25)}
    FP_B = {"fmr": 72, "frsp": 12, "fctiwz": 15, "fneg": 40, "fabs": 264}
    BC = {"beq": (12, 2), "bne": (4, 2), "blt": (12, 0), "bge": (4, 0), "bgt": (12, 1), "ble": (4, 1)}

    def branch_target(self, s, pc):
        s = s.strip()
        m = re.match(r"^import\.(\w+)\.(\w+)$", s)
        if m:
            return ("import", m.group(1), m.group(2))
        return self.value(s, pc)

    def encode(self, mn, args, pc):
        mn = mn.lower()
        a = [x.strip() for x in args.split(",")] if args.strip() else []
        if mn == "nop":
            return 0x60000000
        if mn == "li":
            return (14 << 26) | (self.reg(a[0]) << 21) | (self.value(a[1], pc) & 0xFFFF)
        if mn == "lis":
            return (15 << 26) | (self.reg(a[0]) << 21) | (self.value(a[1], pc) & 0xFFFF)
        if mn == "mr":
            s = self.reg(a[1])
            return (31 << 26) | (s << 21) | (self.reg(a[0]) << 16) | (s << 11) | (444 << 1)
        if mn in self.D_OPS:
            return (self.D_OPS[mn] << 26) | (self.reg(a[0]) << 21) | (self.reg(a[1]) << 16) | (self.value(a[2], pc) & 0xFFFF)
        if mn in self.D_LOGIC:
            return (self.D_LOGIC[mn] << 26) | (self.reg(a[1]) << 21) | (self.reg(a[0]) << 16) | (self.value(a[2], pc) & 0xFFFF)
        if mn in self.D_MEM:
            d, ra = self.mem(a[1], pc)
            return (self.D_MEM[mn] << 26) | (self.reg(a[0]) << 21) | (ra << 16) | d
        if mn in ("cmpwi", "cmplwi", "cmpw", "cmplw"):
            if a[0].lower().startswith("cr"):
                cf, a = self.crf(a[0]), a[1:]
            else:
                cf = 0
            if mn == "cmpwi":
                return (11 << 26) | (cf << 23) | (self.reg(a[0]) << 16) | (self.value(a[1], pc) & 0xFFFF)
            if mn == "cmplwi":
                return (10 << 26) | (cf << 23) | (self.reg(a[0]) << 16) | (self.value(a[1], pc) & 0xFFFF)
            xo = 0 if mn == "cmpw" else 32
            return (31 << 26) | (cf << 23) | (self.reg(a[0]) << 16) | (self.reg(a[1]) << 11) | (xo << 1)
        if mn in self.X_ARITH:
            return (31 << 26) | (self.reg(a[0]) << 21) | (self.reg(a[1]) << 16) | (self.reg(a[2]) << 11) | (self.X_ARITH[mn] << 1)
        if mn in self.X_LOGIC:
            return (31 << 26) | (self.reg(a[1]) << 21) | (self.reg(a[0]) << 16) | (self.reg(a[2]) << 11) | (self.X_LOGIC[mn] << 1)
        if mn == "rlwinm":
            sh, mb, me = (self.value(x, pc) for x in a[2:5])
            return (21 << 26) | (self.reg(a[1]) << 21) | (self.reg(a[0]) << 16) | (sh << 11) | (mb << 6) | (me << 1)
        if mn == "mflr":
            return 0x7C0802A6 | (self.reg(a[0]) << 21)
        if mn == "mtlr":
            return 0x7C0803A6 | (self.reg(a[0]) << 21)
        if mn == "mtctr":
            return 0x7C0903A6 | (self.reg(a[0]) << 21)
        if mn == "cror":
            d, x, y = (self.value(v, pc) for v in a)
            return (19 << 26) | (d << 21) | (x << 16) | (y << 11) | (449 << 1)
        if mn in self.FP_A:
            op, xo = self.FP_A[mn]
            return (op << 26) | (self.reg(a[0]) << 21) | (self.reg(a[1]) << 16) | (self.reg(a[2]) << 11) | (xo << 1)
        if mn in self.FP_MUL:
            op, xo = self.FP_MUL[mn]
            return (op << 26) | (self.reg(a[0]) << 21) | (self.reg(a[1]) << 16) | (self.reg(a[2]) << 6) | (xo << 1)
        if mn in self.FP_B:
            return (63 << 26) | (self.reg(a[0]) << 21) | (self.reg(a[1]) << 11) | (self.FP_B[mn] << 1)
        if mn == "fcmpu":
            cf = self.crf(a[0]) if a[0].lower().startswith("cr") else None
            if cf is None:
                cf, a = 0, ["cr0"] + a
            return (63 << 26) | (cf << 23) | (self.reg(a[1]) << 16) | (self.reg(a[2]) << 11)
        if mn == "blr":
            return 0x4E800020
        if mn == "bctr":
            return 0x4E800420
        if mn == "bctrl":
            return 0x4E800421
        m = re.match(r"^b(eq|ne|lt|ge|gt|le)lr$", mn)
        if m:
            bo, bit = self.BC["b" + m.group(1)]
            cf = self.crf(a[0]) if a else 0
            return (19 << 26) | (bo << 21) | ((4 * cf + bit) << 16) | (16 << 1)
        if mn in ("b", "bl", "ba", "bla"):
            t = self.branch_target(a[0], pc)
            if isinstance(t, tuple):
                if mn != "bl":
                    raise PackError(f"{mn} to an import")
                self.p.import_calls[pc] = (t[1], t[2])
                return 0x48000001  # bl with LI 0: the recompiler takes the call from import_calls
            aa = mn.endswith("a")
            lk = mn in ("bl", "bla")
            off = t if aa else t - pc
            if not aa and not -0x2000000 <= off < 0x2000000:
                raise PackError(f"branch out of range at {pc:08x}")
            if lk or aa:
                if CAVE_BASE <= t < self.p.cave_hi:
                    self.p.entries.add(t)
            return (18 << 26) | (off & 0x03FFFFFC) | (2 if aa else 0) | (1 if lk else 0)
        if mn in self.BC:
            bo, bit = self.BC[mn]
            if a[0].lower().startswith("cr"):
                cf, a = self.crf(a[0]), a[1:]
            else:
                cf = 0
            t = self.value(a[0], pc)
            off = t - pc
            if not -0x8000 <= off < 0x8000:
                raise PackError(f"conditional branch out of range at {pc:08x}")
            return (16 << 26) | (bo << 21) | ((4 * cf + bit) << 16) | (off & 0xFFFC)
        raise PackError(f"unsupported mnemonic {mn}")


def load(pack_dir, overrides=None, imports=None):
    """Assemble a pack. `imports`: set of (lib, name) the image imports (others are still allowed: HLE by name)."""
    vars_ = _read_rules(os.path.join(pack_dir, "rules.txt"))
    for k, v in (overrides or {}).items():
        vars_[k] = v
    p = Pack()
    asm = _Asm(p, vars_, imports or set())
    # items: (kind, address, payload) where kind is "insn" / "data" / "label"
    items = []
    cur = CAVE_BASE
    for fname in sorted(f for f in os.listdir(pack_dir) if f.startswith("patch_") and f.endswith(".asm")):
        raw = open(os.path.join(pack_dir, fname), encoding="utf-8", errors="replace").read().split("\n")
        raw = _rewrite(fname, raw, vars_)
        for ln in raw:
            s = _strip(ln)
            if not s or s.startswith("[") or s.startswith("moduleMatches") or s.startswith(".origin"):
                continue
            m = re.match(r"^(0x[0-9A-Fa-f]+)\s*=\s*(.*)$", s)
            if m:
                addr, rest = int(m.group(1), 16), m.group(2).strip()
                lm = re.match(r"^([A-Za-z_][A-Za-z0-9_.]*):$", rest)
                if lm:
                    p.symbols[lm.group(1)] = addr
                else:
                    items.append(("at", addr, rest))
                continue
            lm = re.match(r"^([A-Za-z_][A-Za-z0-9_.]*):\s*(.*)$", s)
            if lm:
                p.symbols[lm.group(1)] = cur
                s = lm.group(2).strip()
                if not s:
                    continue
            if s.startswith(".align"):
                n = asm.value(s.split(None, 1)[1])
                cur = (cur + n - 1) // n * n
                continue
            if s.startswith((".float", ".int", ".uint", ".ptr")):
                kind, body = s.split(None, 1)
                vals = [v for v in body.split(",") if v.strip()]
                items.append(("data", cur, (kind, vals)))
                cur += 4 * len(vals)
                continue
            items.append(("insn", cur, s))
            cur += 4
    p.cave_hi = (cur + 3) & ~3
    # pass 2: encode (all labels known now)
    for kind, addr, payload in items:
        if kind == "data":
            dk, vals = payload
            for k, v in enumerate(vals):
                if dk == ".float":
                    w = struct.unpack(">I", struct.pack(">f", asm.value(v, allow_float=True)))[0]
                else:
                    w = asm.value(v) & 0xFFFFFFFF
                p.words[addr + 4 * k] = w
            if addr >= CAVE_BASE:
                p.data_ranges.append((addr, addr + 4 * len(vals)))
            continue
        text = payload
        if kind == "at" and text.startswith((".float", ".int", ".uint", ".ptr")):
            dk, body = text.split(None, 1)
            v = asm.value(body, allow_float=(dk == ".float"))
            p.words[addr] = struct.unpack(">I", struct.pack(">f", v))[0] if dk == ".float" else v & 0xFFFFFFFF
            continue
        mn, _, args = text.partition(" ")
        if "\t" in mn:
            mn, _, rest = mn.partition("\t")
            args = rest + " " + args
        try:
            p.source[addr] = text
            p.words[addr] = asm.encode(mn, args, addr) & 0xFFFFFFFF
        except PackError as e:
            raise PackError(f"{addr:08x} '{text}': {e}") from None
    # every cave label targeted by a branch from game code (bla/ba) or a call is an entry; also label each
    # branch target that game code reaches with "ba"
    return p


if __name__ == "__main__":
    import sys

    pk = load(sys.argv[1], {k: int(v) for k, v in (a.split("=") for a in sys.argv[2:])})
    print(f"cave {pk.cave_lo:08x}-{pk.cave_hi:08x}, {len(pk.words)} words, {len(pk.entries)} entries, "
          f"{len(pk.import_calls)} import calls, {len(pk.data_ranges)} data ranges")
    for a in sorted(w for w in pk.words if w < CAVE_BASE or w >= pk.cave_hi):
        print(f"  patch {a:08x} = {pk.words[a]:08x}")
    for a in sorted(pk.entries):
        names = [n for n, v in pk.symbols.items() if v == a]
        print(f"  entry {a:08x} {names}")
    for a, (lib, f) in sorted(pk.import_calls.items()):
        print(f"  import call {a:08x} {lib}.{f}")
