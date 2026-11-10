"""PowerPC (Espresso) -> C++ code generator.

Each guest function becomes `extern "C" void f_XXXXXXXX(PPCContext* __restrict c)`. Instructions
are emitted in address order with a `#line <address>` marker each (host crash backtraces then
symbolize to guest addresses), labels only where something branches. Semantics follow the Cemu
interpreter (src/Cafe/HW/Espresso/Interpreter), including its paired-single rounding rules.
"""
import bisect
import os
import re

from . import ppc
from .analyze import Function, bo_always
from .image import Image
from .regctx import ALL_HELD, ARG_NAMES, LOCAL_ARGS, RET_NAMES, Sync

# Condition register in locals. A function keeps the CR bits it uses in `cr_` (registers, after clang's SROA),
# so a compare feeding a branch is a compare and a jump instead of four byte stores to the context. c->cr is
# synced wherever code outside the function can see it: the bits the function writes are stored before calls,
# returns, tail calls, traps, hooks and probes; the bits it uses are reloaded after calls, hooks and probes.
_RE_TAIL = re.compile(r"\[\[clang::musttail\]\] return f_[0-9a-f]{8}\(c\);")
_RE_CALL = re.compile(r"(?<!return )\bf_[0-9a-f]{8}\(c\);")
_RE_HLE_CALL = re.compile(r"HLE_CALL\([^;]*\);")
_RE_RT_CALL = re.compile(r"rt_call\(c, [^;]*\);")
_RE_HOST = re.compile(r"(?:hook_0x[0-9a-f]{8}|rt_probe)\(c[^;]*\);")
_RE_LEAVE = re.compile(r"HLE_TAIL\([^;]*\);|RETURN\(\);|rt_trap\(c, [^;]*\);|rt_bad_switch\(c, [^;]*\);")
_RE_CR_WRITE_FIELD = re.compile(r"crl_[suf]\(cr_, c, (\d+),")
_RE_CR_WRITE_BIT = re.compile(r"cr_\[(\d+)\] = ")
_RE_CR_BIT = re.compile(r"cr_\[(\d+)\]")


def local_cr(prologue: list[str], body: list[str]) -> list[str]:
    """Rewrites one function's body (lines after the prologue) to keep CR bits in locals; see above."""
    if os.environ.get("BOTW_LOCAL_CR", "1") == "0":
        return prologue + body
    text = "\n".join(body)
    text = (text.replace("c->cr[", "cr_[").replace("set_cr_s(c, ", "crl_s(cr_, c, ")
            .replace("set_cr_u(c, ", "crl_u(cr_, c, ").replace("set_cr_f(c, ", "crl_f(cr_, c, ")
            .replace("CR0(", "CRL0(").replace("ppc_stwcx(c, ", "ppc_stwcx_cr(cr_, c, "))
    written: set[int] = set()
    for f in _RE_CR_WRITE_FIELD.findall(text):
        written.update(range(4 * int(f), 4 * int(f) + 4))
    if "CRL0(" in text or "ppc_stwcx_cr(" in text:
        written.update(range(4))
    written.update(int(b) for b in _RE_CR_WRITE_BIT.findall(text))
    used = written | {int(b) for b in _RE_CR_BIT.findall(text)}
    if not used:
        return prologue + body
    store = " ".join(f"c->cr[{b}] = cr_[{b}];" for b in sorted(written))
    load = " ".join(f"cr_[{b}] = c->cr[{b}];" for b in sorted(used))
    # A called guest function never reads the volatile fields cr0 and cr5-cr7 (EABI: scratch, set before use), so
    # calls skip storing them; cr1 (bit 6 flags float varargs) and the saved fields cr2-cr4 are still stored.
    # Tail calls, HLE calls, hooks and returns keep the full store. BOTW_CALL_CR_ALL=1 stores every bit.
    # Returns skip them too: the caller does not read cr0/cr5-cr7 after a call before setting them.
    store_call = store
    if os.environ.get("BOTW_CALL_CR_ALL", "0") != "1":
        store_call = " ".join(f"c->cr[{b}] = cr_[{b}];" for b in sorted(written) if 4 <= b < 20)
    text = _RE_TAIL.sub(lambda m: f"{{ {store} {m.group(0)} }}", text)
    text = _RE_CALL.sub(lambda m: f"{{ {store_call} {m.group(0)} {load} }}", text)
    text = _RE_HLE_CALL.sub(lambda m: f"{{ {store} {m.group(0)} {load} }}", text)
    text = _RE_RT_CALL.sub(lambda m: f"{store_call} {m.group(0)} {load}", text)
    text = _RE_HOST.sub(lambda m: f"{{ {store} {m.group(0)} {load} }}", text)
    text = _RE_LEAVE.sub(lambda m: f"{{ {store_call if m.group(0) == 'RETURN();' else store} {m.group(0)} }}", text)
    return prologue + [f"  uint8_t cr_[32]; {load}"] + text.split("\n")


# Registers in locals. By the PowerPC EABI a callee never reads the caller's r0/r11/r12/f0/f9-f13/ctr (scratch,
# not arguments) and preserves r1 and r14-r31/f14-f31 (cafe/../scratch abi_audit.py checks the first over every
# function). So a function keeps those in C locals: clang holds them in host registers across calls instead of
# storing every change to c before a call and reloading after it. r1 is written through (see local_regs). Hooks, probes and traps see
# (and hooks/probes may change) every register: all written locals are stored before them and reloaded after.
# Argument/return registers r3-r10/f1-f8 and r2/r13 stay in c. BOTW_LOCAL_REGS=0 = everything in c.
_LOCAL_GPR = frozenset([0, 1, 11, 12] + list(range(14, 32)) + (list(range(3, 11)) if LOCAL_ARGS else []))
_LOCAL_FPR = frozenset([0] + list(range(9, 32)) + (list(range(1, 9)) if LOCAL_ARGS else []))
# BOTW_LOCAL_ARGS: r3-r10/f1-f8/XER[CA] ("A" names) are locals as well. Their stores follow a per-site "dirty" set
# (gen_function arg_dirty: A locals written since they last matched c), written into the text as /*D ...*/ markers
# before each call/exit: a direct call stores the dirty ones its callee reads or delivers (regctx li|exp), an
# indirect/HLE call or a native all dirty ones (and reloads every A local after), a return the dirty EABI results
# RET plus what callers read (exp). BOTW_ARGS_POISON=1 (checking build): at returns, the A registers the function
# wrote that no caller may read get garbage in c.
_A_NAMES = ARG_NAMES + ["ca"]
_A_SET = frozenset(_A_NAMES)
_A_BIT = {n: 1 << k for k, n in enumerate(_A_NAMES)}
_RET = frozenset(RET_NAMES)
_ARGS_POISON = os.environ.get("BOTW_ARGS_POISON") == "1"
_RE_AEV = re.compile(
    r"(?P<tail>\[\[clang::musttail\]\] return f_(?P<tc>[0-9a-f]{8})\(c\);)"
    r"|(?<!return )(?P<call>\bf_(?P<cc>[0-9a-f]{8})\(c\);)"
    r"|(?P<sync>HLE_CALL\(|rt_call\(c, )"
    r"|(?P<exit>HLE_TAIL\(|RETURN\(\);|RETURN_TAIL\(\);)"
    r"|c->r\[(?P<wr>\d+)\] (?:=|\|=)(?!=)"
    r"|c->(?:f|ps1)\[(?P<wf>\d+)\] =(?!=)"
    r"|psq_load\w*\(c->f\[(?P<wp>\d+)\]"
    r"|(?P<wca>c->ca = (?!=)|adde_r\(c->ca)")
_MK = r"(?:/\*D([^*]*)\*/ )?"  # the dirty-set marker before a call or exit
_RE_MK = re.compile(r"/\*D[^*]*\*/ ")


def _abits(names) -> int:
    b = 0
    for n in names:
        b |= _A_BIT.get(n, 0)
    return b


def _anames(b: int) -> list[str]:
    return [n for n in _A_NAMES if b & _A_BIT[n]]
_GPR_NV = frozenset(f"r{n}" for n in range(14, 32))
_RE_GPR = re.compile(r"c->r\[(\d+)\]")
_RE_FPR = re.compile(r"c->(f|ps1)\[(\d+)\]")
_RE_LOCAL_WRITE = re.compile(r"\b(r|f|p)(\d+)_ (?:=|\|=)(?!=)|\b(ctr)_(?: =(?!=)|--)")


def local_regs(prologue: list[str], body: list[str], keep=frozenset(), sync=None,
               exp=frozenset(), natives=frozenset()) -> tuple[list[str], list[str]]:
    """Rewrites one function to keep the registers above in locals; returns (prologue, body). keep: registers
    ("r0", "f12", "ctr"...) this function keeps in the context. Registers exchanged with a caller or callee outside
    the EABI (cafe/regctx.py) are synced at the exchange: sync {callee entry: Sync} (before a call or tail call the
    callee's li|exp are stored, after a call its exp reloaded), exp: what this function delivers (stored at exits)."""
    if os.environ.get("BOTW_LOCAL_REGS", "1") == "0":
        return prologue, body
    text = "\n".join(body)
    sync = sync or {}
    callees = {int(t, 16) for t in _RE_CALLEE.findall(text)}
    carried = set(exp)  # registers this function passes on: a local even when its own code never names it
    for t in callees:
        s = sync.get(t)
        if s is not None:
            carried |= s.li | s.exp
    local_gpr = {n for n in _LOCAL_GPR if f"r{n}" not in keep}
    local_fpr = {n for n in _LOCAL_FPR if f"f{n}" not in keep}
    gprs = sorted(({int(n) for n in _RE_GPR.findall(text)} | {int(x[1:]) for x in carried if x[0] == "r"})
                  & local_gpr)
    fprs = sorted(({int(n) for _, n in _RE_FPR.findall(text)} | {int(x[1:]) for x in carried if x[0] == "f"})
                  & local_fpr)
    ctr = ("c->ctr" in text or "ctr" in carried) and "ctr" not in keep
    ca = LOCAL_ARGS and ("c->ca" in text or "ca" in carried) and "ca" not in keep
    if not gprs and not fprs and not ctr and not ca:
        return prologue, [_RE_MK.sub("", x) for x in body] if LOCAL_ARGS else body
    text = _RE_GPR.sub(lambda m: f"r{m.group(1)}_" if int(m.group(1)) in local_gpr else m.group(0), text)
    text = _RE_FPR.sub(lambda m: (("f" if m.group(1) == "f" else "p") + f"{m.group(2)}_")
                       if int(m.group(2)) in local_fpr else m.group(0), text)
    if ctr:
        text = text.replace("c->ctr", "ctr_")
    if ca:
        text = re.sub(r"c->ca\b", "ca_", text)
    written = set()
    for m in _RE_LOCAL_WRITE.finditer(text):
        written.add("ctr" if m.group(3) else f"{'f' if m.group(1) == 'p' else m.group(1)}{m.group(2)}")
    written.update(f"f{n}" for n in re.findall(r"psq_load(?:_f32)?r\(f(\d+)_,", text))  # by reference
    if ca and re.search(r"\bca_ =(?!=)|adde_r\(ca_", text):
        written.add("ca")
    decl = []
    store_all, load_all = [], []
    for n in gprs:
        decl.append(f"uint32_t r{n}_ = c->r[{n}];")
        load_all.append(f"r{n}_ = c->r[{n}];")
        store_all.append(f"c->r[{n}] = r{n}_;")
    for n in fprs:
        decl.append(f"double f{n}_ = c->f[{n}], p{n}_ = c->ps1[{n}];")
        load_all.append(f"f{n}_ = c->f[{n}]; p{n}_ = c->ps1[{n}];")
        store_all.append(f"c->f[{n}] = f{n}_; c->ps1[{n}] = p{n}_;")
    if ctr:
        decl.append("uint32_t ctr_ = c->ctr;")
        load_all.append("ctr_ = c->ctr;")
        store_all.append("c->ctr = ctr_;")
    if ca:
        decl.append("uint8_t ca_ = c->ca;")
        load_all.append("ca_ = c->ca;")
        store_all.append("c->ca = ca_;")
    # every local, written or not: since the entry load a callee may have stored another value to c
    sa, la = " ".join(store_all), " ".join(load_all)
    # r1 is written through to c (a few writes per function: prologue, epilogue), so calls need no store and
    # direct calls no reload: a callee restores r1 and writes it through too. The GHS prologue helpers that return
    # with r1 moved are only called from functions that keep every register in c (cafe/regctx.py ALL_HELD).
    # HLE and indirect calls still reload it (cheap there, and an HLE longjmp may set it).
    if 1 in gprs:
        text = re.sub(r"\br1_ (?:=|\|=)(?!=)[^;]*;", lambda m: m.group(0) + " c->r[1] = r1_;", text)
    sp_load = "r1_ = c->r[1];" if 1 in gprs else ""

    def wrap(m: re.Match, before: str, after: str = "") -> str:
        if not before and not after:
            return m.group("s")
        return "{ " + " ".join(x for x in (before, m.group("s"), after) if x) + " }"

    def st(names) -> str:  # store the locals among names to c
        out = [f"c->r[{x[1:]}] = {x}_;" for x in sorted(names) if x[0] == "r" and int(x[1:]) in gprs]
        out += [f"c->f[{x[1:]}] = {x}_; c->ps1[{x[1:]}] = p{x[1:]}_;" for x in sorted(names)
                if x[0] == "f" and int(x[1:]) in fprs]
        if ctr and "ctr" in names:
            out.append("c->ctr = ctr_;")
        if ca and "ca" in names:
            out.append("c->ca = ca_;")
        return " ".join(out)

    def ld(names) -> str:  # reload the locals among names from c
        out = [f"{x}_ = c->r[{x[1:]}];" for x in sorted(names) if x[0] == "r" and int(x[1:]) in gprs]
        out += [f"{x}_ = c->f[{x[1:]}]; p{x[1:]}_ = c->ps1[{x[1:]}];" for x in sorted(names)
                if x[0] == "f" and int(x[1:]) in fprs]
        if ctr and "ctr" in names:
            out.append("ctr_ = c->ctr;")
        if ca and "ca" in names:
            out.append("ca_ = c->ca;")
        return " ".join(out)

    _none = Sync(False, frozenset(), frozenset())

    def callee_at(m: re.Match) -> int:
        return int(_RE_CALLEE.search(m.group("s")).group(1), 16)

    def callee(m: re.Match):
        return sync.get(callee_at(m), _none)

    def dirty(m: re.Match) -> set:  # the A locals that may differ from c here (no marker: all of them)
        return set(_A_SET) if m.group(1) is None else set(m.group(1).split())

    def old(names) -> set:
        return set(names) - _A_SET

    def native(m: re.Match) -> bool:
        return LOCAL_ARGS and callee_at(m) in natives

    exp_ret = set(exp) | _RET
    poison = ""
    if _ARGS_POISON:
        bad = sorted((written & _A_SET) - exp_ret)
        poison = " ".join([f"c->r[{x[1:]}] = 0xDEADBEEFu;" for x in bad if x[0] == "r"]
                          + [f"c->f[{x[1:]}] = c->ps1[{x[1:]}] = __builtin_nan(\"\");" for x in bad if x[0] == "f"]
                          + ["c->ca = 2;" for x in bad if x == "ca"])

    def ret(m: re.Match) -> str:
        s = m.group("s")
        if s.startswith("HLE_TAIL("):
            return wrap(m, st(old(exp) | (dirty(m) & _A_SET)))
        return wrap(m, " ".join(x for x in (st(old(exp) | (dirty(m) & exp_ret)), poison) if x))

    def tail(m: re.Match) -> str:
        s = callee(m)
        if native(m):
            return wrap(m, st(old(s.li | s.exp | exp) | (dirty(m) & _A_SET)))
        return wrap(m, st(old(s.li | s.exp | exp) | (dirty(m) & (s.li | s.exp | exp_ret))))

    def call(m: re.Match) -> str:
        s = callee(m)
        if native(m):
            return wrap(m, st(old(s.li | s.exp) | (dirty(m) & _A_SET)), " ".join(x for x in (ld(s.exp), la_a) if x))
        return wrap(m, st(old(s.li | s.exp) | (dirty(m) & (s.li | s.exp))), ld(s.exp))

    la_a = ld(_A_SET)  # after code outside the analysis: every A local again
    # an indirect call's target is unknown: it gets the scratch registers this function wrote, in case it reads one
    sp_ind = st(written & {"r0", "r11", "r12", "f0", "f9", "f10", "f11", "f12", "f13", "ctr"})
    after_ext = " ".join(x for x in (sp_load, la_a) if x)
    text = _RE_HOST.sub(lambda m: wrap_plain(m, sa, la), text)
    text = _RE_TRAP.sub(lambda m: wrap_plain(m, sa), text)
    text = _RX_HLE_CALL.sub(lambda m: wrap(m, st(dirty(m) & _A_SET), after_ext), text)
    text = _RX_RT_CALL.sub(lambda m: wrap(m, " ".join(x for x in (sp_ind, st(dirty(m) & _A_SET)) if x), after_ext),
                           text)
    text = _RX_TAIL.sub(tail, text)
    text = _RX_CALL.sub(call, text)
    text = _RX_RETURN.sub(ret, text)
    if LOCAL_ARGS:
        text = _RE_MK.sub("", text)
    return prologue + ["  " + " ".join(decl)], text.split("\n")


def wrap_plain(m: re.Match, before: str, after: str = "") -> str:
    if not before and not after:
        return m.group(0)
    return "{ " + " ".join(x for x in (before, m.group(0), after) if x) + " }"


_RX_HLE_CALL = re.compile(_MK + r"(?P<s>HLE_CALL\([^;]*\);)")
_RX_RT_CALL = re.compile(_MK + r"(?P<s>rt_call\(c, [^;]*\);)")
_RX_TAIL = re.compile(_MK + r"(?P<s>\[\[clang::musttail\]\] return f_[0-9a-f]{8}\(c\);)")
_RX_CALL = re.compile(_MK + r"(?<!return )(?P<s>\bf_[0-9a-f]{8}\(c\);)")
_RX_RETURN = re.compile(_MK + r"(?P<s>HLE_TAIL\([^;]*\);|RETURN\(\);|RETURN_TAIL\(\);)")


_RE_CALLEE = re.compile(r"\bf_([0-9a-f]{8})\(c\);")
_RE_TRAP = re.compile(r"rt_trap\(c, [^;]*\);|rt_bad_switch\(c, [^;]*\);")
_RE_RETURN = re.compile(r"HLE_TAIL\([^;]*\);|RETURN\(\);|RETURN_TAIL\(\);")


FPCC_DROP = os.environ.get("BOTW_FPCC_DROP", "1") != "0"


def _fpscr_reader(i: ppc.Insn) -> None:
    # fcmp no longer updates FPSCR[FPCC] (ppc.h PPC_FPCC_DROP): no generated code may read the FPSCR
    raise ValueError(f"BOTW_FPCC_DROP: {i.op} at {i.addr:08x} reads the FPSCR; generate with BOTW_FPCC_DROP=0")


def header_defines() -> str:
    """ppc.h switches for the generated files (each BOTW_* =0: the old header code, byte-identical output)."""
    out = ""
    if FPCC_DROP:
        out += "#define PPC_FPCC_DROP 1\n"
    if os.environ.get("BOTW_F2F_FAST", "1") != "0":
        out += "#define PPC_F2F_FAST 1\n"
    if os.environ.get("BOTW_FRES_INLINE", "1") != "0":
        out += "#define PPC_FRES_INLINE 1\n"
    return out


def R(n: int) -> str:
    return f"c->r[{n}]"


def F(n: int) -> str:
    return f"c->f[{n}]"


def P(n: int) -> str:
    return f"c->ps1[{n}]"


def RA0(n: int) -> str:
    return "0u" if n == 0 else R(n)


def sanitize(s: str) -> str:
    return "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in s)


def hle_symbol(lib: str, name: str) -> str:
    return f"hle_{sanitize(lib)}_{sanitize(name)}"


class Generator:
    def __init__(self, img: Image, entries: set[int]):
        self.img = img
        self.entries = entries
        self.unsupported: dict[str, int] = {}
        self.imports_used: dict[str, tuple[str, str]] = {}  # hle symbol -> (lib, name)
        self.probes: set[int] = set()
        self.hooks: set[int] = set()  # host hook_0xADDR(c) runs after the (non-branch) instruction at ADDR
        self.natives: set[int] = set()  # host f_ADDR replaces the function; the recompiled one is emitted as f_ADDR_guest
        self.check_returns = True
        self.outer_back_edges: set[tuple[int, int]] = set()  # of the function being generated
        self.spin_back_edges: set[tuple[int, int]] = set()  # of the function being generated
        self.regctx: dict[int, Sync] = {}  # cafe/regctx.py: registers exchanged at entry/exit, or all in c
        self.keep: frozenset = frozenset()  # of the function being generated
        # TEST: registers every function keeps in the context (BOTW_KEEP_REGS="r0,r11,ctr,..."), for bisecting
        self.keep_all = frozenset(x for x in os.environ.get("BOTW_KEEP_REGS", "").split(",") if x)
        self.prologue_saves: set[int] = set()  # of the function being generated (find_prologue_saves)
        rng = os.environ.get("BOTW_LOCAL_RANGE", "")  # TEST: "lo:hi" hex
        self.local_range = tuple(int(x, 16) for x in rng.split(":")) if rng else None

    # ---- branch conditions --------------------------------------------------------------
    def cond(self, i: ppc.Insn, pre: list[str]) -> str | None:
        """C condition for BO/BI; None means 'always'. CTR decrement goes into `pre`."""
        bo, bi = i.BO, i.BI
        parts = []
        if not bo & 0x04:
            pre.append("c->ctr--;")
            parts.append("c->ctr == 0" if bo & 0x02 else "c->ctr != 0")
        if not bo & 0x10:
            parts.append(f"c->cr[{bi}]" if bo & 0x08 else f"!c->cr[{bi}]")
        if not parts:
            return None
        return " && ".join(f"({p})" for p in parts)

    def call_direct(self, fn: Function, a: int, t: int, callees: set[str]) -> str:
        sym = f"f_{t:08x}"
        callees.add(sym)
        return f"c->lr = 0x{a + 4:08x}u; {sym}(c);"

    def tail(self, t: int, callees: set[str]) -> str:
        sym = f"f_{t:08x}"
        callees.add(sym)
        return f"[[clang::musttail]] return {sym}(c);"

    def ret(self) -> str:
        return "RETURN();"

    def jump(self, a: int, t: int) -> str:
        # The back-edge of an outermost loop is where a background thread hands its emulated core to a
        # waiting higher-priority thread (the console preempts anywhere; the runtime only at these points).
        # Inner loops skip the check: they are the hot ones and end within one outer iteration.
        # A loop that stores nothing to guest memory may be a spin-wait on a word another thread changes. Guest
        # loads are plain C loads, so clang may keep such a word in a register across iterations (a racing
        # non-atomic read may be assumed unchanged) and spin forever: a compiler barrier at its back-edge makes
        # every iteration read memory again (no instruction emitted).
        bar = "PPC_MEM_BARRIER(); " if (t, a) in self.spin_back_edges else ""
        pre = "PREEMPT_POINT(); " if (t, a) in self.outer_back_edges else ""
        return f"{bar}{pre}goto L_{t:08x};"

    @staticmethod
    def find_spin_back_edges(fn: Function) -> set[tuple[int, int]]:
        """Back-edges (head t, branch at a) whose loop body t..a has no store to guest memory."""
        backs = [(i.target, a) for a, i in fn.insns.items()
                 if i.op in ("b", "bc") and not i.LK and i.target is not None and i.target <= a and i.target in fn.insns]
        stores = sorted(x for x, i in fn.insns.items() if i.op.startswith(("st", "psq_st")) or i.op in ("dcbz", "dcbz_l"))
        out = set()
        for t, a in backs:
            k = bisect.bisect_left(stores, t)
            if k == len(stores) or stores[k] > a:
                out.add((t, a))
        return out

    @staticmethod
    def find_outer_back_edges(fn: Function) -> set[tuple[int, int]]:
        backs = [(i.target, a) for a, i in fn.insns.items()
                 if i.op in ("b", "bc") and not i.LK and i.target is not None and i.target <= a and i.target in fn.insns]
        # a loop (head t, back-edge at a) is enclosed when another loop starts before it and ends at or after it
        return {(t, a) for t, a in backs if not any(t2 < t and a2 >= a for t2, a2 in backs)}

    def find_prologue_saves(self, fn: Function) -> set[int]:
        """stmw rN,d(r1) (N >= 14) in the straight-line entry block before any write of rN..r31: the save of the
        caller's registers, droppable when all of them are locals. Empty with BOTW_LOCAL_REGS=0/BOTW_KEEP_SAVES=1.
        BOTW_FSAVE_DROP (default 1): the same for f14-f31, saved by stfd / psq_st / stfs (stfd + ps_merge10 fN,fN,fN
        + stfs for a paired single) to r1-based slots (r1, or a register set from r1 by mr/addi in the block)."""
        if os.environ.get("BOTW_LOCAL_REGS", "1") == "0" or os.environ.get("BOTW_KEEP_SAVES", "0") == "1":
            return set()
        if os.environ.get("BOTW_FSAVE_DROP", "1") != "0":
            return self._find_prologue_saves_fpr(fn)
        written: set[int] = set()
        a = fn.entry
        while a in fn.insns:
            i = fn.insns[a]
            if a != fn.entry and a in fn.labels:
                break
            if i.op == "stmw" and i.rA == 1 and i.rD >= 14:
                regs = range(i.rD, 32)
                if not any(n in written for n in regs) and not any(f"r{n}" in self.keep for n in regs):
                    return {a}
                return set()
            if i.op in ("b", "bc", "bclr", "bcctr", "tw", "twi", "sc"):
                break
            code, _ = self.insn(fn, i, set(), set())
            written.update(int(n) for n in re.findall(r"c->r\[(\d+)\] (?:\|)?=(?!=)", " ".join(code)))
            a += 4
        return set()

    def _find_prologue_saves_fpr(self, fn: Function) -> set[int]:
        """find_prologue_saves with f14-f31 saves too. A dropped save only stored the caller's value, which this
        function never needs (its fN is a local; the epilogue's restore loads are dead, as for stmw)."""
        out: set[int] = set()
        written: set[int] = set()  # GPRs
        fwritten: set[int] = set()  # FPRs written other than by a dropped save swap
        fsaved: set[int] = set()  # FPRs with a dropped save (their ps_merge10 swap may be dropped too)
        sp = {1}  # registers holding a stack address
        stmw_done = False
        a = fn.entry
        while a in fn.insns:
            i = fn.insns[a]
            if a != fn.entry and a in fn.labels:
                break
            if i.op in ("b", "bc", "bclr", "bcctr", "tw", "twi", "sc"):
                break
            if i.op == "stmw" and i.rA == 1 and i.rD >= 14 and not stmw_done:
                stmw_done = True  # as find_prologue_saves: only the first stmw
                regs = range(i.rD, 32)
                if not any(n in written for n in regs) and not any(f"r{n}" in self.keep for n in regs):
                    out.add(a)
                a += 4
                continue
            n = i.rD
            nv = n >= 14 and f"f{n}" not in self.keep and n not in fwritten
            if nv and i.rA in sp and (i.op in ("stfd", "stfs") or (i.op == "psq_st" and i.I == 0)):
                out.add(a)
                fsaved.add(n)
                a += 4
                continue
            if nv and i.op == "ps_merge10" and i.rA == n and i.rB == n and n in fsaved:
                out.add(a)  # the half swap between the two stores of a paired-single save
                a += 4
                continue
            code = " ".join(self.insn(fn, i, set(), set())[0])
            written.update(int(x) for x in re.findall(r"c->r\[(\d+)\] (?:\|)?=(?!=)", code))
            fwritten.update(int(x) for x in re.findall(r"c->(?:f|ps1)\[(\d+)\] =(?!=)", code))
            fwritten.update(int(x) for x in re.findall(r"psq_load\w*\(c->f\[(\d+)\]", code))
            if i.op == "or" and i.rD == i.rB and i.rD in sp:
                sp.add(i.rA)  # mr rA, r1
            elif i.op == "addi" and i.rA in sp:
                sp.add(i.rD)
            else:
                sp -= {x for x in sp if x != 1 and f"c->r[{x}] " in code and re.search(rf"c->r\[{x}\] (?:\|)?=(?!=)", code)}
            a += 4
        return out

    # ---- function ---------------------------------------------------------------------------
    def gen_function(self, fn: Function) -> tuple[str, set[str]]:
        callees: set[str] = set()
        me = self.regctx.get(fn.entry)
        self.keep = (ALL_HELD if me is not None and me.all else frozenset()) | self.keep_all
        if self.local_range and not self.local_range[0] <= fn.entry < self.local_range[1]:
            self.keep |= _GPR_NV  # TEST bisect: r14-r31 locals only inside BOTW_LOCAL_RANGE
        if os.environ.get("BOTW_FSAVE_DROP", "1") != "0":
            self.prologue_saves = set()  # not the previous function's while find_prologue_saves runs insn()
        self.prologue_saves = self.find_prologue_saves(fn)
        guest ="_guest" if fn.entry in self.natives else ""
        prologue = [f'extern "C" void f_{fn.entry:08x}{guest}(PPCContext* __restrict c) {{',
                    "  uint8_t* const M = g_mem; (void)M; PPC_SEPARATE(M, c);"]
        if self.check_returns:
            prologue.append("  const uint32_t lr0 = c->lr; (void)lr0;")
        out: list[str] = []
        addrs = sorted(fn.insns)
        labels = set(fn.labels)
        self.outer_back_edges = self.find_outer_back_edges(fn)
        self.spin_back_edges = self.find_spin_back_edges(fn)
        if addrs and addrs[0] != fn.entry:
            # the body includes code below the entry (shared tails): start at the entry
            labels.add(fn.entry)
            out.append(f"  goto L_{fn.entry:08x};")
        body: list[tuple[int, list[str]]] = []
        falls_at: dict[int, bool] = {}
        for idx, a in enumerate(addrs):
            i = fn.insns[a]
            code, falls = self.insn(fn, i, callees, labels)
            falls_at[a] = falls
            nxt = addrs[idx + 1] if idx + 1 < len(addrs) else None
            if falls and nxt != a + 4:
                if a + 4 in fn.insns:
                    labels.add(a + 4)
                    code.append(f"goto L_{a + 4:08x};")
                elif a + 4 in self.entries:
                    code.append(self.tail(a + 4, callees))
                else:
                    code.append(f'rt_trap(c, 0x{a + 4:08x}u, "fell off function");')
            body.append((a, code))
        if (LOCAL_ARGS and os.environ.get("BOTW_LOCAL_REGS", "1") != "0"
                and not _A_SET <= self.keep):
            body = self.arg_dirty(fn, body, falls_at)
        for a, code in body:
            if a in labels:
                out.append(f"L_{a:08x}:;")
            out.append(f"#line {a} \"botw.ppc\"")
            if a in self.probes:
                out.append(f"  rt_probe(c, 0x{a:08x}u);")
            for line in code:
                out.append("  " + line)
            if a in self.hooks:
                out.append(f"  hook_0x{a:08x}(c);")
                callees.add(f"hook_0x{a:08x}")
        out.append("}")
        prologue, out = local_regs(prologue, out, self.keep, self.regctx, me.exp if me is not None else frozenset(),
                                   self.natives)
        return "\n".join(local_cr(prologue, out)), callees

    def arg_dirty(self, fn: Function, body: list[tuple[int, list[str]]],
                  falls_at: dict[int, bool]) -> list[tuple[int, list[str]]]:
        """BOTW_LOCAL_ARGS: forward may-analysis of the A registers (r3-r10, f1-f8, CA) whose local may differ from
        c: a write makes one dirty; a direct call cleans what it stores or reloads (callee li|exp), an indirect/HLE
        call or native cleans all (all dirty stored, all reloaded). A conditional call cleans nothing. Returns the
        body with a /*D names*/ marker before each call and exit, for local_regs."""
        none = Sync(False, frozenset(), frozenset())
        events: dict[int, list] = {}
        texts: dict[int, str] = {}
        for a, code in body:
            s = "\n".join(code)
            texts[a] = s
            ev = []
            for m in _RE_AEV.finditer(s):
                k = m.lastgroup
                if m.group("tail") is not None:
                    ev.append(("x", m.start(), 0))
                elif m.group("call") is not None:
                    t = int(m.group("cc"), 16)
                    cond = "if (" in s[:m.start()]
                    if t in self.natives:
                        ev.append(("s", m.start(), cond))
                    else:
                        sy = self.regctx.get(t, none)
                        ev.append(("c", m.start(), (cond, _abits(sy.li | sy.exp))))
                elif m.group("sync") is not None:
                    ev.append(("s", m.start(), "if (" in s[:m.start()]))
                elif m.group("exit") is not None:
                    ev.append(("x", m.start(), 0))
                else:
                    if m.group("wr") is not None:
                        b = _A_BIT.get(f"r{m.group('wr')}", 0)
                    elif m.group("wf") is not None:
                        b = _A_BIT.get(f"f{m.group('wf')}", 0)
                    elif m.group("wp") is not None:
                        b = _A_BIT.get(f"f{m.group('wp')}", 0)
                    else:
                        b = _A_BIT["ca"]
                    if b:
                        ev.append(("w", m.start(), b))
                del k
            events[a] = ev

        def run(a: int, d: int, marks: list | None) -> int:
            for kind, pos, x in events[a]:
                if kind == "w":
                    d |= x
                    continue
                if marks is not None:
                    marks.append((pos, d))
                if kind == "c" and not x[0]:
                    d &= ~x[1]
                elif kind == "s" and not x:
                    d = 0
            return d

        succ: dict[int, list[int]] = {}
        for a in texts:
            i = fn.insns[a]
            out = []
            if falls_at[a] and a + 4 in fn.insns:
                out.append(a + 4)
            if i.op in ("b", "bc") and not i.LK and i.target in fn.insns and i.target not in fn.tail_targets:
                out.append(i.target)
            sw = fn.switches.get(a)
            if sw is not None:
                out += [t for t in sw.targets if t in fn.insns]
            succ[a] = out
        d_in: dict[int, int] = {}
        if fn.entry in texts:
            d_in[fn.entry] = 0
            work = [fn.entry]
            while work:
                a = work.pop()
                d = run(a, d_in[a], None)
                for t in succ[a]:
                    old = d_in.get(t)
                    if old is None or d & ~old:
                        d_in[t] = d if old is None else old | d
                        work.append(t)
        allb = (1 << len(_A_NAMES)) - 1
        out_body = []
        for a, code in body:
            marks: list = []
            run(a, d_in.get(a, allb), marks)  # unreached from the entry: everything may be dirty
            if not marks:
                out_body.append((a, code))
                continue
            s = texts[a]
            for pos, d in sorted(marks, reverse=True):
                s = s[:pos] + "/*D" + "".join(" " + n for n in _anames(d)) + "*/ " + s[pos:]
            out_body.append((a, s.split("\n")))
        return out_body

    def unsup(self, i: ppc.Insn) -> list[str]:
        self.unsupported[i.op] = self.unsupported.get(i.op, 0) + 1
        return [f'rt_trap(c, 0x{i.addr:08x}u, "unsupported {i.op} {i.word:08x}");']

    # ---- instructions ---------------------------------------------------------------------
    def insn(self, fn: Function, i: ppc.Insn, callees: set[str], labels: set[int]) -> tuple[list[str], bool]:
        """Returns (C++ lines, falls through)."""
        op = i.op
        a = i.addr
        d, ra, rb = i.rD, i.rA, i.rB
        img = self.img

        # -- branches ---------------------------------------------------------------------
        if op == "b":
            if i.LK:
                imp = img.reloc_import(a)
                if imp is not None:
                    sym = hle_symbol(imp.lib, imp.name)
                    self.imports_used[sym] = (imp.lib, imp.name)
                    return [f'c->lr = 0x{a + 4:08x}u; HLE_CALL({sym}, "{imp.lib}.{imp.name}");'], True
                r = img.reloc_at.get(a)
                if r is not None:
                    tgt = img.reloc_target(a)
                    if tgt == 0:
                        return [f'rt_trap(c, 0x{a:08x}u, "call to undefined weak symbol");'], True
                    return [f'rt_trap(c, 0x{a:08x}u, "unhandled REL24 call");'], True
                if i.target == a + 4:
                    return [f"c->lr = 0x{a + 4:08x}u;"], True
                if not img.is_text(i.target):
                    return [f'rt_trap(c, 0x{a:08x}u, "call outside text");'], True
                return [self.call_direct(fn, a, i.target, callees)], True
            imp = img.reloc_import(a)
            if imp is not None:
                sym = hle_symbol(imp.lib, imp.name)
                self.imports_used[sym] = (imp.lib, imp.name)
                return [f'HLE_TAIL({sym}, "{imp.lib}.{imp.name}");'], False
            if img.reloc_at.get(a) is not None:
                return [f'rt_trap(c, 0x{a:08x}u, "tail call to unresolved symbol");'], False
            t = i.target
            if t in fn.tail_targets:
                return [self.tail(t, callees)], False
            labels.add(t)
            return [self.jump(a, t)], False

        if op == "bc":
            pre: list[str] = []
            cnd = self.cond(i, pre)
            t = i.target
            if i.LK:
                if t == a + 4:
                    body = f"c->lr = 0x{a + 4:08x}u;"  # bcl 20,31,$+4: read PC
                else:
                    body = self.call_direct(fn, a, t, callees)
                return pre + ([f"if ({cnd}) {{ {body} }}"] if cnd else [body]), True
            if t in fn.tail_targets:
                act = self.tail(t, callees)
            else:
                labels.add(t)
                act = self.jump(a, t)
            if cnd is None:
                return pre + [act], False
            return pre + [f"if ({cnd}) {{ {act} }}"], True

        if op == "bclr":
            pre = []
            cnd = self.cond(i, pre)
            if i.LK:
                body = f"{{ uint32_t t_ = c->lr; c->lr = 0x{a + 4:08x}u; rt_call(c, t_); }}"
                return pre + ([f"if ({cnd}) {body}"] if cnd else [body]), True
            if cnd is None:
                return pre + [self.ret()], False
            return pre + [f"if ({cnd}) {{ {self.ret()} }}"], True

        if op == "bcctr":
            pre = []
            cnd = self.cond(i, pre)
            if i.LK:
                body = f"{{ c->lr = 0x{a + 4:08x}u; rt_call(c, c->ctr); }}"
                return pre + ([f"if ({cnd}) {body}"] if cnd else [body]), True
            sw = fn.switches.get(a)
            if sw is not None:
                lines = ["switch (c->ctr) {"]
                for t in sw.targets:
                    labels.add(t)
                    lines.append(f"  case 0x{t:08x}u: goto L_{t:08x};")
                lines.append(f"  default: rt_bad_switch(c, 0x{a:08x}u, c->ctr); return;")
                lines.append("}")
                return lines, False
            body = "{ rt_call(c, c->ctr); RETURN_TAIL(); }"
            if cnd is None:
                return pre + [body], False
            return pre + [f"if ({cnd}) {body}"], True

        if op == "tw":
            if d == 31:
                return [f'rt_trap(c, 0x{a:08x}u, "tw (trap)");'], False
            return [f"if ({self.trap_cond(d, R(ra), R(rb))}) rt_trap(c, 0x{a:08x}u, \"tw\");"], True
        if op == "twi":
            return [f"if ({self.trap_cond(d, R(ra), f'(uint32_t){i.imm}')}) rt_trap(c, 0x{a:08x}u, \"twi\");"], True
        if op in ("illegal", "sc", "rfi"):
            return [f'rt_trap(c, 0x{a:08x}u, "{op} {i.word:08x}");'], False

        lines = self.simple(i)
        if lines is None:
            return self.unsup(i), True
        if LOCAL_ARGS:
            lines = [x.replace("adde_(c, ", "adde_r(c->ca, ") for x in lines]
        return lines, True

    @staticmethod
    def trap_cond(to: int, x: str, y: str) -> str:
        p = []
        if to & 16: p.append(f"(int32_t){x} < (int32_t){y}")
        if to & 8: p.append(f"(int32_t){x} > (int32_t){y}")
        if to & 4: p.append(f"{x} == {y}")
        if to & 2: p.append(f"{x} < {y}")
        if to & 1: p.append(f"{x} > {y}")
        return " || ".join(p) if p else "0"

    # ---- non-branch instructions ------------------------------------------------------------
    def simple(self, i: ppc.Insn) -> list[str] | None:
        op = i.op
        d, ra, rb, rc = i.rD, i.rA, i.rB, i.rC
        imm = i.imm
        a = i.addr
        L: list[str] = []
        rec = (lambda v: L.append(f"CR0({v});")) if i.Rc else (lambda v: None)
        if a in self.prologue_saves and op in ("stfd", "stfs", "psq_st", "ps_merge10"):
            return [f"/* {op} prologue save of f{d}: a local */"]  # find_prologue_saves

        # integer arithmetic (D-form)
        if op == "addi":
            L.append(f"{R(d)} = {RA0(ra)} + (uint32_t){imm};")
            return L
        if op == "addis":
            L.append(f"{R(d)} = {RA0(ra)} + 0x{(imm << 16) & 0xFFFFFFFF:08x}u;")
            return L
        if op in ("addic", "addic."):
            L.append(f"{{ uint32_t a_ = {R(ra)}; {R(d)} = a_ + (uint32_t){imm}; c->ca = {R(d)} < a_; }}")
            if op == "addic.":
                L.append(f"CR0({R(d)});")
            return L
        if op == "subfic":
            L.append(f"{{ uint32_t a_ = {R(ra)}; {R(d)} = adde_(c, ~a_, (uint32_t){imm}, 1); }}")
            return L
        if op == "mulli":
            L.append(f"{R(d)} = (uint32_t)((int32_t){R(ra)} * (int32_t){imm});")
            return L
        if op in ("cmpi", "cmpli", "cmp", "cmpl"):
            f = d
            if op == "cmpi":
                L.append(f"set_cr_s(c, {f}, (int32_t){R(ra)}, {imm});")
            elif op == "cmpli":
                L.append(f"set_cr_u(c, {f}, {R(ra)}, {imm}u);")
            elif op == "cmp":
                L.append(f"set_cr_s(c, {f}, (int32_t){R(ra)}, (int32_t){R(rb)});")
            else:
                L.append(f"set_cr_u(c, {f}, {R(ra)}, {R(rb)});")
            return L
        # logical immediates: rS=d, rA=ra
        if op in ("ori", "oris", "xori", "xoris", "andi.", "andis."):
            v = imm << 16 if op.endswith(("is", "is.")) else imm
            sym = {"o": "|", "x": "^", "a": "&"}[op[0]]
            L.append(f"{R(ra)} = {R(d)} {sym} 0x{v:08x}u;")
            if op.startswith("and"):
                L.append(f"CR0({R(ra)});")
            return L
        # rotates
        if op == "rlwinm":
            m = self.mask(i.mb, i.me)
            sh = imm & 31
            src = R(d) if sh == 0 else f"rotl32({R(d)}, {sh})"
            L.append(f"{R(ra)} = {src} & 0x{m:08x}u;")
            rec(R(ra))
            return L
        if op == "rlwimi":
            m = self.mask(i.mb, i.me)
            sh = imm & 31
            L.append(f"{R(ra)} = (rotl32({R(d)}, {sh}) & 0x{m:08x}u) | ({R(ra)} & 0x{(~m) & 0xFFFFFFFF:08x}u);")
            rec(R(ra))
            return L
        if op == "rlwnm":
            m = self.mask(i.mb, i.me)
            L.append(f"{R(ra)} = rotl32({R(d)}, {R(rb)} & 31) & 0x{m:08x}u;")
            rec(R(ra))
            return L

        # X-form logical: rS=d, rA=ra, rB=rb
        logic = {"and": "{s} & {b}", "or": "{s} | {b}", "xor": "{s} ^ {b}", "nand": "~({s} & {b})",
                 "nor": "~({s} | {b})", "eqv": "~({s} ^ {b})", "andc": "{s} & ~{b}", "orc": "{s} | ~{b}"}
        if op in logic:
            L.append(f"{R(ra)} = {logic[op].format(s=R(d), b=R(rb))};")
            rec(R(ra))
            return L
        if op == "extsb":
            L.append(f"{R(ra)} = (uint32_t)(int32_t)(int8_t){R(d)};")
            rec(R(ra))
            return L
        if op == "extsh":
            L.append(f"{R(ra)} = (uint32_t)(int32_t)(int16_t){R(d)};")
            rec(R(ra))
            return L
        if op == "cntlzw":
            L.append(f"{R(ra)} = {R(d)} ? (uint32_t)__builtin_clz({R(d)}) : 32u;")
            rec(R(ra))
            return L
        if op == "slw":
            L.append(f"{{ uint32_t n_ = {R(rb)} & 63; {R(ra)} = n_ & 32 ? 0u : {R(d)} << n_; }}")
            rec(R(ra))
            return L
        if op == "srw":
            L.append(f"{{ uint32_t n_ = {R(rb)} & 63; {R(ra)} = n_ & 32 ? 0u : {R(d)} >> n_; }}")
            rec(R(ra))
            return L
        if op == "sraw":
            L.append(f"{{ uint32_t n_ = {R(rb)} & 63; int32_t s_ = (int32_t){R(d)}; "
                     f"if (n_ & 32) {{ {R(ra)} = (uint32_t)(s_ >> 31); c->ca = s_ < 0; }} "
                     f"else {{ {R(ra)} = (uint32_t)(s_ >> n_); c->ca = s_ < 0 && n_ && ((uint32_t)s_ & ((1u << n_) - 1)) != 0; }} }}")
            rec(R(ra))
            return L
        if op == "srawi":
            n = imm & 31
            if n == 0:
                L.append(f"{R(ra)} = {R(d)}; c->ca = 0;")
            else:
                L.append(f"{{ int32_t s_ = (int32_t){R(d)}; {R(ra)} = (uint32_t)(s_ >> {n}); "
                         f"c->ca = s_ < 0 && ((uint32_t)s_ & 0x{(1 << n) - 1:x}u) != 0; }}")
            rec(R(ra))
            return L

        # XO-form arithmetic: rD=d, rA=ra, rB=rb
        if op in ("add", "subf", "addc", "subfc", "adde", "subfe", "addze", "subfze", "addme", "subfme",
                  "neg", "mullw", "mulhw", "mulhwu", "divw", "divwu"):
            A, B = R(ra), R(rb)
            if op == "add":
                L.append(f"{R(d)} = add_ov(c, {A}, {B});" if i.OE else f"{R(d)} = {A} + {B};")
            elif op == "subf":
                L.append(f"{R(d)} = sub_ov(c, {B}, {A});" if i.OE else f"{R(d)} = {B} - {A};")
            elif op == "addc":
                L.append(f"{{ uint32_t a_ = {A}, b_ = {B}; {R(d)} = adde_(c, a_, b_, 0); }}")
            elif op == "subfc":
                L.append(f"{{ uint32_t a_ = {A}, b_ = {B}; {R(d)} = adde_(c, ~a_, b_, 1); }}")
            elif op == "adde":
                L.append(f"{{ uint32_t a_ = {A}, b_ = {B}; {R(d)} = adde_(c, a_, b_, c->ca); }}")
            elif op == "subfe":
                L.append(f"{{ uint32_t a_ = {A}, b_ = {B}; {R(d)} = adde_(c, ~a_, b_, c->ca); }}")
            elif op == "addze":
                L.append(f"{R(d)} = adde_(c, {A}, 0, c->ca);")
            elif op == "subfze":
                L.append(f"{R(d)} = adde_(c, ~{A}, 0, c->ca);")
            elif op == "addme":
                L.append(f"{R(d)} = adde_(c, {A}, 0xFFFFFFFFu, c->ca);")
            elif op == "subfme":
                L.append(f"{R(d)} = adde_(c, ~{A}, 0xFFFFFFFFu, c->ca);")
            elif op == "neg":
                L.append(f"{R(d)} = sub_ov(c, 0, {A});" if i.OE else f"{R(d)} = 0u - {A};")
            elif op == "mullw":
                if i.OE:
                    L.append(f"{{ int64_t p_ = (int64_t)(int32_t){A} * (int32_t){B}; {R(d)} = (uint32_t)p_; "
                             f"c->ov = p_ != (int32_t)p_; c->so |= c->ov; }}")
                else:
                    L.append(f"{R(d)} = (uint32_t)((int32_t){A} * (int32_t){B});")
            elif op == "mulhw":
                L.append(f"{R(d)} = (uint32_t)(((int64_t)(int32_t){A} * (int32_t){B}) >> 32);")
            elif op == "mulhwu":
                L.append(f"{R(d)} = (uint32_t)(((uint64_t){A} * {B}) >> 32);")
            elif op == "divw":
                L.append(f"{R(d)} = ppc_divw((int32_t){A}, (int32_t){B});")
            elif op == "divwu":
                L.append(f"{R(d)} = ppc_divwu({A}, {B});")
            if i.OE and op not in ("add", "subf", "neg", "mullw"):
                pass  # carry-form overflow bits are not tracked (not used by this binary)
            rec(R(d))
            return L

        # loads/stores (D-form)
        dform_ld = {"lwz": "LD32({ea})", "lbz": "LD8({ea})", "lhz": "LD16({ea})",
                    "lha": "(uint32_t)(int32_t)(int16_t)LD16({ea})"}
        dform_st = {"stw": "ST32", "stb": "ST8", "sth": "ST16"}
        base = op[:-1] if op.endswith("u") and op[:-1] in dform_ld | dform_st else op
        upd = base != op
        if base in dform_ld or base in dform_st:
            ea = f"({RA0(ra)} + (uint32_t){imm})" if not upd else f"({R(ra)} + (uint32_t){imm})"
            if upd:
                L.append(f"{{ uint32_t ea_ = {ea};")
                ea = "ea_"
            if base in dform_ld:
                L.append(f"{R(d)} = {dform_ld[base].format(ea=ea)};")
            else:
                L.append(f"{dform_st[base]}({ea}, {R(d)});")
            if upd:
                L.append(f"{R(ra)} = ea_; }}")
            return L
        # loads/stores (X-form)
        xform = {"lwzx": ("ld", "LD32({ea})"), "lbzx": ("ld", "LD8({ea})"), "lhzx": ("ld", "LD16({ea})"),
                 "lhax": ("ld", "(uint32_t)(int32_t)(int16_t)LD16({ea})"),
                 "lwbrx": ("ld", "ld32r_(M, {ea})"), "lhbrx": ("ld", "(uint32_t)ld16r_(M, {ea})"),
                 "stwx": ("st", "ST32"), "stbx": ("st", "ST8"), "sthx": ("st", "ST16"),
                 "stwbrx": ("st", "st32r_"), "sthbrx": ("st", "st16r_")}
        xbase = op.replace("ux", "x") if op.endswith("ux") else op
        if xbase in xform:
            upd = op != xbase
            kind, tmpl = xform[xbase]
            ea_e = f"({R(ra)} + {R(rb)})" if upd else f"({RA0(ra)} + {R(rb)})"
            L.append(f"{{ uint32_t ea_ = {ea_e};")
            if kind == "ld":
                L.append(f"{R(d)} = {tmpl.format(ea='ea_')};")
            elif tmpl.endswith("r_"):
                L.append(f"{tmpl}(M, ea_, {R(d)});")
            else:
                L.append(f"{tmpl}(ea_, {R(d)});")
            if upd:
                L.append(f"{R(ra)} = ea_;")
            L.append("}")
            return L
        if op == "stmw" and a in self.prologue_saves:
            # prologue save of the caller's r14-r31: those are this function's own locals (local_regs), the
            # caller's values are never needed. Only the entry block's stmw before any write of those registers:
            # GHS also copies blocks with lmw/stmw. The epilogue lmw stays: its loads are dead and clang drops them.
            return ["/* stmw prologue save of r14-r31: locals */"]
        if op == "lmw":
            L.append(f"{{ uint32_t ea_ = {RA0(ra)} + (uint32_t){imm};")
            for k in range(d, 32):
                L.append(f"{R(k)} = LD32(ea_ + {4 * (k - d)}u);")
            L.append("}")
            return L
        if op == "stmw":
            L.append(f"{{ uint32_t ea_ = {RA0(ra)} + (uint32_t){imm};")
            for k in range(d, 32):
                L.append(f"ST32(ea_ + {4 * (k - d)}u, {R(k)});")
            L.append("}")
            return L
        if op == "lswi":
            n = imm
            L.append(f"{{ uint32_t ea_ = {RA0(ra)};")
            reg = d - 1
            for k in range(n):
                if k % 4 == 0:
                    reg = (reg + 1) % 32
                    L.append(f"{R(reg)} = 0;")
                L.append(f"{R(reg)} |= (uint32_t)LD8(ea_ + {k}u) << {24 - 8 * (k % 4)};")
            L.append("}")
            return L
        if op == "stswi":
            n = imm
            L.append(f"{{ uint32_t ea_ = {RA0(ra)};")
            reg = d - 1
            for k in range(n):
                if k % 4 == 0:
                    reg = (reg + 1) % 32
                L.append(f"ST8(ea_ + {k}u, {R(reg)} >> {24 - 8 * (k % 4)});")
            L.append("}")
            return L
        if op == "lwarx":
            L.append(f"{R(d)} = ppc_lwarx(c, M, {RA0(ra)} + {R(rb)});")
            return L
        if op == "stwcx.":
            L.append(f"ppc_stwcx(c, M, {RA0(ra)} + {R(rb)}, {R(d)});")
            return L

        # floating point loads/stores
        fl = {"lfs": "s", "lfsu": "s", "lfd": "d", "lfdu": "d"}
        fs = {"stfs": "s", "stfsu": "s", "stfd": "d", "stfdu": "d"}
        if op in fl or op in fs:
            upd = op.endswith("u")
            ea = f"({R(ra) if upd else RA0(ra)} + (uint32_t){imm})"
            L.append(f"{{ uint32_t ea_ = {ea};")
            L += self.fp_mem(op[:4] if op.startswith("stf") else op[:3], d)
            if upd:
                L.append(f"{R(ra)} = ea_;")
            L.append("}")
            return L
        fx = {"lfsx": "lfs", "lfsux": "lfs", "lfdx": "lfd", "lfdux": "lfd",
              "stfsx": "stfs", "stfsux": "stfs", "stfdx": "stfd", "stfdux": "stfd"}
        if op in fx:
            upd = op.endswith("ux")
            ea = f"({R(ra) if upd else RA0(ra)} + {R(rb)})"
            L.append(f"{{ uint32_t ea_ = {ea};")
            L += self.fp_mem(fx[op], d)
            if upd:
                L.append(f"{R(ra)} = ea_;")
            L.append("}")
            return L
        if op == "stfiwx":
            L.append(f"ST32({RA0(ra)} + {R(rb)}, (uint32_t)dbits({F(d)}));")
            return L
        if op in ("psq_l", "psq_lu", "psq_st", "psq_stu", "psq_lx", "psq_lux", "psq_stx", "psq_stux"):
            upd = op in ("psq_lu", "psq_stu", "psq_lux", "psq_stux")
            if op in ("psq_l", "psq_lu", "psq_st", "psq_stu"):
                ea = f"({R(ra) if upd else RA0(ra)} + (uint32_t){imm})"
            else:
                ea = f"({R(ra) if upd else RA0(ra)} + {R(rb)})"
            # register halves passed by reference (they may be locals, see local_regs)
            if i.I == 0:
                # GQR0 stays 0 (plain floats, no scale): the OS sets it so and BotW never writes it, so skip
                # the per-access type switch (99.5% of BotW's paired-single loads and stores)
                fnn = "psq_load_f32r" if "_l" in op else "psq_store_f32r"
                L.append(f"{{ uint32_t ea_ = {ea}; {fnn}({F(d)}, {P(d)}, M, ea_, {i.W});")
            else:
                fnn = "psq_loadr" if "_l" in op else "psq_storer"
                L.append(f"{{ uint32_t ea_ = {ea}; {fnn}({F(d)}, {P(d)}, M, ea_, {i.W}, c->gqr[{i.I}]);")
            if upd:
                L.append(f"{R(ra)} = ea_;")
            L.append("}")
            return L

        # SPRs / CR
        if op == "mfspr":
            spr = imm
            if spr == 8:
                L.append(f"{R(d)} = c->lr;")
            elif spr == 9:
                L.append(f"{R(d)} = c->ctr;")
            elif spr == 1:
                L.append(f"{R(d)} = ((uint32_t)c->so << 31) | ((uint32_t)c->ov << 30) | ((uint32_t)c->ca << 29) | c->xer_bc;")
            elif 896 <= spr <= 903 or 912 <= spr <= 919:
                L.append(f"{R(d)} = c->gqr[{spr & 7}];")
            elif spr in (268, 269):
                L.append(f"{R(d)} = (uint32_t)(rt_timebase() >> {32 if spr == 269 else 0});")
            else:
                return None
            return L
        if op == "mtspr":
            spr = imm
            if spr == 8:
                L.append(f"c->lr = {R(d)};")
            elif spr == 9:
                L.append(f"c->ctr = {R(d)};")
            elif spr == 1:
                L.append(f"{{ uint32_t x_ = {R(d)}; c->so = x_ >> 31; c->ov = (x_ >> 30) & 1; c->ca = (x_ >> 29) & 1; c->xer_bc = x_ & 0x7F; }}")
            elif 896 <= spr <= 903 or 912 <= spr <= 919:
                L.append(f"c->gqr[{spr & 7}] = {R(d)};")
            else:
                return None
            return L
        if op == "mftb":
            L.append(f"{R(d)} = (uint32_t)(rt_timebase() >> {32 if imm == 269 else 0});")
            return L
        if op == "mfcr":
            parts = " | ".join(f"((uint32_t)c->cr[{b}] << {31 - b})" for b in range(32))
            L.append(f"{R(d)} = {parts};")
            return L
        if op == "mtcrf":
            crm = imm
            L.append(f"{{ uint32_t v_ = {R(d)};")
            for f in range(8):
                if crm & (0x80 >> f):
                    for k in range(4):
                        b = 4 * f + k
                        L.append(f"c->cr[{b}] = (v_ >> {31 - b}) & 1;")
            L.append("}")
            return L
        if op == "mcrf":
            for k in range(4):
                if d != ra:
                    L.append(f"c->cr[{4 * d + k}] = c->cr[{4 * ra + k}];")
            return L
        if op == "mcrxr":
            L.append(f"c->cr[{4 * d}] = c->so; c->cr[{4 * d + 1}] = c->ov; c->cr[{4 * d + 2}] = c->ca; c->cr[{4 * d + 3}] = 0;")
            L.append("c->so = c->ov = c->ca = 0;")
            return L
        crops = {"crand": "{a} & {b}", "cror": "{a} | {b}", "crxor": "{a} ^ {b}", "crnand": "!({a} & {b})",
                 "crnor": "!({a} | {b})", "creqv": "!({a} ^ {b})", "crandc": "{a} & !{b}", "crorc": "{a} | !{b}"}
        if op in crops:
            L.append(f"c->cr[{d}] = (uint8_t)({crops[op].format(a=f'c->cr[{ra}]', b=f'c->cr[{rb}]')});")
            return L

        # cache / sync
        if op in ("sync", "eieio"):
            L.append("__atomic_thread_fence(__ATOMIC_SEQ_CST);")
            return L
        if op == "isync":
            L.append("__asm__ volatile(\"\" ::: \"memory\");")
            return L
        if op in ("dcbst", "dcbf", "dcbi", "dcbt", "dcbtst", "icbi"):
            return ["/* cache op */"]
        if op in ("dcbz", "dcbz_l"):
            L.append(f"memset(M + (({RA0(ra)} + {R(rb)}) & ~31u), 0, 32);")
            return L

        return self.fpu(i)

    def fp_mem(self, op: str, d: int) -> list[str]:
        if op == "lfs":
            return [f"{F(d)} = {P(d)} = f32_to_f64(LD32(ea_));"]
        if op == "lfd":
            return [f"{F(d)} = bitsd(LD64(ea_));"]
        if op == "stfs":
            return [f"ST32(ea_, f64_to_f32({F(d)}));"]
        if op == "stfd":
            return [f"ST64(ea_, dbits({F(d)}));"]
        raise ValueError(op)

    @staticmethod
    def mask(mb: int, me: int) -> int:
        m1 = 0xFFFFFFFF >> mb
        m2 = 0xFFFFFFFF if me >= 31 else (~(0xFFFFFFFF >> (me + 1))) & 0xFFFFFFFF
        return (m1 & m2) if mb <= me else (m1 | m2)

    # ---- FPU / paired singles ---------------------------------------------------------------
    def fpu(self, i: ppc.Insn) -> list[str] | None:
        op = i.op
        d, a_, b, c_ = i.rD, i.rA, i.rB, i.rC
        L: list[str] = []
        fa, fb, fc = F(a_), F(b), F(c_)
        pa, pb, pc = P(a_), P(b), P(c_)
        if FPCC_DROP and op in ("mffs", "mcrfs"):
            _fpscr_reader(i)

        def single(expr: str) -> list[str]:
            # result rounded to single; ps1 mirrors ps0 (HID2[PSE] set on Cafe)
            return [f"{{ double t_ = round_single({expr}); {F(d)} = t_; {P(d)} = t_; }}"]

        # Float semantics are Cemu's recompiler (PPCRecompilerImlGenFPU.cpp + BackendX64FPU.cpp), op for op:
        # multiply-adds are an unfused mulsd + addsd/subsd (the guest is built with -ffp-contract=off), single forms
        # round once at the end (cvtsd2ss + cvtss2sd) with no 25-bit rounding of frC. fnmadd/fnmadds are not taken
        # by Cemu's recompiler, so they follow its interpreter (PPCInterpreterFPU.cpp).
        dbl = {"fadd": f"{fa} + {fb}", "fsub": f"{fa} - {fb}", "fmul": f"{fa} * {fc}", "fdiv": f"{fa} / {fb}",
               "fmadd": f"{fa} * {fc} + {fb}", "fmsub": f"{fa} * {fc} - {fb}",
               "fnmadd": f"-({fa} * {fc} + {fb})", "fnmsub": f"-({fa} * {fc} - {fb})", "fsqrt": f"sqrt({fb})"}
        sgl = {"fadds": f"{fa} + {fb}", "fsubs": f"{fa} - {fb}", "fmuls": f"{fa} * {fc}",
               "fdivs": f"{fa} / {fb}", "fmadds": f"{fa} * {fc} + {fb}",
               "fmsubs": f"{fa} * {fc} - {fb}", "fnmadds": f"-({fa} * round25({fc}) + {fb})",
               "fnmsubs": f"-({fa} * {fc} - {fb})", "fsqrts": f"sqrt({fb})"}
        if op in dbl:
            L.append(f"{F(d)} = {dbl[op]};")
        elif op in sgl:
            L += single(sgl[op])
        elif op == "fres":
            L.append(f"{{ double t_ = round_single(ppc_fres({fb})); {F(d)} = t_; {P(d)} = t_; }}")
        elif op == "frsqrte":
            L.append(f"{F(d)} = round_single(ppc_frsqrte({fb}));")
        elif op == "frsp":
            L += single(fb)
        elif op == "fsel":
            L.append(f"if ({fa} >= -0.0) {{ {F(d)} = {fc}; {P(d)} = {pc}; }} else {{ {F(d)} = {fb}; {P(d)} = {pb}; }}")
        elif op == "fmr":
            L.append(f"{F(d)} = {fb};")
        elif op == "fneg":
            L.append(f"{F(d)} = bitsd(dbits({fb}) ^ 0x8000000000000000ULL);")
        elif op == "fabs":
            L.append(f"{F(d)} = bitsd(dbits({fb}) & ~0x8000000000000000ULL);")
        elif op == "fnabs":
            L.append(f"{F(d)} = bitsd(dbits({fb}) | 0x8000000000000000ULL);")
        elif op == "fctiwz":
            L.append(f"{F(d)} = bitsd(ppc_fctiwz({fb}));")
        elif op == "fctiw":
            L.append(f"{F(d)} = bitsd(ppc_fctiw({fb}));")
        elif op in ("fcmpu", "fcmpo"):
            L.append(f"set_cr_f(c, {d}, {fa}, {fb});")
            return L
        elif op == "mffs":
            L.append(f"{F(d)} = bitsd((uint64_t)c->fpscr);")
        elif op == "mtfsf":
            fm = i.imm
            mask = 0
            for k in range(8):
                if fm & (0x80 >> k):
                    mask |= 0xF << (28 - 4 * k)
            L.append(f"c->fpscr = ((uint32_t)dbits({fb}) & 0x{mask:08x}u) | (c->fpscr & 0x{(~mask) & 0xFFFFFFFF:08x}u);")
        elif op == "mtfsfi":
            sh = 28 - 4 * d
            L.append(f"c->fpscr = (c->fpscr & ~(0xFu << {sh})) | ({i.imm}u << {sh});")
        elif op == "mtfsb0":
            L.append(f"c->fpscr &= ~(1u << {31 - d});")
        elif op == "mtfsb1":
            L.append(f"c->fpscr |= 1u << {31 - d};")
        elif op == "mcrfs":
            sh = 28 - 4 * a_
            for k in range(4):
                L.append(f"c->cr[{4 * d + k}] = (c->fpscr >> {sh + 3 - k}) & 1;")
            return L
        # paired singles
        elif op == "ps_add":
            L += self.ps2(d, f"{fa} + {fb}", f"{pa} + {pb}")
        elif op == "ps_sub":
            L += self.ps2(d, f"{fa} - {fb}", f"{pa} - {pb}")
        # paired singles as Cemu's recompiler does them: double math, one cvtsd2ss/cvtss2sd rounding per slot, no
        # 25-bit frC rounding, no flush-to-zero; ps_nmadd is not rounded at all there
        elif op == "ps_mul":
            L += self.ps2(d, f"{fa} * {fc}", f"{pa} * {pc}")
        elif op == "ps_div":
            L += self.ps2(d, f"{fa} / {fb}", f"{pa} / {pb}")
        elif op == "ps_madd":
            L += self.ps2(d, f"{fa} * {fc} + {fb}", f"{pa} * {pc} + {pb}")
        elif op == "ps_msub":
            L += self.ps2(d, f"{fa} * {fc} - {fb}", f"{pa} * {pc} - {pb}")
        elif op == "ps_nmadd":
            L.append(f"{{ double x_ = -({fa} * {fc} + {fb}), y_ = -({pa} * {pc} + {pb}); {F(d)} = x_; {P(d)} = y_; }}")
        elif op == "ps_nmsub":
            L += self.ps2(d, f"-({fa} * {fc} - {fb})", f"-({pa} * {pc} - {pb})")
        elif op == "ps_madds0":
            L.append(f"{{ double k_ = {fc}; double x_ = (float)({fa} * k_ + {fb}), y_ = (float)({pa} * k_ + {pb}); {F(d)} = x_; {P(d)} = y_; }}")
        elif op == "ps_madds1":
            L.append(f"{{ double k_ = {pc}; double x_ = (float)({fa} * k_ + {fb}), y_ = (float)({pa} * k_ + {pb}); {F(d)} = x_; {P(d)} = y_; }}")
        elif op == "ps_muls0":
            L.append(f"{{ double k_ = {fc}; double x_ = (float)({fa} * k_), y_ = (float)({pa} * k_); {F(d)} = x_; {P(d)} = y_; }}")
        elif op == "ps_muls1":
            L.append(f"{{ double k_ = {pc}; double x_ = (float)({fa} * k_), y_ = (float)({pa} * k_); {F(d)} = x_; {P(d)} = y_; }}")
        elif op == "ps_sum0":
            L.append(f"{{ double x_ = (float)({fa} + {pb}), y_ = (float)({pc}); {F(d)} = x_; {P(d)} = y_; }}")
        elif op == "ps_sum1":
            L.append(f"{{ double x_ = (float)({fc}), y_ = (float)({fa} + {pb}); {F(d)} = x_; {P(d)} = y_; }}")
        elif op == "ps_sel":
            L.append(f"{{ double x_ = {fa} >= -0.0 ? {fc} : {fb}; double y_ = {pa} >= -0.0 ? {pc} : {pb}; {F(d)} = x_; {P(d)} = y_; }}")
        elif op == "ps_res":
            L.append(f"{{ double x_ = (float)ppc_fres({fb}), y_ = (float)ppc_fres({pb}); {F(d)} = x_; {P(d)} = y_; }}")
        elif op == "ps_rsqrte":
            L.append(f"{{ double x_ = (float)ppc_frsqrte({fb}), y_ = (float)ppc_frsqrte({pb}); {F(d)} = x_; {P(d)} = y_; }}")
        elif op == "ps_mr":
            L.append(f"{{ double x_ = {fb}, y_ = {pb}; {F(d)} = x_; {P(d)} = y_; }}")
        elif op == "ps_neg":
            L.append(f"{{ double x_ = bitsd(dbits({fb}) ^ 0x8000000000000000ULL), y_ = bitsd(dbits({pb}) ^ 0x8000000000000000ULL); {F(d)} = x_; {P(d)} = y_; }}")
        elif op == "ps_abs":
            L.append(f"{{ double x_ = bitsd(dbits({fb}) & ~0x8000000000000000ULL), y_ = bitsd(dbits({pb}) & ~0x8000000000000000ULL); {F(d)} = x_; {P(d)} = y_; }}")
        elif op == "ps_nabs":
            L.append(f"{{ double x_ = bitsd(dbits({fb}) | 0x8000000000000000ULL), y_ = bitsd(dbits({pb}) | 0x8000000000000000ULL); {F(d)} = x_; {P(d)} = y_; }}")
        elif op in ("ps_merge00", "ps_merge01", "ps_merge10", "ps_merge11"):
            x = fa if op[8] == "0" else pa
            y = fb if op[9] == "0" else pb
            L.append(f"{{ double x_ = {x}, y_ = {y}; {F(d)} = x_; {P(d)} = y_; }}")
        elif op in ("ps_cmpu0", "ps_cmpo0"):
            L.append(f"set_cr_f(c, {d}, {fa}, {fb});")
            return L
        elif op in ("ps_cmpu1", "ps_cmpo1"):
            L.append(f"set_cr_f(c, {d}, {pa}, {pb});")
            return L
        else:
            return None
        if i.Rc:
            if FPCC_DROP:
                _fpscr_reader(i)
            L.append("c->cr[4] = c->fpscr >> 31; c->cr[5] = (c->fpscr >> 30) & 1; c->cr[6] = (c->fpscr >> 29) & 1; c->cr[7] = (c->fpscr >> 28) & 1;")
        return L

    @staticmethod
    def ps2(d: int, e0: str, e1: str, ftz: bool = False) -> list[str]:
        if ftz:
            return [f"{{ double x_ = ftz_single({e0}), y_ = ftz_single({e1}); c->f[{d}] = x_; c->ps1[{d}] = y_; }}"]
        return [f"{{ double x_ = (float)({e0}), y_ = (float)({e1}); c->f[{d}] = x_; c->ps1[{d}] = y_; }}"]
