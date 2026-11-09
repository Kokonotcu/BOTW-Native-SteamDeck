"""Which registers each function must keep in the PPCContext instead of C locals (codegen local_regs).

The EABI says a callee never reads the caller's scratch registers (r0/r11/r12/f0/f9-f13/ctr) and preserves r1 and
r14-r31/f14-f31. BotW's compiler does not always keep the first rule: with interprocedural register allocation a
caller keeps a value in r0/f0 across a call to a callee it knows leaves r0 alone, and some code (FPS++ caves) passes
values in r11/r12. Locals keep "left alone" semantics by themselves (the callee cannot touch the caller's local);
what they cannot do is carry a value from caller to callee or back. So per function:
  LI(F)   registers F reads from its caller (live at entry; for r14-r31/f14-f31 not counting the frame saves)
  MW(F)   scratch registers F or its callees may write (indirect calls: every scratch register)
  NC(F)   non-volatile registers F or its callees write without F saving them
  EXP(F)  registers F must deliver to its caller (the caller reads them after the call and F may write them)
A call that carries a register (in: LI of the callee, out: EXP) keeps it in the context in both functions; the rest
are locals. Register sets are bitmasks. compute() returns {entry: Sync}: a register a call carries is synced at
that call only (the caller stores LI|EXP of the callee before it and reloads EXP after; the callee loads its
locals at entry and stores EXP at its exits), so both functions keep it in a local everywhere else.
"""
import collections
import os
import re

NAMES = (["r0", "r11", "r12", "f0", "f9", "f10", "f11", "f12", "f13", "ctr"]
         + [f"r{n}" for n in range(14, 32)] + [f"f{n}" for n in range(14, 32)])
BIT = {n: 1 << i for i, n in enumerate(NAMES)}
SCRATCH = sum(BIT[n] for n in NAMES[:10])
NONVOL = sum(BIT[n] for n in NAMES[10:])
_RE_REF = re.compile(r"c->(r|f|ps1)\[(\d+)\]|c->ctr")
_RE_DEF = re.compile(r"c->(r|f|ps1)\[(\d+)\] = (?!=)|c->ctr = (?!=)")
_RE_DIRECT = re.compile(r"(?<!return )\bf_([0-9a-f]{8})\(c\);")
_RE_TAIL = re.compile(r"musttail\]\] return f_([0-9a-f]{8})\(c\);")
_RE_SAVE = re.compile(r"ST32\(\(c->r\[1\] \+ \(uint32_t\)-?\d+\), c->r\[(\d+)\]\)")
CALL, TAIL, IND = 1, 2, 3
_NOMASK = os.environ.get("BOTW_REGCTX_NOMASK") == "1"  # TEST: track r14-r31/f14-f31 live-ins everywhere
# A value a function may pass through unchanged to a caller that reads it after the call (EXP) is not its live-in:
# the caller stores it before the call (codegen) and, inside each function, it stays live back to where it was
# set, so a callee that delivered it gets a reload (EXPd in _liveness). Counting it as LI (the old way) carried
# it up through every caller chain, mostly over paths where the caller never reads it. BOTW_REGCTX_PT=1: old way.
_PT = os.environ.get("BOTW_REGCTX_PT") == "1"
_RE_NV_STORE_VALUE = re.compile(
    r", (?:c->r\[(?:1[4-9]|2\d|3[01])\]|dbits\(c->f\[(?:1[4-9]|2\d|3[01])\]\)|f64_to_f32\(c->f\[(?:1[4-9]|2\d|3[01])\]\))\)$")


def _bit(m: re.Match) -> int:
    if m.group(0).startswith("c->ctr"):
        return BIT["ctr"]
    return BIT.get(("f" if m.group(1) in ("f", "ps1") else "r") + m.group(2), 0)


def _refs(s: str) -> int:
    b = 0
    for x in _RE_REF.finditer(s):
        b |= _bit(x)
    return b


def def_use(text: str) -> tuple[int, int, int]:
    """(defs, uses, saves) bitmasks: statements in order, a register read before the instruction writes it is a
    use; a non-volatile register stored to an r1-based slot (frame save) is a save, not a use."""
    defs = uses = saves = 0
    frame = "c->r[1] +" in text
    for st in text.replace("{", ";").replace("}", ";").split(";"):
        if _NOMASK:  # TEST: a store whose value is a non-volatile register is a save, whatever the address
            st = _RE_NV_STORE_VALUE.sub(")", st)
        m = _RE_SAVE.search(st)
        if m and int(m.group(1)) >= 14:
            saves |= BIT[f"r{m.group(1)}"]
            continue
        if frame and ("ST64(ea_, dbits(c->f[" in st or "psq_store" in st):
            fm = re.search(r"c->f\[(\d+)\]", st)
            if fm and int(fm.group(1)) >= 14:
                saves |= BIT[f"f{fm.group(1)}"]
                continue
        pl = re.search(r"psq_load(?:_f32)?r\(c->f\[(\d+)\], c->ps1\[\d+\],(.*)", st)
        if pl:
            uses |= _refs(pl.group(2)) & ~defs
            defs |= BIT.get(f"f{pl.group(1)}", 0)
            continue
        lhs = 0
        rest = st
        while True:
            m = _RE_DEF.match(rest.lstrip())
            if not m:
                break
            lhs |= _bit(m)
            rest = rest.lstrip()[m.end():]
        uses |= _refs(rest) & ~defs
        defs |= lhs
    if "c->ctr--" in text and not defs & BIT["ctr"]:
        uses |= BIT["ctr"]
    return defs, uses, saves


class _Fn:
    __slots__ = ("entry", "d", "u", "succ", "kind", "callee", "cond", "writes", "saves", "callees", "indirect")


# paired singles are saved as stfd + ps_merge10 fX,fX,fX (swap) + stfs, restored the same way
_FRAME_ST = {"stw": "r", "stfd": "f", "stfs": "f", "psq_st": "f"}
_FRAME_LD = {"lwz": "r", "lfd": "f", "lfs": "f", "psq_l": "f"}


def _frame_slots(fn) -> tuple[dict[int, int], int]:
    """Preserved registers: stored to an r1 slot and loaded back from the same slot (GHS saves scratch registers
    too when its interprocedural allocation promises callers they survive). Returns ({store addr: bit}, mask)."""
    stores, loads = {}, set()
    for a, i in fn.insns.items():
        if i.rA != 1:
            continue
        if i.op in _FRAME_ST:
            n = f"{_FRAME_ST[i.op]}{i.rD}"
            if n in BIT:
                stores[a] = (BIT[n], i.imm)
        elif i.op in _FRAME_LD:
            n = f"{_FRAME_LD[i.op]}{i.rD}"
            if n in BIT:
                loads.add((BIT[n], i.imm))
    kept = {a: b for a, (b, k) in stores.items() if (b, k) in loads}
    mask = 0
    for b in kept.values():
        mask |= b
    return kept, mask


def _scan(an, gen) -> dict[int, _Fn]:
    fns = {}
    for entry, fn in an.functions.items():
        f = _Fn()
        f.entry = entry
        addrs = sorted(fn.insns)
        idx = {a: k for k, a in enumerate(addrs)}
        f.d, f.u, f.succ, f.kind, f.callee, f.cond = [], [], [], [], [], []
        f.writes = f.saves = 0
        f.callees, f.indirect = set(), False
        save_at, f.saves = _frame_slots(fn)
        tails = []
        for a in addrs:
            i = fn.insns[a]
            code, falls = gen.insn(fn, i, set(), set())
            text = " ".join(code)
            d, u, s = def_use(text)
            if a in save_at:
                u &= ~save_at[a]  # the save reads the register only to restore it
            elif i.op == "ps_merge10" and i.rD == i.rA == i.rB and (BIT.get(f"f{i.rD}", 0) & f.saves
                                                                    or (_NOMASK and i.rD >= 14)):
                b = BIT[f"f{i.rD}"]  # the half swap inside a paired-single save or restore
                u &= ~b
                d &= ~b
            f.writes |= d
            f.saves |= s
            if i.op == "stmw" and i.rA == 1:
                for n in range(max(i.rD, 14), 32):
                    f.saves |= BIT[f"r{n}"]
                    u &= ~BIT[f"r{n}"]  # the frame save reads them only to restore them
            succ = []
            if falls and a + 4 in idx:
                succ.append(idx[a + 4])
            if i.op in ("b", "bc") and not i.LK and i.target in idx and i.target not in fn.tail_targets:
                succ.append(idx[i.target])
            sw = fn.switches.get(a)
            if sw is not None:
                succ += [idx[t] for t in sw.targets if t in idx]
            kind, callee = 0, -1
            m = _RE_TAIL.search(text)
            if m:
                kind, callee = TAIL, int(m.group(1), 16)
            else:
                m = _RE_DIRECT.search(text)
                if m:
                    kind, callee = CALL, int(m.group(1), 16)
                elif "rt_call(c," in text:
                    kind = IND
                    f.indirect = True
            if callee >= 0:
                f.callees.add(callee)
            # falling into the next function's entry: gen_function emits a tail call (GHS save/restore helpers
            # are chains of such entries), so the graph gets one too, as an extra node after this instruction
            fall_tail = falls and a + 4 not in idx and a + 4 in gen.entries
            if fall_tail:
                succ.append(-1)
            f.d.append(d)
            f.u.append(u)
            f.succ.append(tuple(succ))
            f.kind.append(kind)
            f.callee.append(callee)
            f.cond.append(kind != 0 and text.lstrip().startswith("if ("))
            if fall_tail:
                f.callees.add(a + 4)
                tails.append((len(f.d) - 1, a + 4))
        for k, t in tails:  # the extra tail-call nodes go at the end; -1 in succ points at them
            j = len(f.d)
            f.succ[k] = tuple(j if s == -1 else s for s in f.succ[k])
            f.d.append(0)
            f.u.append(0)
            f.succ.append(())
            f.kind.append(TAIL)
            f.callee.append(t)
            f.cond.append(False)
        fns[entry] = f
        f_entry_index[entry] = idx.get(entry, 0)
    return fns


f_entry_index: dict[int, int] = {}


def _fix(fns, base: dict[int, int]) -> dict[int, int]:
    """Least fixpoint of S(F) = base(F) | S(callees)."""
    S = dict(base)
    callers = collections.defaultdict(set)
    for e, f in fns.items():
        for g in f.callees:
            callers[g].add(e)
    work = collections.deque(fns)
    queued = set(fns)
    while work:
        e = work.popleft()
        queued.discard(e)
        new = S[e]
        for g in fns[e].callees:
            new |= S.get(g, 0)
        if new != S[e]:
            S[e] = new
            for k in callers[e]:
                if k not in queued:
                    queued.add(k)
                    work.append(k)
    return S


def _liveness(f: _Fn, LI, MW, NC, exit_live: int, post: list | None, EXPd=None) -> int:
    """Backward liveness. exit_live: registers this function delivers to its caller (live at its returns).
    post collects (caller, callee or None, bitmask) for values a call delivers to this function. EXPd: registers
    each callee delivers; a callee may also pass such a value through unchanged, so with EXPd given a value live
    after the call that the callee delivers is live before it too (and at a tail call all of exit_live is)."""
    n = len(f.d)
    live = [0] * n

    def clob(k):
        if f.kind[k] == IND:
            return 0
        g = f.callee[k]
        return MW.get(g, SCRATCH) | NC.get(g, 0)

    changed = True
    while changed:
        changed = False
        for k in range(n - 1, -1, -1):
            kind = f.kind[k]
            s = f.succ[k]
            out = 0
            for j in s:
                out |= live[j]
            if not s and kind != TAIL:
                out = exit_live
            d, u = f.d[k], f.u[k]
            if kind == TAIL:
                li = u | LI.get(f.callee[k], 0) | (exit_live if EXPd is not None else exit_live & ~clob(k))
                if f.cond[k]:
                    li |= out & ~d
            elif kind:
                li = u | LI.get(f.callee[k], 0) | (out & ~d & ~(0 if f.cond[k] else clob(k)))
                if EXPd is not None and kind == CALL:
                    li |= out & ~d & EXPd.get(f.callee[k], 0)
            else:
                li = u | (out & ~d)
            if li != live[k]:
                live[k] = li
                changed = True
    if post is not None:
        for k in range(n):
            kind = f.kind[k]
            if kind in (CALL, IND):
                out = 0
                for j in f.succ[k]:
                    out |= live[j]
                if not f.succ[k]:
                    out = exit_live
                x = out & ~f.d[k] & clob(k)
                if x:
                    post.append((f.entry, f.callee[k] if kind == CALL else None, x))
            elif kind == TAIL:
                x = exit_live & clob(k)
                if x:
                    post.append((f.entry, f.callee[k], x))
    return live[f_entry_index.get(f.entry, 0)] if n else 0


def _fps_cave(entry: int) -> bool:
    """Graphic pack caves (tools/gfxpack.py, 0x01800000, below the game's .text): entered by bla from the middle
    of a game function, they read and write that function's registers, r14-r31/f14-f31 included."""
    return entry < 0x02000000


def _ghs_helper(entry: int) -> bool:
    """The GHS runtime's register save/restore and prologue helpers: they save and restore the caller's
    registers through r11, move r1, and one returns its caller's address in r31."""
    return 0x04211ffc <= entry < 0x042124a8


def _cave(entry: int) -> bool:
    """Code outside the EABI for r14-r31/f14-f31 (writes them for its caller)."""
    return _fps_cave(entry) or _ghs_helper(entry)


# Caves and helpers keep every register in the context (exactly the old semantics): a restore helper must write
# back the context copy of a register whatever its caller does, and these tiny functions gain nothing from locals.
ALL_HELD = frozenset(NAMES) | {"r1"}


def _names(b: int) -> set[str]:
    return {n for n in NAMES if b & BIT[n]}


def compute(an, gen, log=print, origins=None) -> dict[int, set[str]]:
    fns = _scan(an, gen)
    # A scratch register the function saves and restores is not clobbered for its callers. Indirect calls add
    # nothing: the compiler cannot know their target, so it never keeps or receives a scratch value through one.
    MW = _fix(fns, {e: f.writes & SCRATCH & ~f.saves for e, f in fns.items()})
    # r14-r31/f14-f31 follow the EABI in compiled code (read only to be saved, never returned); only the graphic
    # pack caves, entered by bla from the middle of a game function, read and write that function's registers.
    NC = _fix(fns, {e: (f.writes & NONVOL & ~f.saves) if _cave(e) else 0 for e, f in fns.items()})
    # a compiled function restores r14-r31 for its callers even when a helper it calls writes them
    NC = {e: (v if _cave(e) else 0) for e, v in NC.items()}
    callers = collections.defaultdict(set)
    for e, f in fns.items():
        for g in f.callees:
            callers[g].add(e)
    LI = {e: 0 for e in fns}
    EXP = collections.defaultdict(int)
    post = []
    indirect_returns = 0
    rounds = 0
    for rounds in range(1, 51):
        work = collections.deque(fns)
        queued = set(fns)
        while work:
            e = work.popleft()
            queued.discard(e)
            li = _liveness(fns[e], LI, MW, NC, EXP[e] if _PT else 0, None)
            if not _fps_cave(e) and not _NOMASK:
                li &= ~NONVOL  # compiled code and the GHS helpers read the caller's r14-r31/f14-f31 only to save them
            if li != LI[e]:
                LI[e] = li
                for k in callers[e]:
                    if k not in queued:
                        queued.add(k)
                        work.append(k)
        post = []
        for e, f in fns.items():
            _liveness(f, LI, MW, NC, EXP[e], post, None if _PT else EXP)
        grew = False
        indirect_returns = 0
        if origins is not None and rounds == 1:  # first-round deliveries: the seeds of EXP
            for e, g, x in post:
                for n in _names(x):
                    if len(origins["seed " + n]) < 6:
                        origins["seed " + n].append(f"f_{e:08x}->{'ind' if g is None else f'f_{g:08x}'}")
        for e, g, x in post:
            if g is None:
                indirect_returns += 1
            elif x & ~EXP[g]:
                EXP[g] |= x
                grew = True
        if not grew:
            break
    edges = sum(1 for f in fns.values() for g in f.callees if LI.get(g, 0)) + len(post)
    if origins is not None:  # functions whose own code reads a register at entry (no callee gives it)
        for e, f in fns.items():
            own = LI[e]
            for g in f.callees:
                own &= ~LI.get(g, 0)
            for n in _names(own):
                if len(origins[n]) < 6:
                    origins[n].append(f"f_{e:08x}")
    li_scratch_uncalled = sum(1 for e in fns if (LI[e] & SCRATCH) and not callers[e])
    nv_entry = sum(1 for e in fns if LI[e] & NONVOL)
    out = {}
    for e, f in fns.items():
        # caves and the game functions patched to call into a graphic pack cave (hand-written asm entered from
        # the middle of the function, it may rely on any of its registers): the old semantics, everything in c
        # Callers of the GHS helpers too: a helper writes the context copies of r14-r31 (a restore, or mflr r31 in a
        # prologue helper) that the caller would otherwise never write back for its own caller.
        al = _cave(e) or any(_cave(g) for g in f.callees)
        if al or LI[e] or EXP.get(e, 0):
            out[e] = Sync(al, frozenset(_names(LI[e])), frozenset(_names(EXP.get(e, 0))))
    log(f"regctx: {len(fns)} functions, {edges} carrying call edges, {sum(s.all for s in out.values())} hold every "
        f"register in the context, {sum(1 for s in out.values() if s.li)} take registers at entry, "
        f"{sum(1 for s in out.values() if s.exp)} deliver some at exit; non-volatile live-in in {nv_entry}; "
        f"scratch live-in without direct callers {li_scratch_uncalled}; scratch delivered by indirect calls "
        f"{indirect_returns}; rounds {rounds}")
    return out


class Sync(collections.namedtuple("Sync", "all li exp")):
    """all: the function keeps every register in the context. li: registers it reads from its caller (callers
    store them before the call); exp: registers it delivers (stored at its exits, callers reload them)."""

    def dump(self) -> str:
        return f"{'ALL ' if self.all else ''}i={','.join(sorted(self.li))} o={','.join(sorted(self.exp))}"

    @staticmethod
    def parse(words: list[str]) -> "Sync":
        d = dict(w.split("=", 1) for w in words if "=" in w)
        return Sync("ALL" in words, frozenset(x for x in d.get("i", "").split(",") if x),
                    frozenset(x for x in d.get("o", "").split(",") if x))
