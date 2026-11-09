"""Assembled FPS++ next to its source, decoded with the recompiler's own decoder (cafe.ppc): every line where the
decoded mnemonic differs from the source mnemonic is flagged with '!!'.
uv run python -I tests/gfxpack_listing.py /root/fpspp"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from cafe import ppc  # noqa: E402
from tools import gfxpack  # noqa: E402

ALIAS = {"li": "addi", "lis": "addis", "mr": "or", "nop": "ori", "mflr": "mfspr", "mtlr": "mtspr", "mtctr": "mtspr",
         "blr": "bclr", "bctrl": "bcctr", "bctr": "bcctr", "beqlr": "bclr", "bnelr": "bclr", "ba": "b", "bla": "b",
         "bl": "b", "beq": "bc", "bne": "bc", "blt": "bc", "bgt": "bc", "ble": "bc", "bge": "bc", "cmpwi": "cmpi",
         "cmplwi": "cmpli", "cmpw": "cmp", "cmplw": "cmpl"}
pk = gfxpack.load(sys.argv[1], {"fpsLimitNormal": 60})
bad = 0
for a in sorted(pk.source):
    w = pk.words[a]
    i = ppc.decode(w, a)
    src = pk.source[a]
    want = ALIAS.get(src.split()[0].lower(), src.split()[0].lower())
    ok = i.op == want or i.op.rstrip(".") == want.rstrip(".")
    bad += not ok
    print(f"{'  ' if ok else '!!'} {a:08x} {w:08x}  {ppc.disasm(i) if hasattr(ppc, 'disasm') else i.op:<34} ; {src}")
print(f"{bad} mismatches")
