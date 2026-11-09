#!/bin/sh
# Package the Linux build for Steam Deck: stripped binary + bundled libs that SteamOS lacks.
# usage: tools/deck/package.sh <build dir> <out dir> <debug symbols dir>
set -e
SRC=$(cd "$(dirname "$0")/../.." && pwd)
BUILD=$1
OUT=$2
SYMS=$3
mkdir -p "$OUT/lib" "$SYMS"

cp "$BUILD/BotWRecomp" "$OUT/BotWRecomp"
objcopy --only-keep-debug "$OUT/BotWRecomp" "$SYMS/BotWRecomp.debug"
strip --strip-debug "$OUT/BotWRecomp"
objcopy --add-gnu-debuglink="$SYMS/BotWRecomp.debug" "$OUT/BotWRecomp"

# fmt has a distro-specific soname (Ubuntu libfmt.so.9, Arch libfmt.so.11), so it travels with the game.
# zlib/zstd/X11 stay system copies: Mesa loads them too and must get the Deck's own versions.
for lib in $(ldd "$OUT/BotWRecomp" | awk '/libfmt/ {print $3}'); do
  cp -L "$lib" "$OUT/lib/"
done
patchelf --set-rpath '$ORIGIN/lib' "$OUT/BotWRecomp"

cp "$SRC/runtime/gamecontrollerdb.txt" "$OUT/"
cp "$SRC/tools/deck/botw.sh" "$OUT/"
cp "$SRC/tools/deck/README-DECK.txt" "$OUT/"
chmod +x "$OUT/botw.sh" "$OUT/BotWRecomp"

echo "== needed libraries"
readelf -d "$OUT/BotWRecomp" | awk '/NEEDED|RUNPATH|RPATH/'
echo "== newest symbol versions required"
objdump -T "$OUT/BotWRecomp" "$OUT"/lib/* | grep -o 'GLIBC_[0-9.]*\|GLIBCXX_[0-9.]*\|CXXABI_[0-9.]*' | sort -u -V | tail -6
ls -la "$OUT" "$OUT/lib"
