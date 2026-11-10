#!/bin/sh
# Compile the BotW macOS port (vasist15-source/BOTWRecomp) for Linux x86-64 with the Vulkan renderer, for a Deck test.
# Compile only on the PC. Output: /root/ext/vasist-botw/build/rt-linux/wwhd ; log: scratch/vasist_build.log
# JOBS (default 4) stays low: the recompiler agent builds in parallel.
cd /root/ext/vasist-botw || exit 1
[ -n "$(ls build/gen/code_*.c 2>/dev/null | head -1)" ] || { echo "NO_GEN"; exit 1; }
CC=clang CXX=clang++ cmake -S . -B build/rt-linux -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DWWHD_RENDERER=VULKAN \
  -DWWHD_BUNDLED_DEPS=ON \
  -DCMAKE_C_FLAGS="-march=x86-64-v3 -mtune=znver2" \
  -DCMAKE_CXX_FLAGS="-march=x86-64-v3 -mtune=znver2" || { echo CONFIGURE_FAILED; exit 1; }
time cmake --build build/rt-linux --target wwhd -j"${JOBS:-4}" -- -k 0
rc=$?
ls -la build/rt-linux/wwhd 2>/dev/null
echo "build rc=$rc"
[ $rc -eq 0 ] && echo VASIST_BUILD_OK || echo VASIST_BUILD_FAILED
