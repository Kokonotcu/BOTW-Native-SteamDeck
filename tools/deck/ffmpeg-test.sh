#!/bin/sh
# Test-only FFmpeg (adds the mp4 demuxer) and tests/h264_decode_test against the game's movies.
# usage: tools/deck/ffmpeg-test.sh <game content/Movie folder>
set -e
SRC=$(cd "$(dirname "$0")/../.." && pwd)
V=7.1.1
cd /root
[ -f ffmpeg-$V.tar.xz ] || curl -sSLO https://ffmpeg.org/releases/ffmpeg-$V.tar.xz
if [ ! -f /root/ffmpeg-test/lib/libavformat.a ]; then
  rm -rf ffmpeg-test-src && mkdir ffmpeg-test-src && tar xf ffmpeg-$V.tar.xz -C ffmpeg-test-src --strip-components=1
  cd ffmpeg-test-src
  ./configure --prefix=/root/ffmpeg-test --cc=clang --enable-static --disable-shared --enable-pic \
    --disable-autodetect --disable-everything --disable-programs --disable-doc --disable-network \
    --disable-avdevice --disable-swscale --disable-swresample --disable-avfilter \
    --enable-decoder=h264,hevc --enable-demuxer=mov --enable-bsf=h264_mp4toannexb --enable-protocol=file
  make -j4 >/dev/null
  make install >/dev/null
fi
P=/root/ffmpeg-test
clang++ -std=c++20 -O1 -I$P/include "$SRC/tests/h264_decode_test.cpp" -o /root/h264_decode_test \
  $P/lib/libavformat.a $P/lib/libavcodec.a $P/lib/libavutil.a -lm -lpthread
/root/h264_decode_test "$1"/*.mp4
