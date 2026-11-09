#!/bin/sh
# Minimal static FFmpeg for BotW movies: libavcodec (H.264 decoder) + libavutil.
# HEVC is enabled only because 7.1.x builds aom_film_grain.o, which the H.264 SEI code calls, for HEVC alone.
set -e
export DEBIAN_FRONTEND=noninteractive
apt-get install -y nasm xz-utils
cd /root
V=7.1.1
[ -f ffmpeg-$V.tar.xz ] || curl -sSLO https://ffmpeg.org/releases/ffmpeg-$V.tar.xz
rm -rf ffmpeg-$V && tar xf ffmpeg-$V.tar.xz
cd ffmpeg-$V
./configure --prefix=/root/ffmpeg-min --cc=clang --cxx=clang++ \
  --extra-cflags="-march=x86-64-v3 -mtune=znver2" \
  --enable-static --disable-shared --enable-pic \
  --disable-autodetect --disable-everything --disable-programs --disable-doc --disable-network \
  --disable-avdevice --disable-avformat --disable-swscale --disable-swresample --disable-avfilter \
  --enable-decoder=h264,hevc
make -j4
rm -rf /root/ffmpeg-min
make install
ls -la /root/ffmpeg-min/lib
nm /root/ffmpeg-min/lib/libavcodec.a | grep -q " T ff_aom_uninit_film_grain_params"
echo FFMPEG_OK
