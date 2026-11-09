#!/bin/sh
# Steam Deck launcher for BotW-Native. In Desktop Mode: Steam > Add a Game > Add a Non-Steam Game >
# Browse, pick this file, then play it from Game Mode. Extra arguments are passed to the game
# (for example --fps 60 or --gamepad-view).
HERE=$(cd "$(dirname "$0")" && pwd)

# Game folder with code/ content/ meta/ (and aoc/ for the DLC). BOTW_GAME overrides the search.
if [ -z "$BOTW_GAME" ]; then
  for d in "$HERE/game" "$HERE/../game" "$HOME/Games/BotW/game"; do
    if [ -f "$d/code/U-King.rpx" ]; then BOTW_GAME=$d; break; fi
  done
fi
if [ -z "$BOTW_GAME" ] || [ ! -f "$BOTW_GAME/code/U-King.rpx" ]; then
  echo "BotW game folder not found. Put it next to this script as 'game' (code/ content/ meta/ aoc/) or set BOTW_GAME." >&2
  exit 1
fi
export BOTW_GAME

# Unattended test runs: a one-shot file sets BOTW_USER_ROOT (a separate data folder with a copy of the save, so a
# test never touches the real saves) and BOTW_AUTOPLAY_ARGS (e.g. --script ...). It is deleted as soon as it is
# read, so the next normal launch is unaffected. Written by the developer's test tools over SSH, never by the game.
AUTOPLAY_FILE=${XDG_DATA_HOME:-$HOME/.local/share}/BotWRecomp-autoplay.env
BOTW_AUTOPLAY_ARGS=
if [ -f "$AUTOPLAY_FILE" ]; then
  . "$AUTOPLAY_FILE"
  rm -f "$AUTOPLAY_FILE"
  export BOTW_USER_ROOT
fi

# First run: 30 fps and native 720p suit the Deck (press both sticks, L3+R3, to open the menu and change them).
USER_ROOT=${BOTW_USER_ROOT:-${XDG_DATA_HOME:-$HOME/.local/share}/BotWRecomp}
if [ ! -f "$USER_ROOT/settings.ini" ]; then
  mkdir -p "$USER_ROOT"
  printf 'fps=30\nscale=1\n' > "$USER_ROOT/settings.ini"
fi

# malloc on transparent huge pages: the GPU thread's heap data (texture/buffer caches) missed the TLB ~1 per 1000
# instructions on 4 KB pages; with this its cycles per frame drop ~8% (GPU thread) and ~2% (main), +~100 MB RAM.
# BOTW_MALLOC_THP=0 turns it off. glibc reads it at process start, so it must be set here.
if [ "${BOTW_MALLOC_THP:-1}" != 0 ]; then
  export GLIBC_TUNABLES="${GLIBC_TUNABLES:+$GLIBC_TUNABLES:}glibc.malloc.hugetlb=1"
fi

# Game Mode (gamescope) and Desktop Mode both provide X11; Cemu's Vulkan path is proven there.
if [ -n "$DISPLAY" ] && [ -z "$SDL_VIDEO_DRIVER" ]; then
  export SDL_VIDEO_DRIVER=x11
fi

exec "$HERE/BotWRecomp" --fullscreen $BOTW_AUTOPLAY_ARGS "$@" # extra arguments pass through
