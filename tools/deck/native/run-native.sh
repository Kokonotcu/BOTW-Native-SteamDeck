#!/bin/sh
# Steam shortcut "BotW Native GPU (test)": the BotW macOS port's native GX2 -> Vulkan build, its own save copy
D=/home/deck/Games/BotW-native-test
cd "$D" || exit 1
export LD_LIBRARY_PATH="$D/lib"
# one window (the TV picture), full screen: gamescope otherwise showed the GamePad window ("tap here to play")
export WWHD_DRC_MODE=off WWHD_FULLSCREEN=1
# per-interval render statistics (pipeline creation, preparation, submission) for the slowdown investigation
export WWHD_VK_STATS=1
exec ./wwhd --game /home/deck/Games/BotW/game --base /home/deck/Games/BotW/game --aoc /home/deck/Games/BotW/game/aoc --save "$D/save" > "$D/run-visible.log" 2>&1
