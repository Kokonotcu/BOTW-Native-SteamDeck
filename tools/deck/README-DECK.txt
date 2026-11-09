BotW-Native for Steam Deck
==========================

Built for x86-64-v3 (AVX2, FMA, BMI2, MOVBE) and tuned for the Deck's Zen 2 CPU. It also runs on
other Linux PCs from about 2015 on (Intel Haswell or newer, any AMD Zen).

What is in this folder
  BotWRecomp            the game program (Linux, Vulkan)
  botw.sh               launcher: finds the game files, sets Deck defaults, starts fullscreen
  lib/                  libfmt (SteamOS does not ship this version)
  gamecontrollerdb.txt  controller mappings
  game/                 your merged game files: code/ content/ meta/ (v1.5.0, v208) and aoc/ (DLC)

Install
  1. Copy this whole folder to the Deck, for example to /home/deck/Games/BotW
     (USB drive, SD card, or KDE Connect / network share in Desktop Mode).
  2. In Desktop Mode open a terminal (Konsole) in that folder and run once:
       chmod +x botw.sh BotWRecomp
     (Copies from Windows usually lose the "executable" flag.)
  3. Steam (Desktop Mode) > Games > Add a Non-Steam Game to My Library > Browse >
     pick botw.sh > Add Selected Programs.
  4. Switch to Game Mode and start it from the library.

  To test from Desktop Mode, run ./botw.sh in Konsole. Messages go to the terminal and to the log.

Defaults on the first run (written to ~/.local/share/BotWRecomp/settings.ini)
  60 fps and 720p render (the Deck panel is 1280x800). Press both sticks (L3+R3) to open the
  in-game menu and change them; 30 fps uses roughly half the CPU and GPU time and less battery.
  Extra launch options go after the program in Steam's launch options, e.g.  %command% --fps 30

Files the game writes
  ~/.local/share/BotWRecomp/save/          saves
  ~/.local/share/BotWRecomp/cache/         shaders and pipelines compiled on this Deck
  ~/.local/share/BotWRecomp/runtime.log    only with BOTW_LOG=1 in the launch options
                                           (BOTW_LOG=1 %command%); send it if something goes wrong
  ~/.local/share/BotWRecomp/telemetry/     only when Telemetry is switched on in the L3+R3 menu

Known limits
  - The first time through an area stutters while shaders compile; later runs use the cache.
  - Gyro works with Steam Input on: the game keeps the Deck's motion sensor switched on (Steam turns it
    off in-game). "Gyro turning: World" in the menu measures turns around the real vertical.
  - Tested on the Deck through the opening, Kakariko, Hateno, Wetland, Tarrey Town, Zora's Domain and
    Rito; not a full play-through.

Game files: the folder must be a decrypted dump of your own copy; code/U-King.rpx must have
SHA-256 ba58da5b95ce929e005d058ceb08b9b2788d1ab2bbc8a6c189bbadca0bb34d30 (USA v208).
