cmake_minimum_required(VERSION 3.24)

# APPLE is not set until project(), so pick languages from the host.
if(CMAKE_HOST_SYSTEM_NAME STREQUAL "Darwin")
  set(CMAKE_OSX_ARCHITECTURES arm64)
  set(CMAKE_OSX_DEPLOYMENT_TARGET 13.3)
endif()

# Windows and Linux use Vulkan. Metal stays the Apple backend from upstream BotW Recompiled.
project(BotWRecomp
  DESCRIPTION "Breath of the Wild recompiled for Windows and Linux (Vulkan)"
  LANGUAGES C CXX)

# Build speed helpers, both optional. ccache only helps rebuilds (a first build is always a
# cache miss), so it is used when it is installed and otherwise ignored.
option(BOTW_USE_CCACHE "Use ccache when it is installed" ON)
if(BOTW_USE_CCACHE AND NOT CMAKE_CXX_COMPILER_LAUNCHER)
  find_program(BOTW_CCACHE_PROGRAM ccache)
  if(BOTW_CCACHE_PROGRAM)
    set(CMAKE_C_COMPILER_LAUNCHER ${BOTW_CCACHE_PROGRAM})
    set(CMAKE_CXX_COMPILER_LAUNCHER ${BOTW_CCACHE_PROGRAM})
    message(STATUS "Using ccache: ${BOTW_CCACHE_PROGRAM}")
  endif()
endif()
# Link with lld when asked and the compiler accepts it; falls back to the default linker.
option(BOTW_USE_LLD "Link with lld if the compiler supports -fuse-ld=lld" OFF)
if(BOTW_USE_LLD AND NOT MSVC)
  include(CheckLinkerFlag)
  check_linker_flag(CXX "-fuse-ld=lld" BOTW_HAVE_LLD)
  if(BOTW_HAVE_LLD)
    add_link_options(-fuse-ld=lld)
  endif()
endif()
# Hot functions first (tools/deck/symbol-order.txt: Deck profile samples, hottest first, scratch hot_order.py): the
# code that runs packs into ~20 of 222 MB instead of being spread among cold functions. Needs one section per function.
set(BOTW_SYMBOL_ORDER "" CACHE FILEPATH "lld --symbol-ordering-file for the program's functions (empty = linker order)")
if(BOTW_SYMBOL_ORDER AND BOTW_HAVE_LLD)
  add_compile_options(-ffunction-sections)
  add_link_options("-Wl,--symbol-ordering-file=${BOTW_SYMBOL_ORDER}" -Wl,--no-warn-symbol-ordering)
endif()

# Keep local folder names out of shipped binaries: source and build paths in __FILE__ and
# debug info are rewritten to relative ones. Each flag is used only if the compiler accepts it.
if(NOT MSVC)
  include(CheckCXXCompilerFlag)
  # GCC applies the last matching map, so the (often nested) build folder comes second.
  foreach(_botw_map_dir "${CMAKE_SOURCE_DIR}" "${CMAKE_BINARY_DIR}")
    file(TO_CMAKE_PATH "${_botw_map_dir}" _botw_map_dir)
    if(_botw_map_dir STREQUAL "${CMAKE_BINARY_DIR}" AND NOT _botw_map_dir STREQUAL "${CMAKE_SOURCE_DIR}")
      set(_botw_map_to "build")
    else()
      set(_botw_map_to ".")
    endif()
    foreach(_botw_map_kind file macro)
      string(MAKE_C_IDENTIFIER "BOTW_HAVE_${_botw_map_kind}_PREFIX_MAP" _botw_map_var)
      check_cxx_compiler_flag("-f${_botw_map_kind}-prefix-map=/a=b" ${_botw_map_var})
      if(${_botw_map_var})
        add_compile_options("-f${_botw_map_kind}-prefix-map=${_botw_map_dir}=${_botw_map_to}")
      endif()
    endforeach()
  endforeach()
endif()

if(APPLE)
  enable_language(OBJC)
  enable_language(OBJCXX)
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE RelWithDebInfo)
endif()

# Metal stays the default on Apple. Vulkan is the Windows/Linux backend and can be turned on
# beside Metal (MoltenVK) with -DBOTW_ENABLE_VULKAN=ON.
if(APPLE)
  set(_botw_metal_default ON)
  set(_botw_vulkan_default OFF)
else()
  set(_botw_metal_default OFF)
  set(_botw_vulkan_default ON)
endif()
option(BOTW_ENABLE_METAL "Build the Metal renderer (Apple only)" ${_botw_metal_default})
option(BOTW_ENABLE_VULKAN "Build the Vulkan renderer" ${_botw_vulkan_default})
if(NOT APPLE AND BOTW_ENABLE_METAL)
  message(STATUS "Metal is Apple-only; disabling BOTW_ENABLE_METAL")
  set(BOTW_ENABLE_METAL OFF CACHE BOOL "Build the Metal renderer (Apple only)" FORCE)
endif()
if(NOT APPLE AND NOT BOTW_ENABLE_VULKAN)
  message(FATAL_ERROR "Windows and Linux builds require -DBOTW_ENABLE_VULKAN=ON")
endif()
if(NOT BOTW_ENABLE_METAL AND NOT BOTW_ENABLE_VULKAN)
  message(FATAL_ERROR "Enable BOTW_ENABLE_METAL, BOTW_ENABLE_VULKAN, or both")
endif()

if(APPLE)
  list(APPEND CMAKE_PREFIX_PATH /opt/homebrew /usr/local)
  if(EXISTS /opt/homebrew/opt/openssl@3)
    set(OPENSSL_ROOT_DIR /opt/homebrew/opt/openssl@3)
  endif()
endif()

include(FetchContent)
find_package(SDL3 CONFIG QUIET)
if(NOT SDL3_FOUND)
  set(SDL_SHARED OFF CACHE BOOL "" FORCE)
  set(SDL_STATIC ON CACHE BOOL "" FORCE)
  set(SDL_TEST_LIBRARY OFF CACHE BOOL "" FORCE)
  set(SDL_TESTS OFF CACHE BOOL "" FORCE)
  set(SDL_EXAMPLES OFF CACHE BOOL "" FORCE)
  set(SDL_INSTALL OFF CACHE BOOL "" FORCE)
  set(SDL_INSTALL_DOCS OFF CACHE BOOL "" FORCE)
  FetchContent_Declare(SDL
    GIT_REPOSITORY https://github.com/libsdl-org/SDL.git
    GIT_TAG release-3.4.10
    GIT_SHALLOW TRUE)
  FetchContent_MakeAvailable(SDL)
endif()

find_package(fmt REQUIRED)
find_package(OpenSSL REQUIRED)
find_package(PNG REQUIRED)
find_package(ZLIB REQUIRED)
find_package(Threads REQUIRED)
find_library(ZSTD_LIBRARY zstd REQUIRED)
if(BOTW_ENABLE_VULKAN)
  find_package(Vulkan REQUIRED)
  find_package(glslang CONFIG QUIET)
  if(NOT TARGET glslang::glslang AND NOT TARGET glslang)
    set(ENABLE_OPT OFF CACHE BOOL "" FORCE)
    set(ENABLE_GLSLANG_BINARIES OFF CACHE BOOL "" FORCE)
    set(ENABLE_SPVREMAPPER OFF CACHE BOOL "" FORCE)
    set(GLSLANG_TESTS OFF CACHE BOOL "" FORCE)
    set(GLSLANG_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
    set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
    FetchContent_Declare(glslang
      GIT_REPOSITORY https://github.com/KhronosGroup/glslang.git
      GIT_TAG 15.1.0
      GIT_SHALLOW TRUE)
    FetchContent_MakeAvailable(glslang)
  endif()
endif()

if(UNIX AND NOT APPLE)
  find_package(X11 REQUIRED)
  # Wayland headers are only needed when the renderer is built with HAS_WAYLAND.
  find_package(PkgConfig QUIET)
  if(PkgConfig_FOUND)
    pkg_check_modules(WAYLAND QUIET wayland-client)
  endif()
endif()

# Recompiled guest code (tools/recompile.py writes generated/). Without it, a stub links so the
# Vulkan device can still be brought up.
set(BOTW_PREBUILT_GUEST_DIR "" CACHE PATH "Directory containing prebuilt guest/*.cpp.obj to link instead of recompiling generated/")
if(BOTW_PREBUILT_GUEST_DIR)
  file(GLOB _GUEST_OBJS "${BOTW_PREBUILT_GUEST_DIR}/*.obj" "${BOTW_PREBUILT_GUEST_DIR}/*.o")
else()
file(GLOB GUEST_SOURCES CONFIGURE_DEPENDS
  ${CMAKE_SOURCE_DIR}/generated/fn_*.cpp
  ${CMAKE_SOURCE_DIR}/generated/big_*.cpp
  ${CMAKE_SOURCE_DIR}/generated/huge_*.cpp
  ${CMAKE_SOURCE_DIR}/generated/table.cpp
  ${CMAKE_SOURCE_DIR}/generated/hle_stubs.cpp
  ${CMAKE_SOURCE_DIR}/generated/gfxpack_data.cpp)
if(NOT GUEST_SOURCES)
  set(GUEST_SOURCES ${CMAKE_SOURCE_DIR}/runtime/src/guest_stub.cpp)
  message(STATUS "generated/ is empty: linking a guest stub (graphics init only; the game will not run)")
endif()
# Optimisation level for the recompiled game code (the bulk of the build time).
set(BOTW_GUEST_OPT "-O2" CACHE STRING "Compiler optimisation flag for generated/ game code")
add_library(guest OBJECT ${GUEST_SOURCES})
target_include_directories(guest PRIVATE runtime/include generated)
if(APPLE)
  target_compile_options(guest PRIVATE -O2 -gline-tables-only -w -fwrapv -fno-strict-aliasing -mcpu=apple-m1
    -ffp-contract=off)
else()
  target_compile_options(guest PRIVATE ${BOTW_GUEST_OPT} -g1 -w -fwrapv -fno-strict-aliasing -ffp-contract=off)
endif()
# Per-call-site cache for indirect guest calls (ppc.h rt_call). Needs the __ehdr_start image base: Linux only.
option(BOTW_INLINE_RT_CALL "Per-call-site cache for indirect guest calls" ON)
if(BOTW_INLINE_RT_CALL AND CMAKE_SYSTEM_NAME STREQUAL "Linux")
  target_compile_definitions(guest PRIVATE BOTW_INLINE_RT_CALL)
endif()
file(GLOB BIG_SOURCES CONFIGURE_DEPENDS ${CMAKE_SOURCE_DIR}/generated/big_*.cpp)
file(GLOB HUGE_SOURCES CONFIGURE_DEPENDS ${CMAKE_SOURCE_DIR}/generated/huge_*.cpp)
if(BIG_SOURCES)
  set_source_files_properties(${BIG_SOURCES} PROPERTIES COMPILE_OPTIONS "-O1")
endif()
if(HUGE_SOURCES)
  set_source_files_properties(${HUGE_SOURCES} PROPERTIES COMPILE_OPTIONS "-O0")
endif()
endif()

# Cemu's GPU pieces the native GPU (runtime/src/gx2 + runtime/src/gfx) uses, MPL-2.0: the shader decompiler, the
# address library and the register definitions (runtime/cemu, trimmed: no Latte emulation), with their minimal common
# definitions (cemu_shim.h); Dear ImGui for the renderer's overlay.
file(GLOB_RECURSE CEMU_SOURCES CONFIGURE_DEPENDS ${CMAKE_SOURCE_DIR}/runtime/cemu/*.cpp)
list(FILTER CEMU_SOURCES EXCLUDE REGEX "LatteDecompilerEmitMSL[^/]*\\.cpp$")  # Metal's emitter
list(APPEND CEMU_SOURCES
  ${CMAKE_SOURCE_DIR}/runtime/third_party/imgui/imgui.cpp ${CMAKE_SOURCE_DIR}/runtime/third_party/imgui/imgui_draw.cpp
  ${CMAKE_SOURCE_DIR}/runtime/third_party/imgui/imgui_tables.cpp ${CMAKE_SOURCE_DIR}/runtime/third_party/imgui/imgui_widgets.cpp)
add_library(cemu_gpu OBJECT ${CEMU_SOURCES})
target_include_directories(cemu_gpu PRIVATE runtime/cemu runtime/cemu/Cafe runtime/third_party/imgui)
target_compile_definitions(cemu_gpu PUBLIC ENABLE_VULKAN=1 IMGUI_DISABLE_DEMO_WINDOWS IMGUI_DISABLE_OBSOLETE_FUNCTIONS)
target_compile_options(cemu_gpu PRIVATE -w -O2 "SHELL:-include ${CMAKE_SOURCE_DIR}/runtime/cemu/cemu_shim.h")

file(GLOB_RECURSE RUNTIME_SOURCES CONFIGURE_DEPENDS
  ${CMAKE_SOURCE_DIR}/runtime/src/*.cpp
  ${CMAKE_SOURCE_DIR}/runtime/src/*.mm)
list(FILTER RUNTIME_SOURCES EXCLUDE REGEX "/guest_stub\\.cpp$")
if(NOT APPLE)
  list(FILTER RUNTIME_SOURCES EXCLUDE REGEX "\\.mm$")
endif()
# Linux file systems are case-sensitive like iOS; reuse its case-insensitive game folder lookup.
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  list(APPEND RUNTIME_SOURCES ${CMAKE_SOURCE_DIR}/ios/src/fs_case_ios.cpp)
endif()

if(BOTW_PREBUILT_GUEST_DIR)
  add_executable(BotWRecomp ${RUNTIME_SOURCES} ${_GUEST_OBJS} $<TARGET_OBJECTS:cemu_gpu>)
  set_source_files_properties(${_GUEST_OBJS} PROPERTIES EXTERNAL_OBJECT TRUE GENERATED TRUE)
else()
  add_executable(BotWRecomp ${RUNTIME_SOURCES} $<TARGET_OBJECTS:guest> $<TARGET_OBJECTS:cemu_gpu>)
endif()
target_include_directories(BotWRecomp PRIVATE runtime/include runtime/src generated
  runtime/cemu runtime/cemu/Cafe runtime/third_party runtime/third_party/imgui)
# The native GPU sources use Cemu's Latte headers, which expect Cemu's common definitions first (as cemu_gpu has them).
file(GLOB_RECURSE NATIVE_GPU_SOURCES CONFIGURE_DEPENDS ${CMAKE_SOURCE_DIR}/runtime/src/gx2/*.cpp ${CMAKE_SOURCE_DIR}/runtime/src/gfx/*.cpp)
set_source_files_properties(${NATIVE_GPU_SOURCES} PROPERTIES
  COMPILE_OPTIONS "-include;${CMAKE_SOURCE_DIR}/runtime/cemu/cemu_shim.h"
  COMPILE_DEFINITIONS "VK_NO_PROTOTYPES")  # the renderer loads Vulkan itself (loader.h)
if(APPLE)
  target_compile_options(BotWRecomp PRIVATE -Wall -Wno-unused-function -Wno-unused-variable -fwrapv -mcpu=apple-m1
    $<$<COMPILE_LANGUAGE:OBJCXX>:-fobjc-arc>)
else()
  target_compile_options(BotWRecomp PRIVATE -O3 -g1 -Wall -Wno-unused-function -Wno-unused-variable -fwrapv)
endif()
if(BOTW_ENABLE_VULKAN)
  target_compile_definitions(BotWRecomp PRIVATE ENABLE_VULKAN)
endif()
if(BOTW_ENABLE_METAL)
  target_compile_definitions(BotWRecomp PRIVATE ENABLE_METAL)
endif()

# Community SDL game controller database, copied next to the executable.
# SDL_AddGamepadMappingsFromFile reads it at startup. The compile path is the
# fallback when the process is not started from the build directory.
set(_botw_controller_db "${CMAKE_SOURCE_DIR}/runtime/gamecontrollerdb.txt")
# The compiled-in fallback path is for development runs only, so release builds do not carry
# the local source folder.
option(BOTW_EMBED_SOURCE_PATHS "Compile in the source path of gamecontrollerdb.txt as a fallback" OFF)
if(BOTW_EMBED_SOURCE_PATHS)
  file(TO_CMAKE_PATH "${_botw_controller_db}" _botw_controller_db_fwd)
  target_compile_definitions(BotWRecomp PRIVATE "BOTW_GAMECONTROLLER_DB=\"${_botw_controller_db_fwd}\"")
endif()
add_custom_command(TARGET BotWRecomp POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
          "${_botw_controller_db}" "$<TARGET_FILE_DIR:BotWRecomp>/gamecontrollerdb.txt"
  VERBATIM)

set(_botw_glslang "")
if(BOTW_ENABLE_VULKAN)
  if(TARGET glslang::glslang)
    list(APPEND _botw_glslang glslang::glslang)
  elseif(TARGET glslang)
    list(APPEND _botw_glslang glslang)
  else()
    message(FATAL_ERROR "glslang target not found (need glslang 15.x)")
  endif()
  if(TARGET glslang::glslang-default-resource-limits)
    list(APPEND _botw_glslang glslang::glslang-default-resource-limits)
  elseif(TARGET glslang-default-resource-limits)
    list(APPEND _botw_glslang glslang-default-resource-limits)
  endif()
  if(TARGET glslang::SPIRV)
    list(APPEND _botw_glslang glslang::SPIRV)
  elseif(TARGET SPIRV)
    list(APPEND _botw_glslang SPIRV)
  else()
    message(FATAL_ERROR "glslang SPIRV target not found")
  endif()
endif()

# The glslang build tree exposes <glslang/Public/...> but the renderer (gfx/vulkan/shaders.cpp, GLSL -> SPIR-V)
# includes <glslang/SPIRV/...>, which is how an installed glslang lays the headers out.
if(BOTW_ENABLE_VULKAN AND DEFINED glslang_SOURCE_DIR)
  set(_glslang_inc ${CMAKE_BINARY_DIR}/glslang-include)
  file(MAKE_DIRECTORY ${_glslang_inc}/glslang)
  foreach(_sub Public SPIRV Include)
    if(NOT EXISTS ${_glslang_inc}/glslang/${_sub})
      if(_sub STREQUAL "SPIRV")
        set(_src ${glslang_SOURCE_DIR}/SPIRV)
      else()
        set(_src ${glslang_SOURCE_DIR}/glslang/${_sub})
      endif()
      file(CREATE_LINK ${_src} ${_glslang_inc}/glslang/${_sub} SYMBOLIC)
    endif()
  endforeach()
  target_include_directories(BotWRecomp PRIVATE ${_glslang_inc})
endif()

target_link_libraries(BotWRecomp PRIVATE
  SDL3::SDL3 fmt::fmt OpenSSL::Crypto PNG::PNG ZLIB::ZLIB ${ZSTD_LIBRARY} Threads::Threads
  ${_botw_glslang})
if(BOTW_ENABLE_VULKAN)
  target_link_libraries(BotWRecomp PRIVATE Vulkan::Vulkan)
endif()
if(UNIX AND NOT APPLE)
  target_link_libraries(BotWRecomp PRIVATE ${CMAKE_DL_LIBS} X11::X11)
  # Movies (h264.rpl): Windows decodes with Media Foundation; Linux needs a static FFmpeg built by
  # tools/deck/ffmpeg.sh. Without it the movies play as black frames.
  set(BOTW_FFMPEG_DIR "" CACHE PATH "Prefix of a static FFmpeg (libavcodec + libavutil) for H.264 movies")
  if(BOTW_FFMPEG_DIR)
    target_include_directories(BotWRecomp PRIVATE ${BOTW_FFMPEG_DIR}/include)
    target_compile_definitions(BotWRecomp PRIVATE BOTW_HAVE_FFMPEG)
    target_link_libraries(BotWRecomp PRIVATE ${BOTW_FFMPEG_DIR}/lib/libavcodec.a ${BOTW_FFMPEG_DIR}/lib/libavutil.a m)
  else()
    message(WARNING "BOTW_FFMPEG_DIR is not set: game movies will play as black frames on Linux")
  endif()
endif()
if(WIN32)
  # psapi: GetProcessMemoryInfo (shader-cache log).
  # ntdll: NtQuerySystemInformation in SystemInfoWin.cpp. That file's
  # `#pragma comment(lib, "ntdll.lib")` is MSVC-only; gcc ignores it, so this
  # link is what resolves the symbol. The Windows CI job links SystemInfoWin.cpp
  # into this executable and fails if ntdll is dropped.
  target_link_libraries(BotWRecomp PRIVATE psapi ntdll mfplat mfuuid ole32)
endif()
if(APPLE)
  target_link_libraries(BotWRecomp PRIVATE
    "-framework Metal" "-framework QuartzCore" "-framework Foundation" "-framework AppKit" "-framework MetalFX"
    "-framework CoreGraphics" "-framework ImageIO" "-framework UniformTypeIdentifiers" "-framework VideoToolbox"
    "-framework CoreMedia" "-framework CoreVideo")
endif()


# Separate from the game. SDL3 is already a dependency; the launcher uses it only to capture
# keys and controller buttons with the same names the runtime reads from controls.ini.
if(WIN32)
  set(_botw_manifest "${CMAKE_SOURCE_DIR}/launcher/app.manifest")
  file(TO_CMAKE_PATH "${_botw_manifest}" _botw_manifest_cmake)
  file(WRITE "${CMAKE_BINARY_DIR}/botw_launcher_manifest.rc" "1 24 \"${_botw_manifest_cmake}\"\n")
  # BotWLauncher: guided first-run setup, then Play / Mods / Settings.
  add_executable(BotWLauncher WIN32
    launcher/main.cpp
    launcher/game_build.cpp
    launcher/game_info.cpp
    launcher/mods.cpp
    launcher/setup_wizard.cpp
    launcher/unzip.cpp
    "${CMAKE_BINARY_DIR}/botw_launcher_manifest.rc")
  # BotWLauncherClassic: the full launcher with controls, saves and the Setup Assistant.
  add_executable(BotWLauncherClassic WIN32
    launcher/classic_main.cpp
    launcher/game_info.cpp
    launcher/mods.cpp
    launcher/save_import.cpp
    launcher/setup_wizard.cpp
    launcher/ui.cpp
    launcher/unzip.cpp
    runtime/src/input/controls_file.cpp
    "${CMAKE_BINARY_DIR}/botw_launcher_manifest.rc")
  foreach(_tgt BotWLauncher BotWLauncherClassic)
    target_include_directories(${_tgt} PRIVATE launcher runtime/src)
    target_compile_definitions(${_tgt} PRIVATE SDL_MAIN_HANDLED UNICODE _UNICODE)
    target_compile_options(${_tgt} PRIVATE -O2 -Wall -Wno-unused-function -Wno-unused-variable -fwrapv)
    if(TARGET SDL3::SDL3-shared)
      target_link_libraries(${_tgt} PRIVATE SDL3::SDL3-shared)
    else()
      target_link_libraries(${_tgt} PRIVATE SDL3::SDL3)
    endif()
    target_link_libraries(${_tgt} PRIVATE ZLIB::ZLIB comctl32 comdlg32 shell32 ole32 uuid uxtheme dwmapi gdiplus)
    if(MINGW)
      # GUI subsystem, but the entry point stays main() so SDL_main is not required.
      target_link_options(${_tgt} PRIVATE "-Wl,-e,mainCRTStartup")
    endif()
  endforeach()
  file(GENERATE OUTPUT "${CMAKE_BINARY_DIR}/BotWLauncher.cmd" CONTENT
    "@echo off\r\nset \"PATH=C:\\msys64\\ucrt64\\bin;%PATH%\"\r\nstart \"\" \"%~dp0BotWLauncher.exe\"\r\n")
  message(STATUS "BotWLauncher: Windows launcher (BotWLauncher.exe)")

  add_executable(BotWNativeSetup WIN32
    launcher/installer.cpp
    launcher/ui.cpp
    "${CMAKE_BINARY_DIR}/botw_launcher_manifest.rc")
  set_target_properties(BotWNativeSetup PROPERTIES OUTPUT_NAME "BotW-Native-Setup")
  target_include_directories(BotWNativeSetup PRIVATE launcher)
  target_compile_definitions(BotWNativeSetup PRIVATE UNICODE _UNICODE)
  target_compile_options(BotWNativeSetup PRIVATE -O2 -Wall -Wno-unused-function -Wno-unused-variable -fwrapv)
  target_link_libraries(BotWNativeSetup PRIVATE comctl32 comdlg32 shell32 ole32 uuid uxtheme dwmapi gdiplus)
  if(MINGW)
    target_link_options(BotWNativeSetup PRIVATE "-Wl,-e,mainCRTStartup" -static-libgcc -static-libstdc++ -Wl,-Bstatic,--whole-archive -lwinpthread -Wl,--no-whole-archive,-Bdynamic)
  endif()
endif()

message(STATUS "BotWRecomp backends: Metal=${BOTW_ENABLE_METAL} Vulkan=${BOTW_ENABLE_VULKAN}")


# Unit test for the mod file index (docs/mods.md). Off by default.
option(BOTW_BUILD_TESTS "Build the BotWModTests unit test" OFF)
if(BOTW_BUILD_TESTS)
  add_executable(BotWModTests tests/mods_test.cpp runtime/src/coreinit/mods.cpp runtime/src/log.cpp)
  target_include_directories(BotWModTests PRIVATE runtime/src)
  target_link_libraries(BotWModTests PRIVATE ZLIB::ZLIB)
  enable_testing()
  add_test(NAME mods COMMAND BotWModTests)
endif()

# Release package: `cmake --build <build> --target botw_package` (or
# `cmake --install <build> --prefix <dir> --strip`) copies the shipped programs, stripped of
# debug info, plus gamecontrollerdb.txt and the non-system DLLs they need, into one folder.
set(BOTW_PACKAGE_DIR "${CMAKE_BINARY_DIR}/package" CACHE PATH "Output folder for the botw_package target")
set(_botw_ship BotWRecomp)
if(WIN32)
  list(APPEND _botw_ship BotWLauncher BotWLauncherClassic BotWNativeSetup)
endif()
if(WIN32 OR (UNIX AND NOT APPLE))
  install(TARGETS ${_botw_ship}
    RUNTIME_DEPENDENCIES
      PRE_EXCLUDE_REGEXES "^api-ms-" "^ext-ms-" "^vulkan-1" "^libvulkan"
      POST_EXCLUDE_REGEXES ".*[Ss]ystem32.*" "^/lib" "^/usr/lib"
      DIRECTORIES $ENV{PATH}
    RUNTIME DESTINATION .
    LIBRARY DESTINATION .)
else()
  install(TARGETS ${_botw_ship} RUNTIME DESTINATION . BUNDLE DESTINATION .)
endif()
install(FILES "${CMAKE_SOURCE_DIR}/runtime/gamecontrollerdb.txt" DESTINATION .)
add_custom_target(botw_package
  COMMAND ${CMAKE_COMMAND} --install "${CMAKE_BINARY_DIR}" --prefix "${BOTW_PACKAGE_DIR}" --strip
  DEPENDS ${_botw_ship}
  COMMENT "Packaging stripped release into ${BOTW_PACKAGE_DIR}"
  VERBATIM)
