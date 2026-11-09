// swkbd.rpl (Wii U software keyboard), loaded by the game through OSDynLoad when it asks for text (naming the
// horse). After Cemu's Cafe/OS/libs/swkbd/swkbd.cpp: one input form, its text, and an OK flag. The keyboard itself
// is the Steam Deck's (window.cpp opens it); the text box is drawn by the overlay (overlay_menu.cpp). Every export
// the game resolves is here: one missing export is a fatal stop (crash naming the horse, 2026-11-03).
#include <cstring>
#include <mutex>
#include <string>

#include "hle.h"
#include "swkbd_host.h"

namespace {
constexpr int kMaxForm = 4095;  // UTF-16 units, Cemu's SWKBD_FORM_STRING_MAX_LENGTH - 1

std::mutex g_mu;
std::u16string g_text;
bool g_active = false, g_ok = false, g_keyboard_only = false, g_dirty = false;
int g_max = kMaxForm, g_vis = 0, g_sub_font = 0, g_sub_predict = 0;
uint32_t g_receiver = 0, g_recv_buf = 0, g_recv_units = 0, g_form_buf = 0;

int max_len_locked() {
    if (!g_keyboard_only) return g_max;
    return g_recv_units > 1 ? (int)g_recv_units - 1 : 0;
}

// the game's receiver buffer gets the text on the guest thread (SwkbdCalc / state queries), not from the UI thread
void sync_receiver_locked() {
    if (!g_dirty) return;
    g_dirty = false;
    if (!g_recv_buf || g_recv_units < 2) return;
    const size_t n = std::min<size_t>(g_text.size(), g_recv_units - 1);
    for (size_t i = 0; i < n; i++) mem::w16(g_recv_buf + 2 * (uint32_t)i, g_text[i]);
    mem::w16(g_recv_buf + 2 * (uint32_t)n, 0);
}

void read_receiver_locked(uint32_t arg) {
    g_receiver = mem::r32(arg + 0x00);
    g_recv_buf = mem::r32(arg + 0x04);
    g_recv_units = mem::r32(arg + 0x08);
}

std::u16string read_u16z(uint32_t p, int max) {
    std::u16string s;
    for (int i = 0; p && i < max; i++) {
        const uint16_t ch = mem::r16(p + 2 * (uint32_t)i);
        if (!ch) break;
        s.push_back((char16_t)ch);
    }
    return s;
}
}  // namespace

#define SWKBD_EXPORT(fn, mangled)                                \
    static void swkbd_##fn(PPCContext* c);                       \
    static HleRegistration swkbd_reg_##fn("swkbd", mangled, swkbd_##fn); \
    static void swkbd_##fn(PPCContext* c)

SWKBD_EXPORT(Create, "SwkbdCreate__3RplFPUcQ3_2nn5swkbd10RegionTypeUiP8FSClient") {
    std::lock_guard<std::mutex> l(g_mu);
    g_text.clear();
    g_active = g_ok = g_keyboard_only = g_dirty = false;
    g_receiver = g_recv_buf = g_recv_units = 0;
    g_sub_font = g_sub_predict = 3;  // as Cemu: pretend the font / prediction threads have some work
    RET(1);
}
SWKBD_EXPORT(Destroy, "SwkbdDestroy__3RplFv") {
    std::lock_guard<std::mutex> l(g_mu);
    if (g_active) g_vis = -1;
    g_active = false;
}
SWKBD_EXPORT(SetVersion, "SwkbdSetVersion__3RplFi") {}
SWKBD_EXPORT(MuteAllSound, "SwkbdMuteAllSound__3RplFb") {}
SWKBD_EXPORT(SetControllerRemo, "SwkbdSetControllerRemo__3RplFQ3_2nn5swkbd14ControllerType") {}
SWKBD_EXPORT(SetCursorPos, "SwkbdSetCursorPos__3RplFi") {}
SWKBD_EXPORT(SetSelectFrom, "SwkbdSetSelectFrom__3RplFi") {}
SWKBD_EXPORT(SetEnableOkButton, "SwkbdSetEnableOkButton__3RplFb") {}
SWKBD_EXPORT(SetUserControllerEventObj, "SwkbdSetUserControllerEventObj__3RplFPQ3_2nn5swkbd19IControllerEventObj") {}
SWKBD_EXPORT(SetUserSoundObj, "SwkbdSetUserSoundObj__3RplFPQ3_2nn5swkbd9ISoundObj") {}
SWKBD_EXPORT(InactivateSelectCursor, "SwkbdInactivateSelectCursor__3RplFv") {}
SWKBD_EXPORT(ConfirmUnfixAll, "SwkbdConfirmUnfixAll__3RplFv") {}
SWKBD_EXPORT(DrawTV, "SwkbdDrawTV__3RplFv") {}    // the overlay draws the text box
SWKBD_EXPORT(DrawDRC, "SwkbdDrawDRC__3RplFv") {}
SWKBD_EXPORT(IsCoveredWithSubWindow, "SwkbdIsCoveredWithSubWindow__3RplFv") { RET(0); }
SWKBD_EXPORT(IsSelectCursorActive, "SwkbdIsSelectCursorActive__3RplFv") { RET(0); }
SWKBD_EXPORT(IsDecideCancelButton, "SwkbdIsDecideCancelButton__3RplFPb") { RET(0); }
SWKBD_EXPORT(GetKeyboardCondition, "SwkbdGetKeyboardCondition__3RplFPQ3_2nn5swkbd17KeyboardCondition") { RET(0); }

SWKBD_EXPORT(Calc, "SwkbdCalc__3RplFRCQ3_2nn5swkbd14ControllerInfo") {
    std::lock_guard<std::mutex> l(g_mu);
    sync_receiver_locked();
}
SWKBD_EXPORT(GetStateKeyboard, "SwkbdGetStateKeyboard__3RplFv") {
    std::lock_guard<std::mutex> l(g_mu);
    sync_receiver_locked();
    RET(g_active ? 2 : 0);  // 2 = displayed, 0 = blank
}
SWKBD_EXPORT(GetStateInputForm, "SwkbdGetStateInputForm__3RplFv") {
    std::lock_guard<std::mutex> l(g_mu);
    RET(g_active ? 2 : 0);
}
SWKBD_EXPORT(SetReceiver, "SwkbdSetReceiver__3RplFRCQ3_2nn5swkbd11ReceiverArg") {
    std::lock_guard<std::mutex> l(g_mu);
    read_receiver_locked(ARG(0));
    RET(0);
}
SWKBD_EXPORT(IsKeyboardTarget, "SwkbdIsKeyboardTarget__3RplFPQ3_2nn5swkbd14IEventReceiver") {
    std::lock_guard<std::mutex> l(g_mu);
    RET(g_active && ARG(0) == g_receiver ? 1 : 0);
}
// AppearArg: +0xC8 initial text (UTF-16BE), +0xD0 max text length
SWKBD_EXPORT(AppearInputForm, "SwkbdAppearInputForm__3RplFRCQ3_2nn5swkbd9AppearArg") {
    std::lock_guard<std::mutex> l(g_mu);
    const uint32_t a = ARG(0);
    const int max = (int)mem::r32(a + 0xD0);
    g_max = max <= 0 ? kMaxForm : std::min(max, kMaxForm);
    g_text = read_u16z(mem::r32(a + 0xC8), g_max);
    g_active = true, g_ok = false, g_keyboard_only = false, g_dirty = true;
    g_vis = 1;
    LOG("swkbd: input form (max %d chars)", g_max);
    RET(1);
}
// KeyboardArg: the ReceiverArg sits at +0xA8 (Cemu's SwkbdKeyboardArg_t)
SWKBD_EXPORT(AppearKeyboard, "SwkbdAppearKeyboard__3RplFRCQ3_2nn5swkbd11KeyboardArg") {
    std::lock_guard<std::mutex> l(g_mu);
    read_receiver_locked(ARG(0) + 0xA8);
    g_text.clear();
    g_active = true, g_ok = false, g_keyboard_only = true, g_dirty = true;
    g_vis = 1;
    LOG("swkbd: keyboard (buffer %u chars)", g_recv_units);
    RET(1);
}
SWKBD_EXPORT(DisappearInputForm, "SwkbdDisappearInputForm__3RplFv") {
    std::lock_guard<std::mutex> l(g_mu);
    if (g_active) g_vis = -1;
    g_active = false;
    RET(1);
}
SWKBD_EXPORT(DisappearKeyboard, "SwkbdDisappearKeyboard__3RplFv") {
    std::lock_guard<std::mutex> l(g_mu);
    if (g_active) g_vis = -1;
    g_active = false;
    RET(1);
}
SWKBD_EXPORT(GetInputFormString, "SwkbdGetInputFormString__3RplFv") {
    std::lock_guard<std::mutex> l(g_mu);
    sync_receiver_locked();
    if (!g_form_buf) g_form_buf = mem::sys_alloc(2 * (kMaxForm + 1), 16);
    const size_t n = std::min<size_t>(g_text.size(), kMaxForm);
    for (size_t i = 0; i < n; i++) mem::w16(g_form_buf + 2 * (uint32_t)i, g_text[i]);
    mem::w16(g_form_buf + 2 * (uint32_t)n, 0);
    RET(g_form_buf);
}
SWKBD_EXPORT(SetInputFormString, "SwkbdSetInputFormString__3RplFPCw") {
    std::lock_guard<std::mutex> l(g_mu);
    g_text = read_u16z(ARG(0), g_max);
    g_dirty = true;
}
SWKBD_EXPORT(IsDecideOkButton, "SwkbdIsDecideOkButton__3RplFPb") {
    std::lock_guard<std::mutex> l(g_mu);
    sync_receiver_locked();
    RET(g_ok ? 1 : 0);
}
// DrawStringInfo: six -1 words and a zero byte, as Cemu
SWKBD_EXPORT(GetDrawStringInfo, "SwkbdGetDrawStringInfo__3RplFPQ3_2nn5swkbd14DrawStringInfo") {
    for (uint32_t i = 0; i < 6; i++) mem::w32(ARG(0) + 4 * i, 0xFFFFFFFFu);
    mem::w8(ARG(0) + 0x18, 0);
    RET(0);
}
// an empty learning dictionary, field for field as Cemu's SwkbdInitLearnDic
SWKBD_EXPORT(InitLearnDic, "SwkbdInitLearnDic__3RplFPv") {
    const uint32_t d = ARG(0);
    if (!d) {
        RET(0);
        return;
    }
    constexpr uint32_t kSize = 0xA460, kEntries = 1000, kStride = 0x20, kIndex = kEntries + 1, kMagic = 0x4E4A4443;
    const uint32_t first = 0x48, second = first + kIndex * 2, index_end = second + kIndex * 2, area_end = index_end + kEntries * kStride;
    for (uint32_t i = 0; i < kSize; i += 4) mem::w32(d + i, 0);
    const uint32_t hdr[18] = {kMagic, 0x30000, 0x80020000, kSize - 0x48, 0x2C, 0x6E, 0x6E, index_end, 0, 0,
                              kEntries, kStride, 0, 0, 0, first, second, area_end};
    for (uint32_t i = 0; i < 18; i++) mem::w32(d + 4 * i, hdr[i]);
    mem::w32(d + kSize - 4, kMagic);
    RET(1);
}
SWKBD_EXPORT(IsNeedCalcSubThreadFont, "SwkbdIsNeedCalcSubThreadFont__3RplFv") {
    std::lock_guard<std::mutex> l(g_mu);
    RET(g_sub_font > 0 ? 1 : 0);
}
SWKBD_EXPORT(IsNeedCalcSubThreadPredict, "SwkbdIsNeedCalcSubThreadPredict__3RplFv") {
    std::lock_guard<std::mutex> l(g_mu);
    RET(g_sub_predict > 0 ? 1 : 0);
}
SWKBD_EXPORT(CalcSubThreadFont, "SwkbdCalcSubThreadFont__3RplFv") {
    std::lock_guard<std::mutex> l(g_mu);
    if (g_sub_font > 0) g_sub_font--;
}
SWKBD_EXPORT(CalcSubThreadPredict, "SwkbdCalcSubThreadPredict__3RplFv") {
    std::lock_guard<std::mutex> l(g_mu);
    if (g_sub_predict > 0) g_sub_predict--;
}

// ---- host side (swkbd_host.h) ----
namespace swkbd {
bool active() {
    std::lock_guard<std::mutex> l(g_mu);
    return g_active;
}

std::string text() {
    std::lock_guard<std::mutex> l(g_mu);
    std::string out;
    for (char16_t ch : g_text) {
        const uint32_t u = ch;
        if (u < 0x80) out += (char)u;
        else if (u < 0x800) out += (char)(0xC0 | (u >> 6)), out += (char)(0x80 | (u & 0x3F));
        else out += (char)(0xE0 | (u >> 12)), out += (char)(0x80 | ((u >> 6) & 0x3F)), out += (char)(0x80 | (u & 0x3F));
    }
    return out;
}

void type(const char* s) {
    std::lock_guard<std::mutex> l(g_mu);
    if (!g_active) return;
    const int max = max_len_locked();
    const unsigned char* p = (const unsigned char*)s;
    while (*p) {
        uint32_t u;
        int n;
        if (*p < 0x80) u = *p, n = 1;
        else if ((*p >> 5) == 6) u = *p & 0x1F, n = 2;
        else if ((*p >> 4) == 14) u = *p & 0x0F, n = 3;
        else u = *p & 0x07, n = 4;
        for (int i = 1; i < n; i++) {
            if ((p[i] & 0xC0) != 0x80) { n = i; break; }
            u = (u << 6) | (p[i] & 0x3F);
        }
        p += n;
        if (u < 0x20 || u > 0xFFFF) continue;  // control characters; nothing outside the BMP in a UTF-16 name
        if ((int)g_text.size() < max) g_text.push_back((char16_t)u), g_dirty = true;
    }
}

void backspace() {
    std::lock_guard<std::mutex> l(g_mu);
    if (g_active && !g_text.empty()) g_text.pop_back(), g_dirty = true;
}

void confirm() {
    std::lock_guard<std::mutex> l(g_mu);
    if (g_active) g_ok = true;
}

int take_visibility_change() {
    std::lock_guard<std::mutex> l(g_mu);
    const int v = g_vis;
    g_vis = 0;
    return v;
}
}  // namespace swkbd
