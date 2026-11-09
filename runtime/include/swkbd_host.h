// Software keyboard (swkbd.rpl, runtime/src/coreinit/swkbd.cpp): the host side. The game's text entry (horse
// names, ...) is typed with the Steam Deck keyboard or a physical one; window.cpp feeds the text, the overlay shows it.
#pragma once
#include <string>

namespace swkbd {
bool active();                 // the game is showing its keyboard
std::string text();            // typed so far (UTF-8)
void type(const char* utf8);   // text from the Steam / physical keyboard
void backspace();
void confirm();                // Enter / Start: the game's OK button
int take_visibility_change();  // host main thread: +1 the game opened its keyboard, -1 closed it, 0 no change
}  // namespace swkbd
