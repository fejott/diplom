"""
WiFi Settings Screen for ILI9341 240x320 display.

USB keyboard controls:
  Up / Down    — navigate network list
  Enter        — select network / confirm connect
  Escape       — back / exit
  Backspace    — delete last password character
  F5           — rescan networks
  Any key      — type password character (Shift supported)
"""

import subprocess
import threading
import time
from typing import Optional

from PIL import Image, ImageDraw

try:
    import evdev
    from evdev import InputDevice, ecodes, categorize, KeyEvent
    _EVDEV_OK = True
except ImportError:
    _EVDEV_OK = False

from .tft_display import TFTDisplay, C, _font

# ── US keyboard layout maps ────────────────────────────────────────────────────

_UNSHIFTED = {
    'KEY_A': 'a', 'KEY_B': 'b', 'KEY_C': 'c', 'KEY_D': 'd', 'KEY_E': 'e',
    'KEY_F': 'f', 'KEY_G': 'g', 'KEY_H': 'h', 'KEY_I': 'i', 'KEY_J': 'j',
    'KEY_K': 'k', 'KEY_L': 'l', 'KEY_M': 'm', 'KEY_N': 'n', 'KEY_O': 'o',
    'KEY_P': 'p', 'KEY_Q': 'q', 'KEY_R': 'r', 'KEY_S': 's', 'KEY_T': 't',
    'KEY_U': 'u', 'KEY_V': 'v', 'KEY_W': 'w', 'KEY_X': 'x', 'KEY_Y': 'y',
    'KEY_Z': 'z',
    'KEY_0': '0', 'KEY_1': '1', 'KEY_2': '2', 'KEY_3': '3', 'KEY_4': '4',
    'KEY_5': '5', 'KEY_6': '6', 'KEY_7': '7', 'KEY_8': '8', 'KEY_9': '9',
    'KEY_SPACE': ' ',   'KEY_MINUS': '-',    'KEY_EQUAL': '=',
    'KEY_LEFTBRACE': '[', 'KEY_RIGHTBRACE': ']',
    'KEY_SEMICOLON': ';', 'KEY_APOSTROPHE': "'", 'KEY_GRAVE': '`',
    'KEY_BACKSLASH': '\\', 'KEY_COMMA': ',', 'KEY_DOT': '.', 'KEY_SLASH': '/',
}

_SHIFTED = {
    'KEY_A': 'A', 'KEY_B': 'B', 'KEY_C': 'C', 'KEY_D': 'D', 'KEY_E': 'E',
    'KEY_F': 'F', 'KEY_G': 'G', 'KEY_H': 'H', 'KEY_I': 'I', 'KEY_J': 'J',
    'KEY_K': 'K', 'KEY_L': 'L', 'KEY_M': 'M', 'KEY_N': 'N', 'KEY_O': 'O',
    'KEY_P': 'P', 'KEY_Q': 'Q', 'KEY_R': 'R', 'KEY_S': 'S', 'KEY_T': 'T',
    'KEY_U': 'U', 'KEY_V': 'V', 'KEY_W': 'W', 'KEY_X': 'X', 'KEY_Y': 'Y',
    'KEY_Z': 'Z',
    'KEY_0': ')', 'KEY_1': '!', 'KEY_2': '@', 'KEY_3': '#', 'KEY_4': '$',
    'KEY_5': '%', 'KEY_6': '^', 'KEY_7': '&', 'KEY_8': '*', 'KEY_9': '(',
    'KEY_SPACE': ' ',   'KEY_MINUS': '_',    'KEY_EQUAL': '+',
    'KEY_LEFTBRACE': '{', 'KEY_RIGHTBRACE': '}',
    'KEY_SEMICOLON': ':', 'KEY_APOSTROPHE': '"', 'KEY_GRAVE': '~',
    'KEY_BACKSLASH': '|', 'KEY_COMMA': '<',  'KEY_DOT': '>',  'KEY_SLASH': '?',
}

_SHIFT_KEYS = {'KEY_LEFTSHIFT', 'KEY_RIGHTSHIFT'}


class WiFiScreen:
    """Interactive WiFi settings screen for the TFT display."""

    _ITEM_H    = 30
    _LIST_TOP  = 30
    _MAX_ROWS  = 7

    def __init__(self, display: TFTDisplay):
        self._display  = display
        self._networks: list[str] = []
        self._selected = 0
        self._scroll   = 0
        self._password = ''
        self._shift    = False
        self._mode     = 'list'   # 'list' | 'password' | 'connecting'
        self._status   = ''
        self._keyboard: Optional[InputDevice] = None
        self._running  = False

        self._f_title = _font(13, bold=True)
        self._f_item  = _font(12)
        self._f_small = _font(10)
        self._f_pass  = _font(11)

    # ── Entry point ───────────────────────────────────────────────────────────

    def run(self):
        self._running = True
        self._mode    = 'list'
        self._status  = 'Scanning...'
        self._render()

        threading.Thread(target=self._scan, daemon=True).start()

        if not _EVDEV_OK:
            self._status = 'ERROR: pip install evdev'
            self._render()
            time.sleep(3)
            self._running = False
            return

        self._keyboard = self._find_keyboard()
        if not self._keyboard:
            self._status = 'ERROR: no USB keyboard found'
            self._render()
            time.sleep(3)
            self._running = False
            return

        self._event_loop()
        self._running = False

    # ── Keyboard input ────────────────────────────────────────────────────────

    @staticmethod
    def _find_keyboard() -> Optional[InputDevice]:
        for path in evdev.list_devices():
            dev = InputDevice(path)
            caps = dev.capabilities()
            if ecodes.EV_KEY in caps and ecodes.KEY_ENTER in caps[ecodes.EV_KEY]:
                return dev
        return None

    def _event_loop(self):
        for event in self._keyboard.read_loop():
            if not self._running:
                break
            if event.type != ecodes.EV_KEY:
                continue
            kev = categorize(event)

            if kev.keycode in _SHIFT_KEYS:
                self._shift = kev.keystate in (KeyEvent.key_down, KeyEvent.key_hold)
                continue

            if kev.keystate not in (KeyEvent.key_down, KeyEvent.key_hold):
                continue

            self._handle_key(kev.keycode)

    def _handle_key(self, keycode: str):
        if self._mode == 'list':
            if keycode == 'KEY_UP':
                self._selected = max(0, self._selected - 1)
            elif keycode == 'KEY_DOWN':
                self._selected = min(len(self._networks) - 1, self._selected + 1)
            elif keycode in ('KEY_ENTER', 'KEY_KPENTER'):
                if self._networks:
                    self._password = ''
                    self._status   = ''
                    self._mode     = 'password'
            elif keycode == 'KEY_F5':
                self._networks = []
                self._status   = 'Scanning...'
                threading.Thread(target=self._scan, daemon=True).start()
            elif keycode == 'KEY_ESC':
                self._running = False
                return

        elif self._mode == 'password':
            if keycode in ('KEY_ENTER', 'KEY_KPENTER'):
                threading.Thread(target=self._connect, daemon=True).start()
                return
            elif keycode == 'KEY_ESC':
                self._mode   = 'list'
                self._status = ''
            elif keycode == 'KEY_BACKSPACE':
                self._password = self._password[:-1]
            else:
                ch = (_SHIFTED if self._shift else _UNSHIFTED).get(keycode, '')
                if ch:
                    self._password += ch

        self._render()

    # ── WiFi operations ───────────────────────────────────────────────────────

    def _scan(self):
        try:
            r = subprocess.run(
                ['nmcli', '--terse', '-f', 'SSID,SIGNAL',
                 'dev', 'wifi', 'list', '--rescan', 'yes'],
                capture_output=True, text=True, timeout=20,
            )
            seen: set[str] = set()
            nets: list[str] = []
            for line in r.stdout.splitlines():
                ssid = line.split(':')[0].strip()
                if ssid and ssid not in seen:
                    seen.add(ssid)
                    nets.append(ssid)
            self._networks = nets
            self._selected = 0
            self._scroll   = 0
            self._status   = f'{len(nets)} networks found' if nets else 'No networks found'
        except Exception as exc:
            self._status = f'Scan error: {exc}'
        self._render()

    def _connect(self):
        ssid = self._networks[self._selected]
        self._mode   = 'connecting'
        self._status = f'Connecting to {ssid[:20]}...'
        self._render()
        try:
            r = subprocess.run(
                ['nmcli', 'dev', 'wifi', 'connect', ssid, 'password', self._password],
                capture_output=True, text=True, timeout=30,
            )
            if r.returncode == 0:
                self._status = 'Connected!'
            else:
                err = (r.stderr or r.stdout).strip()
                self._status = 'Failed: ' + err[:26]
        except Exception as exc:
            self._status = f'Error: {exc}'
        self._render()
        time.sleep(2)
        self._running = False

    # ── Rendering ─────────────────────────────────────────────────────────────

    def _render(self):
        W, H = 240, 320
        img  = Image.new('RGB', (W, H), C['bg'])
        draw = ImageDraw.Draw(img)

        # Header bar
        draw.rectangle([(0, 0), (W, 26)], fill=C['header'])
        draw.line([(0, 26), (W, 26)], fill=C['accent'], width=1)
        titles = {
            'list':       'WiFi Networks',
            'password':   'Enter Password',
            'connecting': 'Connecting...',
        }
        title = titles.get(self._mode, 'WiFi')
        tw = draw.textlength(title, font=self._f_title)
        draw.text(((W - tw) / 2, 6), title, font=self._f_title, fill=C['accent'])

        if self._mode == 'list':
            self._draw_list(draw, W, H)
        else:
            self._draw_password(draw, W, H)

        # Status bar
        if self._status:
            draw.rectangle([(0, H - 22), (W, H)], fill=C['header'])
            col = (C['green']  if 'Connected' in self._status else
                   C['orange'] if ('Failed' in self._status or 'Error' in self._status) else
                   C['gray'])
            sw = draw.textlength(self._status, font=self._f_small)
            draw.text(((W - sw) / 2, H - 18), self._status, font=self._f_small, fill=col)

        self._display.display_image(img)

    def _draw_list(self, draw: ImageDraw.ImageDraw, W: int, H: int):
        if not self._networks:
            msg = self._status or 'Scanning...'
            mw  = draw.textlength(msg, font=self._f_item)
            draw.text(((W - mw) / 2, H // 2 - 8), msg, font=self._f_item, fill=C['gray'])
            return

        visible = min(self._MAX_ROWS, (H - 60) // self._ITEM_H)

        if self._selected < self._scroll:
            self._scroll = self._selected
        elif self._selected >= self._scroll + visible:
            self._scroll = self._selected - visible + 1

        y = self._LIST_TOP
        for i in range(self._scroll, min(self._scroll + visible, len(self._networks))):
            ssid   = self._networks[i]
            is_sel = (i == self._selected)
            draw.rectangle([(0, y), (W, y + self._ITEM_H - 1)],
                           fill=C['header'] if is_sel else C['bg'])
            col = C['accent'] if is_sel else C['white']
            pfx = '▶ ' if is_sel else '  '
            draw.text((8, y + 7), pfx + ssid[:22], font=self._f_item, fill=col)
            draw.line([(8, y + self._ITEM_H), (W - 8, y + self._ITEM_H)],
                      fill=C['divider'], width=1)
            y += self._ITEM_H

        hint = '↑↓ Select  Enter OK  Esc Exit  F5 Scan'
        hw = draw.textlength(hint, font=self._f_small)
        draw.text(((W - hw) / 2, H - 40), hint, font=self._f_small, fill=C['gray'])

    def _draw_password(self, draw: ImageDraw.ImageDraw, W: int, H: int):
        ssid = self._networks[self._selected] if self._networks else ''
        draw.text((10, 32), 'Network:', font=self._f_small, fill=C['gray'])
        draw.text((10, 46), ssid[:26], font=self._f_item, fill=C['white'])

        draw.line([(0, 70), (W, 70)], fill=C['divider'], width=1)
        draw.text((10, 76), 'Password:', font=self._f_small, fill=C['gray'])

        # Password box
        masked = '●' * len(self._password)
        shown  = masked[-19:] if len(masked) > 19 else masked
        bx1, by1, bx2, by2 = 8, 92, W - 8, 118
        draw.rectangle([(bx1, by1), (bx2, by2)], fill=C['header'])
        draw.rectangle([(bx1, by1), (bx2, by2)], outline=C['accent'], width=1)
        draw.text((14, by1 + 5), shown or ' ', font=self._f_pass, fill=C['white'])
        cx = 14 + draw.textlength(shown, font=self._f_pass)
        draw.text((cx, by1 + 5), '|', font=self._f_pass, fill=C['accent'])

        hints = ['Type password on keyboard', 'Enter = Connect   Esc = Back']
        for n, h in enumerate(hints):
            hw = draw.textlength(h, font=self._f_small)
            draw.text(((W - hw) / 2, 145 + n * 16), h, font=self._f_small, fill=C['gray'])
