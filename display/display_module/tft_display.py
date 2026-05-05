"""
ILI9341 240x320 TFT display driver using luma.lcd.

Talks directly to the SPI device — no fbcp-ili9341 framebuffer service needed.
Wiring (current physical layout):
  SPI0 CE0  (BCM 8/10/11) = physical pins 24/19/23
  DC   = GPIO 2  (physical 3)   ← shares I2C SDA, move to GPIO 24 (physical 18) when rewiring
  RST  = GPIO 3  (physical 5)   ← shares I2C SCL, move to GPIO 25 (physical 22) when rewiring
  BL   = GPIO 4  (physical 7)   (backlight, active HIGH)
"""

import os
import time
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont


# ── Font helpers ──────────────────────────────────────────────────────────────

_FONT_PATHS = [
    '/usr/share/fonts/truetype/dejavu/DejaVuSansMono{suffix}.ttf',
    '/usr/share/fonts/truetype/liberation/LiberationMono-{suffix}.ttf',
    '/usr/share/fonts/truetype/freefont/FreeMono{suffix}.ttf',
]

def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    suffix = 'Bold' if bold else ''
    suffix2 = 'Bold' if bold else 'Regular'
    for pattern in _FONT_PATHS:
        path = pattern.format(suffix=suffix) if '{}' not in pattern else pattern.format(suffix=suffix2)
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            pass
    return ImageFont.load_default()


# ── Colour palette ────────────────────────────────────────────────────────────

C = {
    'bg':      (10,  15,  30),
    'header':  (18,  28,  58),
    'accent':  (50,  180, 255),
    'white':   (235, 242, 255),
    'gray':    (110, 125, 148),
    'green':   (55,  210, 105),
    'orange':  (255, 165,  50),
    'blue':    (80,  150, 255),
    'divider': (30,  45,  80),
}


# ── Display class ─────────────────────────────────────────────────────────────

class TFTDisplay:
    """
    Renders weather data to an ILI9341 240x320 TFT via luma.lcd.

    Usage:
        display = TFTDisplay()
        display.render({
            'temperature': 22.5, 'humidity': 60.0, 'pressure': 1013.0,
            'latitude': 55.75, 'longitude': 37.61, 'altitude': 155.0,
            'gps_fix': True, 'timestamp': time.time(),
        })
        display.close()
    """

    WIDTH  = 240
    HEIGHT = 320

    def __init__(self,
                 spi_port: int = 0,
                 spi_device: int = 0,
                 gpio_dc: int = 2,
                 gpio_rst: int = 3,
                 gpio_backlight: int = 4):

        from luma.core.interface.serial import spi
        from luma.lcd.device import ili9341

        self._serial = spi(
            port=spi_port,
            device=spi_device,
            gpio_DC=gpio_dc,
            gpio_RST=gpio_rst,
            reset_active_low=True,
        )
        # luma.lcd 2.x only accepts ILI9341 in landscape (320×240); rotate=1
        # makes luma rotate our portrait canvas internally.
        self._device = ili9341(self._serial, width=320, height=240, rotate=1)

        # Backlight via gpiozero (works with lgpio on Trixie)
        self._backlight = None
        if gpio_backlight is not None:
            try:
                from gpiozero import LED
                self._backlight = LED(gpio_backlight)
                self._backlight.on()
            except Exception:
                pass  # backlight will stay on via hardware pull-up on some boards

        self._load_fonts()

    def _load_fonts(self):
        self.f_title = _font(13, bold=True)
        self.f_label = _font(11)
        self.f_value = _font(27, bold=True)
        self.f_small = _font(10)

    # ── Public API ────────────────────────────────────────────────────────────

    def render(self, data: dict):
        """Draw one frame of weather data to the display."""
        img = Image.new('RGB', (self.WIDTH, self.HEIGHT), C['bg'])
        draw = ImageDraw.Draw(img)

        self._draw_header(draw, data.get('timestamp'))
        self._draw_sensors(draw, data)
        self._draw_gps(draw, data)
        self._draw_footer(draw)

        self._device.display(img)

    def display_image(self, img: Image.Image):
        """Push a pre-rendered PIL image directly to the display."""
        self._device.display(img)

    def close(self):
        if self._backlight:
            self._backlight.off()
        self._device.cleanup()

    # ── Drawing helpers ───────────────────────────────────────────────────────

    def _draw_header(self, draw: ImageDraw.ImageDraw, timestamp=None):
        draw.rectangle([(0, 0), (self.WIDTH, 26)], fill=C['header'])
        draw.line([(0, 26), (self.WIDTH, 26)], fill=C['accent'], width=1)

        title = 'WEATHER STATION'
        tw = draw.textlength(title, font=self.f_title)
        draw.text(((self.WIDTH - tw) / 2, 6), title, font=self.f_title, fill=C['accent'])

        if timestamp:
            ts_str = datetime.fromtimestamp(timestamp).strftime('%H:%M:%S')
            tw2 = draw.textlength(ts_str, font=self.f_small)
            draw.text((self.WIDTH - tw2 - 4, 30), ts_str, font=self.f_small, fill=C['gray'])

    def _draw_sensors(self, draw: ImageDraw.ImageDraw, data: dict):
        rows = [
            ('TEMPERATURE', data.get('temperature'), '{:.1f} °C', 'white'),
            ('HUMIDITY',    data.get('humidity'),    '{:.1f} %',        'blue'),
            ('PRESSURE',    data.get('pressure'),    '{:.1f} hPa',      'green'),
        ]
        y = 34
        for label, value, fmt, color_key in rows:
            y += 5
            draw.text((10, y), label, font=self.f_label, fill=C['gray'])
            y += 14
            if value is not None:
                text = fmt.format(value)
                color = C[color_key]
            else:
                text = '---'
                color = C['gray']
            draw.text((10, y), text, font=self.f_value, fill=color)
            y += 32
            draw.line([(8, y), (self.WIDTH - 8, y)], fill=C['divider'], width=1)

    def _draw_gps(self, draw: ImageDraw.ImageDraw, data: dict):
        gps_y = 174
        draw.rectangle([(0, gps_y), (self.WIDTH, self.HEIGHT - 16)], fill=C['header'])
        draw.line([(0, gps_y), (self.WIDTH, gps_y)], fill=C['accent'], width=1)

        gps_fix   = data.get('gps_fix', False)
        latitude  = data.get('latitude')
        longitude = data.get('longitude')
        altitude  = data.get('altitude')

        fix_color = C['green'] if gps_fix else C['orange']
        dot       = '●' if gps_fix else '○'
        fix_text  = f'GPS {dot} {"FIX" if gps_fix else "SEARCHING..."}'
        draw.text((10, gps_y + 5), fix_text, font=self.f_label, fill=fix_color)

        coord_color = C['white'] if gps_fix else C['gray']
        y = gps_y + 20

        if latitude is not None:
            ns = 'N' if latitude >= 0 else 'S'
            draw.text((10, y), f'LAT  {abs(latitude):9.5f}° {ns}', font=self.f_small, fill=coord_color)
        y += 14

        if longitude is not None:
            ew = 'E' if longitude >= 0 else 'W'
            draw.text((10, y), f'LON  {abs(longitude):9.5f}° {ew}', font=self.f_small, fill=coord_color)
        y += 14

        if altitude is not None:
            draw.text((10, y), f'ALT  {altitude:.1f} m', font=self.f_small, fill=coord_color)

    def _draw_footer(self, draw: ImageDraw.ImageDraw):
        draw.line([(0, self.HEIGHT - 16), (self.WIDTH, self.HEIGHT - 16)], fill=C['divider'], width=1)
        ver = 'Pi4 aarch64  ILI9341 240x320'
        tw = draw.textlength(ver, font=self.f_small)
        draw.text(((self.WIDTH - tw) / 2, self.HEIGHT - 13), ver, font=self.f_small, fill=C['gray'])
