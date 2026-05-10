"""
ILI9341 240x320 TFT display driver using luma.lcd.

Talks directly to the SPI device — no fbcp-ili9341 framebuffer service needed.
Wiring:
  SPI0 CE0  (BCM 8/10/11) = physical pins 24/19/23
  DC   = GPIO 24 (physical 18)
  RST  = GPIO 23 (physical 16)
  BL   = GPIO 4  (physical 7)   (backlight, active HIGH)

Layout (portrait 240×320):
  0–26   header bar  "WEATHER STATION"
  26–38  timestamp
  38–143 sensors  (TEMP / HUM / PRESSURE, 35 px each)
  143–195 GPS section  (compact: 52 px)
  195–308 Forecast section  (113 px — fits online API + precip line)
  308–320 footer
"""

import time
from datetime import datetime
from typing import Optional

from PIL import Image, ImageDraw, ImageFont


# ── Font helpers ───────────────────────────────────────────────────────────────

_FONT_PATHS = [
    '/usr/share/fonts/truetype/dejavu/DejaVuSansMono{suffix}.ttf',
    '/usr/share/fonts/truetype/liberation/LiberationMono-{suffix}.ttf',
    '/usr/share/fonts/truetype/freefont/FreeMono{suffix}.ttf',
]


def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    suffix = 'Bold' if bold else ''
    suffix2 = 'Bold' if bold else 'Regular'
    for pattern in _FONT_PATHS:
        path = pattern.format(suffix=suffix) if '{}' not in pattern \
               else pattern.format(suffix=suffix2)
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            pass
    return ImageFont.load_default()


# ── Colour palette ─────────────────────────────────────────────────────────────

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
    'purple':  (160, 100, 255),
    'cyan':    (0,   210, 175),   # online API indicator
}


# ── Display class ──────────────────────────────────────────────────────────────

class TFTDisplay:
    """
    Renders weather + GPS + forecast to an ILI9341 240×320 TFT via luma.lcd.

    Usage:
        display = TFTDisplay()
        display.render({
            'temperature': 22.5, 'humidity': 60.0, 'pressure': 1013.0,
            'latitude': 55.75, 'longitude': 37.61, 'altitude': 155.0,
            'gps_fix': True, 'timestamp': time.time(),
            'forecast': forecast_result,   # ForecastResult or None
            'data_count': 142,             # rows collected so far
        })
        display.close()
    """

    WIDTH  = 240
    HEIGHT = 320

    def __init__(self,
                 spi_port: int = 0,
                 spi_device: int = 0,
                 gpio_dc: int = 24,
                 gpio_rst: int = 23,
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

        self._backlight = None
        if gpio_backlight is not None:
            try:
                from gpiozero import LED
                self._backlight = LED(gpio_backlight)
                self._backlight.on()
            except Exception:
                pass

        self._load_fonts()

    def _load_fonts(self) -> None:
        self.f_title = _font(13, bold=True)
        self.f_label = _font(10)
        self.f_value = _font(18, bold=True)
        self.f_small = _font(10)
        self.f_tiny  = _font(9)

    # ── Public API ─────────────────────────────────────────────────────────────

    def render(self, data: dict) -> None:
        """Draw one full frame from sensor + forecast data."""
        img  = Image.new('RGB', (self.WIDTH, self.HEIGHT), C['bg'])
        draw = ImageDraw.Draw(img)

        self._draw_header(draw, data.get('timestamp'))
        self._draw_sensors(draw, data)
        self._draw_gps(draw, data)
        self._draw_forecast(draw, data.get('forecast'), data.get('data_count', 0))
        self._draw_footer(draw)

        self._device.display(img)

    def display_image(self, img: Image.Image) -> None:
        """Push a pre-rendered PIL image directly to the display."""
        self._device.display(img)

    def close(self) -> None:
        if self._backlight:
            self._backlight.off()
        self._device.cleanup()

    # ── Section renderers ──────────────────────────────────────────────────────

    def _draw_header(self, draw: ImageDraw.ImageDraw, timestamp: Optional[float]) -> None:
        """Top bar with title and timestamp.  Occupies y 0–38."""
        W = self.WIDTH
        draw.rectangle([(0, 0), (W, 26)], fill=C['header'])
        draw.line([(0, 26), (W, 26)], fill=C['accent'], width=1)

        title = 'WEATHER STATION'
        tw = draw.textlength(title, font=self.f_title)
        draw.text(((W - tw) / 2, 5), title, font=self.f_title, fill=C['accent'])

        if timestamp:
            ts = datetime.fromtimestamp(timestamp).strftime('%H:%M:%S')
            tw2 = draw.textlength(ts, font=self.f_tiny)
            draw.text((W - tw2 - 4, 28), ts, font=self.f_tiny, fill=C['gray'])

    def _draw_sensors(self, draw: ImageDraw.ImageDraw, data: dict) -> None:
        """Three sensor rows.  Occupies y 38–143."""
        rows = [
            ('TEMPERATURE', data.get('temperature'), '{:.1f} °C',  'white'),
            ('HUMIDITY',    data.get('humidity'),    '{:.1f} %',   'blue'),
            ('PRESSURE',    data.get('pressure'),    '{:.1f} hPa', 'green'),
        ]
        y = 38
        for label, value, fmt, col_key in rows:
            draw.text((10, y + 2), label, font=self.f_label, fill=C['gray'])
            text  = fmt.format(value) if value is not None else '---'
            color = C[col_key]       if value is not None else C['gray']
            draw.text((10, y + 14), text, font=self.f_value, fill=color)
            y += 35
            draw.line([(8, y), (self.WIDTH - 8, y)], fill=C['divider'], width=1)
        # y ends at 143

    def _draw_gps(self, draw: ImageDraw.ImageDraw, data: dict) -> None:
        """GPS section.  Occupies y 143–195 (compact)."""
        W    = self.WIDTH
        top  = 143
        draw.rectangle([(0, top), (W, top + 14)], fill=C['header'])
        draw.line([(0, top), (W, top)], fill=C['accent'], width=1)

        gps_fix   = data.get('gps_fix', False)
        latitude  = data.get('latitude')
        longitude = data.get('longitude')
        altitude  = data.get('altitude')

        fix_col  = C['green'] if gps_fix else C['orange']
        dot      = '●' if gps_fix else '○'
        fix_text = f'GPS {dot} {"FIX" if gps_fix else "SEARCHING..."}'
        draw.text((10, top + 2), fix_text, font=self.f_label, fill=fix_col)

        y    = top + 17
        ccol = C['white'] if gps_fix else C['gray']

        if gps_fix and latitude is not None and longitude is not None:
            ns = 'N' if latitude  >= 0 else 'S'
            ew = 'E' if longitude >= 0 else 'W'
            # Lat + lon on one line to save vertical space
            draw.text((10, y),
                      f'{abs(latitude):.4f}°{ns}  {abs(longitude):.4f}°{ew}',
                      font=self.f_tiny, fill=ccol)
            y += 13
            if altitude is not None:
                draw.text((10, y), f'ALT {altitude:.0f} m',
                          font=self.f_tiny, fill=ccol)
        else:
            draw.text((10, y), 'Ожидание сигнала...', font=self.f_tiny, fill=C['gray'])

        # Bottom divider — forecast starts below here
        draw.line([(0, 195), (W, 195)], fill=C['divider'], width=1)

    # ── Forecast helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _precip_emoji(prob: float) -> str:
        if prob >= 0.7: return '🌧'
        if prob >= 0.4: return '🌦'
        if prob >= 0.2: return '🌤'
        return '☀'

    @staticmethod
    def _pressure_arrow(trend: float) -> str:
        if trend > 0.5:  return '▲'
        if trend < -0.5: return '▼'
        return '—'

    def _draw_forecast(self, draw: ImageDraw.ImageDraw, forecast, data_count: int) -> None:
        """Forecast section — unified table for all modes.  Occupies y 195–308 (113 px).

        Layout (after 14 px header bar):
          y+0   column header:  Вр / Темп / Осадки
          y+13  +1h row
          y+26  +2h row
          y+39  +3h row
          y+52  pressure trend
          y+65  forecast text
          y+78  confidence + valid_until
          y+91  data counter (rule-based only, while LSTM not ready)
        All lines use f_tiny (9 pt).  Total content height ≤ 99 px — fits in 113 px section.
        """
        W   = self.WIDTH
        top = 195
        draw.line([(0, top), (W, top)], fill=C['accent'], width=1)
        draw.rectangle([(0, top), (W, top + 14)], fill=C['header'])

        # ── No forecast ───────────────────────────────────────────────────────
        if forecast is None:
            draw.text((10, top + 2), 'ПРОГНОЗ', font=self.f_label, fill=C['purple'])
            draw.text((10, top + 18), 'Нет данных', font=self.f_small, fill=C['gray'])
            return

        method   = forecast.method
        internet = getattr(forecast, 'internet_available', None)
        gps_used = getattr(forecast, 'gps_used', False)
        lstm_rdy = getattr(forecast, 'lstm_ready', True)

        # ── Header bar label ──────────────────────────────────────────────────
        if method == 'online_api':
            label, col = '🌐 Online API',    C['cyan']
        elif method == 'lstm':
            label, col = '🤖 Авт. LSTM',     C['purple']
        elif method == 'rule-based':
            if internet is False:
                label = '📡 Правила: нет сети'
            else:
                label = '📡 Правила'
            col = C['orange']
        else:  # insufficient_data
            draw.text((10, top + 2), '📡 Сбор данных...', font=self.f_label, fill=C['orange'])
            y = top + 17
            try:
                import config
                min_r = config.FORECAST_MIN_READINGS
            except Exception:
                min_r = 500
            draw.text((10, y), f'Накоплено: {data_count}/{min_r}',
                      font=self.f_tiny, fill=C['gray'])
            return

        draw.text((10, top + 2), label, font=self.f_label, fill=col)

        # ── Unified table (all active modes) ─────────────────────────────────
        y = top + 17

        # Column header
        draw.text((10, y), 'Вр    Темп    Осадки', font=self.f_tiny, fill=C['gray'])
        y += 13

        # Three hourly rows
        for h, temp, prob in (
            (1, forecast.temp_in_1h, forecast.precip_prob_1h),
            (2, forecast.temp_in_2h, forecast.precip_prob_2h),
            (3, forecast.temp_in_3h, forecast.precip_prob_3h),
        ):
            t_str = f'{temp:+.1f}°C' if temp is not None else '  N/A'
            if prob is not None:
                p_str = f'{self._precip_emoji(prob)}{round(prob * 100):3d}%'
            else:
                p_str = '  N/A'
            row = f'+{h}ч  {t_str:<7} {p_str}'
            draw.text((10, y), row, font=self.f_tiny, fill=C['white'])
            y += 13

        # Pressure trend
        arrow = self._pressure_arrow(forecast.pressure_trend)
        draw.text((10, y),
                  f'Давл: {arrow} {forecast.pressure_trend:+.1f} hPa/ч',
                  font=self.f_tiny, fill=C['blue'])
        y += 13

        # Forecast text (truncated to ~32 chars to fit screen width)
        draw.text((10, y), forecast.forecast_text[:32], font=self.f_tiny, fill=C['white'])
        y += 13

        # Confidence + valid_until
        draw.text((10, y),
                  f'Точн:{forecast.confidence * 100:.0f}%  '
                  f'До {forecast.valid_until.strftime("%H:%M")}',
                  font=self.f_tiny, fill=C['gray'])
        y += 13

        # Data counter — rule-based only, while LSTM is not yet ready
        if method == 'rule-based' and not lstm_rdy:
            try:
                import config
                min_r = config.FORECAST_MIN_READINGS
            except Exception:
                min_r = 500
            draw.text((10, y), f'Накоплено: {data_count}/{min_r}',
                      font=self.f_tiny, fill=C['gray'])

    def _draw_footer(self, draw: ImageDraw.ImageDraw) -> None:
        """Thin footer bar at the very bottom.  Occupies y 308–320."""
        draw.line([(0, 308), (self.WIDTH, 308)], fill=C['divider'], width=1)
        ver = 'ILI9341  Pi4 aarch64'
        tw  = draw.textlength(ver, font=self.f_tiny)
        draw.text(((self.WIDTH - tw) / 2, 310), ver, font=self.f_tiny, fill=C['divider'])
