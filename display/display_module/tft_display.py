"""
Low-level rendering layer for the ILI9341 240x320 TFT display.

Works with fbcp-ili9341, which mirrors /dev/fb0 to the SPI screen.
In headless mode  → draws directly to the Linux framebuffer via SDL.
In desktop mode   → opens a 240x320 window (fbcp-ili9341 picks it up).
"""

import os
import time
import pygame
from datetime import datetime


class TFTDisplay:
    WIDTH  = 240
    HEIGHT = 320

    # Refresh rate for weather data (2 Hz is more than enough)
    FPS = 2

    C = {
        'bg':      (10,  15,  30),
        'header':  (18,  28,  58),
        'accent':  (50,  180, 255),
        'white':   (235, 242, 255),
        'gray':    (110, 125, 148),
        'green':   (55,  210, 105),
        'orange':  (255, 165,  50),
        'red':     (255,  70,  70),
        'blue':    (80,  150, 255),
        'divider': (30,  45,  80),
    }

    def __init__(self, fb_device: str = '/dev/fb0'):
        if not os.environ.get('DISPLAY'):
            os.putenv('SDL_FBDEV', fb_device)
            os.putenv('SDL_VIDEODRIVER', 'fbcon')
            os.putenv('SDL_NOMOUSE', '1')

        pygame.init()
        pygame.mouse.set_visible(False)

        if os.environ.get('DISPLAY'):
            self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
            pygame.display.set_caption('Weather Station')
        else:
            self.screen = pygame.display.set_mode(
                (self.WIDTH, self.HEIGHT),
                pygame.FULLSCREEN | pygame.NOFRAME,
            )

        self.clock = pygame.time.Clock()
        self._load_fonts()

    def _load_fonts(self):
        for name in ('DejaVuSansMono', 'FreeMono', 'LiberationMono', 'Courier', None):
            try:
                self.f_title = pygame.font.SysFont(name, 13, bold=True)
                self.f_label = pygame.font.SysFont(name, 11)
                self.f_value = pygame.font.SysFont(name, 27, bold=True)
                self.f_small = pygame.font.SysFont(name, 10)
                break
            except Exception:
                continue

    # ── Public API ────────────────────────────────────────────────────────────

    def render(self, data: dict) -> bool:
        """Draw one frame.  Returns False if the app should exit."""
        self.screen.fill(self.C['bg'])
        self._draw_header(data.get('timestamp'))
        self._draw_sensors(data)
        self._draw_gps(data)
        self._draw_footer()
        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                return False
        return True

    def tick(self):
        self.clock.tick(self.FPS)

    def close(self):
        pygame.quit()

    # ── Private drawing helpers ───────────────────────────────────────────────

    def _draw_header(self, timestamp=None):
        pygame.draw.rect(self.screen, self.C['header'], (0, 0, self.WIDTH, 26))
        pygame.draw.line(self.screen, self.C['accent'], (0, 26), (self.WIDTH, 26), 1)

        title = self.f_title.render('WEATHER STATION', True, self.C['accent'])
        self.screen.blit(title, (self.WIDTH // 2 - title.get_width() // 2, 6))

        if timestamp:
            ts_str = datetime.fromtimestamp(timestamp).strftime('%H:%M:%S')
            ts = self.f_small.render(ts_str, True, self.C['gray'])
            self.screen.blit(ts, (self.WIDTH - ts.get_width() - 4, 30))

    def _draw_sensors(self, data):
        rows = [
            ('TEMPERATURE', data.get('temperature'), '{:.1f} °C', 'white'),
            ('HUMIDITY',    data.get('humidity'),    '{:.1f} %',  'blue'),
            ('PRESSURE',    data.get('pressure'),    '{:.1f} hPa','green'),
        ]

        y = 34
        for label, value, fmt, color_key in rows:
            y += 5
            lbl = self.f_label.render(label, True, self.C['gray'])
            self.screen.blit(lbl, (10, y))
            y += 14

            if value is not None:
                val_surf = self.f_value.render(fmt.format(value), True, self.C[color_key])
            else:
                val_surf = self.f_value.render(fmt.format(0).replace('0', '-'), True, self.C['gray'])
            self.screen.blit(val_surf, (10, y))
            y += 32

            pygame.draw.line(self.screen, self.C['divider'], (8, y), (self.WIDTH - 8, y), 1)

    def _draw_gps(self, data):
        gps_y = 174
        pygame.draw.rect(self.screen, self.C['header'],
                         (0, gps_y, self.WIDTH, self.HEIGHT - gps_y - 16))
        pygame.draw.line(self.screen, self.C['accent'],
                         (0, gps_y), (self.WIDTH, gps_y), 1)

        gps_fix   = data.get('gps_fix', False)
        latitude  = data.get('latitude')
        longitude = data.get('longitude')
        altitude  = data.get('altitude')

        fix_color  = self.C['green'] if gps_fix else self.C['orange']
        dot        = '●' if gps_fix else '○'
        fix_text   = f'GPS {dot} {"FIX" if gps_fix else "SEARCHING..."}'
        fix_surf   = self.f_label.render(fix_text, True, fix_color)
        self.screen.blit(fix_surf, (10, gps_y + 5))

        coord_color = self.C['white'] if gps_fix else self.C['gray']
        y = gps_y + 20

        if latitude is not None:
            ns  = 'N' if latitude >= 0 else 'S'
            lat = self.f_small.render(f'LAT  {abs(latitude):9.5f}° {ns}', True, coord_color)
            self.screen.blit(lat, (10, y))
        y += 14

        if longitude is not None:
            ew  = 'E' if longitude >= 0 else 'W'
            lon = self.f_small.render(f'LON  {abs(longitude):9.5f}° {ew}', True, coord_color)
            self.screen.blit(lon, (10, y))
        y += 14

        if altitude is not None:
            alt = self.f_small.render(f'ALT  {altitude:.1f} m', True, coord_color)
            self.screen.blit(alt, (10, y))

    def _draw_footer(self):
        pygame.draw.line(self.screen, self.C['divider'],
                         (0, self.HEIGHT - 16), (self.WIDTH, self.HEIGHT - 16), 1)
        ver = self.f_small.render('Pi4 aarch64  ILI9341 240x320', True, self.C['gray'])
        self.screen.blit(ver, (self.WIDTH // 2 - ver.get_width() // 2, self.HEIGHT - 13))
