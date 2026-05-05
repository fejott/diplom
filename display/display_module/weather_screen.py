"""
WeatherScreen — continuous display loop for the weather station.

Usage (in main.py or a dedicated process):

    from display import WeatherScreen

    def get_data():
        return {
            'temperature': bme280.temperature,
            'humidity':    bme280.humidity,
            'pressure':    bme280.pressure,
            'latitude':    gps.latitude,
            'longitude':   gps.longitude,
            'altitude':    gps.altitude,
            'gps_fix':     gps.has_fix,
        }

    screen = WeatherScreen(data_source=get_data, update_interval=5.0)
    screen.run()          # blocks; handles SIGINT/SIGTERM cleanly

Or update from another thread:
    screen = WeatherScreen()
    screen.update(temperature=22.5, humidity=60.0)
    screen.run()
"""

import signal
import threading
import time

from .tft_display import TFTDisplay


class WeatherScreen:
    def __init__(self, data_source=None, update_interval: float = 5.0,
                 fb_device: str = '/dev/fb0'):
        self._display         = TFTDisplay(fb_device=fb_device)
        self._data_source     = data_source
        self._update_interval = update_interval
        self._data: dict      = {}
        self._lock            = threading.Lock()
        self._running         = False

        signal.signal(signal.SIGTERM, self._stop)
        signal.signal(signal.SIGINT,  self._stop)

    # ── Thread-safe data injection ────────────────────────────────────────────

    def update(self, **kwargs):
        """Push new sensor values from any thread."""
        with self._lock:
            self._data.update(kwargs)

    # ── Main loop ─────────────────────────────────────────────────────────────

    def run(self):
        self._running = True
        last_poll = 0.0

        try:
            while self._running:
                now = time.monotonic()

                if self._data_source and (now - last_poll) >= self._update_interval:
                    self._poll()
                    last_poll = now

                with self._lock:
                    snapshot = dict(self._data)
                snapshot['timestamp'] = time.time()

                if not self._display.render(snapshot):
                    break

                self._display.tick()
        finally:
            self._display.close()

    # ── Internals ─────────────────────────────────────────────────────────────

    def _poll(self):
        try:
            fresh = self._data_source()
            if isinstance(fresh, dict):
                with self._lock:
                    self._data.update(fresh)
        except Exception:
            pass  # keep showing last known data on transient sensor errors

    def _stop(self, *_):
        self._running = False


# ── Standalone test (run with: python -m display.weather_screen) ──────────────
if __name__ == '__main__':
    import math
    t0 = time.time()

    def synthetic_sensor():
        elapsed = time.time() - t0
        return {
            'temperature': 22.5  + math.sin(elapsed / 30) * 8,
            'humidity':    58.0  + math.cos(elapsed / 20) * 12,
            'pressure':    1013.2 + math.sin(elapsed / 60) * 4,
            'latitude':    55.75580,
            'longitude':   37.61730,
            'altitude':    155.4,
            'gps_fix':     (int(elapsed) % 30) > 8,
        }

    screen = WeatherScreen(data_source=synthetic_sensor, update_interval=2.0)
    screen.run()
