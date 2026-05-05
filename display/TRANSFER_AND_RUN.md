# Display setup — Pi 4 aarch64 Trixie + ILI9341

## Why not fbcp-ili9341?

`fbcp-ili9341` captures the display using the **DispmanX API** (`vc_dispmanx_*`),
which is part of the VideoCore userland.  That API was **removed from
64-bit Raspberry Pi OS** — and `libraspberrypi-dev` is no longer in the
Trixie package repos.  Recompiling won't help: the underlying library simply
doesn't exist.

The replacement is **luma.lcd**, a pure-Python ILI9341 driver that talks
directly to `/dev/spidev0.0`.  No binary to build, no framebuffer service
to keep running.

---

## Step 1 — Run the setup script on the Pi

The files are already in `~/diplom/display/` via your git pull.

```bash
cd ~/diplom/display
bash fix_and_build.sh
```

This will:
- Enable SPI in `/boot/firmware/config.txt`
- Add you to the `spi` and `gpio` groups
- Install `luma.lcd`, `lgpio`, `pillow`

**Reboot afterwards** if SPI was just enabled:
```bash
sudo reboot
```

---

## Step 2 — Install the display module

```bash
cd ~/diplom/display
bash install_display_module.sh
```

Verify that `/dev/spidev0.0` exists before continuing.

---

## Step 3 — Test

```bash
cd ~/diplom/weather_station
python3 -m display.weather_screen
```

You should see animated synthetic weather data on the TFT.  Ctrl-C to stop.

---

## Step 4 — Integrate with your weather station

In `~/diplom/weather_station/main.py`:

```python
from display import WeatherScreen

def get_sensor_data():
    return {
        'temperature': bme280.temperature,   # °C
        'humidity':    bme280.humidity,       # %
        'pressure':    bme280.pressure,       # hPa
        'latitude':    gps.latitude,
        'longitude':   gps.longitude,
        'altitude':    gps.altitude,
        'gps_fix':     gps.has_fix,
    }

screen = WeatherScreen(data_source=get_sensor_data, update_interval=5.0)
screen.run()
```

Or from a separate thread — call `screen.update(temperature=..., ...)` from
your sensor loop and `screen.run()` from the main thread.

---

## Step 5 — Install as boot service (optional)

```bash
cd ~/diplom/display
bash install_service.sh
```

---

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `/dev/spidev0.0` missing | Add `dtparam=spi=on` to `/boot/firmware/config.txt`, reboot |
| `Permission denied /dev/spidev*` | `sudo usermod -aG spi $USER`, reboot |
| `Cannot open GPIO` | `sudo usermod -aG gpio $USER`, reboot |
| Display shows nothing / white | Check DC=GPIO24 RST=GPIO25; try `gpio_rst=None` if no reset wire |
| `luma.lcd` import error | `pip install luma.lcd lgpio` (inside your venv) |
| Garbled colors | Add `bgr_mode=True` to `ili9341(...)` constructor call in tft_display.py |

---

## Wiring reference

| Display pin | Pi GPIO (BCM) | Pi physical pin |
|-------------|--------------|-----------------|
| VCC         | 3.3V         | 1               |
| GND         | GND          | 6               |
| CS          | GPIO 8 (CE0) | 24              |
| RESET       | GPIO 25      | 22              |
| DC/RS       | GPIO 24      | 18              |
| SDI (MOSI)  | GPIO 10      | 19              |
| SCK         | GPIO 11      | 23              |
| LED (BL)    | GPIO 18      | 12              |
