# How to transfer and run on your Pi

## Step 1 — Copy files to the Pi

From Windows PowerShell (replace `pi@raspberrypi.local` with your Pi's address):

```powershell
# Copy all scripts and the display module
scp fix_and_build.sh install_service.sh install_display_module.sh pi@raspberrypi.local:~/
scp -r display_module pi@raspberrypi.local:~/
```

Or use WinSCP / VS Code Remote SSH — just drop the files in your home folder.

---

## Step 2 — Fix and build fbcp-ili9341

SSH into the Pi, then:

```bash
chmod +x fix_and_build.sh install_service.sh install_display_module.sh
bash fix_and_build.sh
```

Expected output:
```
[1/4] Installing dependencies...
[2/4] Patching CMakeLists.txt for aarch64...
  ✓ Fix 1: /opt/vc paths made conditional
  ✓ Fix 2: 32-bit ARM flags wrapped in arch guard
  ✓ Fix 3: aarch64 ARMv8-A fallback added
[3/4] Running cmake + make...
...
[4/4] Done.
  Binary: ~/fbcp-ili9341/build/fbcp-ili9341
```

### If bcm_host is not found
```bash
sudo apt-get install libraspberrypi-dev
# if that fails on Bookworm:
sudo apt-get install libpigpio-dev   # alternative
```

### Test the binary (display should light up, mirroring the framebuffer)
```bash
sudo ~/fbcp-ili9341/build/fbcp-ili9341
```

---

## Step 3 — Set framebuffer resolution to 240×320

Edit `/boot/firmware/config.txt` (Bookworm) or `/boot/config.txt` (Bullseye):

```ini
[all]
hdmi_group=2
hdmi_mode=87
hdmi_cvt=240 320 60 1 0 0 0
hdmi_force_hotplug=1
```

Then reboot.  After reboot, `fbset` should show `240x320`.

---

## Step 4 — Install as a boot service

```bash
bash install_service.sh
```

---

## Step 5 — Install the display module into the weather station

```bash
bash install_display_module.sh
```

This copies `display/__init__.py`, `tft_display.py`, and `weather_screen.py`
into `~/diplom/weather_station/display/` and installs pygame.

---

## Step 6 — Integrate with your weather station

In `~/diplom/weather_station/main.py` (adapt to match your sensor API):

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
screen.run()  # blocks; Ctrl-C or SIGTERM exits cleanly
```

Or, if sensor reading runs in its own thread, push data manually:

```python
screen = WeatherScreen()
# from your sensor thread:
screen.update(temperature=22.5, humidity=60.1, pressure=1013.0)
# then in main thread:
screen.run()
```

---

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `bcm_host.h not found` | `sudo apt install libraspberrypi-dev` |
| `bcm_host` link error | `sudo apt install libraspberrypi0` |
| pygame: `no video device` | Make sure fbcp-ili9341 is running; also try `export SDL_VIDEODRIVER=fbcon SDL_FBDEV=/dev/fb0` |
| Display all white/black | Try `-DSPI_BUS_CLOCK_DIVISOR=8` or `=10` (increase if 6 is too fast for your wiring) |
| No GPIO access | Run with `sudo`, or add user to `gpio` group: `sudo usermod -aG gpio $USER` |
