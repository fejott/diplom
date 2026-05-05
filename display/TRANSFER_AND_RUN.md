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

```bash
cd ~/diplom
git pull
cd display
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

Verify that `/dev/spidev0.0` exists before continuing:
```bash
ls /dev/spidev*
```

---

## Step 3 — Test with synthetic data

```bash
cd ~/diplom/weather_station
python3 -m display.weather_screen
```

You should see animated synthetic weather data on the TFT.  Ctrl-C to stop.

---

## Step 4 — Run the full weather station

`main.py` is already integrated with the TFT display.  It will automatically
use the display if `luma.lcd` is available:

```bash
cd ~/diplom/weather_station
python3 main.py
```

Both the terminal and the TFT display will update every 5 seconds.
To run without the TFT (terminal only):

```bash
python3 main.py --no-tft
```

---

## Step 5 — Install as a boot service

```bash
cd ~/diplom/display
bash install_service.sh
```

This creates `/etc/systemd/system/weather-station.service` which runs
`main.py` on boot (terminal + TFT display together).

```bash
sudo systemctl status weather-station
journalctl -u weather-station -f
```

---

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `/dev/spidev0.0` missing | Add `dtparam=spi=on` to `/boot/firmware/config.txt`, reboot |
| `Permission denied /dev/spidev*` | `sudo usermod -aG spi $USER`, reboot |
| `Cannot open GPIO` | `sudo usermod -aG gpio $USER`, reboot |
| Display shows nothing / white | Check DC=GPIO24 RST=GPIO25; try `gpio_rst=None` if no reset wire |
| `luma.lcd` import error | `pip install luma.lcd lgpio pillow` (inside your venv) |
| Garbled colors | Add `bgr_mode=True` to `ili9341(...)` in `display/tft_display.py` |
| TFT works, terminal blank | Run `python3 main.py --no-tft` to confirm sensors work without display |

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
