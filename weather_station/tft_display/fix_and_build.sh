#!/bin/bash
# fix_and_build.sh  (UPDATED)
#
# fbcp-ili9341 cannot run on 64-bit Raspberry Pi OS Trixie:
#   • It requires the VideoCore DispmanX API (vc_dispmanx_*), which was
#     removed from 64-bit / Trixie.  The package libraspberrypi-dev no
#     longer exists on Trixie and DispmanX headers are not available.
#
# This script sets up the luma.lcd Python driver instead.
# luma.lcd speaks SPI directly to the ILI9341 — no C++ binary needed.
#
# Run:  bash fix_and_build.sh
set -e

echo "=== ILI9341 display setup (luma.lcd, aarch64 Trixie) ==="
echo ""
echo "NOTE: fbcp-ili9341 is incompatible with 64-bit Raspberry Pi OS Trixie."
echo "      Using luma.lcd Python driver instead (direct SPI, no DispmanX)."
echo ""

# ── 1. Enable SPI ────────────────────────────────────────────────────────────
echo "[1/4] Enabling SPI interface..."

CONFIG="/boot/firmware/config.txt"
[ -f "$CONFIG" ] || CONFIG="/boot/config.txt"

if grep -q "^dtparam=spi=on" "$CONFIG"; then
    echo "  SPI already enabled in $CONFIG"
else
    echo "  Adding dtparam=spi=on to $CONFIG"
    echo "" | sudo tee -a "$CONFIG" > /dev/null
    echo "dtparam=spi=on" | sudo tee -a "$CONFIG" > /dev/null
    echo "  REBOOT REQUIRED after this script finishes!"
fi

# ── 2. Permissions ───────────────────────────────────────────────────────────
echo ""
echo "[2/4] Adding $USER to spi and gpio groups..."
sudo usermod -aG spi,gpio "$USER"
echo "  Done (takes effect after next login/reboot)"

# ── 3. System packages ───────────────────────────────────────────────────────
echo ""
echo "[3/4] Installing system packages..."
sudo apt-get install -y python3-lgpio python3-gpiozero python3-pil \
                        fonts-dejavu-core 2>&1 | grep -E "(installed|upgraded|already)" || true

# ── 4. Python packages ───────────────────────────────────────────────────────
echo ""
echo "[4/4] Installing Python packages (luma.lcd, pillow, lgpio)..."

pip install luma.lcd lgpio pillow 2>&1 | tail -5

echo ""
echo "=== Setup complete ==="
echo ""
echo "Next steps:"
echo "  1. Reboot if SPI was just enabled: sudo reboot"
echo "  2. Run the station:"
echo "       cd ~/diplom/weather_station"
echo "       python3 main.py            # with TFT display"
echo "       python3 main.py --no-tft   # terminal only"
