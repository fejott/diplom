#!/bin/bash
# install_display_module.sh
# Verifies that the TFT display module is ready to use.
# The files are already in place via git — this script just checks
# that dependencies are installed and SPI is enabled.
# Run AFTER fix_and_build.sh.
set -e

WS_DIR="$HOME/diplom/weather_station"
MODULE_DIR="$WS_DIR/tft_display/display_module"

echo "=== TFT display module check ==="
echo ""

# ── Check module files ────────────────────────────────────────────────────────
echo "[1/3] Checking module files in $MODULE_DIR..."
for F in __init__.py tft_display.py wifi_screen.py; do
    if [ -f "$MODULE_DIR/$F" ]; then
        echo "  OK  $F"
    else
        echo "  MISSING  $F  (run: cd ~/diplom && git pull)"
    fi
done

# ── Check SPI device ──────────────────────────────────────────────────────────
echo ""
echo "[2/3] Checking SPI device..."
if ls /dev/spidev* > /dev/null 2>&1; then
    echo "  SPI OK: $(ls /dev/spidev*)"
else
    echo "  WARNING: /dev/spidev* not found — enable SPI in raspi-config and reboot"
fi

# ── Check Python packages ─────────────────────────────────────────────────────
echo ""
echo "[3/3] Checking Python packages..."
for PKG in luma.lcd PIL gpiozero; do
    python3 -c "import ${PKG//./_}" 2>/dev/null \
        && echo "  OK  $PKG" \
        || echo "  MISSING  $PKG  (run: bash fix_and_build.sh)"
done

echo ""
echo "=== Done. To run the station: ==="
echo "  cd $WS_DIR"
echo "  python3 main.py            # with TFT display"
echo "  python3 main.py --no-tft   # terminal only"
