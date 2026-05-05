#!/bin/bash
# install_display_module.sh
# Copies the display module into ~/diplom/weather_station/display/
# Run AFTER fix_and_build.sh (which installs luma.lcd).
set -e

WS_DIR="$HOME/diplom/weather_station"
DISPLAY_DIR="$WS_DIR/display"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "=== Installing display module into $DISPLAY_DIR ==="

if [ ! -d "$WS_DIR" ]; then
    echo "ERROR: $WS_DIR not found."
    exit 1
fi

mkdir -p "$DISPLAY_DIR"
cp "$SCRIPT_DIR/display_module/__init__.py"       "$DISPLAY_DIR/"
cp "$SCRIPT_DIR/display_module/tft_display.py"    "$DISPLAY_DIR/"
cp "$SCRIPT_DIR/display_module/weather_screen.py" "$DISPLAY_DIR/"

echo "Files installed to $DISPLAY_DIR"
echo ""
echo "Verify SPI device is present:"
ls /dev/spidev* 2>/dev/null && echo "  SPI OK" || echo "  WARNING: /dev/spidev* not found — enable SPI and reboot"
echo ""
echo "Test the display:"
echo "  cd $WS_DIR"
echo "  python3 -m display.weather_screen"
