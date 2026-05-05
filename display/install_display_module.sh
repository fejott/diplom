#!/bin/bash
# install_display_module.sh
# Copies the display module into ~/diplom/weather_station/display/
# and installs pygame.
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
cp "$SCRIPT_DIR/display_module/__init__.py"      "$DISPLAY_DIR/"
cp "$SCRIPT_DIR/display_module/tft_display.py"   "$DISPLAY_DIR/"
cp "$SCRIPT_DIR/display_module/weather_screen.py" "$DISPLAY_DIR/"

echo "Files copied to $DISPLAY_DIR"

echo ""
echo "Installing pygame..."
pip3 install --break-system-packages pygame 2>/dev/null \
    || pip3 install pygame

echo ""
echo "Done. To test the display (fbcp-ili9341 must be running):"
echo "  cd $WS_DIR"
echo "  python3 -m display.weather_screen"
echo ""
echo "To integrate into your main.py, add:"
echo "  from display import WeatherScreen"
echo "  screen = WeatherScreen(data_source=get_sensor_data, update_interval=5.0)"
echo "  screen.run()"
