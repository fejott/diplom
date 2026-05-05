#!/bin/bash
# install_service.sh
# Installs the weather station display as a systemd service.
# Run AFTER fix_and_build.sh and install_display_module.sh.
set -e

WS_DIR="$HOME/diplom/weather_station"
VENV="$WS_DIR/../venv"
PYTHON="python3"
[ -f "$VENV/bin/python3" ] && PYTHON="$VENV/bin/python3"

SERVICE_FILE=/etc/systemd/system/weather-display.service

echo "=== Installing weather display service ==="

sudo tee "$SERVICE_FILE" > /dev/null << EOF
[Unit]
Description=Weather Station TFT Display
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$WS_DIR
ExecStart=$PYTHON -m display.weather_screen
Restart=always
RestartSec=5
Environment=PYTHONUNBUFFERED=1

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable weather-display
sudo systemctl start weather-display

echo "Service installed and started."
echo "  Status : sudo systemctl status weather-display"
echo "  Logs   : journalctl -u weather-display -f"
