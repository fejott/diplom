#!/bin/bash
# install_service.sh
# Installs the weather station as a systemd service that starts on boot.
# Run AFTER fix_and_build.sh and install_display_module.sh.
set -e

WS_DIR="$HOME/diplom/weather_station"
SERVICE_FILE=/etc/systemd/system/weather-station.service

# Find the right Python (prefer venv)
PYTHON="python3"
for CANDIDATE in "$WS_DIR/venv/bin/python3" "$HOME/diplom/venv/bin/python3" "$HOME/venv/bin/python3"; do
    [ -f "$CANDIDATE" ] && PYTHON="$CANDIDATE" && break
done

echo "=== Installing weather-station systemd service ==="
echo "  Working dir : $WS_DIR"
echo "  Python      : $PYTHON"
echo ""

if [ ! -d "$WS_DIR" ]; then
    echo "ERROR: $WS_DIR not found."
    exit 1
fi

sudo tee "$SERVICE_FILE" > /dev/null << EOF
[Unit]
Description=Weather Station (BME280 + GPS + TFT display)
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$WS_DIR
ExecStart=$PYTHON main.py
Restart=always
RestartSec=5
Environment=PYTHONUNBUFFERED=1

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable weather-station
sudo systemctl start weather-station

echo "Service installed and started."
echo "  Status : sudo systemctl status weather-station"
echo "  Logs   : journalctl -u weather-station -f"
echo "  Stop   : sudo systemctl stop weather-station"
echo "  Restart: sudo systemctl restart weather-station"
