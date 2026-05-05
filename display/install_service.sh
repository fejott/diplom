#!/bin/bash
# install_service.sh
# Installs fbcp-ili9341 as a systemd service that starts at boot.
# Must run AFTER fix_and_build.sh succeeds.
set -e

BINARY="$HOME/fbcp-ili9341/build/fbcp-ili9341"

if [ ! -f "$BINARY" ]; then
    echo "ERROR: $BINARY not found. Run fix_and_build.sh first."
    exit 1
fi

sudo cp "$BINARY" /usr/local/bin/fbcp-ili9341

sudo tee /etc/systemd/system/fbcp-ili9341.service > /dev/null << EOF
[Unit]
Description=fbcp-ili9341 SPI display framebuffer copy
After=network.target

[Service]
Type=simple
ExecStart=/usr/local/bin/fbcp-ili9341
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable fbcp-ili9341
sudo systemctl start fbcp-ili9341

echo "Service installed and started."
echo "  Status : sudo systemctl status fbcp-ili9341"
echo "  Logs   : journalctl -u fbcp-ili9341 -f"

# ── Framebuffer resolution hint ──────────────────────────────────────────────
echo ""
echo "IMPORTANT: set HDMI to 240x320 in /boot/config.txt (or /boot/firmware/config.txt)"
echo "for a 1:1 pixel mapping.  Add these lines under [all]:"
echo ""
echo "  hdmi_group=2"
echo "  hdmi_mode=87"
echo "  hdmi_cvt=240 320 60 1 0 0 0"
echo "  hdmi_force_hotplug=1"
echo ""
echo "Then reboot."
