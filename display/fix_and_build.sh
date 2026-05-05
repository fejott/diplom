#!/bin/bash
# fix_and_build.sh
# Run on Raspberry Pi 4 (aarch64) to patch and build fbcp-ili9341.
# Usage: bash fix_and_build.sh
set -e

FBCP_DIR="$HOME/fbcp-ili9341"
BUILD_DIR="$FBCP_DIR/build"

echo "=== fbcp-ili9341 aarch64 fix + build ==="
echo "Working in: $FBCP_DIR"

# ── 1. Dependencies ─────────────────────────────────────────────────────────
echo ""
echo "[1/4] Installing dependencies..."
sudo apt-get update -q
sudo apt-get install -y cmake libraspberrypi-dev

# ── 2. Patch CMakeLists.txt ──────────────────────────────────────────────────
echo ""
echo "[2/4] Patching CMakeLists.txt for aarch64..."

cd "$FBCP_DIR"
cp -n CMakeLists.txt CMakeLists.txt.bak && echo "  Backup created: CMakeLists.txt.bak" || echo "  Backup already exists, skipping."

python3 - << 'PYEOF'
import sys

with open('CMakeLists.txt', 'r') as f:
    src = f.read()

changes = 0

# ── Fix 1: /opt/vc include/link paths → conditional ─────────────────────────
OLD = 'include_directories(/opt/vc/include)\nlink_directories(/opt/vc/lib)'
NEW = (
    'if(EXISTS "/opt/vc/include")\n'
    '  include_directories(/opt/vc/include)\n'
    '  link_directories(/opt/vc/lib)\n'
    'else()\n'
    '  # 64-bit Pi OS: VideoCore headers are in system paths\n'
    '  include_directories(/usr/include)\n'
    '  link_directories(/usr/lib/aarch64-linux-gnu)\n'
    'endif()'
)
if OLD in src:
    src = src.replace(OLD, NEW, 1)
    changes += 1
    print('  ✓ Fix 1: /opt/vc paths made conditional')
else:
    print('  ~ Fix 1: /opt/vc paths already patched or not found — skipping')

# ── Fix 2: 32-bit ARM compiler flags → conditional on arch ──────────────────
OLD2 = ('set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}'
        ' -marm -mabi=aapcs-linux -mhard-float -mfloat-abi=hard'
        ' -mlittle-endian -mtls-dialect=gnu2 -funsafe-math-optimizations")')
NEW2 = (
    'if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")\n'
    '  # 64-bit ARM: the 32-bit-only flags (-marm, -mabi=aapcs-linux,\n'
    '  # -mhard-float, -mfloat-abi=hard, -mtls-dialect=gnu2) are invalid.\n'
    '  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}'
    ' -mlittle-endian -funsafe-math-optimizations")\n'
    'else()\n'
    '  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}'
    ' -marm -mabi=aapcs-linux -mhard-float -mfloat-abi=hard'
    ' -mlittle-endian -mtls-dialect=gnu2 -funsafe-math-optimizations")\n'
    'endif()'
)
if OLD2 in src:
    src = src.replace(OLD2, NEW2, 1)
    changes += 1
    print('  ✓ Fix 2: 32-bit ARM flags wrapped in arch guard')
else:
    print('  ~ Fix 2: flags already patched or not found — skipping')

# ── Fix 3: aarch64 fallback → default ARMV8A when board revision unknown ────
ANCHOR = 'option(SINGLE_CORE_BOARD'
FALLBACK = (
    '# On aarch64 with an unrecognised board revision, default to ARMv8-A.\n'
    'if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64"'
    ' AND NOT DEFAULT_TO_ARMV6Z AND NOT DEFAULT_TO_ARMV7A AND NOT DEFAULT_TO_ARMV8A)\n'
    '  message(STATUS "aarch64 detected: defaulting to ARMv8-A target.")\n'
    '  set(DEFAULT_TO_ARMV8A ON)\n'
    'endif()\n\n'
)
if ANCHOR in src and FALLBACK not in src:
    src = src.replace(ANCHOR, FALLBACK + ANCHOR, 1)
    changes += 1
    print('  ✓ Fix 3: aarch64 ARMv8-A fallback added')
else:
    print('  ~ Fix 3: fallback already present — skipping')

if changes == 0:
    print('  Nothing to patch — CMakeLists.txt may already be fixed.')
    sys.exit(0)

with open('CMakeLists.txt', 'w') as f:
    f.write(src)

print(f'  Applied {changes} patch(es) successfully.')
PYEOF

# ── 3. Build ─────────────────────────────────────────────────────────────────
echo ""
echo "[3/4] Running cmake + make..."

mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

cmake .. \
  -DILI9341=ON \
  -DSPI_BUS_CLOCK_DIVISOR=6 \
  -DGPIO_TFT_DATA_CONTROL=24 \
  -DGPIO_TFT_RESET_PIN=25 \
  -DBACKLIGHT_CONTROL=ON \
  -DGPIO_TFT_BACKLIGHT=18

make -j"$(nproc)"

echo ""
echo "[4/4] Done."
echo "  Binary: $BUILD_DIR/fbcp-ili9341"
echo ""
echo "Quick test (Ctrl-C to stop):"
echo "  sudo $BUILD_DIR/fbcp-ili9341"
echo ""
echo "Install as service:"
echo "  bash install_service.sh"
