"""
Central configuration for the Weather Station application.
All hardware constants and tunable parameters live here.
"""

# I2C
I2C_BUS: int = 1
BME280_ADDRESS: int = 0x76  # change to 0x77 if 0x76 fails

# GPS UART
GPS_PORT: str = "/dev/ttyS0"  # use /dev/ttyAMA0 for older Pi models
GPS_BAUDRATE: int = 9600
GPS_TIMEOUT: int = 1          # seconds per read() call
GPS_FIX_TIMEOUT: int = 10     # seconds to wait for a valid fix before giving up

# Application timing
UPDATE_INTERVAL: int = 5      # seconds between screen refresh

# Logging
LOG_FILE: str = "weather_station.log"
LOG_MAX_BYTES: int = 5 * 1024 * 1024   # 5 MB per log file
LOG_BACKUP_COUNT: int = 3
