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

# Forecasting — storage
DB_PATH: str = "weather_history.db"
MODEL_PATH: str = "forecasting/lstm_model.tflite"
SCALER_PATH: str = "forecasting/scaler_params.json"
METRICS_PATH: str = "forecasting/metrics.json"

# Forecasting — LSTM
SEQUENCE_LENGTH: int = 72           # number of readings fed to the model
FORECAST_STEPS: list = [36, 72, 216]  # reading-offsets for +1h/+2h/+3h predictions
FORECAST_MIN_READINGS: int = 500    # min DB rows before LSTM is considered ready
LSTM_UNITS: int = 32
LSTM_LAYERS: int = 2
MAX_RAM_MB: int = 400

# Forecasting — retraining schedule
LSTM_RETRAIN_INTERVAL: int = 3600   # seconds between retrains
RETRAIN_THRESHOLD: int = 100        # new readings needed to trigger a retrain
