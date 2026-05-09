"""ForecastResult dataclass — returned by both LSTM and rule-based forecasters."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Literal, Optional


@dataclass
class ForecastResult:
    """Unified forecast output for any forecasting method.

    Attributes:
        method:          Which engine produced this result.
        forecast_text:   Human-readable forecast in Russian.
        confidence:      0.0–1.0 estimate of prediction reliability.
        pressure_trend:  hPa change over the forecast horizon.
        temp_in_1h:      Predicted temperature after step 1 (°C), or None.
        temp_in_2h:      Predicted temperature after step 2 (°C), or None.
        temp_in_3h:      Predicted temperature after step 3 (°C), or None.
        pressure_in_1h:  Predicted pressure after step 1 (hPa), or None.
        pressure_in_2h:  Predicted pressure after step 2 (hPa), or None.
        pressure_in_3h:  Predicted pressure after step 3 (hPa), or None.
        valid_until:     Datetime until which this forecast is considered valid.
        model_version:   String tag identifying the model ("lstm_v1", "rule_v1").
    """

    method: Literal["lstm", "rule-based", "insufficient_data"]
    forecast_text: str
    confidence: float
    pressure_trend: float
    temp_in_1h: Optional[float]
    temp_in_2h: Optional[float]
    temp_in_3h: Optional[float]
    pressure_in_1h: Optional[float]
    pressure_in_2h: Optional[float]
    pressure_in_3h: Optional[float]
    valid_until: datetime
    model_version: str
