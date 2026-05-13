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
        pressure_trend:  hPa/hour pressure change over the forecast horizon.
        temp_in_1h:      Predicted temperature +1 h (°C), or None.
        temp_in_2h:      Predicted temperature +2 h (°C), or None.
        temp_in_3h:      Predicted temperature +3 h (°C), or None.
        precip_prob_1h:  Precipitation probability +1 h (0.0–1.0), or None.
        precip_prob_2h:  Precipitation probability +2 h (0.0–1.0), or None.
        precip_prob_3h:  Precipitation probability +3 h (0.0–1.0), or None.
        pressure_in_1h:  Predicted pressure +1 h (hPa), or None.
        pressure_in_2h:  Predicted pressure +2 h (hPa), or None.
        pressure_in_3h:  Predicted pressure +3 h (hPa), or None.
        valid_until:     Datetime until which this forecast is considered valid.
        model_version:   String tag identifying the model ("lstm_v1", "rule_v1").
    """

    method: Literal["lstm", "lstm_corrected", "rule-based", "insufficient_data", "online_api"]
    forecast_text: str
    confidence: float
    pressure_trend: float
    temp_in_1h:     Optional[float]
    temp_in_2h:     Optional[float]
    temp_in_3h:     Optional[float]
    precip_prob_1h: Optional[float]
    precip_prob_2h: Optional[float]
    precip_prob_3h: Optional[float]
    pressure_in_1h: Optional[float]
    pressure_in_2h: Optional[float]
    pressure_in_3h: Optional[float]
    valid_until: datetime
    model_version: str
    correction_applied:        bool           = False
    correction_delta_temp_1h:  Optional[float] = None
    correction_delta_pres_1h:  Optional[float] = None
