"""Optional feature creation from confirmed canonical columns."""

import pandas as pd

from .utils import get_logger

LOG = get_logger(__name__)


def add_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Derive time features when source fields exist; retain optional source features."""
    result = frame.copy()
    if "event_date" in result:
        dates = pd.to_datetime(result["event_date"], errors="coerce")
        result["month"] = dates.dt.month.astype("Int64")
        result["weekend"] = (dates.dt.dayofweek >= 5).where(dates.notna()).astype("boolean")
    if "event_time" in result:
        times = pd.to_datetime(result["event_time"].astype("string"), format="%H:%M", errors="coerce")
        result["hour"] = times.dt.hour.astype("Int64")
        hour = result["hour"]
        result["night"] = ((hour < 6) | (hour >= 20)).where(hour.notna()).astype("boolean")
        result["time_band"] = pd.cut(hour, bins=[-1, 5, 11, 17, 23], labels=["madrugada", "manana", "tarde", "noche"]).astype("string")
    expected = ("collision_type", "vulnerable_user", "motorcycle", "alcohol", "municipality", "state", "vehicle_count")
    unavailable = [name for name in expected if name not in result]
    if unavailable:
        LOG.info("Optional features unavailable: %s", ", ".join(unavailable))
    return result
