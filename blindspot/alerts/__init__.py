"""Alert logic (WP 3.x): walking corridor, turning a depth map into alerts, naive baseline.

An alert function has the signature ``fn(depth_m, intrinsics, cfg) -> list[Alert]``.
"""

from blindspot.alerts.corridor import CorridorConfig, corridor_alerts
from blindspot.alerts.naive import naive_alerts

ALERT_LOGICS = {"corridor": corridor_alerts, "naive": naive_alerts}

__all__ = ["ALERT_LOGICS", "CorridorConfig", "corridor_alerts", "naive_alerts"]
