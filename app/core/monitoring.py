import time
from app.core.logging import get_logger

logger = get_logger(__name__)


class PerformanceTracker:
    """
    Track performance metrics across the application
    """

    def __init__(self):
        self.metrics = {}

    def track(self, operation_name: str):
        """Context manager to track an operation"""
        return OperationTimer(self, operation_name)

    def record(self, operation_name: str, duration: float, success: bool = True):
        """Record a metric"""
        if operation_name not in self.metrics:
            self.metrics[operation_name] = {
                "count": 0,
                "total_time": 0,
                "successes": 0,
                "failures": 0,
                "avg_time": 0,
                "min_time": float("inf"),
                "max_time": 0,
            }

        m = self.metrics[operation_name]
        m["count"] += 1
        m["total_time"] += duration
        m["avg_time"] = m["total_time"] / m["count"]
        m["min_time"] = min(m["min_time"], duration)
        m["max_time"] = max(m["max_time"], duration)

        if success:
            m["successes"] += 1
        else:
            m["failures"] += 1

    def get_metrics(self) -> dict:
        """Get all recorded metrics"""
        return self.metrics

    def log_summary(self):
        """Log a summary of all metrics"""
        logger.info("Performance Summary:")
        for operation, stats in self.metrics.items():
            logger.info(
                f"  {operation}: "
                f"{stats['count']} calls, "
                f"avg {stats['avg_time']:.2f}s, "
                f"min {stats['min_time']:.2f}s, "
                f"max {stats['max_time']:.2f}s, "
                f"success rate {stats['successes']}/{stats['count']}"
            )


class OperationTimer:
    """Context manager for timing operations"""

    def __init__(self, tracker: PerformanceTracker, operation_name: str):
        self.tracker = tracker
        self.operation_name = operation_name
        self.start_time = None
        self.success = True

    def __enter__(self):
        self.start_time = time.time()
        logger.debug(f"Starting {self.operation_name}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - self.start_time
        self.success = exc_type is None

        self.tracker.record(self.operation_name, duration, self.success)

        if self.success:
            logger.debug(f"{self.operation_name} took {duration:.2f}s")
        else:
            logger.error(f"{self.operation_name} failed after {duration:.2f}s")

        return False


# Global tracker instance
performance_tracker = PerformanceTracker()
