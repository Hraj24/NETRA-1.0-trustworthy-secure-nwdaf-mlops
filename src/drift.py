import numpy as np
from collections import deque

class MovingAverageDriftDetector:
    """
    Detects concept drift using sliding window mean comparison.
    Drift is confirmed only after consecutive violations.
    """

    def __init__(self, window_size=50, threshold=0.3, min_drift_windows=3):
        self.window_size = window_size
        self.threshold = threshold
        self.min_drift_windows = min_drift_windows

        self.window = deque(maxlen=window_size)
        self.reference_mean = None
        self.drift_counter = 0

    def update(self, value: float) -> bool:
        self.window.append(value)

        # Not enough data yet
        if len(self.window) < self.window_size:
            return False

        current_mean = np.mean(self.window)

        # Initialize baseline
        if self.reference_mean is None:
            self.reference_mean = current_mean
            return False

        relative_change = abs(current_mean - self.reference_mean) / (
            self.reference_mean + 1e-6
        )

        if relative_change > self.threshold:
            self.drift_counter += 1
        else:
            # Adapt baseline if system stabilizes
            self.drift_counter = 0
            self.reference_mean = current_mean

        return self.drift_counter >= self.min_drift_windows
