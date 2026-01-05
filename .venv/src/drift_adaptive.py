from river.drift import ADWIN
from river.drift.binary import DDM


"""
    Adaptive Windowing drift detector (statistical)
"""
class ADWINDriftDetector:
    def __init__(self, delta=0.002):
        self.detector = ADWIN(delta=delta)

    def update(self, value: float) -> bool:
        self.detector.update(value)
        return self.detector.drift_detected

"""
    Drift Detection Method (error-rate based)
"""
class DDMDriftDetector:
    def __init__(self):
        self.detector = DDM()

    def update(self, value: float) -> bool:
        error = 1 if value > 20 else 0
        self.detector.update(error)
        return self.detector.drift_detected
