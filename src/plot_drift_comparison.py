import numpy as np
import matplotlib.pyplot as plt

from src.drift import MovingAverageDriftDetector
from src.drift_adaptive import ADWINDriftDetector, DDMDriftDetector


def simulate(detector, T=400):
    """
    Simulate SLA stream and record drift detection time
    """
    sla_values = []
    drift_points = []

    for t in range(T):
        # Stable phase → Drift phase
        metric = 10 if t < 200 else 30
        sla_values.append(metric)

        if detector.update(metric):
            drift_points.append(t)

    return sla_values, drift_points


def plot_comparison():
    detectors = {
        "Moving Average": MovingAverageDriftDetector(
            window_size=50, threshold=0.3, min_drift_windows=3
        ),
        "ADWIN": ADWINDriftDetector(delta=0.002),
        "DDM": DDMDriftDetector()
    }

    plt.figure(figsize=(12, 6))

    for name, detector in detectors.items():
        sla, drift_points = simulate(detector)

        plt.plot(sla, linewidth=1.8, label=f"{name} – SLA")

        if drift_points:
            plt.axvline(
                drift_points[0],
                linestyle="--",
                linewidth=2,
                label=f"{name} drift @ t={drift_points[0]}"
            )

    # True drift region
    plt.axvspan(200, len(sla), color="orange", alpha=0.12, label="True drift region")

    plt.xlabel("Time")
    plt.ylabel("SLA Metric")
    plt.title("Drift Detection Comparison: MA vs ADWIN vs DDM")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    plot_comparison()
