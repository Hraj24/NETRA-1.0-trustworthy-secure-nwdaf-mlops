import numpy as np
import matplotlib.pyplot as plt

from src.drift import MovingAverageDriftDetector
from src.drift_adaptive import ADWINDriftDetector, DDMDriftDetector


def simulate(T=500):
    ma = MovingAverageDriftDetector(
        window_size=50, threshold=0.3, min_drift_windows=3
    )
    adwin = ADWINDriftDetector(delta=0.002)
    ddm = DDMDriftDetector()

    times, sla = [], []
    ma_flags, adwin_flags, ddm_flags = [], [], []

    for t in range(T):
        metric = 10 if t < 200 else 30  # controlled drift
        times.append(t)
        sla.append(metric)

        ma_flags.append(ma.update(metric))
        adwin_flags.append(adwin.update(metric))
        ddm_flags.append(ddm.update(metric))

    return (
        np.array(times),
        np.array(sla),
        ma_flags,
        adwin_flags,
        ddm_flags,
    )


def plot_hybrid_drift():
    times, sla, ma_flags, adwin_flags, ddm_flags = simulate()

    # Detection times
    def first_true(flags):
        idx = [i for i, f in enumerate(flags) if f]
        return idx[0] if idx else None

    t_ma = first_true(ma_flags)
    t_adwin = first_true(adwin_flags)
    t_ddm = first_true(ddm_flags)

    plt.figure(figsize=(12, 6))

    # SLA
    plt.plot(times, sla, linewidth=2, label="SLA metric")

    # True drift region
    plt.axvspan(200, times[-1], color="orange", alpha=0.12, label="True drift region")

    # Early warnings (ADWIN / DDM)
    if t_adwin is not None:
        plt.axvline(t_adwin, color="purple", linestyle=":", linewidth=2,
                    label=f"ADWIN early warning (t={t_adwin})")

    if t_ddm is not None:
        plt.axvline(t_ddm, color="green", linestyle=":", linewidth=2,
                    label=f"DDM early warning (t={t_ddm})")

    # Primary rollback trigger (Moving Average)
    if t_ma is not None:
        plt.axvline(t_ma, color="red", linestyle="--", linewidth=2.5,
                    label=f"MA rollback trigger (t={t_ma})")

        plt.axvspan(t_ma, times[-1], color="red", alpha=0.08,
                    label="Rollback active")

        plt.annotate(
            "Rollback triggered",
            xy=(t_ma, sla[t_ma]),
            xytext=(t_ma + 25, sla[t_ma] + 6),
            arrowprops=dict(arrowstyle="->", color="red")
        )

    plt.xlabel("Time")
    plt.ylabel("SLA metric value")
    plt.title("Hybrid Drift Detection: MA Rollback with ADWIN/DDM Early Warnings")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    plot_hybrid_drift()






