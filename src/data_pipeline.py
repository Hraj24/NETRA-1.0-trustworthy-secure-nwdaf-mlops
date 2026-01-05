# import numpy as np
# import pandas as pd
# from pathlib import Path

# def generate_synthetic_traffic(n_samples=10000, seed=42):
#     rng = np.random.default_rng(seed)

#     time_of_day = rng.integers(0, 24, size=n_samples)
#     slice_type = rng.choice([0, 1, 2], size=n_samples)  # eMBB, URLLC, mMTC
#     jitter = rng.normal(loc=5 + 0.5 * slice_type, scale=2, size=n_samples)
#     packet_loss = rng.beta(2 + slice_type, 10, size=n_samples)
#     throughput = rng.normal(loc=100 + 10 * time_of_day, scale=20, size=n_samples)

#     # target: future load (next 5‑minute traffic)
#     future_load = throughput * (1 + 0.01 * time_of_day) + 3 * jitter + 100 * packet_loss
#     future_load += rng.normal(0, 10, size=n_samples)

#     df = pd.DataFrame({
#         "time_of_day": time_of_day,
#         "slice_type": slice_type,
#         "jitter": jitter,
#         "packet_loss": packet_loss,
#         "throughput": throughput,
#         "future_load": future_load,
#     })
#     return df

# if __name__ == "__main__":
#     df = generate_synthetic_traffic()
#     Path("data").mkdir(parents=True, exist_ok=True)
#     df.to_csv("data/traffic_synthetic.csv", index=False)
#     print("Saved data/traffic_synthetic.csv")





import numpy as np
import pandas as pd
from pathlib import Path

def generate_domain_traffic(
    n_samples: int,
    seed: int,
    domain: str,
) -> pd.DataFrame:
    """
    Generate synthetic slice traffic for a given domain:
    - urban: higher throughput, moderate jitter, higher load
    - rural: lower throughput, lower jitter, lower load
    - iot: low throughput, higher jitter variance, many small packets
    """
    rng = np.random.default_rng(seed)

    # Common features
    time_of_day = rng.integers(0, 24, size=n_samples)  # 0–23 hours
    # 0: eMBB, 1: URLLC, 2: mMTC
    slice_type = rng.choice([0, 1, 2], size=n_samples)

    if domain == "urban":
        # Higher throughput, moderate jitter
        base_throughput = 200 + 15 * time_of_day       # Mbps
        throughput = rng.normal(loc=base_throughput, scale=30, size=n_samples)
        jitter = rng.normal(loc=5 + 0.3 * slice_type, scale=1.5, size=n_samples)   # ms
        packet_loss = rng.beta(2, 18, size=n_samples)   # low loss
    elif domain == "rural":
        # Lower throughput, slightly more variable loss, low jitter
        base_throughput = 80 + 5 * time_of_day
        throughput = rng.normal(loc=base_throughput, scale=20, size=n_samples)
        jitter = rng.normal(loc=4 + 0.2 * slice_type, scale=1.0, size=n_samples)
        packet_loss = rng.beta(2.5, 12, size=n_samples)
    elif domain == "iot":
        # Many small IoT packets: low throughput, higher jitter variance
        base_throughput = 30 + 2 * time_of_day
        throughput = rng.normal(loc=base_throughput, scale=10, size=n_samples)
        jitter = rng.normal(loc=8 + 0.5 * slice_type, scale=3.0, size=n_samples)
        packet_loss = rng.beta(3, 10, size=n_samples)
    else:
        raise ValueError(f"Unknown domain: {domain}")

    # Clamp to reasonable ranges
    throughput = np.clip(throughput, 1, None)
    jitter = np.clip(jitter, 0.1, None)
    packet_loss = np.clip(packet_loss, 0.0, 0.3)

    # Target: future load (next 5‑minute traffic)
    # Domain‑dependent coefficients to make patterns different
    if domain == "urban":
        future_load = (
            throughput * (1 + 0.015 * time_of_day)
            + 4 * jitter
            + 120 * packet_loss
        )
    elif domain == "rural":
        future_load = (
            throughput * (1 + 0.01 * time_of_day)
            + 3 * jitter
            + 80 * packet_loss
        )
    else:  # iot
        future_load = (
            throughput * (1 + 0.008 * time_of_day)
            + 6 * jitter   # jitter more influential
            + 60 * packet_loss
        )

    # Add noise
    future_load += rng.normal(0, 8, size=n_samples)

    df = pd.DataFrame(
        {
            "time_of_day": time_of_day,
            "slice_type": slice_type,
            "jitter": jitter,
            "packet_loss": packet_loss,
            "throughput": throughput,
            "future_load": future_load,
            "domain": domain,
        }
    )
    return df


def generate_all_domains(n_samples_per_domain: int = 5000) -> None:
    df_urban = generate_domain_traffic(
        n_samples=n_samples_per_domain,
        seed=42,
        domain="urban",
    )
    df_rural = generate_domain_traffic(
        n_samples=n_samples_per_domain,
        seed=43,
        domain="rural",
    )
    df_iot = generate_domain_traffic(
        n_samples=n_samples_per_domain,
        seed=44,
        domain="iot",
    )
    Path("data").mkdir(parents=True, exist_ok=True)
    df_urban.to_csv("data/domain_a.csv", index=False)  # urban
    df_rural.to_csv("data/domain_b.csv", index=False)  # rural
    df_iot.to_csv("data/domain_c.csv", index=False)    # IoT


    print("Saved data/domain_a.csv (urban)")
    print("Saved data/domain_b.csv (rural)")
    print("Saved data/domain_c.csv (iot)")
    


if __name__ == "__main__":
    generate_all_domains()
