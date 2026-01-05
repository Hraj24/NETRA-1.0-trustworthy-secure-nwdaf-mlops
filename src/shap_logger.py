import csv
import os
from datetime import datetime

class ShapLogger:
    def __init__(self, path="logs/shap_log.csv"):
        self.path = path
        os.makedirs("logs", exist_ok=True)

        if not os.path.exists(self.path):
            with open(self.path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "timestamp",
                    "time_of_day",
                    "slice_type",
                    "jitter",
                    "packet_loss",
                    "throughput",
                    "rollback_active"
                ])

    def log(self, shap_values: dict, rollback_active: bool):
        with open(self.path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.utcnow().isoformat(),
                shap_values["time_of_day"],
                shap_values["slice_type"],
                shap_values["jitter"],
                shap_values["packet_loss"],
                shap_values["throughput"],
                rollback_active
            ])
