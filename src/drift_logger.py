import csv
import os
from datetime import datetime


class DriftLogger:
    def __init__(self, log_file="drift_events.csv"):
        self.log_file = log_file
        self._initialize_file()

    def _initialize_file(self):
        if not os.path.exists(self.log_file):
            with open(self.log_file, mode="w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "timestamp",
                    "detector",
                    "sla_metric",
                    "model_version",
                    "action"
                ])

    def log(self, detector_name, sla_metric, model_version, action):
        with open(self.log_file, mode="a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.now().isoformat(),
                detector_name,
                round(float(sla_metric), 3),
                model_version,
                action
            ])
