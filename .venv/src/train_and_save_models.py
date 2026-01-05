import os
import joblib
import numpy as np

# import your builders
from models import build_rf, build_fnn   # adjust filename if needed

# Create dummy regression data (simulating NWDAF metrics)
X = np.random.rand(1000, 10)
y = np.sum(X, axis=1) + np.random.normal(0, 0.1, size=1000)

# Build models
stable_model = build_rf()
bad_model = build_fnn()

# Train models
stable_model.fit(X, y)
bad_model.fit(X, y[::-1])      # intentionally wrong labels = "bad" model

# Ensure models folder exists
os.makedirs("models", exist_ok=True)

# Save using joblib
joblib.dump(stable_model, "models/model_v1.pkl")
joblib.dump(bad_model, "models/model_v2_bad.pkl")

print("✅ Models trained and saved successfully")
