import json
import joblib
import skl2onnx
from skl2onnx import to_onnx
import numpy as np
import pandas as pd

def package_rf():
    df = pd.read_csv("data/traffic_synthetic.csv")
    X = df.drop(columns=["future_load"])
    feature_names = list(X.columns)

    model = joblib.load("models/model_v1.pkl")
    onnx_model = to_onnx(model, X.iloc[:1].astype(np.float32).values)

    onnx_path = "models/nwdaf_rf.onnx"
    with open(onnx_path, "wb") as f:
        f.write(onnx_model.SerializeToString())

    metadata = {
        "model_name": "nwdaf_rf_traffic_predictor",
        "version": "1.0.0",
        "input_schema": feature_names,
        "output": "future_load",
        "validity": "synthetic_traffic_v1",
        "slice_constraints": ["embb", "urllc", "mmtc"],
    }
    with open("models/nwdaf_rf_meta.json", "w") as f:
        json.dump(metadata, f, indent=2)

if __name__ == "__main__":
    package_rf()
