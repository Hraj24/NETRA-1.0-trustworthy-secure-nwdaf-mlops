# import pandas as pd
# import shap
# from models import build_rf

# def train_and_explain():
#     df = pd.read_csv("data/traffic_synthetic.csv")
#     X = df.drop(columns=["future_load"])
#     y = df["future_load"]

#     model = build_rf()
#     model.fit(X, y)

#     explainer = shap.TreeExplainer(model)
#     shap_values = explainer.shap_values(X.sample(100, random_state=42))

#     shap.summary_plot(shap_values, X.sample(100, random_state=42), show=False)
#     # Save figure instead of showing
#     import matplotlib.pyplot as plt
#     plt.tight_layout()
#     plt.savefig("reports/shap_summary.png")

# if __name__ == "__main__":
#     train_and_explain()







import pandas as pd
import numpy as np
import shap
import joblib
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDRegressor

# ----------------------------------------
# Load FL Global Model
# ----------------------------------------
model_data = joblib.load("models/fl_global_model.pkl")

model = SGDRegressor()
model.coef_ = np.array(model_data["coef"])
model.intercept_ = np.array(model_data["intercept"])

# ----------------------------------------
# Load public / synthetic data
# ----------------------------------------
df = pd.read_csv("data/traffic_synthetic.csv")

SLICE_MAP = {"eMBB": 0, "URLLC": 1, "mMTC": 2}
if "slice_type" in df.columns:
    df["slice_type"] = df["slice_type"].map(SLICE_MAP)

# ----------------------------------------
# Prepare features
# ----------------------------------------
X = df.drop(columns=["future_load"]).values

# Add placeholder for model compatibility
X_full = np.hstack([X, np.zeros((X.shape[0], 1))])

# ❗ REMOVE placeholder for SHAP (zero variance)
X_shap = X_full[:, :-1]

feature_names = [
    "time_of_day",
    "slice_type",
    "jitter",
    "packet_loss",
    "throughput",
]

# ----------------------------------------
# SHAP Linear Explainer (NEW API)
# ----------------------------------------
masker = shap.maskers.Independent(X_full)

explainer = shap.LinearExplainer(
    model,
    masker
)

shap_values_full = explainer.shap_values(X_full[:100])

# Drop placeholder feature
shap_values = shap_values_full[:, :-1]
X_plot = X_full[:100, :-1]

X_plot = X_shap[:100]

# ----------------------------------------
# Global Explanation
# ----------------------------------------
shap.summary_plot(
    shap_values,
    X_plot,
    feature_names=feature_names,
    show=False
)

plt.tight_layout()
plt.savefig("reports/shap_summary_fl.png", dpi=300)
plt.close()

# # ----------------------------------------
# # Local Explanation (single instance)
# # ----------------------------------------
# shap.force_plot(
#     explainer.expected_value,
#     shap_values[0],
#     X_plot[0],
#     feature_names=feature_names,
#     matplotlib=True
# )

# plt.savefig("reports/shap_force_fl.png", dpi=300)
# plt.close()

# print("✅ SHAP explanations generated successfully (no NaNs)")


# ----------------------------------------
# Local explanation (BAR plot – stable)
# ----------------------------------------
shap.plots.bar(
    shap.Explanation(
        values=shap_values[0],
        base_values=explainer.expected_value,
        data=X_plot[0],
        feature_names=feature_names,
    ),
    show=False
)

plt.tight_layout()
plt.savefig("reports/shap_local_bar_fl.png", dpi=300)
plt.close()
print("✅ SHAP explanations generated successfully (no NaNs)")


