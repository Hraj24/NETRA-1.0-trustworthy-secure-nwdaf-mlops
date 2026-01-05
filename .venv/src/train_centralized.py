import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error,r2_score,mean_absolute_percentage_error,max_error
from math import sqrt
from models import build_rf, build_fnn

def train_and_eval(model_name="rf"):
    df = pd.read_csv("data/traffic_synthetic.csv")
    X = df.drop(columns=["future_load"])
    y = df["future_load"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    if model_name == "rf":
        model = build_rf()
    else:
        model = build_fnn()

    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    rmse = sqrt(mean_squared_error(y_test, preds))
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    mape = mean_absolute_percentage_error(y_test, preds) * 100
    max_err = max_error(y_test,preds)

    print(f"Model: {model_name}, RMSE={rmse:.3f}, MAE={mae:.3f}, R²={r2:.2f}, MAPE(%)={mape:.1f}, Max Error={max_err:.1f}")
    return model

if __name__ == "__main__":
    train_and_eval("rf")
    train_and_eval("fnn")
