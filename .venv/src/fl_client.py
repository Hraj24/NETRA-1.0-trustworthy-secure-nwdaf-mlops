import csv
import os
import flwr as fl
import numpy as np
import pandas as pd
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder, StandardScaler
from math import sqrt

class TrafficClient(fl.client.NumPyClient):
    def __init__(self, path, domain_name):
        self.domain_name = domain_name
        df = pd.read_csv(path)

        # Encode all categorical columns
        for col in df.columns:
            if df[col].dtype == object:
                df[col] = LabelEncoder().fit_transform(df[col])

        X = df.drop(columns=["future_load"]).values
        y = df["future_load"].values

        # Scale features (IMPORTANT)
        self.scaler = StandardScaler()
        X = self.scaler.fit_transform(X)

        n = int(0.8 * len(X))
        self.X_train, self.X_test = X[:n], X[n:]
        self.y_train, self.y_test = y[:n], y[n:]

        self.model = SGDRegressor(
            max_iter=1,
            learning_rate="constant",
            eta0=0.01,
            random_state=42,
        )

        self.model.partial_fit(self.X_train, self.y_train)

        # RMSE log
        self.log_file = f"logs/fl_rmse_{domain_name}.csv"
        os.makedirs("logs", exist_ok=True)
        if not os.path.exists(self.log_file):
            with open(self.log_file, "w", newline="") as f:
                csv.writer(f).writerow(["round", "rmse"])

        self.round = 0

    def get_parameters(self, config):
        return [self.model.coef_, self.model.intercept_]

    def set_parameters(self, parameters):
        self.model.coef_ = parameters[0]
        self.model.intercept_ = parameters[1]

    def fit(self, parameters, config):
        self.round += 1
        self.set_parameters(parameters)
        self.model.partial_fit(self.X_train, self.y_train)
        return self.get_parameters(config), len(self.X_train), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        preds = self.model.predict(self.X_test)
        rmse = sqrt(mean_squared_error(self.y_test, preds))

        # Log RMSE per round
        with open(self.log_file, "a", newline="") as f:
            csv.writer(f).writerow([self.round, rmse])

        return float(rmse), len(self.X_test), {"rmse": rmse}


def start_client(csv_path, domain_name):
    fl.client.start_numpy_client(
        server_address="127.0.0.1:8080",
        client=TrafficClient(csv_path, domain_name),
    )
