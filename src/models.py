from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor

def build_rf():
    return RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        n_jobs=-1,
        random_state=42,
    )

def build_fnn():
    return MLPRegressor(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        learning_rate_init=1e-3,
        max_iter=50,
        random_state=42,
    )
