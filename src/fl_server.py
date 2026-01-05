# import flwr as fl

# if __name__ == "__main__":
#     strategy = fl.server.strategy.FedAvg(
#         fraction_fit=1.0,
#         fraction_evaluate=1.0,
#         min_fit_clients=3,
#         min_evaluate_clients=3,
#         min_available_clients=3,
#     )

#     fl.server.start_server(
#         server_address="127.0.0.1:8080",
#         config=fl.server.ServerConfig(num_rounds=5),
#         strategy=strategy,
#     )




import flwr as fl
import joblib
from typing import List, Tuple, Optional
from flwr.common import Parameters, FitRes
from flwr.server.client_proxy import ClientProxy


# --------------------------------------------------
# Custom FedAvg Strategy to Save Global Model
# --------------------------------------------------
class SaveModelFedAvg(fl.server.strategy.FedAvg):
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[BaseException],
    ) -> Optional[Tuple[Parameters, dict]]:

        aggregated = super().aggregate_fit(
            server_round, results, failures
        )

        if aggregated is None:
            return None

        # 🔑 IMPORTANT: unpack return value
        aggregated_parameters, aggregated_metrics = aggregated

        # Convert Parameters → NumPy arrays
        weights = fl.common.parameters_to_ndarrays(aggregated_parameters)

        # Save global FL model
        model_data = {
            "coef": weights[0],
            "intercept": weights[1],
        }

        joblib.dump(model_data, "models/fl_global_model.pkl")
        print(f"[SERVER] Saved FL global model at round {server_round}")

        return aggregated_parameters, aggregated_metrics


# --------------------------------------------------
# Start Flower server
# --------------------------------------------------
if __name__ == "__main__":
    strategy = SaveModelFedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=3,
        min_evaluate_clients=3,
        min_available_clients=3,
    )

    fl.server.start_server(
        server_address="127.0.0.1:8080",
        config=fl.server.ServerConfig(num_rounds=5),
        strategy=strategy,
    )


