import datetime
import torch
from flwr.app import ArrayRecord, ConfigRecord, Context
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg

from src.task import Net, set_seed

# Create ServerApp
app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    """
    Main entry point for the ServerApp.
    Code borrowed with modifications from Flower's tutorials.
    """

    # Read run config
    fraction_train: float = context.run_config["fraction-train"]
    num_rounds: int = context.run_config["num-server-rounds"]
    lr: float = context.run_config["lr"]
    seed: int = context.run_config["seed"]
    set_seed(seed)

    # Load global model (ResNet18 via Net)
    global_model = Net()
    arrays = ArrayRecord(global_model.state_dict())

    # Initialize FedAvg strategy
    strategy = FedAvg(fraction_train=fraction_train)

    # Start strategy, run FedAvg for `num_rounds`
    start_time = datetime.datetime.now()
    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        train_config=ConfigRecord({"lr": lr}),
        num_rounds=num_rounds,
    )

    end_time = datetime.datetime.now()
    duration = end_time - start_time
    print(f"\nFederated Training Duration: {duration}")

    # Save final model to disk
    print("\nSaving final model to disk...")
    state_dict = result.arrays.to_torch_state_dict()
    torch.save(state_dict, "final_model.pt")
