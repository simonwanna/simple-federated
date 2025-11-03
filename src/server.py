import datetime
import torch
from flwr.app import ArrayRecord, ConfigRecord, Context
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg
import wandb

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
    local_epochs: int = context.run_config["local-epochs"]
    seed: int = context.run_config["seed"]
    set_seed(seed)
    
    # Initialize wandb
    wandb.init(
        project="smpl-fed",
        config=context.run_config,
        name=f"fedavg-r{num_rounds}-loc{local_epochs}-fr{fraction_train}-lr{lr}-{datetime.datetime.now():%Y%m%d-%H%M%S}",
    )

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
    
    # Log all results to W&B after aggregation to simplify comparison with centralised approach
    all_rounds = sorted(
        set(result.train_metrics_clientapp.keys()) | set(result.evaluate_metrics_clientapp.keys())
    )
    for round_num in all_rounds:
        payload = {}

        if round_num in result.train_metrics_clientapp:
            tm = result.train_metrics_clientapp[round_num]
            payload["train/loss"] = float(tm["train_loss"])

        if round_num in result.evaluate_metrics_clientapp:
            em = result.evaluate_metrics_clientapp[round_num]
            payload["eval/loss"] = float(em["eval_loss"])
            payload["eval/acc"] = float(em["eval_acc"])

        if payload:
            wandb.log(payload, step=round_num)
    
    wandb.finish()
