import pathlib
import datetime
import tomllib
import torch

from task import (
    Net,
    load_data,
    test as test_fn,
    train_epoch as train_fn,
    set_seed,
)


def main(context):
    """Centralized training and evaluation."""

    # Seed everything
    seed = context["seed"]
    set_seed(seed)

    # Load the model
    model = Net()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Using device: {device}")

    # Initialize optimizer and criterion
    optimizer = torch.optim.Adam(model.parameters(), lr=context["lr"])
    criterion = torch.nn.CrossEntropyLoss().to(device)

    # Load the data
    trainloader, testloader = load_data(
        partition_id=0, num_partitions=1, bs=context["batch-size"], seed=seed
    )

    # Train the model
    start = datetime.datetime.now()
    for epoch in range(context["global-epochs"]):
        train_loss = train_fn(
            model,
            optimizer,
            criterion,
            trainloader,
            device,
            centralized=True,
        )

        # Evaluate the model
        test_loss, test_accuracy = test_fn(model, testloader, device, centralized=True)
        print(
            f"Epoch {epoch + 1} - Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}"
        )

    end = datetime.datetime.now()
    duration = end - start
    print(f"Centralized Training Duration: {duration}")


if __name__ == "__main__":
    # Load FL context from pyproject.toml
    pyproject_path = pathlib.Path(__file__).parent.parent / "pyproject.toml"
    with open(pyproject_path, "rb") as f:
        pyproject_data = tomllib.load(f)
    context = pyproject_data["tool"]["flwr"]["app"]["config"]

    # Run centralized training and evaluation
    main(context)
