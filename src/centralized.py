import pathlib
import tomllib
import torch

from task import (
    Net,
    load_data,
    test as test_fn,
    train as train_fn,
)


def main(context):
    """Centralized training and evaluation."""

    # Load the model
    model = Net()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load the data
    trainloader, testloader = load_data(partition_id=0, num_partitions=1)

    # Train the model
    train_fn(
        model,
        trainloader,
        context["global-epochs"],
        context["lr"],
        device,
        centralized=True,
    )

    # Evaluate the model
    test_loss, test_accuracy = test_fn(model, testloader, device, centralized=True)

    print(f"Centralized Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")


if __name__ == "__main__":
    
    # Load FL context from pyproject.toml
    pyproject_path = pathlib.Path(__file__).parent.parent / "pyproject.toml"
    with open(pyproject_path, "rb") as f:
        pyproject_data = tomllib.load(f)
    context = pyproject_data["tool"]["flwr"]["app"]["config"]

    # Run centralized training and evaluation
    main(context)
