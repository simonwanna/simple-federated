# Simple Federated Learning Example

### Setup
Install [uv](https://docs.astral.sh/uv/). \
Run `uv sync` to set up virtual env and install dependencies. \
Customise training parameters in `pyproject.toml` if needed.

### Run Flower server
Activate the virtual environment: `source .venv/bin/activate` \
To run the Flower server, execute: `flwr run .`
To run the centralised training, execute: `uv run src/centralized.py`