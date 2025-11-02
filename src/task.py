import os
import random

import numpy as np
import torch
import torch.nn as nn
import tqdm
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor
from torchvision import models


def set_seed(seed: int, deterministic: bool = True):
    # Ensure cuBLAS uses deterministic workspace (must be set before CUDA init)
    # Additionally, set CUBLAS workspace config: export CUBLAS_WORKSPACE_CONFIG=:16:8
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.use_deterministic_algorithms(True, warn_only=True)


def seed_worker(worker_id):
    # Only used if num_workers > 0
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)


class Net(nn.Module):
    """ResNet18 adapted for CIFAR-10 (3x32x32).
    - 3x3 conv, stride=1, no maxpool (common CIFAR stem)"""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        m = models.resnet18(weights=None, num_classes=num_classes)
        # CIFAR-10 stem
        m.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        m.maxpool = nn.Identity()
        self.model = m

    def forward(self, x):
        return self.model(x)


pytorch_transforms = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


def apply_transforms(batch):
    """Apply transforms to the partition from FederatedDataset."""
    batch["img"] = [pytorch_transforms(img) for img in batch["img"]]
    return batch


def load_data(partition_id: int, num_partitions: int, bs: int = 32, seed: int = 42):
    """
    Load partition CIFAR10 data deterministically.
    Code borrowed with modifications from Flower's tutorials.
    """
    # Only initialize `FederatedDataset` once
    partitioner = IidPartitioner(num_partitions=num_partitions)
    fds = FederatedDataset(
        dataset="uoft-cs/cifar10",
        partitioners={"train": partitioner},
    )
    partition = fds.load_partition(partition_id)

    # Divide data on each node: 80% train, 20% test (seeded)
    partition_train_test = partition.train_test_split(test_size=0.2, seed=seed)

    # Deterministic transforms and shuffling
    partition_train_test = partition_train_test.with_transform(apply_transforms)

    # Deterministic shuffling via generator; different per partition
    g = torch.Generator(device="cpu").manual_seed(seed + int(partition_id))

    # Use num_workers=0 for determinism; if increased, keep seed_worker
    trainloader = DataLoader(
        partition_train_test["train"],
        batch_size=bs,
        shuffle=True,
        generator=g,
        num_workers=0,
        persistent_workers=False,
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=seed_worker,
    )
    testloader = DataLoader(
        partition_train_test["test"],
        batch_size=bs,
        shuffle=False,
        num_workers=0,
        persistent_workers=False,
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=seed_worker,
    )
    return trainloader, testloader


def train(net, trainloader, epochs, lr, device):
    """
    Train the model on the training set.
    Code borrowed with modifications from Flower's tutorials.
    """
    net.to(device)  # move model to GPU if available
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    total_loss = 0.0
    for _ in range(epochs):
        epoch_loss = train_epoch(net, optimizer, criterion, trainloader, device)
        total_loss += epoch_loss
    avg_trainloss = total_loss / epochs
    return avg_trainloss


def train_epoch(net, optimizer, criterion, trainloader, device, centralized=False):
    """Train one epoch."""
    net.to(device)
    net.train()
    running_loss = 0.0
    for batch in tqdm.tqdm(
        trainloader, postfix="Training", leave=False, disable=not centralized
    ):
        images = batch["img"].to(device)
        labels = batch["label"].to(device)
        optimizer.zero_grad()
        loss = criterion(net(images), labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    avg_trainloss = running_loss / len(trainloader)
    return avg_trainloss


def test(net, testloader, device, centralized=False):
    """
    Validate the model on the test set.
    Code borrowed with modifications from Flower's tutorials.
    """
    net.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    with torch.no_grad():
        for batch in tqdm.tqdm(
            testloader, postfix="Evaluating", leave=False, disable=not centralized
        ):
            images = batch["img"].to(device)
            labels = batch["label"].to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
    accuracy = correct / len(testloader.dataset)
    loss = loss / len(testloader)
    return loss, accuracy
