import copy
from collections.abc import Callable

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from ml.train import format_for_cnn
from torch.utils.data import DataLoader, TensorDataset


def run_fine_tuning(
    *,
    model,
    label_encoder,
    collection_features: list[np.ndarray],
    collection_labels: list[str],
    device: torch.device,
    epochs: int,
    learning_rate: float,
    on_started: Callable[[int], None] | None = None,
    on_epoch: Callable[[int, float, float], None] | None = None,
) -> tuple[bool, str]:
    if not collection_features or not collection_labels:
        return False, "No collection data available for fine-tuning."

    x_all = format_for_cnn(np.asarray(collection_features, dtype=np.float32))
    y_encoded = np.asarray(label_encoder.transform(collection_labels), dtype=np.int64)

    total_samples = len(y_encoded)
    if total_samples < 10:
        return False, "Fine-tuning requires at least 10 samples for safe validation."

    rng = np.random.default_rng(seed=42)
    indices = np.arange(total_samples)
    rng.shuffle(indices)
    val_size = max(1, int(total_samples * 0.2))
    if val_size >= total_samples:
        val_size = total_samples - 1

    val_indices = indices[:val_size]
    train_indices = indices[val_size:]

    x_train = x_all[train_indices]
    y_train = y_encoded[train_indices]
    x_val = x_all[val_indices]
    y_val = y_encoded[val_indices]

    x_tensor = torch.from_numpy(x_train).float().to(device)
    y_tensor = torch.from_numpy(y_train).to(device)
    train_loader = DataLoader(
        TensorDataset(x_tensor, y_tensor),
        batch_size=32,
        shuffle=True,
    )

    x_val_tensor = torch.from_numpy(x_val).float().to(device)
    y_val_tensor = torch.from_numpy(y_val).to(device)

    criterion = nn.CrossEntropyLoss()

    def evaluate_on_validation() -> tuple[float, float]:
        model.eval()
        with torch.no_grad():
            logits = model(x_val_tensor)
            loss = criterion(logits, y_val_tensor)
            predictions = logits.argmax(dim=1)
            accuracy = (predictions == y_val_tensor).float().mean().item()
        return float(loss.item()), float(accuracy)

    for parameter in model.conv1.parameters():
        parameter.requires_grad = False
    for parameter in model.conv2.parameters():
        parameter.requires_grad = False
    for parameter in model.fc1.parameters():
        parameter.requires_grad = True
    for parameter in model.fc2.parameters():
        parameter.requires_grad = True

    optimizer = optim.Adam(
        [
            {"params": model.fc2.parameters()},
            {"params": model.fc1.parameters(), "weight_decay": 0.01},
        ],
        lr=learning_rate,
    )

    baseline_state = copy.deepcopy(model.state_dict())
    _, baseline_accuracy = evaluate_on_validation()

    if on_started is not None:
        on_started(epochs)

    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        total = 0
        correct = 0

        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += float(loss.item()) * inputs.size(0)
            predictions = outputs.argmax(dim=1)
            total += targets.size(0)
            correct += int((predictions == targets).sum().item())

        epoch_loss = running_loss / max(total, 1)
        epoch_accuracy = correct / max(total, 1)
        if on_epoch is not None:
            on_epoch(epoch + 1, epoch_loss, epoch_accuracy)

    _, tuned_accuracy = evaluate_on_validation()

    if tuned_accuracy + 1e-6 < baseline_accuracy:
        model.load_state_dict(baseline_state)
        model.eval()
        return (
            True,
            (
                "Fine-tuning rejected: validation accuracy dropped "
                f"from {baseline_accuracy:.3f} to {tuned_accuracy:.3f}. "
                "Kept previous model weights."
            ),
        )

    model.eval()
    return (
        True,
        (
            f"Fine-tuning accepted on {device.type.upper()}. "
            f"Validation accuracy {baseline_accuracy:.3f} -> {tuned_accuracy:.3f}. "
            "Demo session uses updated weights in-memory only (not saved to disk)."
        ),
    )
