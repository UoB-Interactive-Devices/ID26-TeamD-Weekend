import copy
import csv
import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[3]
DATA_FILES = [
    ROOT / "data" / "processed_gesture_data.csv",
    ROOT / "data" / "processed_gesture_data2.csv",
]
MODEL_PATH = ROOT / "model" / "gesture_cnn_model.pth"
ENCODER_PATH = ROOT / "model" / "label_encoder.pkl"
TOTAL_SENSORS = 91


def load_data_grouped_by_session(filenames):
    X, y, sessions = [], [], []

    for filename in filenames:
        try:
            with filename.open("r", encoding="utf-8") as f:
                reader = csv.reader(f)
                next(reader)

                for row in reader:
                    if len(row) >= TOTAL_SENSORS + 3:
                        label = row[TOTAL_SENSORS]
                        X.append([float(val) for val in row[:TOTAL_SENSORS]])
                        y.append(label)
                        sessions.append(f"{filename}_{row[TOTAL_SENSORS + 2]}")
        except FileNotFoundError:
            print(f"Could not find {filename}, skipping.")

    return np.array(X, dtype=np.float32), np.array(y), np.array(sessions)


def format_for_cnn(X_flat):
    front = X_flat[:, :49].reshape(-1, 1, 7, 7)
    back = X_flat[:, 49:].reshape(-1, 1, 6, 7)

    back_padded = np.pad(
        back,
        ((0, 0), (0, 0), (0, 1), (0, 0)),
        mode="constant",
        constant_values=0,
    )

    X_cnn = np.concatenate([front, back_padded], axis=1)
    return X_cnn.astype(np.float32)


class GestureCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(2, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.dropout1 = nn.Dropout(0.2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 3 * 3, 64)
        self.dropout2 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        if self.training:
            x = x + torch.randn_like(x) * 0.05
            x = self._random_spatial_shift(x, max_shift=1)

        x = torch.relu(self.conv1(x))
        x = self.pool(x)
        x = self.dropout1(x)
        x = torch.relu(self.conv2(x))
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        return x

    @staticmethod
    def _random_spatial_shift(x, max_shift=1):
        if max_shift <= 0:
            return x

        shifted = x.clone()
        for index in range(x.size(0)):
            shift_y = int(
                torch.randint(-max_shift, max_shift + 1, (1,), device=x.device).item()
            )
            shift_x = int(
                torch.randint(-max_shift, max_shift + 1, (1,), device=x.device).item()
            )
            shifted[index] = torch.roll(
                shifted[index],
                shifts=(shift_y, shift_x),
                dims=(-2, -1),
            )
        return shifted


def main():
    import matplotlib.pyplot as plt
    from sklearn.metrics import (
        ConfusionMatrixDisplay,
        classification_report,
        confusion_matrix,
    )
    from sklearn.model_selection import GroupShuffleSplit
    from sklearn.preprocessing import LabelEncoder
    from sklearn.utils.class_weight import compute_class_weight

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    X_flat, y, session_groups = load_data_grouped_by_session(DATA_FILES)

    if len(X_flat) == 0:
        print("No valid samples found.")
        return

    X = format_for_cnn(X_flat)

    label_encoder = LabelEncoder()
    y_encoded = np.asarray(label_encoder.fit_transform(y), dtype=np.int64)
    num_classes = len(label_encoder.classes_)

    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(X, y_encoded, session_groups))

    X_train, X_test = torch.tensor(X[train_idx]), torch.tensor(X[test_idx])
    y_train, y_test = (
        torch.tensor(y_encoded[train_idx], dtype=torch.long),
        torch.tensor(y_encoded[test_idx], dtype=torch.long),
    )

    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    model = GestureCNN(num_classes).to(device)
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(y_train.numpy()),
        y=y_train.numpy(),
    )
    weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights_tensor)
    optimizer = optim.Adam(
        [
            {"params": model.conv1.parameters()},
            {"params": model.conv2.parameters()},
            {"params": model.fc2.parameters()},
            {"params": model.fc1.parameters(), "weight_decay": 0.01},
        ],
        lr=0.001,
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3, min_lr=1e-5
    )

    epochs = 40
    patience = 8
    best_val_acc = 0.0
    patience_counter = 0
    best_model_weights = None

    for epoch in range(epochs):
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            train_total += targets.size(0)
            train_correct += predicted.eq(targets).sum().item()

        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0

        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)

                val_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()

        val_acc = val_correct / val_total
        print(
            f"Epoch {epoch + 1}/{epochs} | Val Acc: {val_acc * 100:.2f}% | Val Loss: {val_loss / val_total:.4f}"
        )

        scheduler.step(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            best_model_weights = copy.deepcopy(model.state_dict())
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

    if best_model_weights is not None:
        model.load_state_dict(best_model_weights)

    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.numpy())

    print("\nEvaluating on hold-out chunks...")
    print(
        classification_report(
            all_targets, all_preds, target_names=label_encoder.classes_
        )
    )

    print("\nConfusion matrix...")
    cm = confusion_matrix(all_targets, all_preds, normalize="true")
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=label_encoder.classes_
    )
    fig, ax = plt.subplots(figsize=(10, 8))
    disp.plot(cmap=plt.get_cmap("Blues"), ax=ax, values_format=".2f")
    plt.title("Gesture Classification Confusion Matrix")
    plt.tight_layout()
    plt.show()

    torch.save(model.state_dict(), MODEL_PATH)
    with ENCODER_PATH.open("wb") as f:
        pickle.dump(label_encoder, f)
    print(f"Model saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()
