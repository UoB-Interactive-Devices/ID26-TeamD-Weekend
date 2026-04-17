import pickle
from pathlib import Path

import torch

from ml.train import GestureCNN


def load_label_encoder(encoder_path: Path):
    if not encoder_path.exists():
        raise FileNotFoundError(f"Label encoder not found at {encoder_path}")
    with encoder_path.open("rb") as file:
        return pickle.load(file)


def load_model(model_path: Path, num_classes: int, device: torch.device):
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}")

    model = GestureCNN(num_classes).to(device)

    try:
        state_dict = torch.load(
            model_path,
            map_location=device,
            weights_only=True,
        )
    except TypeError:
        state_dict = torch.load(model_path, map_location=device)

    model.load_state_dict(state_dict)
    model.eval()
    return model
