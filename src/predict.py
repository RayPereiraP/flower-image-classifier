import torch
from PIL import Image

from src.dataset import val_transform
from src.model import build_model


def load_model(path, device="cpu"):
    checkpoint = torch.load(path, map_location=device)
    class_names = checkpoint["class_names"]
    model = build_model(len(class_names), pretrained=False)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device).eval()
    return model, class_names


@torch.no_grad()
def predict(model, class_names, image: Image.Image, device="cpu") -> dict[str, float]:
    x = val_transform()(image.convert("RGB")).unsqueeze(0).to(device)
    probs = torch.softmax(model(x), dim=1)[0]
    return {name: float(p) for name, p in zip(class_names, probs)}