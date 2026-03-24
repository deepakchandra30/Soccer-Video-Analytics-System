"""Feature extraction: load pre-extracted .npy or run ResNet on video."""
import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from tqdm import tqdm
import wandb


def load_and_validate_features(feature_path, expected_dim=512):
    """Load .npy feature file and validate shape (2D, correct dim, enough frames)."""
    features = np.load(feature_path)
    assert features.ndim == 2, f"Expected 2D, got {features.ndim}D from {feature_path}"
    assert features.shape[1] == expected_dim, (
        f"Dim mismatch: expected {expected_dim}, got {features.shape[1]} in {feature_path}"
    )
    assert features.shape[0] > 4500, (
        f"Too few frames ({features.shape[0]}) in {feature_path}"
    )
    return features


def extract_features(video_path, output_path, model_name="resnet50",
                     fps=2, batch_size=32, device=None):
    """Run ResNet on video frames and save (N, 2048) features to .npy."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    from decord import VideoReader, cpu as decord_cpu

    if model_name == "resnet50":
        model = models.resnet50(weights="IMAGENET1K_V1")
    elif model_name == "resnet101":
        model = models.resnet101(weights="IMAGENET1K_V1")
    else:
        raise ValueError(f"Unknown model: {model_name}")

    model.fc = torch.nn.Identity()
    model = model.to(device).eval()

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    vr = VideoReader(video_path, ctx=decord_cpu(0))
    video_fps = vr.get_avg_fps()
    step = max(1, int(round(video_fps / fps)))
    frame_indices = list(range(0, len(vr), step))

    all_features = []
    for i in tqdm(range(0, len(frame_indices), batch_size), desc=f"Extracting {model_name}"):
        batch_idx = frame_indices[i:i + batch_size]
        frames = vr.get_batch(batch_idx).asnumpy()
        tensors = torch.stack([transform(f) for f in frames]).to(device)
        with torch.no_grad():
            feats = model(tensors).cpu().numpy()
        all_features.append(feats)

    features = np.concatenate(all_features, axis=0)
    np.save(output_path, features)

    if wandb.run is not None:
        wandb.config.update({"feature_dim": features.shape[1], "n_frames": features.shape[0]})

    return features
