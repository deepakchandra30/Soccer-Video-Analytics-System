"""NMS post-processing and prediction formatting for SoccerNet evaluation."""
import json

import numpy as np
import torch
from SoccerNet.Evaluation.utils import INVERSE_EVENT_DICTIONARY_V2


def nms_detections(
    frame_scores,
    nms_window=30,
    confidence_threshold=0.005,
    framerate=2,
    half=1,
):
    """Per-class non-maximum suppression on frame-level scores."""
    T, num_classes = frame_scores.shape
    predictions = []

    for c in range(num_classes):
        scores = frame_scores[:, c]
        for t in range(T):
            # Lowered threshold so the evaluator can build a proper PR curve!
            if scores[t] < confidence_threshold:
                continue

            # Check if this is the local maximum within the NMS window
            win_start = max(0, t - nms_window // 2)
            win_end = min(T, t + nms_window // 2 + 1)

            if scores[t] == scores[win_start:win_end].max():
                # Convert frame index to MM:SS gameTime string
                total_seconds = int(t / framerate)
                minutes = total_seconds // 60
                seconds = total_seconds % 60
                game_time_str = f"{half} - {minutes:02d}:{seconds:02d}"

                predictions.append(
                    {
                        "gameTime": game_time_str,  # SDK expects this
                        "label": INVERSE_EVENT_DICTIONARY_V2[c],
                        "position": str(int(t * 1000 / framerate)),
                        "half": str(half),
                        "confidence": float(scores[t]),
                    }
                )

    return predictions


def save_predictions(predictions, output_path):
    """Write predictions in SoccerNet SDK format."""
    with open(output_path, "w") as f:
        json.dump({"predictions": predictions}, f, indent=2)


def sliding_window_inference(model, features, window_size=40, stride=20, device="cpu"):
    """Run model on overlapping windows and average predictions.

    Supports models whose temporal output length may differ from window_size
    (e.g., SlowFast temporal downsampling).

    Args:
        model: temporal spotting model
        features: (N, feat_dim) tensor for one half
        window_size: input window length in frames
        stride: step between windows
        device: torch device string

    Returns:
        (N, num_classes) numpy array (background class excluded).
    """
    N = int(features.shape[0])
    if N == 0:
        return np.zeros((0, 0), dtype=np.float32)

    score_sum = None
    count = np.zeros(N, dtype=np.float32)

    model.eval()
    with torch.no_grad():
        start = 0
        while start < N:
            end = min(start + window_size, N)
            actual_len = end - start

            # Slice current window
            window = features[start:end]

            # Pad short tail window to fixed input size
            if window.shape[0] < window_size:
                pad = torch.zeros(
                    window_size - window.shape[0],
                    window.shape[1],
                    dtype=window.dtype,
                )
                window = torch.cat([window, pad], dim=0)

            # Forward
            window = window.unsqueeze(0).to(device)  # (1, W, D)
            logits = model(window)
            probs = torch.softmax(logits, dim=-1)[0].detach().cpu().numpy()  # (T_out, C+1)
            event_probs = probs[:, 1:]  # remove background -> (T_out, C)

            if score_sum is None:
                score_sum = np.zeros((N, event_probs.shape[1]), dtype=np.float32)

            # ---- Robust temporal alignment (fix for broadcast error) ----
            pred_len = int(event_probs.shape[0])

            # If model returns more/less than actual_len, align by interpolation
            if pred_len != actual_len:
                # Linear interpolation from pred_len -> actual_len for each class
                x_old = np.linspace(0.0, 1.0, pred_len, dtype=np.float32)
                x_new = np.linspace(0.0, 1.0, actual_len, dtype=np.float32)
                resized = np.empty((actual_len, event_probs.shape[1]), dtype=np.float32)
                for c in range(event_probs.shape[1]):
                    resized[:, c] = np.interp(x_new, x_old, event_probs[:, c]).astype(np.float32)
                event_chunk = resized
            else:
                event_chunk = event_probs[:actual_len]

            score_sum[start:end] += event_chunk
            count[start:end] += 1.0

            if end >= N:
                break
            start += stride

    count[count == 0] = 1.0
    return score_sum / count[:, None]