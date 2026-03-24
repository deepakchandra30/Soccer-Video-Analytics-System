"""NMS post-processing and prediction formatting for SoccerNet evaluation."""
import json

import numpy as np
import torch
from SoccerNet.Evaluation.utils import INVERSE_EVENT_DICTIONARY_V2


def nms_detections(frame_scores, nms_window=30, confidence_threshold=0.2,
                   framerate=2, half=1):
    """Per-class NMS on frame-level scores, returns prediction dicts."""
    T, num_classes = frame_scores.shape
    predictions = []

    for c in range(num_classes):
        scores = frame_scores[:, c]
        for t in range(T):
            if scores[t] < confidence_threshold:
                continue
            # check if this is the local maximum within the NMS window
            win_start = max(0, t - nms_window // 2)
            win_end = min(T, t + nms_window // 2 + 1)
            if scores[t] == scores[win_start:win_end].max():
                predictions.append({
                    "label": INVERSE_EVENT_DICTIONARY_V2[c],
                    "half": half,
                    "position": int(t * 1000 / framerate),
                    "confidence": float(scores[t]),
                })

    return predictions


def save_predictions(predictions, output_path):
    """Write predictions in SoccerNet SDK format."""
    with open(output_path, "w") as f:
        json.dump({"predictions": predictions}, f, indent=2)


def sliding_window_inference(model, features, window_size=40, stride=20,
                             device="cpu"):
    """Run model on overlapping windows, average predictions, return (N, num_classes) array."""
    N = features.shape[0]
    score_sum = None
    count = np.zeros(N, dtype=np.float32)

    model.eval()
    with torch.no_grad():
        start = 0
        while start < N:
            end = min(start + window_size, N)
            window = features[start:end]

            if window.shape[0] < window_size:
                pad = torch.zeros(window_size - window.shape[0], window.shape[1])
                window = torch.cat([window, pad], dim=0)

            window = window.unsqueeze(0).to(device)
            logits = model(window)  # (1, W, C+1)
            probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()

            # skip background (index 0), keep event classes
            event_probs = probs[:, 1:]

            if score_sum is None:
                score_sum = np.zeros((N, event_probs.shape[1]), dtype=np.float32)

            actual_len = end - start
            score_sum[start:end] += event_probs[:actual_len]
            count[start:end] += 1

            if end >= N:
                break
            start += stride

    count[count == 0] = 1
    return score_sum / count[:, None]
