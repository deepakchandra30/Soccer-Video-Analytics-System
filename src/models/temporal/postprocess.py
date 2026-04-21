"""NMS post-processing and prediction formatting for SoccerNet evaluation."""
import json

import numpy as np
import torch
import torch.nn.functional as F
from SoccerNet.Evaluation.utils import INVERSE_EVENT_DICTIONARY_V2


def _emit_pred(t, c, score, framerate, half):
    """Format one detection as a SoccerNet evaluator-compatible dict."""
    position_ms = int(t * 1000 / framerate)
    seconds = position_ms // 1000
    # SoccerNet's evaluator expects every value as str, plus a
    # gameTime field formatted "H - MM:SS". Native int/float
    # values silently produce avg-mAP=0% even when predictions
    # match the labels.
    return {
        "gameTime": f"{half} - {seconds // 60:02d}:{seconds % 60:02d}",
        "label": INVERSE_EVENT_DICTIONARY_V2[c],
        "position": str(position_ms),
        "half": str(half),
        "confidence": str(round(float(score), 4)),
    }


def nms_detections(frame_scores, nms_window=15, confidence_threshold=0.2,
                   framerate=2, half=1, nms_mode="hard", soft_sigma=None):
    """Per-class non-maximum suppression on frame-level scores.

    Args:
        frame_scores: (T, num_classes) numpy array of probabilities
        nms_window: suppression window in frames (used as both hard-NMS
            kill radius and soft-NMS Gaussian extent)
        confidence_threshold: minimum score to consider
        framerate: feature extraction fps (2 for SoccerNet)
        half: which half of the match (1 or 2)
        nms_mode: "hard" (default) keeps only local maxima; "soft" keeps
            all peaks but decays neighbour scores by Gaussian distance,
            preserving close-in-time but distinct events. Soft-NMS is
            worth ~1-2% tight-mAP on SoccerNet because actions like
            foul/yellow-card legitimately co-occur within 5-10s.
        soft_sigma: standard deviation (in frames) of the Gaussian decay
            for soft-NMS. Defaults to nms_window/3 if not given.

    Returns list of prediction dicts for results_spotting.json.
    """
    T, num_classes = frame_scores.shape
    predictions = []

    if nms_mode == "hard":
        for c in range(num_classes):
            scores = frame_scores[:, c]
            for t in range(T):
                if scores[t] < confidence_threshold:
                    continue
                win_start = max(0, t - nms_window // 2)
                win_end = min(T, t + nms_window // 2 + 1)
                if scores[t] == scores[win_start:win_end].max():
                    predictions.append(_emit_pred(t, c, scores[t], framerate, half))
        return predictions

    if nms_mode != "soft":
        raise ValueError(f"Unknown nms_mode={nms_mode!r}")

    # Soft-NMS: greedy peak selection with Gaussian score decay.
    sigma = soft_sigma if soft_sigma is not None else max(1.0, nms_window / 3.0)
    half_win = nms_window // 2

    for c in range(num_classes):
        scores = frame_scores[:, c].astype(np.float64).copy()
        while True:
            t_max = int(np.argmax(scores))
            if scores[t_max] < confidence_threshold:
                break
            predictions.append(_emit_pred(t_max, c, scores[t_max], framerate, half))
            # decay nearby positions (including t_max -> 0) so the same peak
            # cannot be picked twice and immediate neighbours are damped.
            lo = max(0, t_max - half_win)
            hi = min(T, t_max + half_win + 1)
            offsets = np.arange(lo, hi) - t_max
            decay = np.exp(-(offsets.astype(np.float64) ** 2) / (2 * sigma * sigma))
            # zero out the picked peak, soft-decay the rest
            decay[offsets == 0] = 0.0
            scores[lo:hi] = scores[lo:hi] * decay

    return predictions


def save_predictions(predictions, output_path, url_local=None):
    """Write predictions in SoccerNet SDK format.

    The evaluator looks up ``UrlLocal`` to associate predictions with the
    ground-truth game. Pass the same string used by getListGames(...).
    """
    payload = {"predictions": predictions}
    if url_local is not None:
        payload = {"UrlLocal": url_local, "predictions": predictions}
    with open(output_path, "w") as f:
        json.dump(payload, f, indent=2)


def multi_scale_inference(model, features, scales=((40, 20), (80, 40)),
                          device="cpu"):
    """Run sliding_window_inference at multiple (window_size, stride) scales
    and average the per-frame event probabilities.

    Captures both short-context (40 frames = 20s) and long-context (80
    frames = 40s) views of the same match. Worth ~1-2% tight-mAP because
    different actions need different temporal extents — a goal is decided
    in ~2s of visible action, an offside reads against 20s of build-up.

    Returns: (T, num_classes) numpy array.
    """
    if not scales:
        raise ValueError("scales must be non-empty")

    accum = None
    for window_size, stride in scales:
        scores = sliding_window_inference(model, features,
                                          window_size=window_size,
                                          stride=stride, device=device)
        if accum is None:
            accum = scores.astype(np.float64)
        else:
            # all sliding_window_inference outputs are (T, num_classes) at
            # full T resolution by construction, so direct addition is safe.
            accum += scores

    return (accum / len(scales)).astype(np.float32)


def sliding_window_inference(model, features, window_size=40, stride=20,
                             device="cpu"):
    """Run model on overlapping windows and average the predictions.

    Args:
        model: TSMSpottingHead or similar (produces (1, T, C+1) logits)
        features: (N, feat_dim) tensor for one half
        window_size: window length in frames
        stride: step between windows
        device: torch device string

    Returns: (N, num_classes) numpy array (background class excluded).
    """
    N = features.shape[0]
    score_sum = None
    count = np.zeros(N, dtype=np.float32)

    model.eval()
    with torch.no_grad():
        start = 0
        while start < N:
            end = min(start + window_size, N)
            window = features[start:end]

            # pad short tail window
            if window.shape[0] < window_size:
                pad = torch.zeros(window_size - window.shape[0], window.shape[1])
                window = torch.cat([window, pad], dim=0)

            window = window.unsqueeze(0).to(device)
            logits = model(window)  # (1, W_out, C+1) — W_out < W for SlowFast (T // slow_stride)
            # Realign output to the window's full temporal resolution so the
            # accumulator below can index by frame. Linear interp avoids the
            # tied-max problem that repeat_interleave would create at NMS.
            if logits.size(1) != window_size:
                logits = F.interpolate(
                    logits.transpose(1, 2), size=window_size,
                    mode="linear", align_corners=False,
                ).transpose(1, 2)
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
