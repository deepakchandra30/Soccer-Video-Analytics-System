"""Loss functions for temporal action spotting."""
import math

import torch
from SoccerNet.Evaluation.utils import EVENT_DICTIONARY_V2


def get_class_weights(num_classes=17, bg_weight=0.05):
    """Uniform class weights — kept for backward-compat / smoke tests.

    Prefer ``compute_class_weights_from_matches`` during real training: it
    produces per-class inverse-frequency weights that lift rare events
    (red card, yellow->red, penalty) by ~10x over common ones (ball out of
    play, throw-in), which is worth a few mAP points on tight-mAP because the
    per-class AP average is dragged down by zero-score rare classes under
    uniform weighting.
    """
    weights = torch.ones(num_classes + 1)
    weights[0] = bg_weight
    return weights


def compute_class_weights_from_matches(matches, num_classes=17,
                                       bg_weight=0.05, smoothing="sqrt",
                                       cap=10.0):
    """Inverse-frequency class weights derived from real training annotations.

    Args:
        matches: iterable of ``(features, annotations, half1_len)`` tuples as
            produced by ``train_tsm.load_matches``. Only ``annotations`` is
            read here.
        num_classes: number of event classes (17 for SoccerNet-v2).
        bg_weight: weight for the background class (index 0).
        smoothing: how to temper the raw inverse-frequency scale. ``"sqrt"``
            uses ``sqrt(max_count / count)`` — the usual median-frequency
            balancing that keeps gradients stable; ``"linear"`` uses the
            raw ``max_count / count`` and is more aggressive.
        cap: maximum weight magnitude after smoothing. Prevents a single
            near-zero-count class (e.g. yellow->red in a tiny split) from
            producing 1000x gradients and blowing up training.

    Returns:
        ``torch.FloatTensor`` of shape ``(num_classes + 1,)`` suitable for
        ``nn.CrossEntropyLoss(weight=...)``.
    """
    counts = [0] * num_classes
    for m in matches:
        anns = m[1] if isinstance(m, tuple) else m.get("annotations", [])
        for ann in anns:
            label = ann.get("label", "")
            if label in EVENT_DICTIONARY_V2:
                counts[EVENT_DICTIONARY_V2[label]] += 1

    # Any class that was entirely absent in the training split gets weight 1.
    # We can't compute a reliable inverse frequency, and loud upweighting of
    # a class we've literally never labelled would just inject noise.
    safe_counts = [c if c > 0 else 1 for c in counts]
    max_count = max(safe_counts)

    raw = [max_count / c for c in safe_counts]
    if smoothing == "sqrt":
        weights_per_event = [min(math.sqrt(w), cap) for w in raw]
    elif smoothing == "linear":
        weights_per_event = [min(w, cap) for w in raw]
    else:
        raise ValueError(f"Unknown smoothing={smoothing!r}")

    out = torch.ones(num_classes + 1)
    out[0] = bg_weight
    for i, w in enumerate(weights_per_event):
        out[i + 1] = w
    return out
