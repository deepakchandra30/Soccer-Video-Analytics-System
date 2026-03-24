"""Two-stage inference pipeline: TSM coarse detection -> SlowFast refinement."""
import torch
import numpy as np

from src.models.temporal.postprocess import (
    nms_detections,
    sliding_window_inference,
)


class TwoStagePipeline:
    """Chains a lightweight coarse detector with a fine classifier.

    Stage 1 (TSM): scans the full match cheaply, generates candidate windows
    with high recall / moderate precision.

    Stage 2 (SlowFast): re-classifies only the candidate windows at higher
    temporal resolution for better precision.
    """
    def __init__(self, coarse_model, fine_model, config, device="cpu"):
        self.coarse = coarse_model
        self.fine = fine_model
        self.config = config
        self.device = device
        self._candidate_frame_count = 0

    def detect_candidates(self, features, half=1):
        """Run coarse detector and return candidate event windows."""
        scores = sliding_window_inference(
            self.coarse, features,
            window_size=self.config["coarse_window_size"],
            stride=self.config["coarse_stride"],
            device=self.device,
        )

        raw_preds = nms_detections(
            scores,
            nms_window=self.config["coarse_nms_window"],
            confidence_threshold=self.config["coarse_confidence_threshold"],
            framerate=self.config.get("framerate", 2),
            half=half,
        )

        candidates = []
        framerate = self.config.get("framerate", 2)
        pad = self.config.get("fine_pad_frames", 20)
        N = features.shape[0]

        for pred in raw_preds:
            center_frame = int(pred["position"] * framerate / 1000)
            start = max(0, center_frame - pad)
            end = min(N, center_frame + pad)
            candidates.append({
                "start_frame": start,
                "end_frame": end,
                "half": half,
                "predictions": [pred],
            })

        return candidates

    def classify_candidates(self, features, candidates):
        """Re-classify candidate windows with the fine model."""
        self.fine.eval()
        refined = []

        for cand in candidates:
            start = cand["start_frame"]
            end = cand["end_frame"]
            half = cand["half"]
            window = features[start:end]

            # pad to minimum size for the fine model
            min_len = self.config.get("fine_pad_frames", 20) * 2
            if window.shape[0] < min_len:
                pad_size = min_len - window.shape[0]
                pad = torch.zeros(pad_size, window.shape[1])
                window = torch.cat([window, pad], dim=0)

            # ensure length is divisible by slow_stride (4)
            remainder = window.shape[0] % 4
            if remainder != 0:
                pad_size = 4 - remainder
                pad = torch.zeros(pad_size, window.shape[1])
                window = torch.cat([window, pad], dim=0)

            with torch.no_grad():
                inp = window.unsqueeze(0).to(self.device)
                logits = self.fine(inp)  # (1, T//4, 18)
                probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()

            event_probs = probs[:, 1:]  # (T_out, 17)

            fine_preds = nms_detections(
                event_probs,
                nms_window=self.config["fine_nms_window"],
                confidence_threshold=self.config["fine_confidence_threshold"],
                framerate=self.config.get("framerate", 2),
                half=half,
            )

            # adjust positions: fine model predictions are relative to the window
            stride = 4  # slow_stride
            for p in fine_preds:
                # re-map from fine model frame index to global frame index
                fine_frame = int(p["position"] * self.config.get("framerate", 2) / 1000)
                global_frame = start + fine_frame * stride
                p["position"] = int(global_frame * 1000 / self.config.get("framerate", 2))

            refined.extend(fine_preds)

        return refined

    def run(self, half1_features, half2_features):
        """Run full two-stage pipeline on a match, returns prediction dicts."""
        self._candidate_frame_count = 0
        all_preds = []

        for half_idx, features in enumerate([half1_features, half2_features], 1):
            candidates = self.detect_candidates(features, half=half_idx)

            for c in candidates:
                self._candidate_frame_count += c["end_frame"] - c["start_frame"]

            if len(candidates) > 0:
                refined = self.classify_candidates(features, candidates)
                all_preds.extend(refined)

        return all_preds

    def get_candidate_frame_count(self):
        """Total frames processed by the fine stage in the last run()."""
        return self._candidate_frame_count
