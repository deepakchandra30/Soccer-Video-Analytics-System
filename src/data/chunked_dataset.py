"""Chunked training dataset for temporal action spotting."""
import random

import numpy as np
import torch
from torch.utils.data import Dataset
from SoccerNet.Evaluation.utils import EVENT_DICTIONARY_V2


class ChunkedSoccerNetDataset(Dataset):
    """Yields fixed-size feature chunks with frame-level event labels.

    Implements event-centric sampling: most chunks are centered on annotated
    events so the model sees enough positive examples despite the extreme
    class imbalance (99%+ background frames).

    Each match is a 3-tuple ``(features, annotations, half1_len)`` where
    features is the concatenation of both halves along the time axis.  Frame
    indices for half-2 annotations are offset by ``half1_len`` so targets
    land at the correct position in the concatenated array.  Legacy
    2-tuples are still accepted with a default offset of ``45*60*framerate``
    but trigger a warning because real half lengths vary.
    """
    def __init__(self, matches, chunk_size=40, event_ratio=0.7,
                 num_classes=17, framerate=2, feat_dim=512):
        self.matches = []
        for m in matches:
            if len(m) == 3:
                self.matches.append(m)
            else:
                feats, anns = m
                default_half1 = 45 * 60 * framerate
                print(f"[ChunkedSoccerNetDataset] WARNING: match supplied "
                      f"without half1_len; assuming {default_half1} frames. "
                      f"Half-2 events may be misaligned.")
                self.matches.append((feats, anns, default_half1))

        self.chunk_size = chunk_size
        self.event_ratio = event_ratio
        self.num_classes = num_classes
        self.framerate = framerate
        self.feat_dim = feat_dim

        # Precompute (concat_frame, class_index) pairs per match once.
        # Using the concat index directly means __getitem__ never has to
        # re-parse gameTime or worry about which half an event belongs to.
        self._event_frames = []
        for feats, anns, half1_len in self.matches:
            frames = []
            for ann in anns:
                frame = self._ann_to_concat_frame(ann, half1_len)
                if frame is None or not (0 <= frame < feats.shape[0]):
                    continue
                label = ann.get("label", "")
                if label in EVENT_DICTIONARY_V2:
                    frames.append((frame, EVENT_DICTIONARY_V2[label] + 1))
            self._event_frames.append(frames)

    def __len__(self):
        return sum(f.shape[0] // self.chunk_size for f, _, _ in self.matches)

    def __getitem__(self, idx):
        match_idx = idx % len(self.matches)
        features, _, _ = self.matches[match_idx]
        T = features.shape[0]

        # decide whether to sample around an event or randomly
        event_frames = self._event_frames[match_idx]
        if random.random() < self.event_ratio and len(event_frames) > 0:
            center, _ = random.choice(event_frames)
            start = max(0, center - self.chunk_size // 2)
        else:
            start = random.randint(0, max(0, T - self.chunk_size))

        end = min(start + self.chunk_size, T)
        chunk = features[start:end]

        # pad if chunk is shorter than chunk_size
        if chunk.shape[0] < self.chunk_size:
            pad = np.zeros((self.chunk_size - chunk.shape[0], self.feat_dim),
                           dtype=np.float32)
            chunk = np.concatenate([chunk, pad])

        # build frame-level targets from precomputed (frame, class) pairs.
        targets = np.zeros(self.chunk_size, dtype=np.int64)
        for frame, cls in event_frames:
            local = frame - start
            if 0 <= local < self.chunk_size:
                targets[local] = cls

        return {
            "features": torch.FloatTensor(chunk),
            "targets": torch.LongTensor(targets),
        }

    def _ann_to_concat_frame(self, ann, half1_len):
        """Frame index in the concatenated (half1 + half2) feature array.

        Mirrors SoccerNet's own ``label2vector`` for within-half precision
        (``position`` in ms when present, else ``gameTime`` seconds), then
        adds ``half1_len`` for half-2 events so they index into the second
        half of the concatenated array rather than colliding with half 1.
        """
        gt = str(ann.get("gameTime", ""))
        if " - " in gt:
            try:
                half = int(gt.split(" - ")[0])
            except ValueError:
                half = int(ann.get("half", 1) or 1)
        else:
            half = int(ann.get("half", 1) or 1)

        within_half = None
        if "position" in ann:
            try:
                within_half = int(self.framerate * int(ann["position"]) / 1000)
            except (TypeError, ValueError):
                within_half = None
        if within_half is None and " - " in gt:
            try:
                time_str = gt.split(" - ", 1)[1]
                minutes, seconds = map(int, time_str.split(":"))
                within_half = self.framerate * (minutes * 60 + seconds)
            except (ValueError, IndexError):
                return None
        if within_half is None:
            return None

        return within_half + (0 if half == 1 else half1_len)
