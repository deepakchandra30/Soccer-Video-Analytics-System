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
    """
    def __init__(self, matches, chunk_size=40, event_ratio=0.7,
                 num_classes=17, framerate=2, feat_dim=512):
        self.matches = matches        # list of (features_np, annotations)
        self.chunk_size = chunk_size
        self.event_ratio = event_ratio
        self.num_classes = num_classes
        self.framerate = framerate
        self.feat_dim = feat_dim

        # pre-compute event frame indices per match for faster sampling
        self._event_frames = []
        for feats, anns in self.matches:
            frames = []
            for ann in anns:
                f = self._parse_gametime(ann)
                if 0 <= f < feats.shape[0]:
                    label = ann.get("label", "")
                    if label in EVENT_DICTIONARY_V2:
                        frames.append((f, EVENT_DICTIONARY_V2[label] + 1))
            self._event_frames.append(frames)

    def __len__(self):
        return sum(f.shape[0] // self.chunk_size for f, _ in self.matches)

    def __getitem__(self, idx):
        match_idx = idx % len(self.matches)
        features, annotations = self.matches[match_idx]
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

        # build frame-level targets
        targets = np.zeros(self.chunk_size, dtype=np.int64)
        for ann in annotations:
            frame = self._parse_gametime(ann)
            local = frame - start
            
            # WIDEN THE TARGET FOR COARSE DETECTION 
            # (Applies label to +/- 2 frames around the exact timestamp)
            for offset in range(-2, 3):
                if 0 <= local + offset < self.chunk_size:
                    label = ann.get("label", "")
                    if label in EVENT_DICTIONARY_V2:
                        targets[local + offset] = EVENT_DICTIONARY_V2[label] + 1

        return {
            "features": torch.FloatTensor(chunk),
            "targets": torch.LongTensor(targets),
        }

    def _parse_gametime(self, event):
        """Convert 'H - MM:SS' gameTime string to frame index."""
        game_time = event.get("gameTime", "1 - 00:00")
        parts = game_time.split(" - ")
        time_str = parts[1] if len(parts) > 1 else parts[0]
        minutes, seconds = map(int, time_str.split(":"))
        return self.framerate * (minutes * 60 + seconds)