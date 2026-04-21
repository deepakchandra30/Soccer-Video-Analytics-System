"""Wrappers around SoccerNet's downloader for features and video."""
from SoccerNet.Downloader import SoccerNetDownloader


FEATURE_FILES = {
    "pca512": ["1_ResNET_TF2_PCA512.npy", "2_ResNET_TF2_PCA512.npy"],
    "resnet50": ["1_ResNET_TF2.npy", "2_ResNET_TF2.npy"],
    "baidu": ["1_baidu_soccer_embeddings.npy", "2_baidu_soccer_embeddings.npy"],
}


def download_features(local_directory="data/", splits=None, features="pca512"):
    """Download SoccerNet pre-extracted features + labels.

    ``features`` selects which feature set to pull:
      - "pca512" (default): ResNet-152 → PCA 512, 2fps. Tight-mAP ceiling ~35-40%.
      - "resnet50": full ResNet 2048-dim, 2fps. ~+3-5 pts over pca512.
      - "baidu": Baidu 8576-dim embeddings, 1fps (from the 2021 challenge).
                 Documented ~+15 pts tight-mAP over pca512. ~50GB per split.

    Bug-fix 2026-04-13: also fetch Labels-v2.json. The official evaluator
    `SoccerNet.Evaluation.ActionSpotting.evaluate(..., version=2)` loads
    Labels-v2.json by name. Previously this downloader fetched only
    Labels-v3.json, which produced a silent FileNotFoundError inside the
    evaluator and a reported `avg-mAP tight = 0.0%`. Both labels are
    needed: v2 for the official evaluator, v3 for richer per-event
    metadata used by downstream analytics.
    """
    if splits is None:
        splits = ["train", "valid", "test"]

    if features not in FEATURE_FILES:
        raise ValueError(
            f"Unknown features={features!r}; pick one of {list(FEATURE_FILES)}"
        )

    dl = SoccerNetDownloader(LocalDirectory=local_directory)
    dl.downloadGames(
        files=FEATURE_FILES[features] + ["Labels-v2.json", "Labels-v3.json"],
        split=splits,
    )


def download_video(local_directory="data/", password="", splits=None):
    """Download raw .mkv match videos. NDA password required.

    Get the password by submitting the NDA at https://www.soccer-net.org/data
    """
    if not password:
        raise ValueError("NDA password required — submit form at soccer-net.org/data")

    if splits is None:
        splits = ["train", "valid", "test"]

    dl = SoccerNetDownloader(LocalDirectory=local_directory)
    dl.password = password
    dl.downloadGames(files=["1_720p.mkv", "2_720p.mkv"], split=splits)
