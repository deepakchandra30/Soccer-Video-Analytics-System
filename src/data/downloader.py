"""Wrappers around SoccerNet's downloader for features and video."""
from SoccerNet.Downloader import SoccerNetDownloader


def download_features(local_directory="data/", splits=None):
    """Download pre-extracted ResNet-152 PCA-512 features + labels.

    No NDA needed. Features are (N, 512) float32 at 2fps.
    """
    if splits is None:
        splits = ["train", "valid", "test"]

    dl = SoccerNetDownloader(LocalDirectory=local_directory)
    dl.downloadGames(
        files=["1_ResNET_TF2_PCA512.npy", "2_ResNET_TF2_PCA512.npy", "Labels-v3.json"],
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
