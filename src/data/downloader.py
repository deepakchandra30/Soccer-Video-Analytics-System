"""Wrappers around SoccerNet's downloader for features and video."""
import time
from SoccerNet.Downloader import SoccerNetDownloader


def download_features(local_directory="data/", splits=None, retries=5, backoff=10):
    """Download pre-extracted ResNet-152 PCA-512 features + labels.

    No NDA needed. Features are (N, 512) float32 at 2fps.
    Automatically retries on network errors with exponential backoff.
    """
    if splits is None:
        splits = ["train", "valid", "test"]

    dl = SoccerNetDownloader(LocalDirectory=local_directory)
    files = ["1_ResNET_TF2_PCA512.npy", "2_ResNET_TF2_PCA512.npy", "Labels-v2.json", "Labels-v3.json"]

    for attempt in range(1, retries + 1):
        try:
            dl.downloadGames(files=files, split=splits)
            return
        except Exception as exc:
            if attempt == retries:
                raise
            wait = backoff * attempt
            print(f"[downloader] Attempt {attempt}/{retries} failed: {exc}")
            print(f"[downloader] Retrying in {wait}s (already-downloaded files will be skipped)...")
            time.sleep(wait)


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
