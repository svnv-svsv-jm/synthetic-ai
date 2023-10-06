__all__ = ["download_hubert"]

import wget, os
from pathlib import Path
from ..audiolm_pytorch import HubertWithKmeans


def download_hubert(loc: str = None) -> HubertWithKmeans:
    """Downloads Hubert checkpoints and initializes the wav2vec model."""
    loc = loc or __file__
    # Set up
    here = Path(loc).resolve().parent
    url_hubert_base_ls960 = "https://dl.fbaipublicfiles.com/hubert/hubert_base_ls960.pt"
    url_hubert_base_ls960_L9_km500 = "https://dl.fbaipublicfiles.com/hubert/hubert_base_ls960_L9_km500.bin"
    # Download
    hubert_base_ls960 = os.path.join(here, "hubert_base_ls960.pt")
    if not Path(hubert_base_ls960).exists():
        wget.download(url_hubert_base_ls960, hubert_base_ls960, bar=bar_custom)
    hubert_base_ls960_L9_km500 = os.path.join(here, "hubert_base_ls960_L9_km500.bin")
    if not Path(hubert_base_ls960_L9_km500).exists():
        wget.download(url_hubert_base_ls960_L9_km500, hubert_base_ls960_L9_km500, bar=bar_custom)
    # Init
    wav2vec = HubertWithKmeans(  # type: ignore
        checkpoint_path=hubert_base_ls960,
        kmeans_path=hubert_base_ls960_L9_km500,
    )
    return wav2vec


def bar_custom(current: int, total: int, width: int = 80) -> None:
    """Progress bar for wget downloads."""
    print(f"Downloading: {current / total * 100}% [{current}/ {total}] bytes")
