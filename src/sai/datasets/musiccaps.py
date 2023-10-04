__all__ = ["MusicCaps", "load_musiccaps"]

from lightning.pytorch.utilities.types import TRAIN_DATALOADERS
from loguru import logger
import typing as ty

import subprocess
import os
from pathlib import Path
from datasets import load_dataset, Audio, Dataset
import lightning.pytorch as pl
import torch
from torch import Tensor
from torch.utils.data import random_split, Subset, DataLoader, Dataset as PytDataset


class MusicCapsCollateFn:
    """Collates batch from MusicCaps."""

    def __call__(self, batch: ty.List[Tensor]) -> Tensor:
        """Stack Tensors of different shapes."""
        logger.trace(f"Received {batch}")
        return torch.stack(batch)


class MusicCapsDataset(PytDataset):
    """."""

    def __init__(
        self,
        root: ty.Union[str, Path] = ".data/music_data",
        split: ty.Sequence[ty.Union[int, float]] = (0.6, 0.2, 0.2),
        num_proc: int = 4,
        sampling_rate: int = 44100,
        writer_batch_size: int = 1000,
        samples_to_load: int = None,
        **kwargs: ty.Any,
    ) -> None:
        # Inputs
        self.root = root
        self.num_proc = num_proc
        self.sampling_rate = sampling_rate
        self.writer_batch_size = writer_batch_size
        self.samples_to_load = samples_to_load
        self.split = split
        self.loader_kwargs = kwargs
        # Dataset
        self.ds = load_musiccaps(
            root=self.root,
            num_proc=self.num_proc,
            sampling_rate=self.sampling_rate,
            writer_batch_size=self.writer_batch_size,
            samples_to_load=self.samples_to_load,
        )

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int) -> ty.Dict[str, ty.Any]:
        sample: ty.Dict[str, ty.Any] = self.ds[idx]
        array = sample["audio"]["array"]
        tensor = torch.Tensor(array).float()
        sample["audio"]["array"] = tensor
        return sample


class MusicCaps(pl.LightningDataModule):
    """MusicCaps dataset."""

    def __init__(
        self,
        root: ty.Union[str, Path] = ".data/music_data",
        split: ty.Sequence[ty.Union[int, float]] = (0.6, 0.2, 0.2),
        num_proc: int = 4,
        sampling_rate: int = 44100,
        writer_batch_size: int = 1000,
        samples_to_load: int = None,
        **kwargs: ty.Any,
    ) -> None:
        """
        Args:
            split (str):
                Dataset split.
            num_proc (int, optional):
                How many processes to use for the loading. Defaults to 4.
            sampling_rate (int, optional):
                Sampling rate for the audio, keep in 44100. Defaults to 44100.
            writer_batch_size (int, optional):
                How many examples to keep in memory per worker. Reduce if OOM.. Defaults to 1000.
            samples_to_load (int):
                How many samples to load
        """
        super().__init__()
        # Inputs
        self.root = root
        self.num_proc = num_proc
        self.sampling_rate = sampling_rate
        self.writer_batch_size = writer_batch_size
        self.samples_to_load = samples_to_load
        self.split = split
        self.loader_kwargs = kwargs
        # Dataset
        self.dataset: MusicCapsDataset
        self.train: Subset
        self.val: Subset
        self.test: Subset
        # Collate fn for loaders
        self.collate_fn = None  # MusicCapsCollateFn()

    def prepare_data(self) -> None:
        """Downloads the data."""
        self.dataset = MusicCapsDataset(
            root=self.root,
            num_proc=self.num_proc,
            sampling_rate=self.sampling_rate,
            writer_batch_size=self.writer_batch_size,
            samples_to_load=self.samples_to_load,
        )

    def setup(self, stage: str = None) -> None:
        """Creates training, val, test splits."""
        if self.dataset is None:
            self.prepare_data()
        self.train, self.val, self.test = random_split(
            self.dataset,
            lengths=self.split,
        )

    def train_dataloader(self) -> DataLoader:
        return self._loader(self.train)

    def val_dataloader(self) -> DataLoader:
        return self._loader(self.train)

    def test_dataloader(self) -> DataLoader:
        return self._loader(self.train)

    def _loader(self, ds: Dataset) -> DataLoader:
        """Create DataLoader from Dataset."""
        return DataLoader(ds, collate_fn=self.collate_fn, **self.loader_kwargs)


def load_musiccaps(
    root: ty.Union[str, Path] = ".data/music_data",
    split: str = "train",
    num_proc: int = 4,
    sampling_rate: int = 44100,
    writer_batch_size: int = 1000,
    samples_to_load: int = None,
) -> Dataset:
    """
    Args:
        split (str):
            Dataset split.
        num_proc (int, optional):
            How many processes to use for the loading. Defaults to 4.
        sampling_rate (int, optional):
            Sampling rate for the audio, keep in 44100. Defaults to 44100.
        writer_batch_size (int, optional):
            How many examples to keep in memory per worker. Reduce if OOM.. Defaults to 1000.
        samples_to_load (int):
            How many samples to load
    Returns:
        Dataset: _description_
    """
    # Load dataset
    ds = load_dataset("google/MusicCaps", split=split)
    # Just select some samples
    if samples_to_load is not None:
        ds = ds.select(range(samples_to_load))
    # Use `.map()` to download the dataset
    ds = ds.map(
        Process(root),
        num_proc=num_proc,
        writer_batch_size=writer_batch_size,
        keep_in_memory=False,
    ).cast_column("audio", Audio(sampling_rate=sampling_rate))
    return ds


def download_clip(
    video_identifier: str,
    output_filename: str,
    start_time: str,
    end_time: str,
    num_attempts: int = 5,
    url_base: str = "https://www.youtube.com/watch?v=",
) -> ty.Tuple[bool, str]:
    """Downloads audio from one YouTube videoclip.
    Args:
        video_identifier (str):
            Video's ID.
        output_filename (str):
            Name of the output file.
        start_time (str):
            Start time of video.
        end_time (str):
            End time of video.
        num_attempts (int, optional):
            Number of attempts to download video. Defaults to 5.
        url_base (_type_, optional):
            Base URL of video identifier. Defaults to "https://www.youtube.com/watch?v=".
    Returns:
        ty.Tuple[bool, str]: status, message.
    """
    # Status
    status = False
    # Command
    command = f"""
        yt-dlp --quiet --no-warnings -x --audio-format wav -f bestaudio -o "{output_filename}" --download-sections "*{start_time}-{end_time}" {url_base}{video_identifier}
    """.strip()
    # Try to download
    attempts = 0
    while True:
        try:
            output = subprocess.check_output(command, shell=True, stderr=subprocess.STDOUT)
            logger.trace(output)
        except subprocess.CalledProcessError as err:
            attempts += 1
            if attempts == num_attempts:
                return status, err.output
        else:
            break
    # Check if the video was successfully saved.
    status = os.path.exists(output_filename)
    return status, "Downloaded"


class Process:
    """Callable to download and process audio files."""

    def __init__(
        self,
        data_dir: ty.Union[str, Path] = ".data/music_data",
    ) -> None:
        """
        Args:
            data_dir (ty.Union[str, Path], optional): _description_. Defaults to ".data/music_data".
        """
        self.data_dir = data_dir

    def __call__(
        self,
        example: ty.Dict[str, ty.Any],
    ) -> ty.Dict[str, ty.Any]:
        """
        Args:
            example (ty.Dict[str, ty.Any]): _description_
        Returns:
            ty.Dict[str, ty.Any]: _description_
        """
        if isinstance(self.data_dir, str):
            self.data_dir = Path(self.data_dir)
            self.data_dir.mkdir(exist_ok=True, parents=True)
        outfile_path = str(self.data_dir / f"{example['ytid']}.wav")
        if not os.path.exists(outfile_path):
            status = False
            status, log = download_clip(
                example["ytid"],
                outfile_path,
                example["start_s"],
                example["end_s"],
            )
            logger.trace(log)
        else:
            status = True
        example["audio"] = outfile_path
        example["download_status"] = status
        return example
