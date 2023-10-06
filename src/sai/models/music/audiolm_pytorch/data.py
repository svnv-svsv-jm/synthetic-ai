import typing as ty
from loguru import logger

from pathlib import Path
from functools import wraps

from beartype import beartype
from beartype.typing import Tuple, Union, Optional
from beartype.door import is_bearable

from einops import rearrange, reduce

import torchaudio
from torchaudio.functional import resample
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader

from .utils import curtail_to_multiple


# helper functions


def exists(val):
    return val is not None


def cast_tuple(val, length=1):
    return val if isinstance(val, tuple) else ((val,) * length)


def is_unique(arr):
    return len(set(arr)) == len(arr)


# dataset functions


class SoundDataset(Dataset):
    """SoundDataset."""

    @beartype
    def __init__(
        self,
        folder: str,
        target_sample_hz: Union[int, Tuple[int, ...]],
        exts: ty.Sequence[str] = ["flac", "wav", "mp3", "webm"],
        max_length: Optional[int] = None,
        seq_len_multiple_of: Optional[Union[int, Tuple[Optional[int], ...]]] = None,
    ) -> None:
        """
        Args:
            folder (str): _description_
            target_sample_hz (Union[ int, Tuple[int, ...] ]):
                target sample hz must be specified, or a tuple of them if one wants to return multiple resampled
            oratupleofthemifonewantstoreturnmultipleresampledexts (ty.Sequence[str], optional): _description_. Defaults to ["flac", "wav", "mp3", "webm"].
            max_length (Optional[ int ], optional):
                max length would apply to the highest target_sample_hz, if there are multiple. Defaults to None.
            iftherearemultipleseq_len_multiple_of (Optional[Union[int, Tuple[Optional[int], ...]]], optional): _description_. Defaults to None.
        """
        super().__init__()
        path = Path(folder)
        assert path.exists(), f"Folder {folder} does not exist."

        files = [file for ext in exts for file in path.glob(f"**/*.{ext}")]
        assert len(files) > 0, "No sound files found!"

        self.files = files

        self.max_length = max_length
        self.target_sample_hz = cast_tuple(target_sample_hz)
        num_outputs = len(self.target_sample_hz)

        # strategy, if there are multiple target sample hz, would be to resample to the highest one first
        # apply the max lengths, and then resample to all the others

        self.max_target_sample_hz = max(self.target_sample_hz)
        self.seq_len_multiple_of = cast_tuple(seq_len_multiple_of, num_outputs)

        assert len(self.target_sample_hz) == len(self.seq_len_multiple_of)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx: int):
        file = self.files[idx]

        data, sample_hz = torchaudio.load(file)
        assert isinstance(data, torch.Tensor)
        assert (
            data.size(0) > 0
        ), f"data.size()={data.size()}! One of your audio file ({file}) is empty. please remove it from your folder."
        assert (
            data.numel() > 0
        ), f"One of your audio file ({file}) is empty. please remove it from your folder."

        if data.shape[0] > 1:
            # the audio has more than 1 channel, convert to mono
            data = reduce(data, "c ... -> 1 ...", "mean")

        # first resample data to the max target freq

        data = resample(data, sample_hz, self.max_target_sample_hz)
        sample_hz = self.max_target_sample_hz

        # then curtail or pad the audio depending on the max length

        max_length = self.max_length
        audio_length = data.size(1)

        if exists(max_length):
            if audio_length > max_length:
                max_start = audio_length - max_length
                start = torch.randint(0, max_start, (1,))
                data = data[:, start : start + max_length]
            else:
                data = F.pad(data, (0, max_length - audio_length), "constant")

        data = rearrange(data, "1 ... -> ...")

        # resample if target_sample_hz is not None in the tuple

        num_outputs = len(self.target_sample_hz)
        data = cast_tuple(data, num_outputs)

        data_tuple = tuple(
            resample(d, sample_hz, target_sample_hz)
            for d, target_sample_hz in zip(data, self.target_sample_hz)
        )

        output: ty.List[torch.Tensor] = []

        # process each of the data resample at different frequencies individually for curtailing to multiple

        for data, seq_len_multiple_of in zip(data_tuple, self.seq_len_multiple_of):
            if exists(seq_len_multiple_of):
                data = curtail_to_multiple(data, seq_len_multiple_of)

            output.append(data.float())

        # cast from list to tuple

        output = tuple(output)

        # return only one audio, if only one target resample freq

        if num_outputs == 1:
            x = output[0]
            logger.trace(f"Sample: {x.size()}")
            return output[0]

        return output


# dataloader functions


def collate_one_or_multiple_tensors(fn):
    @wraps(fn)
    def inner(data):
        is_one_data = not isinstance(data[0], tuple)

        if is_one_data:
            data = fn(data)
            return (data,)

        outputs = []
        for datum in zip(*data):
            if is_bearable(datum, Tuple[str, ...]):
                output = list(datum)
            else:
                output = fn(datum)

            outputs.append(output)

        return tuple(outputs)

    return inner


@collate_one_or_multiple_tensors
def curtail_to_shortest_collate(data: ty.List[torch.Tensor]) -> torch.Tensor:
    """Curtails to shortest sequence."""
    min_len = min(*[datum.shape[0] for datum in data])
    data = [datum[:min_len] for datum in data]
    return torch.stack(data)


@collate_one_or_multiple_tensors
def pad_to_longest_fn(data: ty.List[torch.Tensor]) -> torch.Tensor:
    """Pads to longest sequence."""
    return pad_sequence(data, batch_first=True)


def get_dataloader(ds: Dataset, pad_to_longest: bool = True, **kwargs: ty.Any) -> DataLoader:
    """Creates DataLoader from Dataset with custom `collate_fn`."""
    collate_fn = pad_to_longest_fn if pad_to_longest else curtail_to_shortest_collate
    return DataLoader(ds, collate_fn=collate_fn, **kwargs)
