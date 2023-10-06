# pylint: disable=no-member
import pytest
from loguru import logger
import sys
import typing as ty

from sai.models.music.audiolm_pytorch import (
    SoundStream,
    SoundStreamTrainer,
    SemanticTransformer,
    SemanticTransformerTrainer,
)
from sai.models.music.hubert import download_hubert


def test_semantic_transformer() -> None:
    """Test we can initialize the model."""
    wav2vec = download_hubert()
    semantic_transformer = SemanticTransformer(
        num_semantic_tokens=wav2vec.codebook_size,
        dim=1024,
        depth=6,
        flash_attn=True,
    )
    trainer = SemanticTransformerTrainer(
        transformer=semantic_transformer,
        wav2vec=wav2vec,
        folder=".data/music_data",
        batch_size=4,
        data_max_length=320 * 32,
        num_train_steps=1,
        accelerate_kwargs=dict(device_placement=False, cpu=True),
    )
    trainer.train()


def test_soundstream() -> None:
    """Test we can initialize the model."""
    # Model
    soundstream = SoundStream(
        codebook_size=1024,
        rq_num_quantizers=8,
        rq_groups=2,  # this paper proposes using multi-headed residual vector quantization - https://arxiv.org/abs/2305.02765
        attn_window_size=128,  # local attention receptive field at bottleneck
        attn_depth=2,  # 2 local attention transformer blocks - the soundstream folks were not experts with attention, so i took the liberty to add some. encodec went with lstms, but attention should be better
    ).cpu()
    # Trainer
    trainer = SoundStreamTrainer(
        soundstream,
        folder=".data/music_data",
        batch_size=4,
        grad_accum_every=8,  # effective batch size of 32
        data_max_length_seconds=2,  # train on 2 second audio
        num_train_steps=1,
        accelerate_kwargs=dict(device_placement=False, cpu=True),
    )
    trainer.train()


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="TRACE")
    pytest.main([__file__, "-x", "-s", "--pylint"])
