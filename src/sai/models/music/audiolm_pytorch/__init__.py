import torch
from packaging import version

if version.parse(torch.__version__) >= version.parse("2.0.0"):
    from einops._torch_specific import allow_ops_in_compiled_graph

    allow_ops_in_compiled_graph()  # type: ignore

from .soundstream import SoundStream, AudioLMSoundStream, MusicLMSoundStream
from .encodec import EncodecWrapper

from .audiolm_pytorch import (
    AudioLM,
    SemanticTransformer,
    CoarseTransformer,
    FineTransformer,
    FineTransformerWrapper,
    CoarseTransformerWrapper,
    SemanticTransformerWrapper,
)

from .hubert_kmeans import HubertWithKmeans

from .trainer import (
    SoundStreamTrainer,
    SemanticTransformerTrainer,
    FineTransformerTrainer,
    CoarseTransformerTrainer,
)

from .audiolm_pytorch import get_embeds
