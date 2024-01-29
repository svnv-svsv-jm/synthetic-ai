__all__ = ["CategoricalLSTMModel", "TimeGAN"]

from loguru import logger
import typing as ty, os
import torch
import torch.nn as nn
from torch import Tensor
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from lightning.pytorch.loggers import TensorBoardLogger
import lightning.pytorch.callbacks as cb
import lightning.pytorch as pl
from torchmetrics import Accuracy, Metric, AUROC
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

from sai.utils import find_logger, f_one_hot


class TransformerLanguageModel(nn.Module):
    """TransformerLanguageModel."""

    def __init__(
        self,
        vocab_size: int,
        embedding_size: int = None,
        num_layers: int = 4,
        nhead: int = 4,
        out: int = None,
        padding_idx: ty.Optional[int] = 0,
        input_is_one_hot: bool = True,
        num_decoder_layers: int = 6,
    ):
        """_summary_
        Args:
            vocab_size (int): _description_
            embedding_size (int, optional): _description_. Defaults to None.
            num_layers (int, optional): _description_. Defaults to 4.
            nhead (int, optional): _description_. Defaults to 4.
            out (int, optional): Output size. Defaults to `vocab_size`.
            padding_idx (ty.Optional[int], optional): _description_. Defaults to 0.
        """
        super().__init__()
        self.input_is_one_hot = input_is_one_hot
        self.fc: nn.Module
        embedding_size = int(nhead * 4)
        if not self.input_is_one_hot:
            self.vocab_size = vocab_size + 1
            self.embedding = nn.Embedding(self.vocab_size, embedding_size, padding_idx=padding_idx)
        else:
            self.vocab_size = vocab_size
            self.preprocess = nn.Linear(self.vocab_size, embedding_size)
        self.transformer = nn.Transformer(
            d_model=embedding_size,
            nhead=nhead,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_decoder_layers,
            batch_first=True,
        )
        if out is None:
            out = self.vocab_size
        if self.input_is_one_hot:
            self.fc = nn.Sequential(nn.Linear(embedding_size, out), nn.Softmax(dim=-1))
        else:
            self.fc = nn.Linear(embedding_size, out)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass."""
        B = x.size(0)
        L = x.size(1)
        if not self.input_is_one_hot:
            embedded: Tensor = self.embedding(x)
            embedded = embedded.squeeze(2)
        else:
            embedded = self.preprocess(x.float())
        logger.trace(f"embedded: {embedded.size()} | requires_grad={embedded.requires_grad}")
        output: Tensor = self.transformer(embedded, embedded)
        logger.trace(f"output: {output.size()} | requires_grad={output.requires_grad}")
        output = output.view(B, L, -1)
        logger.trace(f"output: {output.size()} | requires_grad={output.requires_grad}")
        output = self.fc(output)
        logger.trace(f"output: {output.size()} | requires_grad={output.requires_grad}")
        return output


class CategoricalLSTMModel(pl.LightningModule):
    """Time-series classifier."""

    def __init__(
        self,
        lr: float = 1e-5,
        checkpoint_kwargs: ty.Dict[str, ty.Any] = {},
        **kwargs: ty.Any,
    ) -> None:
        """
        Args:
            lr (float, optional): Learning rate. Defaults to 1e-5.
        """
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.model = TransformerLanguageModel(**kwargs)
        self.loss = nn.CrossEntropyLoss()
        self.checkpoint_kwargs = checkpoint_kwargs
        # metrics
        num_classes = self.model.vocab_size
        task = "multiclass" if num_classes > 2 else "binary"
        self.train_accuracy: Metric = Accuracy(task, num_classes=num_classes)  # type: ignore
        self.val_accuracy: Metric = Accuracy(task, num_classes=num_classes)  # type: ignore
        self.test_accuracy: Metric = Accuracy(task, num_classes=num_classes)  # type: ignore
        self.train_auroc: Metric = AUROC(task, num_classes=num_classes)  # type: ignore
        self.val_auroc: Metric = AUROC(task, num_classes=num_classes)  # type: ignore
        self.test_auroc: Metric = AUROC(task, num_classes=num_classes)  # type: ignore

    def forward(self, x: Tensor) -> Tensor:
        x = self.model(x)
        return x

    def training_step(self, batch: Tensor, batch_idx: int = None) -> Tensor:
        return self.step(batch, "train")

    def validation_step(self, batch: Tensor, batch_idx: int = None) -> Tensor:
        return self.step(batch, "val")

    def test_step(self, batch: Tensor, batch_idx: int = None) -> Tensor:
        return self.step(batch, "test")

    def on_train_epoch_start(self) -> None:
        self.log_generated_sequence(tag="train")

    def on_validation_epoch_start(self) -> None:
        self.log_generated_sequence(tag="val")

    def on_test_epoch_start(self) -> None:
        self.log_generated_sequence(tag="test")

    def configure_optimizers(self) -> ty.Dict[str, ty.Any]:  # type: ignore
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer,
            mode="min",
            verbose=True,
            patience=2,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "loss/train",
            },
        }

    def configure_callbacks(self) -> ty.List[pl.Callback]:  # type: ignore
        """Configure checkpoint."""
        callbacks = []
        self.checkpoint_kwargs.setdefault("monitor", "loss/train")
        self.checkpoint_kwargs.setdefault("mode", "min")
        self.checkpoint_kwargs.setdefault("save_top_k", 3)
        self.checkpoint_kwargs.setdefault("save_last", True)
        self.checkpoint_kwargs.setdefault("save_on_train_epoch_end", True)
        logger.info(f"Creating {cb.ModelCheckpoint} with parameters: {self.checkpoint_kwargs}")
        ckpt_cb_val: pl.Callback = cb.ModelCheckpoint(**self.checkpoint_kwargs)
        callbacks.append(ckpt_cb_val)
        early = cb.EarlyStopping(
            monitor="loss/train",
            patience=5,
            mode="min",
            check_on_train_epoch_end=True,
            strict=False,
        )
        callbacks.append(early)
        return callbacks

    def __step__(
        self,
        batch: Tensor,  # has shape (B, L, D)
    ) -> ty.Tuple[Tensor, Tensor, Tensor]:
        B = batch.size(0)
        L = batch.size(1)
        D = batch.size(2)
        logger.trace(f"Batch: {B, L, D}")
        inputs = batch[:, 0 : L - 1, :].reshape(B, L - 1, D)
        targets = batch[:, 1:L, :].reshape(B, L - 1, D)
        logger.trace(f"inputs={inputs.size()} | targets={targets.size()}")
        outputs: Tensor = self.model(inputs)
        outputs = outputs.reshape(-1, self.model.vocab_size).float()
        if self.model.input_is_one_hot:
            targets = targets.reshape(-1, self.model.vocab_size)
            targets = targets.argmax(dim=-1).reshape(-1)
        else:
            targets = targets.reshape(-1)
        targets = targets.long()
        logger.trace(f"outputs={outputs.size()} | targets={targets.size()}")
        loss: Tensor = self.loss(outputs, targets)
        return loss, outputs, targets

    def step(
        self,
        batch: Tensor,  # has shape (B, L, D)
        tag: str = "train",
    ) -> Tensor:
        """_summary_
        Args:
            batch (Tensor): _description_
        Returns:
            Tensor: _description_
        """
        # Get loss and log
        loss, outputs, targets = self.__step__(batch)
        self.log(f"loss/{tag}", loss)
        # Log accuracy metrics
        if tag.lower() in ["train", "training"]:
            acc = self.train_accuracy
            auc = self.train_auroc
        elif tag.lower() in ["val", "validation"]:
            acc = self.val_accuracy
            auc = self.val_auroc
        else:
            acc = self.test_accuracy
            auc = self.test_auroc
        acc.update(outputs, targets)  # type: ignore
        auc.update(outputs, targets)  # type: ignore
        self.log(f"accuracy/{tag}", acc)
        self.log(f"auroc/{tag}", auc)
        return loss

    def generate_sequence(
        self,
        start_token: ty.Union[int, ty.List[int]] = None,
        length: int = 45,
        save_dir: str = None,
    ) -> Tensor:
        """Generating sequences using the trained model."""
        if start_token is None:
            start_token = torch.randint(low=0, high=int(self.model.vocab_size) - 1, size=(4,)).tolist()
        if isinstance(start_token, int):
            start_token = [start_token]
        if save_dir is not None:
            Path.mkdir(Path(save_dir), exist_ok=True, parents=True)
        all_seqs = torch.Tensor([])
        logger.debug(f"Start token(s): {start_token}")
        for i, s in enumerate(start_token):
            generated_sequence = [s]
            for _ in range(length - 1):
                input_seq = torch.tensor(generated_sequence, dtype=torch.long).unsqueeze(0)
                if self.model:
                    input_seq = f_one_hot(input_seq, num_classes=self.model.vocab_size)
                output = self(input_seq)
                next_token = output[0, -1, :].argmax().item()
                generated_sequence.append(next_token)
            gen_seq = Tensor(generated_sequence)
            all_seqs = torch.cat((all_seqs, gen_seq.unsqueeze(0)), dim=0)
            if save_dir is not None:
                name = f"gen={i}-token={s}"
                f = os.path.join(save_dir, f"{name}.pt")
                torch.save(gen_seq, f)
                f = os.path.join(save_dir, f"{name}.csv")
                df = pd.DataFrame(gen_seq.view(-1, 1).cpu().numpy())
                df.to_csv(f, header=False, index=False)
        logger.debug(f"Returning sequences ({all_seqs.size()})")
        return all_seqs

    def log_generated_sequence(self, tag: str) -> None:
        """Logs the generated sequences."""
        # Return if no trainer
        if hasattr(self, "trainer") and isinstance(self.trainer, pl.Trainer):
            global_step = self.trainer.global_step
        else:
            logger.debug("No trainer.")
            return
        # Return if no logger
        pl_logger: TensorBoardLogger = find_logger(self.trainer, logger_type=TensorBoardLogger)  # type: ignore
        if pl_logger is None:
            logger.debug("No SummaryWriter.")
            return
        writer: SummaryWriter = pl_logger.experiment
        # Generate sequence
        logger.debug(f"Generating sequences for {tag}")
        start_token = torch.randint(low=0, high=int(self.model.vocab_size) - 1, size=(4,)).tolist()
        save_dir = os.path.join(pl_logger.log_dir, f"global_step={global_step}")
        seqs = self.generate_sequence(start_token=start_token, length=45, save_dir=save_dir)
        logger.debug(f"Got {seqs.size()}")
        fig, axs = plt.subplots(len(start_token))
        fig.suptitle("Generated-sequences")
        for i in range(len(axs)):
            seq = seqs[i]
            logger.debug(f"Plotting ({i}): {seq}")
            axs[i].plot(seq.cpu().numpy(), linestyle="dotted", marker="*")
            axs[i].set_ylim([0, self.model.vocab_size])
        writer.add_figure(  # type: ignore
            f"Generated-sequence/{tag}",
            fig,
            global_step=global_step,
        )


class TimeGAN(pl.LightningModule):
    """TimeGAN (with transformers)."""

    def __init__(
        self,
        lr: float = 1e-5,
        n_critic: int = 3,
        checkpoint_kwargs: ty.Dict[str, ty.Any] = {},
        **kwargs: ty.Any,
    ) -> None:
        """
        Args:
            lr (float, optional):
                Learning rate. Defaults to 1e-5.
            n_critic (int, optional):
                Frequency for the Discriminator training. See :method:`lightning.pytorch.LightningModule.configure_optimizers()` method's documentation. Defaults to 5.
        """
        super().__init__()
        self.save_hyperparameters()
        self.checkpoint_kwargs = checkpoint_kwargs
        self.lr = lr
        self.n_critic = n_critic
        self.generator = TransformerLanguageModel(**kwargs)
        self.discriminator = TransformerLanguageModel(out=2, padding_idx=None, **kwargs)
        self.loss = nn.BCELoss()
        self.B: int
        self.L: int
        self.D: int
        # Important: This property activates manual optimization.
        self.automatic_optimization = False

    def forward(
        self,
        x: ty.Union[Tensor, ty.Sequence[int], torch.Size] = None,
        batch_size: int = 1,
        sequence_length: int = None,
    ) -> Tensor:
        """
        Args:
            x (Tensor | Sequence[int] | torch.Size, optional):
                If `Tensor`, only needed to get its size. Otherwise just input a `Sequence[int] | torch.Size` object.
            batch_size (int, optional):
                Batch size. Defaults to 1.
            sequence_length (int, optional):
                Sequence length. Defaults to None.
        Returns:
            Tensor: Generated sequence.
        """
        self.D = self.discriminator.vocab_size
        if x is None:
            if batch_size is None:
                batch_size = self.B
            x = torch.zeros(size=(batch_size, self.L if sequence_length is None else sequence_length, self.D))
        elif not isinstance(x, Tensor):
            x = torch.zeros(size=x)
        noise = torch.randint_like(x, low=0, high=self.D).to(x.device)  # type: ignore
        X: Tensor = self.generator(noise)
        X = X.softmax(dim=-1)
        return X

    def generate(
        self,
        pad_value: int = None,
        to_df: bool = True,
        columns: ty.Sequence[str] = None,
        **kwargs: ty.Any,
    ) -> ty.Union[Tensor, pd.DataFrame]:
        """Generate a sequence of tokens.

        If a `pad_value` is encountered, all subsequent generated values are also set equal to `pad_value`, as no original sample ends with anything else than `pad_value`.
        """
        X: Tensor = self(**kwargs)  # (B, L, D)
        X = X.argmax(dim=-1)  # (B, L)
        if pad_value is not None:
            for i, x in enumerate(X):  # x: (L,)
                idx = (x == pad_value).nonzero(as_tuple=True)[0]
                if len(idx) > 0:
                    index = idx[0]
                    X[i, index:] = pad_value
        if to_df:
            df_ = torch.Tensor([])
            for i, x in enumerate(X):
                idx = torch.ones_like(x)
                g = torch.cat((idx.view(-1, 1), x.view(-1, 1)), dim=-1)
                df_ = torch.cat((df_, g), dim=0)
            n_cols = df_.size(-1)
            if columns is None:
                columns = [f"col_{i}" for i in range(n_cols)]
            df = pd.DataFrame(df_.detach().cpu().numpy(), columns=columns)
            return df
        return X

    def configure_callbacks(self) -> ty.List[pl.Callback]:  # type: ignore
        """Configure checkpoint."""
        callbacks = []
        self.checkpoint_kwargs.setdefault("monitor", "loss/train")
        self.checkpoint_kwargs.setdefault("mode", "min")
        self.checkpoint_kwargs.setdefault("save_top_k", 3)
        self.checkpoint_kwargs.setdefault("save_last", True)
        self.checkpoint_kwargs.setdefault("save_on_train_epoch_end", True)
        logger.info(f"Creating {cb.ModelCheckpoint} with parameters: {self.checkpoint_kwargs}")
        ckpt_cb_val: pl.Callback = cb.ModelCheckpoint(**self.checkpoint_kwargs)
        callbacks.append(ckpt_cb_val)
        early = cb.EarlyStopping(
            monitor="loss-G/train",
            patience=5,
            mode="min",
            check_on_train_epoch_end=True,
            strict=False,
        )
        callbacks.append(early)
        return callbacks

    def configure_optimizers(self) -> tuple:
        """Optimizers."""
        opt_g = torch.optim.Adam(self.generator.parameters(), lr=self.lr)
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=self.lr)
        return (
            {
                "optimizer": opt_d,
                "frequency": self.n_critic,
                "lr_scheduler": {
                    "scheduler": optim.lr_scheduler.ReduceLROnPlateau(opt_d),
                    "monitor": "loss-D/train",
                    "strict": False,
                },
            },
            {
                "optimizer": opt_g,
                "frequency": 1,
                "lr_scheduler": {
                    "scheduler": optim.lr_scheduler.ReduceLROnPlateau(opt_g),
                    "monitor": "loss-G/train",
                    "strict": False,
                },
            },
        )

    def training_step(self, batch: Tensor, batch_idx: int) -> Tensor:
        """Implementation follows the PyTorch tutorial: https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html"""
        D_opt, G_opt = self.configure_optimizers()
        for optimizer_idx, opt in enumerate((D_opt, G_opt)):
            # Skip to next optimizer if frequency is not right
            frequency: int = opt["frequency"]
            if batch_idx % frequency != 0:
                continue
            # Run optimization step
            loss = self.step(batch, "train", optimizer_idx)
            optimizer: torch.optim.Adam = opt["optimizer"]
            optimizer.zero_grad()
            self.manual_backward(loss)
            optimizer.step()
        return loss

    def on_train_epoch_end(self) -> None:
        """Call schedulers."""
        D_opt, G_opt = self.configure_optimizers()
        for _, opt in enumerate((D_opt, G_opt)):
            scheduler: optim.lr_scheduler.ReduceLROnPlateau = opt["lr_scheduler"]["scheduler"]
            monitor = opt["lr_scheduler"]["monitor"]
            try:
                scheduler.step(self.trainer.callback_metrics[monitor])
            except Exception as ex:
                logger.warning(ex)

    def validation_step(self, batch: Tensor, batch_idx: int) -> Tensor:
        return self.step(batch, "val")

    def test_step(self, batch: Tensor, batch_idx: int) -> Tensor:
        return self.step(batch, "test")

    def step(
        self,
        batch: Tensor,  # has shape (B, L, D)
        tag: str,
        optimizer_idx: int = None,
    ) -> Tensor:
        """Common step."""
        self.B, self.L, self.D = batch.size()
        logger.trace(f"Batch: {batch.size()}")
        # Discriminator update
        if optimizer_idx == 0:
            logger.trace("Discriminator update...")
            # Loss on real
            loss_real = self.loss_on_real(batch)
            # Loss on fake
            fake: Tensor = self(batch)
            output = self.discriminator(fake.detach())
            logger.trace(f"Output on fake: {output.size()}|{output.requires_grad}")
            label = torch.zeros_like(output).to(output.device)
            loss_fake: Tensor = self.loss(output, label)
            logger.trace(f"Loss fake: {loss_fake}")
            loss = loss_real + loss_fake
            self.log(f"loss-D/{tag}", loss, prog_bar=True)
            return loss
        # Generator update
        logger.trace("Generator update...")
        fake = self(batch.size())
        output = self.discriminator(fake)
        label = torch.ones_like(output).to(output.device)
        loss_gen: Tensor = self.loss(output, label)
        self.log(f"loss-G/{tag}", loss_gen, prog_bar=True)
        # If val or test, also evaluate the D's loss over original data
        if optimizer_idx is None:
            loss_real = self.loss_on_real(batch)
            self.log(f"loss-D/{tag}", loss_real)
        # Total loss
        loss = loss_gen
        return loss

    def loss_on_real(self, batch: Tensor) -> Tensor:
        """Loss on real data."""
        output: Tensor = self.discriminator(batch)
        logger.trace(f"Output on real: {output.size()}|{output.requires_grad}")
        label = torch.ones_like(output).to(output.device)
        loss_real: Tensor = self.loss(output, label)
        logger.trace(f"Loss real: {loss_real}")
        return loss_real
