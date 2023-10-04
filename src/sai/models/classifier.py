__all__ = ["Classifier"]

from loguru import logger
import typing as ty

import torch
from torch import Tensor
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics import Accuracy, Metric, AUROC

from sai.nn import MLP


class Classifier(pl.LightningModule):
    """Dummy classifier."""

    def __init__(
        self,
        num_classes: int,
        loss: str = "bce",
        lr: float = 1e-5,
        **kwargs: ty.Any,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        kwargs.setdefault("last_activation", torch.nn.Softmax(dim=1))
        self.layers = MLP(out_features=num_classes, **kwargs)
        self.num_classes = num_classes
        # consider using LogSoftmax with NLLLoss instead of Softmax with CrossEntropyLoss
        task = "multiclass" if num_classes > 2 else "binary"
        self.loss: torch.nn.Module
        if isinstance(loss, str):
            if loss.lower() in ["nll", "nllloss", "nl_loss"]:
                self.loss = torch.nn.NLLLoss()
            elif loss.lower() in ["bce", "bceloss", "bce_loss"]:
                self.loss = torch.nn.BCELoss()
                task = "binary"
            else:
                self.loss = torch.nn.NLLLoss()
        elif isinstance(loss, torch.nn.Module):
            self.loss = loss
        else:
            raise TypeError(f"Unrecognized input for loss: {type(loss)}.")
        # metrics
        self.accuracy: Metric = Accuracy(task, num_classes=num_classes)  # type: ignore
        self.auroc: Metric = AUROC(task, num_classes=num_classes)  # type: ignore
        # others
        self.lr = lr

    def configure_optimizers(self) -> dict:  # type: ignore
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, verbose=True)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "loss/train",
                "strict": False,
            },
        }

    def forward(self, x: ty.Union[Tensor, ty.Tuple[Tensor, ty.Any]]) -> Tensor:  # type: ignore # pylint: disable=arguments-differ
        x = x[0] if isinstance(x, (tuple, list)) else x
        assert torch.is_tensor(x), f"x must be a tensor but found of type {type(x)}"  # type: ignore
        x_vectorized = x.view(x.size(0), -1)
        output: Tensor = self.layers(x_vectorized)
        return output

    def training_step(  # type: ignore  # pylint: disable=arguments-differ
        self,
        batch: ty.Tuple[Tensor, Tensor],
        batch_nb: int,  # pylint: disable=unused-argument
    ) -> Tensor:
        return self.step(batch, batch_nb, "train")

    def validation_step(  # type: ignore  # pylint: disable=arguments-differ
        self,
        batch: ty.Tuple[Tensor, Tensor],
        batch_nb: int,  # pylint: disable=unused-argument
    ) -> Tensor:
        return self.step(batch, batch_nb, "val")

    def test_step(  # type: ignore  # pylint: disable=arguments-differ
        self,
        batch: ty.Tuple[Tensor, Tensor],
        batch_nb: int,  # pylint: disable=unused-argument
    ) -> Tensor:
        return self.step(batch, batch_nb, "test")

    def step(
        self,
        batch: ty.Tuple[Tensor, Tensor],
        batch_nb: int,  # pylint: disable=unused-argument
        tag: str,
    ) -> Tensor:
        """Common step."""
        x, y = batch
        output: Tensor = self(x)
        target = y.view(-1, 1)
        logger.trace(f"Output: {output.size()}")
        logger.trace(f"Target: {target.size()}")
        index = target.long().view(-1)
        onehot: Tensor = F.one_hot(index, num_classes=self.num_classes)
        logger.trace(f"One-hot: {onehot.size()}")
        loss: Tensor = self.loss(output, onehot.float())
        preds = output.detach().argmax(1).view(-1, 1)
        with torch.no_grad():
            self.accuracy.update(preds, target)  # type: ignore
            self.auroc.update(preds, target)  # type: ignore
            self.log(f"loss/{tag}", loss, prog_bar=True)
            self.log(f"acc/{tag}", self.accuracy, prog_bar=True)
            self.log(f"auroc/{tag}", self.auroc, prog_bar=True)
        return loss
