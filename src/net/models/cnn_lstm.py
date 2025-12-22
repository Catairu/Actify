import torch
import lightning as lit
from torchmetrics import Accuracy

import torch.nn as nn
import torch.nn.functional as F
import wandb
from sklearn.metrics import classification_report
from hydra.utils import instantiate


class Cnn_Lstm(lit.LightningModule):
    def __init__(self, cfg):
        super(Cnn_Lstm, self).__init__()
        self.save_hyperparameters()
        self.cfg = cfg
        self.depth = cfg.depth

        self.embed = instantiate(cfg.embed)
        self.features = nn.Sequential(
            *[instantiate(cfg.block) for _ in range(self.depth - 1)]
        )
        self.lstm_block = instantiate(cfg.lstm_block)
        self.unembed = instantiate(cfg.unembed)

        self.train_acc = Accuracy(task="multiclass", num_classes=cfg.num_classes)
        self.val_acc = Accuracy(task="multiclass", num_classes=cfg.num_classes)
        self.test_acc = Accuracy(task="multiclass", num_classes=cfg.num_classes)

        self.class_names = [
            "Walking",
            "Upstairs",
            "Downstairs",
            "Sitting",
            "Standing",
            "Laying",
        ]
        self.test_preds = []
        self.test_labels = []

    def forward(self, x, return_cnn_features=False):
        x = self.embed(x)
        x = self.features(x)
        x = x.permute(0, 2, 1)
        lstm_out = self.lstm_block(x)
        logits = self.unembed(lstm_out)
        return logits

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", self.train_acc(y_hat, y), prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log("val_loss", loss)
        self.log("val_acc", self.val_acc(y_hat, y), prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)

        loss = F.cross_entropy(y_hat, y)
        self.log("test_loss", loss)
        self.log("test_acc", self.test_acc(y_hat, y), prog_bar=True)

        self.test_preds.append(torch.argmax(y_hat, dim=1))
        self.test_labels.append(y)

    def configure_optimizers(self):
        optimizer = instantiate(self.cfg.optimizer, self.parameters())
        return optimizer

    def on_test_epoch_end(self):
        all_preds = torch.cat(self.test_preds)
        all_labels = torch.cat(self.test_labels)

        if isinstance(self.logger, lit.pytorch.loggers.WandbLogger):
            self.logger.experiment.log(
                {
                    "confusion_matrix_interactive": wandb.plot.confusion_matrix(
                        probs=None,
                        y_true=all_labels.flatten().tolist(),
                        preds=all_preds.flatten().tolist(),
                        class_names=self.class_names,
                        title="Confusion Matrix Final",
                    )
                }
            )

        self.test_preds.clear()
        self.test_labels.clear()
