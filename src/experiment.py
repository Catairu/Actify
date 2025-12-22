import hydra
from omegaconf import DictConfig, OmegaConf
import lightning as lit
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import GroupKFold
import numpy as np
import wandb
import os

from net import get_model
from net.models.convae_lstm import ConvAE, LSTMClassifier
from dataset.loader import load_har


def get_num_params(module):
    """Returns the number of parameters in a Lightning module."""
    return sum(p.numel() for p in module.parameters())


def train_standard_model(cfg, train_loader, val_loader, wandb_logger, fold_idx=None):
    """Handles standard training (e.g. CNN-LSTM)."""
    model = get_model(cfg.net)

    filename_suffix = f"fold{fold_idx}" if fold_idx is not None else "final"

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=cfg.net.callbacks.get("patience", 8),
            mode="min",
        ),
        ModelCheckpoint(
            monitor="val_acc",
            mode="max",
            filename=f"{cfg.net.name}-{filename_suffix}-{{epoch}}-{{val_acc:.4f}}",
            save_top_k=1,
        ),
    ]

    trainer = Trainer(
        logger=wandb_logger,
        callbacks=callbacks,
        deterministic=True,
        **cfg.net.trainer,
    )

    trainer.fit(model, train_loader, val_loader)

    best_path = trainer.checkpoint_callback.best_model_path
    best_model = type(model).load_from_checkpoint(best_path, cfg=cfg.net)

    return best_model, trainer


def train_convae_two_stage(cfg, train_loader, val_loader, wandb_logger, fold_idx=None):
    """
    Handles two-stage training:
    1. Train the Autoencoder
    2. Freeze the Encoder -> Train the LSTM classifier
    """
    prefix = f"fold{fold_idx}" if fold_idx is not None else "final"

    print(f"[{prefix}] Phase 1: Training ConvAE...")
    conv_ae = ConvAE(cfg.net)

    ae_checkpoint = ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        filename=f"AE-{prefix}-{{epoch}}-{{val_loss:.4f}}",
    )
    ae_earlystop = EarlyStopping(
        monitor="val_loss",
        patience=cfg.net.callbacks.get("patience_AE", 20),
        mode="min",
    )

    ae_trainer = Trainer(
        logger=wandb_logger,
        callbacks=[ae_checkpoint, ae_earlystop],
        deterministic=True,
        **cfg.net.trainer,
    )

    ae_trainer.fit(conv_ae, train_loader, val_loader)

    best_ae_path = ae_checkpoint.best_model_path
    conv_ae = ConvAE.load_from_checkpoint(best_ae_path, cfg=cfg.net)
    pretrained_encoder = conv_ae.encoder

    for param in pretrained_encoder.parameters():
        param.requires_grad = False

    print(f"[{prefix}] Phase 2: Training LSTM Classifier...")
    lstm_model = LSTMClassifier(cfg.net, pretrained_encoder)

    lstm_checkpoint = ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        filename=f"LSTM-{prefix}-{{epoch}}-{{val_acc:.4f}}",
    )
    lstm_earlystop = EarlyStopping(
        monitor="val_loss",
        patience=cfg.net.callbacks.get("patience_LSTM", 20),
        mode="min",
    )

    lstm_trainer = Trainer(
        logger=wandb_logger,
        callbacks=[lstm_checkpoint, lstm_earlystop],
        deterministic=True,
        **cfg.net.trainer,
    )

    lstm_trainer.fit(lstm_model, train_loader, val_loader)

    best_lstm_path = lstm_checkpoint.best_model_path
    best_lstm_model = LSTMClassifier.load_from_checkpoint(
        best_lstm_path, pretrained_encoder=pretrained_encoder, cfg=cfg.net
    )

    return best_lstm_model, lstm_trainer


@hydra.main(config_path="../cfg", config_name="base", version_base=None)
def main(cfg: DictConfig):
    seed_everything(cfg.seed, workers=True)
    wandb_logger = WandbLogger(**cfg.wandb)

    hyperparams_dict = OmegaConf.to_container(cfg, resolve=True)
    wandb_logger.log_hyperparams(hyperparams_dict)

    train_dataset, test_loader = load_har(**cfg.net.dataset, load_all=True, cfg=cfg)

    k = cfg.get("k_folds", 5)
    groups = train_dataset.subject_ids.numpy()
    gkf = GroupKFold(n_splits=k)

    fold_accs = []
    dummy_X = np.zeros(len(train_dataset))

    is_two_stage = (
        "convae" in cfg.net.get("name", "").lower()
        or cfg.get("two_stage_training", False)
    )

    print(f"\n>>> PIPELINE STARTED. Two-Stage Mode: {is_two_stage} (K={k})")

    for fold, (train_idx, val_idx) in enumerate(
        gkf.split(X=dummy_X, y=None, groups=groups)
    ):
        print(f"\n============ STARTING FOLD {fold + 1}/{k} ============")

        train_ds = Subset(train_dataset, train_idx)
        val_ds = Subset(train_dataset, val_idx)

        train_loader = DataLoader(
            train_ds,
            batch_size=cfg.net.dataset.batch_size,
            shuffle=True,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=cfg.net.dataset.batch_size,
        )

        if is_two_stage:
            best_model, trainer = train_convae_two_stage(
                cfg, train_loader, val_loader, wandb_logger, fold_idx=fold
            )
        else:
            best_model, trainer = train_standard_model(
                cfg, train_loader, val_loader, wandb_logger, fold_idx=fold
            )

        val_metrics = trainer.validate(best_model, val_loader)[0]
        acc = val_metrics["val_acc"]
        fold_accs.append(acc)

        print(f"Fold {fold + 1} Result: val_acc = {acc:.4f}")
        wandb_logger.log_metrics(
            {f"fold_{fold}_best_val_acc": acc, "fold": fold}
        )

    avg_val_acc = np.mean(fold_accs)
    std_val_acc = np.std(fold_accs)

    print("\n============ K-FOLD FINISHED ============")
    print(f"Average K-fold val_acc = {avg_val_acc:.4f} ± {std_val_acc:.4f}")

    wandb_logger.log_metrics(
        {
            "kfold_avg_val_acc": avg_val_acc,
            "kfold_std_val_acc": std_val_acc,
        }
    )

    print("\n============ TRAINING FINAL MODEL ON FULL TRAIN SET ============")

    full_train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.net.dataset.batch_size,
        shuffle=True,
    )

    if is_two_stage:
        final_model, final_trainer = train_convae_two_stage(
            cfg,
            full_train_loader,
            test_loader,
            wandb_logger,
            fold_idx=None,
        )
    else:
        final_model = get_model(cfg.net)
        final_trainer = Trainer(
            logger=wandb_logger,
            deterministic=True,
            **cfg.net.trainer,
        )
        final_trainer.fit(final_model, full_train_loader)

    print("\n============ FINAL TEST EVALUATION ============")
    test_results = final_trainer.test(final_model, test_loader)
    print(test_results)

    wandb_logger.log_metrics(
        {"final_test_acc": test_results[0]["test_acc"]}
    )

    wandb.config.update(
        {
            "num_params": get_num_params(final_model),
            "pipeline_mode": "two_stage" if is_two_stage else "standard",
        },
        allow_val_change=True,
    )

    wandb.finish()


if __name__ == "__main__":
    os.environ["HYDRA_FULL_ERROR"] = "1"
    main()
