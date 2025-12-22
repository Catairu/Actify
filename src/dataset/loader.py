import torch
from torch.utils.data import Dataset, DataLoader, random_split
from pathlib import Path
import numpy as np


# ==========================================
#                LOADERS
# ==========================================


class HarDataset(Dataset):
    SIGNALS = [
        "body_acc_x",
        "body_acc_y",
        "body_acc_z",
        "body_gyro_x",
        "body_gyro_y",
        "body_gyro_z",
        "total_acc_x",
        "total_acc_y",
        "total_acc_z",
    ]

    def __init__(self, data_dir: Path, split: str = "train", transform=None):
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform

        self.X = self.load_signals()
        self.y = self.load_labels()
        self.subject_ids = self.load_subjects()

        if self.transform:
            self.X = self.transform(self.X)

    def load_signals(self):
        signals_dir = self.data_dir / f"UCI HAR Dataset/{self.split}/Inertial Signals"
        X_signals = []
        for sig in self.SIGNALS:
            file_path = signals_dir / f"{sig}_{self.split}.txt"
            if not file_path.exists():
                raise FileNotFoundError(f"File non trovato: {file_path}")
            data = np.loadtxt(file_path)  # (N, 128)
            X_signals.append(data)
        X = np.stack(X_signals, axis=1)  # (N, 9, 128)
        return torch.tensor(X, dtype=torch.float32)

    def load_labels(self):
        labels_file = self.data_dir / f"UCI HAR Dataset/{self.split}/y_{self.split}.txt"
        if not labels_file.exists():
            raise FileNotFoundError(f"Label file non trovato: {labels_file}")
        y = np.loadtxt(labels_file).astype(int).squeeze() - 1
        return torch.tensor(y, dtype=torch.long)

    def load_subjects(self):
        subjects_file = (
            self.data_dir / f"UCI HAR Dataset/{self.split}/subject_{self.split}.txt"
        )
        subjects = np.loadtxt(subjects_file).astype(int).squeeze()
        return torch.tensor(subjects, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def load_har(data_dir: Path, batch_size=32, val_split=0.2, load_all=False, cfg="a"):
    g = torch.Generator()
    g.manual_seed(cfg.seed)
    full_train_dataset = HarDataset(
        data_dir, split="train", transform=normalize_signals
    )

    val_size = int(val_split * len(full_train_dataset))
    train_size = len(full_train_dataset) - val_size
    train_dataset, val_dataset = random_split(
        full_train_dataset, [train_size, val_size]
    )

    test_dataset = HarDataset(data_dir, split="test", transform=normalize_signals)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, generator=g
    )
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    if load_all:
        return full_train_dataset, test_loader

    return train_loader, val_loader, test_loader


def normalize_signals(X):
    mean = X.mean(dim=(0, 1), keepdim=True)
    std = X.std(dim=(0, 1), keepdim=True)
    return (X - mean) / (std + 1e-8)
