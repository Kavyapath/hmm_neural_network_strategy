"""GRU take/skip classifier. One raw logit per candidate trade; BCEWithLogitsLoss,
no sigmoid before the loss. Sigmoid applied only at inference for a score."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

GRU_BLOCKS=2

@dataclass
class GRUConfig:
    n_features: int
    lookback: int = 30
    hidden_size: int = 32
    dropout: float = 0.1
    lr: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    max_epochs: int = 50
    patience: int = 5          # early stopping
    seed: int = 42


class TakeSkipGRU(nn.Module):
    """GRU_BLOCKS blocks of (GRU -> GELU -> Dropout), dropout skipped after
    the last block, then a linear head on the final timestep. A manual
    forward loop, not nn.Sequential — nn.GRU returns (output, h_n), and
    GELU/Dropout can't consume that tuple directly."""

    def __init__(self, n_features: int, hidden_size: int = 32, dropout: float = 0.1):
        super().__init__()
        self.blocks = nn.ModuleList([
            nn.GRU(
                input_size=n_features if i == 0 else hidden_size,
                hidden_size=hidden_size, num_layers=1, batch_first=True,
            )
            for i in range(GRU_BLOCKS)
        ])
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [batch, lookback, n_features] -> logits: [batch]"""
        for i, gru in enumerate(self.blocks):
            x, _ = gru(x)                # x: [batch, lookback, hidden_size]
            x = self.activation(x)
            if i < len(self.blocks) - 1:
                x = self.dropout(x)
        h_last = x[:, -1, :]             # last timestep of the final block's output
        logit = self.head(h_last).squeeze(-1)   # [batch]
        return logit

    def take_prob(self, logits: torch.Tensor) -> torch.Tensor:
        """Sigmoid for INFERENCE only — never before the loss."""
        return torch.sigmoid(logits)


class CandidateWindowDataset(Dataset):
    """Each item: (window [lookback, F], label scalar). Built from a labeled
    trade dataset (Step 5's build_labeled_dataset output, one row per
    resolved trade, tagged with a `ticker` column) and `feature_frames`, a
    dict keyed by that same ticker, each value a per-ticker feature frame
    indexed by date (regime + technical indicators, concatenated already).
    Keeping frames per-ticker (instead of concatenating them) avoids a
    non-unique DatetimeIndex once multiple tickers share calendar dates."""

    def __init__(
        self,
        labeled_trades: "pd.DataFrame",
        feature_frames: dict[str, "pd.DataFrame"],
        lookback: int,
    ):
        self.lookback = lookback
        self.samples = []
        for _, trade in labeled_trades.iterrows():
            sig_date = trade["signal_date"]
            feature_frame = feature_frames[trade["ticker"]]
            feat_idx = feature_frame.index
            if sig_date not in feat_idx:
                continue
            pos = feat_idx.get_loc(sig_date)
            if pos - lookback + 1 < 0:
                continue  # not enough history before this signal, drop
            window = feature_frame.iloc[pos - lookback + 1 : pos + 1].to_numpy(dtype=np.float32)
            self.samples.append((window, np.float32(trade["label"])))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int):
        window, label = self.samples[i]
        return torch.from_numpy(window), torch.tensor(label)


def estimate_pos_weight(labels: np.ndarray) -> torch.Tensor:
    """Optional class weighting, estimated from TRAINING data only (per the
    brief) — pass this as pos_weight to BCEWithLogitsLoss if classes are
    imbalanced. Returns 1.0 (no reweighting) if labels are balanced or empty."""
    n_pos = labels.sum()
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return torch.tensor(1.0)
    return torch.tensor(n_neg / n_pos, dtype=torch.float32)


def train_take_skip_model(
    train_ds: CandidateWindowDataset,
    val_ds: CandidateWindowDataset,
    cfg: GRUConfig,
    device: torch.device,
) -> tuple[TakeSkipGRU, dict]:
    torch.manual_seed(cfg.seed)
    model = TakeSkipGRU(cfg.n_features, cfg.hidden_size, cfg.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    train_labels = np.array([lbl for _, lbl in train_ds.samples])
    pos_weight = estimate_pos_weight(train_labels).to(device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=32, shuffle=False)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_loss = float("inf")
    best_state = None
    epochs_no_improve = 0

    for epoch in range(cfg.max_epochs):
        model.train()
        running = 0.0
        n_correct_train = 0
        for windows, labels in train_loader:
            windows, labels = windows.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(windows)
            loss = loss_fn(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            running += loss.item() * len(labels)
            # accuracy from the SAME forward pass used for the loss/update
            # above — i.e. it reflects the model's state during this batch's
            # training step, not a separate post-hoc eval pass.
            n_correct_train += ((logits.detach() >= 0) == (labels >= 0.5)).sum().item()
        train_loss = running / len(train_ds)
        train_acc = n_correct_train / max(len(train_ds), 1)

        model.eval()
        running_val = 0.0
        n_correct = 0
        with torch.no_grad():
            for windows, labels in val_loader:
                windows, labels = windows.to(device), labels.to(device)
                logits = model(windows)
                running_val += loss_fn(logits, labels).item() * len(labels)
                n_correct += ((logits >= 0) == (labels >= 0.5)).sum().item()
        val_loss = running_val / max(len(val_ds), 1)
        val_acc = n_correct / max(len(val_ds), 1)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= cfg.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history
