import argparse
import csv
import time
from pathlib import Path

import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

from .datasets import list_wavs, RAVDESSMelDataset, INDEX_TO_NAME, filename_to_emotion_id, EMOTION_ID_TO_INDEX
from .model import SmallCNN, CNNBiLSTM
from .utils import set_seed, ensure_dir, save_json

from .wave_datasets import RAVDESSWaveDataset
from .wavlm_model import WavLMFineTuner


def resolve_device(device_str: str) -> torch.device:
    if device_str == "cpu":
        return torch.device("cpu")
    if device_str == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_epoch(model, loader, criterion, optimizer, device, train: bool):
    model.train() if train else model.eval()

    losses = []
    y_true, y_pred = [], []

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        with torch.set_grad_enabled(train):
            logits = model(x)
            loss = criterion(logits, y)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        losses.append(loss.item())
        preds = torch.argmax(logits, dim=1)
        y_true.extend(y.detach().cpu().tolist())
        y_pred.extend(preds.detach().cpu().tolist())

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")
    return float(sum(losses) / max(1, len(losses))), float(acc), float(f1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    model_type = cfg.get("model_type", "cnn")
    set_seed(int(cfg["seed"]))

    run_name = cfg["run_name"]
    out_dir = Path("outputs") / run_name
    ckpt_dir = out_dir / "checkpoints"
    ensure_dir(str(ckpt_dir))
    ensure_dir(str(out_dir))

    device = resolve_device(cfg.get("device", "auto"))

    # -----------------------
    # 1) List files + labels
    # -----------------------
    wavs = list_wavs(cfg["data_root"])
    if len(wavs) == 0:
        raise RuntimeError(f"No .wav files found under: {cfg['data_root']}")

    files, labels = [], []
    for w in wavs:
        eid = filename_to_emotion_id(w)
        if eid is None or eid not in EMOTION_ID_TO_INDEX:
            continue
        files.append(w)
        labels.append(EMOTION_ID_TO_INDEX[eid])

    # -----------------------
    # 2) Stratified splits
    # -----------------------
    X_train, X_tmp, y_train, y_tmp = train_test_split(
        files, labels,
        test_size=(1.0 - cfg["train_split"]),
        random_state=cfg["seed"],
        stratify=labels
    )
    val_ratio = cfg["val_split"] / (cfg["val_split"] + cfg["test_split"])
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp,
        test_size=(1.0 - val_ratio),
        random_state=cfg["seed"],
        stratify=y_tmp
    )

    # -----------------------
    # 3) Build datasets/loaders (mel vs waveform)
    # -----------------------
    if model_type == "wavlm_ft":
        max_seconds = float(cfg.get("max_seconds", 5.0))
        ds_train = RAVDESSWaveDataset(X_train, sample_rate=cfg["sample_rate"], max_seconds=max_seconds)
        ds_val   = RAVDESSWaveDataset(X_val,   sample_rate=cfg["sample_rate"], max_seconds=max_seconds)
        ds_test  = RAVDESSWaveDataset(X_test,  sample_rate=cfg["sample_rate"], max_seconds=max_seconds)
    else:
        ds_train = RAVDESSMelDataset(
            X_train, sample_rate=cfg["sample_rate"],
            n_mels=cfg["n_mels"], n_fft=cfg["n_fft"],
            hop_length=cfg["hop_length"], win_length=cfg["win_length"],
            f_min=cfg["f_min"], f_max=cfg["f_max"],
            max_frames=cfg["max_frames"],
            use_specaugment=cfg["use_specaugment"],
            specaugment_time_mask=cfg["specaugment_time_mask"],
            specaugment_freq_mask=cfg["specaugment_freq_mask"],
        )
        ds_val = RAVDESSMelDataset(
            X_val, sample_rate=cfg["sample_rate"],
            n_mels=cfg["n_mels"], n_fft=cfg["n_fft"],
            hop_length=cfg["hop_length"], win_length=cfg["win_length"],
            f_min=cfg["f_min"], f_max=cfg["f_max"],
            max_frames=cfg["max_frames"],
            use_specaugment=False
        )
        ds_test = RAVDESSMelDataset(
            X_test, sample_rate=cfg["sample_rate"],
            n_mels=cfg["n_mels"], n_fft=cfg["n_fft"],
            hop_length=cfg["hop_length"], win_length=cfg["win_length"],
            f_min=cfg["f_min"], f_max=cfg["f_max"],
            max_frames=cfg["max_frames"],
            use_specaugment=False
        )

    dl_train = DataLoader(ds_train, batch_size=cfg["batch_size"], shuffle=True,
                          num_workers=cfg["num_workers"], pin_memory=True)
    dl_val = DataLoader(ds_val, batch_size=cfg["batch_size"], shuffle=False,
                        num_workers=cfg["num_workers"], pin_memory=True)
    dl_test = DataLoader(ds_test, batch_size=cfg["batch_size"], shuffle=False,
                         num_workers=cfg["num_workers"], pin_memory=True)

    # -----------------------
    # 4) Model selection
    # -----------------------
    if model_type == "cnn":
        model = SmallCNN(num_classes=8, dropout=cfg["dropout"]).to(device)

    elif model_type == "cnn_bilstm":
        model = CNNBiLSTM(
            num_classes=8,
            dropout=cfg["dropout"],
            lstm_hidden=int(cfg.get("lstm_hidden", 128)),
            lstm_layers=int(cfg.get("lstm_layers", 1)),
            bidirectional=bool(cfg.get("bidirectional", True)),
        ).to(device)

    elif model_type == "wavlm_ft":
        model = WavLMFineTuner(
            model_name=cfg["wavlm_model_name"],
            num_classes=8,
            dropout=float(cfg.get("wavlm_dropout", 0.1))
        ).to(device)

    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    # -----------------------
    # 5) Loss + optimizer
    # -----------------------
    criterion = nn.CrossEntropyLoss()

    if model_type == "wavlm_ft":
        # Stage 1: freeze encoder, train head
        model.freeze_encoder()

        head_lr = float(cfg.get("head_lr", 5e-4))
        encoder_lr = float(cfg.get("encoder_lr", 5e-5))

        optimizer = torch.optim.AdamW([
            {"params": model.head.parameters(), "lr": head_lr, "weight_decay": cfg["weight_decay"]},
            {"params": [p for p in model.encoder.parameters() if p.requires_grad],
             "lr": encoder_lr, "weight_decay": cfg["weight_decay"]},
        ])
        unfreeze_epoch = int(cfg.get("unfreeze_epoch", 3))
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
        unfreeze_epoch = None

    # -----------------------
    # 6) Logging setup
    # -----------------------
    log_path = out_dir / "train_log.csv"
    with open(log_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "train_acc", "train_macro_f1",
                         "val_loss", "val_acc", "val_macro_f1"])

    best_val_f1 = -1.0
    patience = 0
    patience_limit = int(cfg.get("early_stopping_patience", 5))

    # -----------------------
    # 7) Training loop
    # -----------------------
    for epoch in range(1, int(cfg["epochs"]) + 1):
        # Unfreeze WavLM encoder at chosen epoch
        if model_type == "wavlm_ft" and epoch == unfreeze_epoch:
            model.unfreeze_encoder()
            head_lr = float(cfg.get("head_lr", 5e-4))
            encoder_lr = float(cfg.get("encoder_lr", 5e-5))
            optimizer = torch.optim.AdamW([
                {"params": model.head.parameters(), "lr": head_lr, "weight_decay": cfg["weight_decay"]},
                {"params": model.encoder.parameters(), "lr": encoder_lr, "weight_decay": cfg["weight_decay"]},
            ])
            print(f"Unfroze WavLM encoder at epoch {epoch}.")

        t0 = time.time()
        train_loss, train_acc, train_f1 = run_epoch(model, dl_train, criterion, optimizer, device, train=True)
        val_loss, val_acc, val_f1 = run_epoch(model, dl_val, criterion, optimizer, device, train=False)
        dt = time.time() - t0

        with open(log_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, train_loss, train_acc, train_f1, val_loss, val_acc, val_f1])

        ckpt = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "config": cfg
        }
        torch.save(ckpt, ckpt_dir / f"epoch_{epoch}.pt")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience = 0
            torch.save(ckpt, ckpt_dir / "best.pt")
        else:
            patience += 1

        print(
            f"[Epoch {epoch:02d}] "
            f"train loss={train_loss:.4f} acc={train_acc:.3f} f1={train_f1:.3f} | "
            f"val loss={val_loss:.4f} acc={val_acc:.3f} f1={val_f1:.3f} | "
            f"time={dt:.1f}s"
        )

        if patience >= patience_limit:
            print("Early stopping triggered.")
            break

    # -----------------------
    # 8) Final test using best checkpoint
    # -----------------------
    best = torch.load(ckpt_dir / "best.pt", map_location=device)
    model.load_state_dict(best["model_state"])
    test_loss, test_acc, test_f1 = run_epoch(model, dl_test, criterion, optimizer, device, train=False)

    metrics = {
        "best_val_macro_f1": float(best_val_f1),
        "test_loss": float(test_loss),
        "test_acc": float(test_acc),
        "test_macro_f1": float(test_f1),
        "label_map": INDEX_TO_NAME,
    }
    save_json(str(out_dir / "metrics.json"), metrics)
    print("Saved metrics to", out_dir / "metrics.json")
    print(metrics)


if __name__ == "__main__":
    main()