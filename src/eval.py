import argparse
from pathlib import Path

import torch
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, f1_score, accuracy_score
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from .datasets import RAVDESSMelDataset, INDEX_TO_NAME, list_wavs, filename_to_emotion_id, EMOTION_ID_TO_INDEX
from .model import SmallCNN, CNNBiLSTM
from .utils import save_json

from .wave_datasets import RAVDESSWaveDataset
from .wavlm_model import WavLMFineTuner


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=str, required=True, help="outputs/<run_name>")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    ckpt_path = run_dir / "checkpoints" / "best.pt"
    if not ckpt_path.exists():
        raise RuntimeError(f"Missing checkpoint: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg = ckpt["config"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_type = cfg.get("model_type", "cnn")

    # -----------------------
    # 1) Create model
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

    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # -----------------------
    # 2) Recreate same split logic as training
    # -----------------------
    wavs = list_wavs(cfg["data_root"])
    files, labels = [], []
    for w in wavs:
        eid = filename_to_emotion_id(w)
        if eid is None or eid not in EMOTION_ID_TO_INDEX:
            continue
        files.append(w)
        labels.append(EMOTION_ID_TO_INDEX[eid])

    _, X_tmp, _, y_tmp = train_test_split(
        files, labels,
        test_size=(1.0 - cfg["train_split"]),
        random_state=cfg["seed"],
        stratify=labels
    )
    val_ratio = cfg["val_split"] / (cfg["val_split"] + cfg["test_split"])
    _, X_test, _, y_test = train_test_split(
        X_tmp, y_tmp,
        test_size=(1.0 - val_ratio),
        random_state=cfg["seed"],
        stratify=y_tmp
    )

    # -----------------------
    # 3) Create dataset / loader depending on model type
    # -----------------------
    if model_type == "wavlm_ft":
        max_seconds = float(cfg.get("max_seconds", 5.0))
        ds_test = RAVDESSWaveDataset(X_test, sample_rate=cfg["sample_rate"], max_seconds=max_seconds)
    else:
        ds_test = RAVDESSMelDataset(
            X_test, sample_rate=cfg["sample_rate"],
            n_mels=cfg["n_mels"], n_fft=cfg["n_fft"],
            hop_length=cfg["hop_length"], win_length=cfg["win_length"],
            f_min=cfg["f_min"], f_max=cfg["f_max"],
            max_frames=cfg["max_frames"], use_specaugment=False
        )

    dl_test = DataLoader(
        ds_test,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=cfg["num_workers"]
    )

    # -----------------------
    # 4) Predict on test set
    # -----------------------
    y_true, y_pred = [], []
    with torch.no_grad():
        for x, y in dl_test:
            x = x.to(device)
            logits = model(x)
            preds = torch.argmax(logits, dim=1).cpu().tolist()
            y_true.extend(y.tolist())
            y_pred.extend(preds)

    acc = float(accuracy_score(y_true, y_pred))
    f1 = float(f1_score(y_true, y_pred, average="macro"))

    # -----------------------
    # 5) Confusion matrix + report
    # -----------------------
    cm = confusion_matrix(y_true, y_pred, labels=list(range(8)))
    disp = ConfusionMatrixDisplay(cm, display_labels=[INDEX_TO_NAME[i] for i in range(8)])
    fig, ax = plt.subplots(figsize=(10, 8))
    disp.plot(ax=ax, xticks_rotation=45, values_format="d", colorbar=False)
    plt.tight_layout()
    out_png = run_dir / "confusion_matrix.png"
    plt.savefig(out_png, dpi=160)
    plt.close(fig)

    report = classification_report(
        y_true, y_pred,
        target_names=[INDEX_TO_NAME[i] for i in range(8)],
        digits=3
    )

    out = {"test_acc": acc, "test_macro_f1": f1, "classification_report": report}
    save_json(str(run_dir / "eval_metrics.json"), out)

    print("Saved:", out_png)
    print("Saved:", run_dir / "eval_metrics.json")
    print(report)


if __name__ == "__main__":
    main()