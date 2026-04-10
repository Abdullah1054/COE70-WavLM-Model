# Speech Emotion Recognition — Custom CNN (Student C)

This repo is a runnable, end-to-end PyTorch training pipeline for a **custom lightweight CNN**
that classifies emotions from **cleaned RAVDESS** audio. It covers:

- Load cleaned `.wav` files (RAVDESS naming expected)
- Convert to **Mel spectrograms**
- **Pad/trim** to a fixed number of time frames (to match your earlier MCR description)
- Train a small **2-block CNN**
- Log metrics, save checkpoints, and run evaluation (accuracy + macro-F1 + confusion matrix)

## 1) Setup

Create a venv and install requirements:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate

pip install -r requirements.txt
```

## 2) Dataset layout (expected)

Point the config to your cleaned audio root folder. Example structure:

```
data/cleaned_ravdess/
  Actor_01/
    03-01-01-01-01-01-01.wav
    ...
  Actor_02/
    ...
```

RAVDESS emotion id is the **3rd field** in the filename (e.g., `03-01-05-02-...` → emotion id `05`).

## 3) Train

Edit `configs/config.yaml` and set:

- `data_root`: path to cleaned wav files
- `sample_rate`: match your preprocessing output (e.g., 16000 or 22050)

Then run:

```bash
python -m src.train --config configs/config.yaml
```

Artifacts go to `outputs/<run_name>/`:
- `checkpoints/best.pt`
- `metrics.json`
- `train_log.csv`

## 4) Evaluate

```bash
python -m src.eval --run_dir outputs/<run_name>
```

This will write:
- `confusion_matrix.png`
- `eval_metrics.json`
