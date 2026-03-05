# CQG — Controlled Question Generation

Implementation of the ACL 2022 paper:
> **"CQG: A Simple and Effective Controlled Generation Framework for Multi-hop Question Generation"**

Given a multi-hop **fact** paragraph and a target **answer**, the model generates a natural-language question whose answer is the given one.  
It uses a BERT encoder with a custom Transformer decoder and a FLAG-guided cross-attention mechanism.

---

## Project Structure

```
CQG/
├── data/                  ← dataset files (not in git, download separately)
│   ├── train.json
│   ├── dev.json
│   └── test.json
├── model/                 ← model architecture
│   ├── Decoder.py         ← Transformer decoder + FLAG attention
│   ├── Generator.py       ← greedy / beam-search generation
│   └── mbart.py
├── utils/
│   ├── config.py          ← Config dataclass
│   └── util.py            ← optimizer, scheduler, gradient clipping
├── docs/
│   └── CHANGES.md         ← full migration & optimization change log
├── dataset.py             ← data loading, tokenisation, collation
├── main.py                ← entry point (train + evaluate)
├── requirements.txt
└── README.md
```

**Runtime outputs** (generated automatically, not committed to git):

| File / Folder | What it is |
|---|---|
| `model.pth` | Best model checkpoint saved during training |
| `model_files/` | Per-run model folder (mirrors `model.pth`) |
| `case.json` | Test-set predictions written after evaluation |

---

## Requirements

- Python 3.10 or newer (tested on 3.13)
- CPU-only is supported; CUDA is used automatically if available

### Install dependencies

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

The required NLTK data (`punkt_tab`, `wordnet`, `omw-1.4`) is downloaded automatically the first time you run `main.py`.

---

## Dataset

Download the training set from Google Drive:  
<https://drive.google.com/file/d/1Lgqlxp3mkXJyOi6fZZfFPZ2MeQTenMFK/view?usp=sharing>

Place all three files in the `data/` folder:

```
data/train.json   (~910 MB, ~90 000 samples)
data/dev.json
data/test.json
```

---

## Running the Project

### Activate the virtual environment first

```powershell
.\venv\Scripts\Activate.ps1
```

### Quick smoke test (minimal load, just verify the pipeline works)

```powershell
python main.py --train_num 2 --dev_num 1 --test_num 1 --num_epochs 1 --batch_size 2 --max_seq_length 64 --generated_max_length 30
```

### Recommended training run (laptop / CPU)

```powershell
python main.py --num_epochs 20 --batch_size 4
```

### Test / evaluate only (load a saved checkpoint)

```powershell
python main.py --mode test --batch_size 4
```

---

## Parameter Reference

| Parameter | Default | Description |
|---|---|---|
| `--mode` | `train` | `train` to train then evaluate; `test` to load `model.pth` and evaluate only |
| `--num_epochs` | `50` | Number of full passes over the training data |
| `--batch_size` | `8` | Samples processed per step — lower this if you run out of memory |
| `--learning_rate` | `2e-5` | AdamW learning rate |
| `--early_stop` | `8` | Stop training if dev BLEU does not improve for this many epochs |
| `--train_num` | `-1` | Number of training samples to use; `-1` = all (~90 k) |
| `--dev_num` | `-1` | Number of validation samples; `-1` = all |
| `--test_num` | `-1` | Number of test samples; `-1` = all |
| `--max_seq_length` | `100` | Max BERT tokens for the **input** (fact + answer); longer inputs are truncated |
| `--generated_max_length` | `150` | Max tokens the model may generate for the output question |
| `--device` | auto | `cpu` or `cuda:0`; auto-detected if not specified |
| `--train_file` | `data/train.json` | Path to training data |
| `--dev_file` | `data/dev.json` | Path to validation data |
| `--test_file` | `data/test.json` | Path to test data |
| `--fp16` | `0` | Set to `1` to enable mixed-precision training (requires CUDA) |
| `--seed` | `42` | Random seed for reproducibility |

---

## Expected Output

During training you will see a per-epoch progress bar and validation BLEU:

```
--training batch: 100%|████| 11237/11237  loss: 2.34
Finish epoch: 3, loss: 26312.45, mean loss: 2.34
The dev bleu is: 18.42
[Model Info] Saving the best model...
```

After the final epoch the test set is evaluated and predictions are printed:

```
####PREDICTIIONS####
['when was the film directed by the person born in london released?']
####TARGETS####
['what year was the film released?']
The test bleu is: 14.72
```

A fully trained model typically achieves **BLEU ~15–25** on the test set.  
All predictions are also saved to `case.json` for manual inspection.

---

## Notes

- This repository is a **migrated version** of the original codebase.  
  The original depended on `fastNLP 0.6.0` which is incompatible with Python 3.12+.  
  All fastNLP dependencies have been removed and replaced with inline implementations.  
  See [docs/CHANGES.md](docs/CHANGES.md) for the full change log.
- Mixed-precision training (`--fp16 1`) only works with a CUDA-capable GPU.
