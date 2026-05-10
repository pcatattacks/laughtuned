# LaughTuned

Fine-tuning Mistral-7B for comedy writing on Guardian news articles, using
DPO and KTO preference optimization implemented from scratch.

CS 5788 (Cornell, Spring 2026) final project.

## Layout

- `config.py` — single source of truth for hyperparameters and paths.
- `data/` — Guardian ingestion, prompt construction, generation, judging,
  dataset prep.
- `models/` — model loading, log-prob helper, DPO/KTO losses, training loop.
- `eval/` — automated metrics, eval-set generation, cross-judge.
- `utils/` — Drive helpers, logging, checkpointing.
- `demo.ipynb` — Colab-runnable demo of the full pipeline.

## Running on Colab

1. Open `demo.ipynb` in Colab (or attach a Colab runtime from
   VS Code / Cursor).
2. The first code cell clones this repo into `/content/laughtuned`,
   installs `requirements.txt`, mounts Google Drive, and creates the
   artifact tree under `/content/drive/MyDrive/LaughTuned/`.
3. Set `CONFIG["guardian_api_key"]` and `CONFIG["anthropic_api_key"]`
   from Colab secrets — do **not** commit keys.
4. Run cells top-to-bottom.

## Code vs artifacts

Code lives in this git repo (small text files). Data, generations,
preferences, checkpoints, metrics, and figures live on Google Drive
(too large/binary for git):

```
/content/drive/MyDrive/LaughTuned/
├── data/{articles,prompts,generations,preferences,splits,eval}/
├── checkpoints/
├── metrics/
├── figures/
└── ref_log_probs/
```

## Type checking

```
pyright
```

Run from the repo root. Configured in `pyrightconfig.json`.
