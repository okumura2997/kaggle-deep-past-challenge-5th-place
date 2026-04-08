# Deep Past Challenge - Translate Akkadian to English: 5th-Place Solution

This repository contains the 5th-place solution for the Kaggle competition [Deep Past Challenge - Translate Akkadian to English](https://www.kaggle.com/competitions/deep-past-initiative-machine-translation).

The accompanying Kaggle solution write-up is available [here](https://www.kaggle.com/competitions/deep-past-initiative-machine-translation/writeups/5th-solution).

The project focuses on Old Assyrian machine translation with a multi-stage pipeline that combines continued pretraining, fine-tuning on EvaCun, PDF-derived parallel data, pseudo-labeling, back-translation, and final tablet-context fine-tuning.

## Environment

The runtime environment is defined in [docker/Dockerfile](docker/Dockerfile).

The Docker base image is `nvcr.io/nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04`, so the reference environment is Ubuntu 24.04 with CUDA 12.8.1 and cuDNN devel.

This setup has been validated on a single NVIDIA H200 GPU.

## Setup

1. Build and run the environment from `docker/Dockerfile`.
2. Create a `.env` file at the repository root and set `OPENROUTER_API_KEY`. This key is used by the sentence-aligned PDF extraction pipeline (`bash scripts/run_aligned_extraction_pipeline.sh`).
3. Log in to Weights & Biases before starting training:

```bash
wandb login
```

## Data Preparation

For end-to-end reproduction, all expected inputs under `input/` are required before running preprocessing:

- `input/deep-past-initiative-machine-translation/`
- `input/deep-past-initiative-machine-translation_old/`
- `input/evacun/`
- `input/old-assyrian-grammars-and-other-resources/`
- `input/pdfs/` with the PDF files used for PDF extraction in this solution

Run preprocessing before any training job:

```bash
bash scripts/preprocess.sh
```

This script runs the base dataset preprocessing and the EvaCun preprocessing pipeline, producing the processed training tables used by later stages, including `data/train_processed.csv` and `data/evacun/train_processed.csv`.

EvaCun is available from Zenodo: <https://zenodo.org/records/17220688>.

Dataset note: `input/deep-past-initiative-machine-translation_old` corresponds to the dataset version available at the time of [this Kaggle discussion post](https://www.kaggle.com/competitions/deep-past-initiative-machine-translation/discussion/674136), while `input/deep-past-initiative-machine-translation` is the final competition dataset. The preprocessing pipeline also uses a subset of parallel transliteration/translation pairs from the old dataset version to replace selected rows in the final competition training data.

## PDF Extraction

There are two PDF extraction pipelines in this repository, and both are part of the full reproduction flow:

Prepare the PDF files used by this solution under `input/pdfs/` before running these steps. The PDF set follows the one described in the Kaggle solution write-up: <https://www.kaggle.com/competitions/deep-past-initiative-machine-translation/writeups/5th-solution>.

1. Tablet-level parallel corpus extraction used before the tablet fine-tuning stages:

```bash
bash scripts/pdf_extraction.sh
```

2. Sentence-aligned extraction used for the final fine-tuning stage. This step requires `OPENROUTER_API_KEY` and produces the data consumed by `scripts/xl/09_final_ft.sh`:

```bash
bash scripts/run_aligned_extraction_pipeline.sh
```

## Synthetic Translation Generation

To generate synthetic English passages that resemble scholarly translations of Akkadian transliterations, used as the source side for back-translation, run:

```bash
bash scripts/generate_translation_like_english.sh
```

## Training Pipeline

The main training scripts are organized under [scripts/xl](scripts/xl). After the data-generation steps above, run them in numeric order. The overall pipeline is:

### Stage 1

Continued pretraining on transliterations from `published_texts.csv`, followed by fine-tuning on EvaCun, and then fine-tuning on PDF-derived tablet-level data.

Relevant scripts:

- `scripts/xl/01_span_corruption.sh`
- `scripts/xl/02_evacan_ft.sh`
- `scripts/xl/03_tablet_ft.sh`

### Stage 2

Train the reverse-direction models used for back-translation, then generate synthetic data via pseudo-labeling and back-translation.

Relevant scripts:

- `scripts/xl/04_bt_evacan_ft.sh`
- `scripts/xl/05_bt_tablet_ft.sh`
- `scripts/xl/06_generate_pseudo_labels.sh`
- `scripts/xl/07_generate_back_translation.sh`

### Stage 3

Fine-tuning on synthetic data from the EvaCun checkpoint, followed by final fine-tuning on PDF data with tablet context.

Relevant scripts:

- `scripts/xl/08_pseudo_bt_pretrain.sh`
- `scripts/xl/09_final_ft.sh`
- `scripts/xl/10_average_weights.sh`

The weights from the final fine-tuning stage are available on Kaggle at [yukiokumura1/byt5-xl-final-ft](https://www.kaggle.com/datasets/yukiokumura1/byt5-xl-final-ft).

## Inference

For inference and submission code, refer to the Kaggle notebook [5th-solution](https://www.kaggle.com/code/ebinan92/5th-solution).

## Directory Structure

```text
.
├── input/
├── data/
├── scripts/
│   └── xl/
├── src/
├── outputs/
└── docker/
```

- `input/`: required raw competition and auxiliary data.
- `data/`: generated preprocessing artifacts, extracted PDF pairs, pseudo-labels, and synthetic data.
- `scripts/xl/`: primary training pipeline to run in numeric order.
- `src/`: preprocessing, extraction, training, and inference implementations.
- `outputs/`: saved checkpoints from pretraining and fine-tuning.
- `docker/`: container definition used for reproduction.

## Notes

- The `scripts/xl` pipeline is the primary reference for reproducing the final training flow.
- For users with less available VRAM, `scripts/large` provides an alternative pipeline based on ByT5-large.
