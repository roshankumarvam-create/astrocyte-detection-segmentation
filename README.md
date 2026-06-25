# Astrocyte Detection & Segmentation at Whole-Slide Scale

<p align="left">
  <img src="https://img.shields.io/badge/Task-Detection%20%2B%20Segmentation-1565C0?style=flat-square">
  <img src="https://img.shields.io/badge/Modality-IHC%20%2F%20GFAP-7B1FA2?style=flat-square">
  <img src="https://img.shields.io/badge/Detection-YOLOv5-EE4C2C?style=flat-square">
  <img src="https://img.shields.io/badge/Segmentation-UNet%20%2B%20SimCLR-1E88E5?style=flat-square">
  <img src="https://img.shields.io/badge/SSL-Self--Supervised-6A1B9A?style=flat-square">
  <img src="https://img.shields.io/badge/Serving-MLflow%20%2B%20FastAPI-009688?style=flat-square">
  <img src="https://img.shields.io/badge/code-private%20(institutional%20IP)-555?style=flat-square">
</p>

> Built at the **Sudha Gopalakrishnan Brain Centre, IIT Madras**, on the institute's large-scale human brain histology archive. This repo documents the system; source and trained weights are held under institutional IP.

---

## The problem

Human brain histology is acquired as **gigapixel whole-slide images (WSI)** — a single section can exceed 200,000 × 300,000 pixels. **GFAP-stained astrocytes** must be located and outlined across thousands of such sections to study glial morphology and distribution. Manual annotation is infeasible at that scale, and off-the-shelf cell models fail on the staining and density characteristics of human cortex. Two things make it hard:

- **Annotation cost** — pixel-accurate masks for astrocytes are expensive, so labeled data is scarce.
- **Morphology** — astrocytes are branching, process-bearing cells that overlap and cluster densely, which is exactly where bounding-box-only detection breaks down.

The system below addresses both: a high-recall detector produces ground truth cheaply, and a **self-supervised** encoder lets a segmentation model learn fine morphology from very few annotated masks.

## Pipeline overview

![End-to-end pipeline](./assets/full-workflow_1.png)

The WSI is tiled, astrocytes are detected with YOLOv5, those detections are converted directly into segmentation masks, and a UNet — whose encoder is pre-trained on unlabeled tissue with SimCLR — produces the final per-cell mask used for density and clinical readouts.

---

## Stage A — Detection (YOLOv5)

The slide is streamed and split into **512 × 512** tiles so detection fits in GPU memory. A YOLOv5 detector is trained on hand-curated GFAP+ astrocyte crops under a strict annotation protocol — *box only around clearly visible astrocytes, soma must be visible, no blurring* — and validated at an **IoU threshold of 0.45**.

Against expert manual annotation the detector reproduces astrocyte locations across morphological variants (stellate, hypertrophic, fibrous), recovering cells at high recall while keeping false positives low enough to serve as training ground truth for the next stage. This detection layer is the **supervisory signal** for segmentation — its boxes become masks rather than being discarded.

## Stage B — Masks directly from detections

![Mask generation](./assets/mask-generation.png)

Instead of a separate weak-label pipeline, segmentation masks are **constructed directly from the detection annotations (COCO format)**. For every bounding box:

1. Extract the image crop
2. Convert to grayscale
3. Apply **gamma correction (γ = 3)** to lift the GFAP signal
4. **Otsu threshold** to separate cell from background
5. Remove small spurious objects
6. Paste the binary crop back into a full-image mask

The result is a **full-image binary mask** (`0 = background`, `1 = astrocyte`) — drop-in ground truth for UNet, with no manual pixel labeling.

## Stage C — Self-supervised pre-training (SimCLR)

![SimCLR pre-training](./assets/simclr-pipeline.png)

Because annotated masks are scarce, a **ResNet18** encoder is first pre-trained on **unlabeled** brain-tissue tiles using **SimCLR** contrastive learning. Two augmented views of each tile are pushed through a shared encoder and a 512 → 128 MLP projection head, and an **NT-Xent** loss (τ = 0.1) pulls matching views together while pushing others apart.

| Setting | Value |
|---|---|
| Backbone | ResNet18 → MLP head (512 → 128) |
| Loss | NT-Xent, τ = 0.1 |
| Augmentations | random resized crop (224), flip, color jitter, grayscale, Gaussian blur |
| Optimizer | LARS · lr 1e-3 · batch 128 · 200 epochs |

The learned weights initialize the segmentation encoder, so the segmenter starts already understanding tissue structure.

## Stage D — Segmentation (UNet + SimCLR)

The segmentation model is a **UNet** whose encoder is the SimCLR-pretrained ResNet18. Four upsampling blocks with **skip connections** from the encoder stages recover spatial detail, and a final upsampling layer returns to the full **512 × 512** resolution. Training uses **`BCEWithLogitsLoss(pos_weight = 10)`** to counter the heavy background/foreground imbalance.

| Setting | Value |
|---|---|
| Encoder | ResNet18, initialized from SimCLR |
| Loss | BCEWithLogitsLoss, pos_weight = 10 |
| Optimizer | Adam · lr 3e-4 · ReduceLROnPlateau (factor 0.1, patience 3) |
| Inference | sigmoid → threshold 0.5 (optional multi-checkpoint ensemble) |

Inference applies a sigmoid to the logits; an ensemble that averages probabilities across checkpoints is available for extra robustness. The full encoder/decoder architecture, training setup, and performance are shown in the [end-to-end workflow diagram](#pipeline-overview) above.

---

## Results

| Model | Task | Best metric | Notes |
|---|---|---|---|
| **UNet + SimCLR** | Segmentation | **Val loss 0.0352** | best performing; train loss 0.0309 @ epoch 3 |
| MedSAM | Segmentation | Dice 0.8063 | baseline comparison |

The best checkpoint is logged to **MLflow** and registered in the model registry.

> **Why segmentation over detection?** With overlapping, densely distributed astrocytes, boxes collide and merge — pixel masks recover the true per-cell shape that morphometrics depend on.

## Deployment

- **Packaging** — the trained model is wrapped with `mlflow.pyfunc` for standardized, versioned inference.
- **Inference API (FastAPI)** — `/api/health`, `/api/get_mask`, `/api/model/info`, with gamma-correction preprocessing applied before inference. Tile size 512 × 512, threshold 0.5, FP32.

## Clinical application — stroke tissue classification

Segmentation outputs feed a downstream readout:

- estimate **astrocyte density** per region,
- generate **WSI-level density heatmaps**,
- classify tissue as **NEGATIVE · SPARSE · DENSE**.

## Tech stack

`PyTorch` · `YOLOv5` · `SimCLR (contrastive SSL)` · `UNet / ResNet18` · `OpenCV / scikit-image` · `MLflow` · `FastAPI` · `NumPy / pandas` · `CUDA`

## Why it matters

This is end-to-end ML systems work: a detector that operates at gigapixel scale, **detection annotations reused as segmentation supervision** (no manual pixel labeling), **self-supervised pre-training** that makes a data-scarce segmentation task tractable, and a packaged serving layer a domain expert can actually use. The same pattern — detect → derive masks → pre-train → segment → quantify — generalizes to most high-resolution biomedical imaging problems.

---

<sub>Code and trained weights are private under IIT Madras / SGBC institutional agreements. A technical walkthrough is available on request.</sub>
