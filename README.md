# Hyperbolic CLIP for Synthetic MRI Detection

## Overview

This project investigates the detection of AI-generated synthetic brain MRI images using deep learning approaches. We compare traditional CNNs (ResNet50), vision-language models (CLIP), and hyperbolic representations for binary classification of real versus synthetic medical images. Additionally, we explore one-class anomaly detection methods that train exclusively on real images, detecting synthetic images as out-of-distribution samples. The comprehensive evaluation spans 9 experiments across supervised classification, zero-shot inference, and anomaly detection paradigms.

## Dataset

| Property | Value |
|----------|-------|
| Total Images | 5,121 |
| Real Images | 1,226 |
| Synthetic Images | 3,895 |
| Resolution | 224×224 |
| Classes | Binary (Real/Synthetic) |
| Sources | 8 |

**Real sources**: CERMEP, TCGA, UPenn  
**Synthetic generators**: GAN, Latent Diffusion Model (LDM), Medical Latent Synthesizer (MLS)

The dataset was cleaned (blank removal, deduplication, pHash quarantine) and split 70/15/15 at the subject level. See [docs/phase_1_dataset_preparation.md](docs/phase_1_dataset_preparation.md) for details.

## Experiments Conducted

| # | Experiment | Description |
|---|------------|-------------|
| 1 | **ResNet50 Baseline** | Standard CNN binary classifier trained on real vs synthetic images |
| 2 | **ResNet50 Cross-Generator** | Leave-one-generator-out evaluation to test generalization |
| 3 | **CLIP Zero-Shot** | Frozen CLIP with text prompts for classification without training |
| 4 | **Hyperbolic Zero-Shot** | CLIP embeddings projected to Poincaré ball with hyperbolic distance |
| 5 | **CLIP Linear Probe** | Frozen CLIP features with trained linear classifier head |
| 6 | **CLIP Fine-Tune** | End-to-end fine-tuning of CLIP with contrastive learning |
| 7 | **Hyperbolic CLIP** | Fine-tuned CLIP with Poincaré ball projection and hyperbolic loss |
| 8 | **Real-Only CLIP (Euclidean)** | One-class anomaly detection using Deep SVDD loss |
| 9 | **Real-Only CLIP (Hyperbolic)** | Hyperbolic anomaly detection with Fréchet mean center |

## Final Model Comparison

| Model | Accuracy | Precision | Recall | F1 | AUROC | AUPRC | PPV | NPV |
|-------|----------|-----------|--------|-----|-------|-------|-----|-----|
| ResNet50 Baseline | 0.9974 | 1.0000 | 0.9966 | 0.9983 | 1.0000 | 1.0000 | 1.0000 | 0.9893 |
| ResNet50 Cross-Generator | 0.9951 | 1.0000 | 0.9653 | 0.9815 | 1.0000 | 1.0000 | 1.0000 | — |
| CLIP Zero-Shot | 0.7647 | 0.7632 | 1.0000 | 0.8657 | 0.8138 | 0.9275 | 0.7632 | 1.0000 |
| Hyperbolic Zero-Shot | 0.7582 | 0.7582 | 1.0000 | 0.8625 | 0.6033 | 0.8390 | 0.7582 | 0.0000 |
| CLIP Linear Probe | 0.9765 | 0.9982 | 0.9707 | 0.9843 | 0.9986 | 0.9996 | 0.9982 | 0.9154 |
| **CLIP Fine-Tune** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** |
| Hyperbolic CLIP | 0.9974 | 0.9966 | 1.0000 | 0.9983 | 1.0000 | 1.0000 | 0.9966 | 1.0000 |
| Real-Only (Euclidean) | 0.9882 | 0.9847 | 1.0000 | 0.9923 | 0.9996 | 0.9999 | 0.9847 | 1.0000 |
| Real-Only (Hyperbolic) | 0.9843 | 0.9830 | 0.9966 | 0.9897 | 0.9987 | 0.9996 | 0.9830 | 0.9887 |

## Results Visualization

![Model Performance Comparison](assets/results/model_performance_comparison.png)

## Key Observations

- **CNN baseline achieves near-perfect performance** (99.74% accuracy), demonstrating that synthetic MRI detection is feasible with standard architectures.
- **Zero-shot methods fail on real images**, achieving only 0–3% accuracy on the real class while detecting 100% of synthetic images.
- **CLIP fine-tuning achieves perfect classification** (100% on all metrics), outperforming all other approaches.
- **Hyperbolic geometry provides marginal improvement** in NPV (1.0 vs 0.9893) compared to Euclidean baselines.
- **One-class anomaly detection is viable**, achieving 98.4–98.8% accuracy without seeing synthetic examples during training.

## Repository Structure

```
├── configs/                  # Experiment configuration files
├── scripts/                  # Training and evaluation scripts
├── src/                      # Model source code and datasets
├── experiments/              # Experiment outputs and results
├── assets/                   # Visualizations and plots
├── docs/                     # Documentation
├── data/                     # Dataset setup instructions
└── README.md
```

## Running Experiments

```bash
# Supervised classification
python scripts/run_clip_finetune.py
python scripts/run_hyperbolic_clip.py

# Anomaly detection
python scripts/run_real_only_clip_euclidean.py
python scripts/run_real_only_clip_hyperbolic.py

# Zero-shot evaluation
python scripts/run_clip_zero_shot.py
```

## License

MIT License — see [LICENSE](LICENSE) for details.
