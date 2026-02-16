# Quadruplet Augmentation (QuadAug)

Official implementation of our paper:

> **Quadruplet Augmentation with Attribute and Structure Invariance for Online Continual Learning**

------

## 📖 Overview

Online Continual Learning (OCL) aims to train models on non-i.i.d. streaming data **without access to task boundaries** during both training and inference. Unlike traditional continual learning, OCL requires models to update in an online manner while preserving previously acquired knowledge.

However, existing OCL approaches suffer from two fundamental challenges:

1. **Shortcut Feature Trap**
    Models overfit to session-specific shortcut features, leading to catastrophic forgetting when distribution shifts.
2. **Limited Plasticity**
    Over-regularization strategies to prevent forgetting often restrict the model’s ability to learn new knowledge.

------

## 🔍 Key Insight

We reveal that these two issues correspond to two essential invariance requirements in OCL:

- **Attribute Invariance**
   The model must capture object attributes that remain stable across sessions.
- **Structure Invariance**
   The model must preserve the relational structure among attributes during continuous updates.

To address this, we analyze OCL from both:

- **Causal Modeling Perspective**
- **Fourier Domain Perspective**

------

## 🚀 Proposed Method: QuadAug

We propose **Quadruplet Augmentation (QuadAug)** — a principled framework that enforces attribute and structure invariance via coordinated data- and channel-level augmentation.

QuadAug consists of four modules:

| Module    | Goal                          | Mechanism                                |
| --------- | ----------------------------- | ---------------------------------------- |
| **P-aug** | Preserve attribute invariance | Phase-domain intervention                |
| **A-aug** | Preserve attribute invariance | Amplitude-domain intervention            |
| **I-aug** | Enforce channel independence  | Inter-channel discrepancy constraint     |
| **S-aug** | Enforce channel sufficiency   | Adversarial channel sufficiency learning |

------

## 🧠 Theoretical Foundation

### 1️⃣ Fine-grained Structural Causal Model

We build a structural causal model for OCL, decomposing latent factors into:

- **Session-invariant class-related factors**
- **Session-specific class-related factors**
- **Class-irrelevant confounders**

This formulation enables us to explicitly intervene on confounding factors while preserving invariant attributes.

------

### 2️⃣ Fourier Perspective on Knowledge Transfer

We analyze the roles of:

- **Amplitude** → captures attribute-related statistical properties
- **Phase** → captures structural and semantic information

Based on this observation, we design a **bidirectional Fourier-based augmentation strategy** (P-aug & A-aug), which:

- Intervenes on subtle confounders
- Preserves invariant object attributes
- Enhances robustness to session shifts

------

### 3️⃣ Structure Invariance Decomposition

We decompose structure invariance into two necessary conditions:

- **Channel Independence**
- **Channel Sufficiency**

These are enforced via:

- **I-aug**: Inter-channel discrepancy constraint
- **S-aug**: Adversarial learning between a channel sufficiency detector and classifiers

This ensures stable relational representations across sessions.

------

## 📊 Experimental Results

QuadAug achieves consistent improvements across:

- **4 Sequential datasets**
- **3 Blurry datasets**

### Performance Gains

- **Seq-CIFAR10**: +2.3% ~ +6.9%
- **CIFAR10-Blurry30**: +1.3% ~ +2.7%

QuadAug demonstrates strong robustness under both clear task boundaries and highly overlapping blurry scenarios.

------

# 📚 Dependencies

- `torch >= 2.1.0`
- `torchvision`
- `numpy`
- `kornia >= 0.7.0`
- `Pillow`
- `timm == 0.9.8`
- `tqdm`
- `onedrivedownloader`
- `ftfy`
- `regex`
- `pyyaml`

Install dependencies:

```
pip install -r requirements.txt
```

Or run directly with:

```
uv run python main.py ...
```

------

# ⚙️ Setup & Usage

## 🚀 Running Experiments

Use `main.py` to run all experiments.

To automatically load the best hyperparameters reported in the paper:

```
--load_best_args
```

### 🔬 Reproducing Paper Results

```
python main.py --dataset seq-cifar10 --model quadaug --buffer_size 1000 --load_best_args

python main.py --dataset seq-cifar100 --model quadaug --buffer_size 1000 --load_best_args

python main.py --dataset seq-cifar10-blurry --model quadaug --buffer_size 1000 --load_best_args

python main.py --dataset seq-miniimg --model quadaug --buffer_size 1000 --load_best_args
```

------

# 🧩 Extending the Framework

## ➕ Add a New Model

1. Add your model to the `models/` folder.
2. Register it in:

```
models/__init__.py
```

Modify the `get_all_models()` function accordingly.

------

## ➕ Add a New Dataset

1. Add your dataset class to the `datasets/` folder.
2. Register it in:

```
datasets/__init__.py
```

Modify the `get_all_datasets()` function accordingly.

------

## 📁 Dataset Path Configuration

The automatic dataset download path is defined in:

```
utils/conf.py
```

Modify:

```
base_path_dataset()
```

to your local dataset directory before running experiments.

------

# 🏗 Project Structure

```
.
├── models/           # Model implementations
├── datasets/         # Dataset definitions
├── utils/            # Utility functions
├── main.py           # Entry point
└── requirements.txt
```

