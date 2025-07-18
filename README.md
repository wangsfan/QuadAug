## Introduction
This is our implementation of our paper *Quadruplet Augmentation with Attribute and Structure Invariance for Online Continual Learning*. 

**Abstract**:
Online Continual Learning (OCL) aims to learn from non-independently and identically distributed streaming data without relying on task boundaries during the training and testing stages. Previous OCL methods tend to suffer from two issues: shortcut feature trap and limited plasticity. We reveal that the two issues lead to two requirements of OCL: attribute invariance and structure invariance, where the former requires to capture the attributes of objects which maintain invariance during all sessions of OCL, and the latter requires to capture the relation of different attributes during OCL. From the causal analysis and Fourier transform perspectives, we propose the Quadruplet Augmentation (QuadAug), which preserves attribute and structure invariance by data and channel augmentation with four modules: P-aug, A-aug, CI-aug, CS-aug. First, we build a fine-grained structural causal model of OCL, and isolate the session-invariant attributes from confounding factors. Then, based on the observation of different roles of amplitude and phase components of Fourier domain during knowledge transfer, we propose a bidirectional data augmentation strategy, which effectively intervening on subtle confounding factors of OCL and preserves attribute invariance (P-aug and A-aug). Finally, we decompose the structure invariance into two necessary conditions: channel independence and channel sufficiency, and preserve channel independence via an inter channel discrepancy constraint (CI-aug) and channel sufficiency via adversarial learning between a channel sufficiency detector and classifiers (CS-aug), facilitating structure invariance across different sessions. Experimental results show that, QuadAug produces significant improvement against traditional OCL methods on three sequential datasets and three blurry datasets, with 2.3% to 6.9% improvement on Seq-CIFAR10 and 1.3% to 2.7% improvement on CIFAR10-Blurry30.

## 📚 Dependencies
- torch>=2.1.0
- numpy
- torchvision
- kornia>=0.7.0
- Pillow
- timm==0.9.8
- tqdm
- onedrivedownloader
- ftfy
- regex
- pyyaml 


## ⚙️ Setup

- 📥 Install with `pip install -r requirements.txt` or run it directly with `uv run python main.py ...`
- 🚀 Use `main.py` to run experiments.
- 🧩 New models can be added to the `models/` folder.
- 📊 New datasets can be added to the `datasets/` folder.


## 🧪 Examples


### Run a model

- Use python main.py to run experiments.
- Use argument --load_best_args to use the best hyperparameters for each of the evaluation setting from the paper.
- To reproduce the results for QuadAug in the paper run the following python main.py --dataset <dataset> --model quadaug --buffer_size <buffer_size> --load_best_args

```bash
python main.py --dataset seq-cifar10 --model quadaug --buffer_size 1000 --load_best_args
python main.py --dataset seq-cifar100 --model quadaug --buffer_size 1000 --load_best_args
python main.py --dataset seq-cifar10-blurry --model quadaug --buffer_size 1000 --load_best_args
python main.py --dataset seq-miniimg --model quadaug --buffer_size 1000 --load_best_args
```

### Build a new model/dataset

New models can be added to the models/ folder. New datasets can be added to the datasets/ folder.




