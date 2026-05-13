# Few-Shot Open-Set Audio Classification via Transductive Prototype Refinement and Class Logit Enhancement

 Drawing on latent-inlierness weighting and decoupled scoring for unknown-class samples, we propose a two-phase transductive method operating over a frozen audio encoder. First, each query sample is assigned a latent inlierness score that down-weights likely unknown-class samples, so that prototype refinement is driven primarily by known-class evidence. The refined prototypes are then directly optimized on a transductive loss combining support cross-entropy, inlierness-weighted conditional entropy minimization, and inlierness-weighted marginal entropy maximization, while open-set rejection uses a prior-adaptive free-energy score that adjusts its threshold with the prior proportion of unknown-class samples, decoupling detection from classification. Experiments on three audio datasets show our method achieves state-of-the-art results for few-shot open-set audio classification under multiple experimental conditions.

**Transductive Few-Shot Open-Set Audio Recognition in ROLE**



Evaluated on **ESC-50**, **FSD-Kaggle2018**, and **UrbanSound8K** across 1-shot / 5-shot settings and 20% / 50% / 80% outlier ratios.

## Repository Structure

```
ROLE/
 models/
   ├── role.py                  # ROLE — Prototype Reﬁnement-based Outlier-Logit Enhance-ment  
   └── ast_feature_extractor.py # Frozen AST audio encoder (feature extraction)
 data/
   ├── dataset_registry.py      # Dataset factory: get_dataset() / get_cache_dir()
   ├── episode_sampler.py       # Few-shot open-set episode sampler
   ├── cache_embeddings.py      # Load / save AST embedding caches
   ├── build_dataset_cache.py   # Build embedding cache from raw audio (run once)
   ├── esc50_dataloader.py      # ESC-50 dataset loader
   ├── fsdkaggle_dataloader.py  # FSD-Kaggle2018 dataset loader
   ├── urbansound_dataloader.py # UrbanSound8K dataset loader
   ├── audio_preprocessing.py   # Audio preprocessing utilities
   └── base_dataset.py          # Abstract base class for all datasets
 evaluation/
   ├── evaluator.py             # FewShotEvaluator — cross-fold eval loop
   └── metrics.py               # AUROC, inlier Accuracy, AUPR, ECE
 experiments/
   └── run_full_benchmark_worker.py  # Benchmark runner (single method x ratios)
 requirements.txt
```

## Setup

```bash
pip install -r requirements.txt
```

The audio backbone (`MIT/ast-finetuned-audioset-10-10-0.4593`) is downloaded automatically from HuggingFace on first use.

## Datasets
- **ESC-50**: 50 classes of environmental sounds, 40 clips per class.
```
https://github.com/karolpiczak/ESC-50
```
- **FSD-Kaggle2018**: 41 classes of Freesound audio
```
https://zenodo.org/records/2552860
```
- **UrbanSound8K**: 10 classes of urban sounds, 8732 clips total.
```
https://zenodo.org/records/1203745
```

## Build Embedding Cache

Extract and cache AST embeddings for each dataset (required before running experiments):

```bash
python data/build_dataset_cache.py --dataset esc50
python data/build_dataset_cache.py --dataset fsd
python data/build_dataset_cache.py --dataset urbansound
```

By default, embeddings are stored under `cache/<dataset>_embeddings/`. Pass `--batch_size` to control GPU memory usage.

## Run Benchmark

Evaluate a single method across all datasets, shots, and a given outlier ratio:

```bash
python experiments/run_full_benchmark_worker.py --method ROLE --ratio 20pct
```

`--ratio` can be `20pct`, `50pct`, `80pct`, or `all` (runs all three sequentially).  
Results are saved to `results/full_benchmark/<method>_<ratio>_<timestamp>.json`.


## Episode Configuration

| Parameter | Value |
|-----------|-------|
| Ways (inlier) | 5 |
| Ways (outlier) | 5 |
| Shots | 1 / 5 |
| Query per inlier class | 4 |
| Query per outlier class | 1 / 4 / 16 (for 20% / 50% / 80%) |
| Episodes per condition | 300 x 5 folds = 1500 |

## Citation
Anonymous
