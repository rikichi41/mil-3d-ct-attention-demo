# 3D CT Multiple Instance Learning Attention Demo

This repository is a public, privacy-safe demo derived from a research project on
patient-level classification of 3D chest CT scans.

The original research data are not included because they contain private medical
information. This demo uses synthetic 3D volumes to show the core modeling idea:

- A patient is treated as a bag.
- A short 3D CT block is treated as an instance.
- A 3D CNN extracts block-level features.
- Max pooling or attention pooling aggregates block features into one patient-level prediction.

## Why Multiple Instance Learning?

Full 3D CT volumes can be too large to process as a single tensor under realistic
GPU memory constraints. One practical strategy is to split each volume into
short 3D blocks, such as 16 slices per block, and learn a patient-level label
from the set of blocks.

This turns the task into a multiple instance learning problem:

```text
patient = bag
3D block = instance
patient label = bag label
```

The key question is how to aggregate instance-level information. This demo
compares:

- `max`: the most suspicious block determines the patient prediction.
- `attention`: the model learns a soft importance weight for each block.
- `gated_attention`: a common extension of attention-based MIL.

## What Is Included

```text
.
├── README.md
├── configs/
│   └── synthetic_demo.json
├── docs/
│   └── research_note.md
├── src/
│   └── mil3d_demo/
│       ├── data.py
│       ├── models.py
│       └── train.py
├── train_demo.py
└── requirements.txt
```

## Quick Start

Create an environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run a small CPU demo:

```bash
python train_demo.py --model attention --epochs 5 --device cpu
```

Compare with max pooling:

```bash
python train_demo.py --model max --epochs 5 --device cpu
```

Try gated attention:

```bash
python train_demo.py --model gated_attention --epochs 5 --device cpu
```

## Synthetic Data

The synthetic dataset creates small 3D patient bags. Positive patients contain
one or more lesion-like Gaussian blobs in a small subset of blocks. Negative
patients do not contain those blobs.

This is not a medical simulation. It is only a safe toy task that demonstrates
the learning setup without using private CT images or labels.

## Privacy And Reproducibility

This repository intentionally excludes:

- Raw CT images
- DICOM files
- Patient identifiers
- Clinical labels
- Checkpoints trained on private data
- Experiment logs from the private research environment

The demo can be run end-to-end with synthetic data only.

## License

This demo code is released under the MIT License.

## Research Context

In the private research project, the main focus is not only improving a backbone
3D CNN, but also evaluating whether attention-based patient-level aggregation is
more suitable than max aggregation when only patient-level labels are available.

MedicalNet-style pretrained 3D CNN weights can be used as a backbone
initialization, but this public demo keeps the model small and self-contained.
