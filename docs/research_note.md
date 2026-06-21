# Research Note

This repository is a simplified public demonstration of a private 3D CT
classification project.

## Problem Setting

The private project handles chest CT volumes where labels are available at the
patient level. Because each full 3D volume can be large, the volume is divided
into short 3D blocks. Each block is encoded by a 3D CNN, then block features are
aggregated into one patient-level prediction.

## MIL Formulation

```text
patient = bag
3D block = instance
patient label = bag label
```

This setting is naturally handled as multiple instance learning because the
training label is not assigned to each block.

## Aggregation Methods

Max aggregation is a simple baseline: the most abnormal-looking block dominates
the prediction.

Attention aggregation learns a weight for each block and forms a weighted
patient representation. This can be more flexible when abnormal findings are
distributed across multiple blocks or when a single maximum score is unstable.

## Why This Public Version Uses Synthetic Data

The original CT data and labels are private medical data. They are not included
in this repository. The synthetic task is only intended to demonstrate the
modeling pattern, code organization, and training loop.
