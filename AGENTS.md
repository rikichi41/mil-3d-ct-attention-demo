# AGENTS.md

## Purpose

This repository is a public, privacy-safe demo of 3D CT multiple-instance learning (MIL).
It is intended to show the modeling idea and code structure without exposing private research data.

When working in this repository, treat it as a portfolio/public demo repository, not as the private research workspace.

## Communication Style

- Explain changes in Japanese when talking with the user.
- Keep code identifiers, file names, commands, branch names, and package names in English.
- Pull request titles and descriptions may be written in Japanese.
- Public-facing README/docs may stay in English unless the user asks to localize them.

## Privacy And Safety Rules

Never add private or sensitive research materials to this repository.

Do not commit:

- Real CT images, DICOM files, or other medical image data.
- Patient identifiers or clinical labels.
- `.npy`, `.npz`, memmap caches, or derived arrays from private data.
- Model checkpoints or pretrained weights trained on private data.
- Private experiment logs, patient-level prediction CSVs, or block-level prediction CSVs.
- Server paths, credentials, tokens, private keys, or other secret information.

This demo should remain runnable with synthetic data only.

## Scientific Claims

- Do not describe this repository as a clinically validated system.
- Do not imply that the synthetic demo results represent real medical performance.
- When explaining results, clearly state that the demo data are synthetic and used only to demonstrate the MIL workflow.
- Keep public claims conservative and reproducible.

## Repository Structure

Important files:

- `README.md`: public-facing overview and quick start.
- `train_demo.py`: command-line entry point for running the synthetic demo.
- `configs/synthetic_demo.json`: default synthetic-data/demo configuration.
- `src/mil3d_demo/data.py`: synthetic bag/block dataset generation.
- `src/mil3d_demo/models.py`: 3D CNN backbone and MIL pooling models.
- `src/mil3d_demo/train.py`: training and evaluation logic.
- `docs/research_note.md`: short public research-context note.

## Development Guidelines

- Keep the demo small enough to run on CPU for quick checks.
- Prefer clear, readable code over complex research-only optimizations.
- Keep dependencies minimal and listed in `requirements.txt`.
- If adding new examples, use synthetic or fully public toy data only.
- If adding generated outputs, keep them small and non-sensitive.
- Do not copy code, logs, config files, or result files from the private research repository unless they have been carefully sanitized.

## Git Guidelines

- Check `git status` before editing and before committing.
- Stage only files related to the current task.
- Do not use `git add .` unless the repository has been inspected and the user explicitly wants all changes included.
- Commit messages should be short and descriptive.
- Pushing to `main` is acceptable for small documentation/demo maintenance tasks if the user requested it.
  For larger behavior changes, prefer a feature branch and PR.

## Recommended Checks

For code changes, run at least a small CPU smoke test when feasible:

```bash
python train_demo.py --model attention --epochs 1 --device cpu
```

For documentation-only changes, inspect the rendered Markdown mentally and ensure it does not include private information.
