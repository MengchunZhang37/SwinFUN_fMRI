# SwinFUN_fMRI

This repository contains:
- **`SwiFUN_original/`** — Original implementation used by the literature ([paper](https://direct.mit.edu/imag/article/doi/10.1162/imag_a_00440/126557/Predicting-task-related-brain-activity-from), [GitHub](https://github.com/Transconnectome/SwiFUN)).
- **`SwiFUN_clean/`** — Debugged & cleaned implementation (recommended for running).
- **`cleaned_data/`** — Sample **data-preparation structure** based on the open-source dataset **DS000030** (layout reference only).

## Repository Layout
```text
SwiFUN_original/            # Reference only
SwiFUN_clean/
  ├─ env/
  │   ├─ requirements.txt
  │   ├─ py39.yaml
  │   └─ build_env.sh       # Slurm job script to build the environment
  ├─ project/               # All Python code
  ├─ sample_scripts/        # Only for 3D Swin Transformer
  └─ reproduce.ipynb        # Reproducible pipeline entry point
cleaned_data/               # Sample data directory layout (DS000030-based)



## How to Run

### 1. Set up the environment
```bash
cd SwiFUN_clean/env
sbatch build_env.sh

This job creates the Conda environment defined by py39.yaml and installs packages from requirements.txt.
When it finishes, activate the environment as printed in the job output (e.g., conda activate py39).
If you need a Jupyter kernel locally:

```bash
pip install ipykernel
python -m ipykernel install --user --name py39 --display-name "Python (py39)"


### 2. Prepare data
Follow the layout shown in cleaned_data/ (DS000030-based), or update paths in the notebook accordingly.

### 3. Run the pipeline
```bash
SwiFUN_clean/reproduce.ipynb



