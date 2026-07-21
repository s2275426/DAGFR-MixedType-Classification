# Classification with Mixed-Type Predictors and Doubly Adaptive Group-Fused Regularisation

This repository contains the official Python implementation and dataset pipeline for reproducing the numerical simulations (Section 4) and real-world application (Section 5) presented in the paper:

> **Classification with Mixed-Type Predictors and Doubly Adaptive Group-Fused Regularisation**  
> **Author:** Sitong Zhang  
> **Affiliation:** School of Informatics, University of Edinburgh  

---

## Path Configuration Notice

> **IMPORTANT BEFORE RUNNING:**  
> Before executing any scripts, please inspect the file path variables (e.g., `data_path`, `output_dir`, or file path strings in script headers) across the Python scripts in both `Simulation_Comparision/` and `Application/` and update them to match your local directory paths.

---

## Requirements & Environment

- **Python Version:** Python 3.9 or higher
- **Dependencies:**
  - `numpy` — Numerical computations and vector/matrix operations
  - `pandas` — Data manipulation and dataset aggregation
  - `scipy` — Optimization routines, statistical functions, and B-spline representations (`scipy.interpolate.BSpline`)
  - `scikit-learn` — Baseline models, cross-validation, feature scaling, and classification metrics
  - `joblib` — Multi-core parallel processing (`Parallel`, `delayed`)

You can install all necessary packages via `pip`:

```bash
pip install numpy pandas scipy scikit-learn joblib
```

---

## Repository Structure & Execution Guide

The repository is structured into two main working directories corresponding to the two experimental sections of the paper: `Simulation_Comparision/` (Section 4) and `Application/` (Section 5).

```text
.
├── Simulation_Comparision/          # Section 4: Controlled Simulation Experiments
│   ├── dgp.py                       # Data generation process for simulation setups
│   ├── lasso_simulation.py          # Baseline 1: Adaptive Lasso Classifier
│   ├── kernel_simulation.py         # Baseline 2: Weighted Kernel Classifier
│   ├── spline_simulation.py         # Baseline 3: Penalised B-Spline Framework
│   ├── dagfr_simulation.py          # Proposed: DAGFR Method
│   └── run_all_comparision.py       # Benchmark aggregation & performance evaluation
│
└── Application/                     # Section 5: MotionSense Real Data Application
    ├── Data/                        # Data processing & feature integration
    │   ├── main 01.41.58.py         # Automated data merging script
    │   ├── A_DeviceMotion_data/     # [External] Downloaded from MotionSense repository
    │   ├── data_subjects_info.csv   # [External] Downloaded from MotionSense repository
    │   └── combined_devices_data.csv# [Generated] Output from running main 01.41.58.py
    └── Experiment/                  # Preprocessing & model fitting
        ├── motionsense_preprocess.py# MotionSense preprocessing & windowing pipeline
        ├── Adaptive Lasso.py        # Adaptive Lasso framework
        ├── Kernel.py                # Weighted Kernel framework
        ├── Spline.py                # P-Spline framework
        └── DAGFR.py                 # Proposed DAGFR framework
```

---

### 1. Numerical Simulations (Section 4)

To reproduce the synthetic evaluation results reported in Section 4:

1. Navigate to the simulation directory:
   ```bash
   cd Simulation_Comparision
   ```

2. Generate the synthetic benchmark datasets:
   ```bash
   python dgp.py
   ```

3. Run individual method simulations sequentially:
   ```bash
   python lasso_simulation.py
   python kernel_simulation.py
   python spline_simulation.py
   python dagfr_simulation.py
   ```

4. Aggregate benchmark metrics across all methods:
   ```bash
   python run_all_comparision.py
   ```
   This outputs the consolidated comparative performance table (including classification accuracy, feature selection metrics, and parameter dimensions) discussed in Section 4.

---

### 2. Real Data Application (Section 5: MotionSense Benchmark)

To reproduce the real-world wearable sensor human activity recognition (HAR) evaluation reported in Section 5:

#### Step 1: Download Raw Dataset
Due to file size constraints, raw dataset files are not tracked in this GitHub repository. Please fetch the official MotionSense dataset from its source repository:

1. Clone or download the MotionSense repository from GitHub:
   ```bash
   git clone https://github.com/mmalekzadeh/motion-sense.git
   ```
2. Copy the folder `A_DeviceMotion_data/` and the metadata file `data_subjects_info.csv` into the `Application/Data/` directory of this repository:
   ```text
   Application/Data/
   ├── A_DeviceMotion_data/
   ├── data_subjects_info.csv
   └── main 01.41.58.py
   ```

#### Step 2: Data Merging & Preprocessing
1. Navigate to `Application/Data` and execute the data merging script:
   ```bash
   cd Application/Data
   python "main 01.41.58.py"
   ```
   *This script merges `A_DeviceMotion_data/` and `data_subjects_info.csv` to generate `combined_devices_data.csv`.*

2. Navigate to `Application/Experiment`:
   ```bash
   cd ../Experiment
   ```

3. Execute feature preprocessing and windowing:
   ```bash
   python motionsense_preprocess.py
   ```

#### Step 3: Model Fitting & Evaluation
Run model evaluations sequentially to compute performance metrics and parameter compression statistics:
```bash
python "Adaptive Lasso.py"
python Kernel.py
python Spline.py
python DAGFR.py
```
*Each script outputs its fitted parameters, test classification performance, and feature compression statistics corresponding to the MotionSense empirical benchmark in Section 5.*
