# adaptive-self-assessment-sim

A simulation framework for machine-learning-driven adaptive self-assessment systems, 
designed to reduce response burden while maintaining assessment accuracy.

## Overview
This repository provides a simulation framework for a **machine-learning-based adaptive self-assessment system**.  
The system dynamically selects or complements assessment items based on response patterns and previous assessment results, aiming to **achieve the same assessment accuracy with fewer items** compared to using the full set of items.

The framework supports:
- **WS1 — Single-session self-assessment**
- **WS2 — Two-session self-assessment (previous + current session)**

## Motivation
Structured self-assessment can support self-understanding and personal growth.  
However, a major challenge is **assessment fatigue caused by a large number of questions or items**, which can reduce motivation and response quality.

This project simulates an **adaptive self-assessment mechanism** that:
- presents items sequentially (currently random selection),
- complements unanswered items using machine learning when reliability is sufficient,
- predicts an overall score after all items are answered or complemented.

The objective is to **reduce the number of required responses without degrading overall prediction performance**.

## Adaptive-self-assessment Algorithm
The decision process is controlled by **RC**, which is a confidence threshold used to determine whether unanswered items can be safely complemented by the model.

The pseudo-code below summarizes the overall process:

```pseudo
Inputs:
    I      : full set of items
    Pra    : previous overall score (WS2 only, else null)
    Pca    : previous item results (WS2 only, else null)
    Rc     : confidence threshold for unanswered item completion

Outputs:
    pred_R  : predicted overall score
    Ca      : final set of answered or complemented items

Initialization:
    C  ← I  # unanswered item set
    Ca ← {} # answered or complemented item list

Loop:
    while C is not empty:
        ci ← select_next_item(C)   # e.g., random, entropy-based, or model-guided
        ans ← query_response(ci)  # ask the user and record response
        Ca[ci] = ans
        remove ci from C

        # predict unanswered items
        (pred_C, conf_C) ← completion_model.predict(C, Ca, Pra, Pca)

        for each j in C:
            if conf_C[j] ≥ Rc:
                Ca[j] = pred_C[j]
                remove j from C

    # after all item scores are obtained (answer or completion), predict overall score using Ca, Pra, Pca
    (pred_R, conf_R) ← overall_estimator.predict(Ca, Pra, Pca)

Return pred_R, Ca
```

## Requirements
### Clone repositories
```bash
git clone https://github.com/yuta-wakui/adaptive-self-assessment-sim.git
cd adaptive-self-assessment-sim
```

### (Recommended) Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate # Windows: venv\Scripts\activate
```
### Install (development mode)
```bash
pip install -e .
```

## Configuration

Simulations are fully controlled by a YAML configuration file.

As an example, a YAML file for running simulations on the provided sample dataset is available at  
[configs/config.yaml](configs/config.yaml).

A typical configuration structure is shown below:

```yaml
mode: ws2  # ws1 or ws2

data:
  common:
    skill_name: "sample"
    id_col: "ID"
    ignore_items:
      - "reflection_length"

  ws1:
    input_path: "data/sample/ws1/ws1_data_sample.csv"
    overall_col: "overall_score"
    item_cols:
      - "item_1"
      - "item_2"
      - "item_3"

  ws2:
    input_path: "data/sample/ws2/ws2_data_sample.csv"
    past_overall_col: "past_overall"
    past_item_cols:
      - "past_item_1"
      - "past_item_2"
    current_overall_col: "current_overall"
    current_item_cols:
      - "current_item_1"
      - "current_item_2"
  
model:
  item_model:
    type: "logistic_regression"
    params:
      max_iter: 1000
  overall_model:
    type: "logistic_regression"
    params:
      max_iter: 1000

thresholds:
  RC: 0.8   # confidence threshold for item completion
  RI: 0.7   # confidence thresholds for accepting overall

cv:
  folds: 5
  stratified: true
  random_seed: 42

results:
  save_csv: true
  output_dir: "outputs/results"
  timestamped: true
  filename_suffix: "sample"
  save_fold_results: true

logging:
  save_logs: true
  log_dir: "outputs/logs"
  timestamped: true
```

### Key Sections Explained
`mode`

Select the simulation type:
- `ws1`: single-session simulation
- `ws2`: two-session simulation (use past + current data)

---
`data`

Specifies dataset structure.

Common settings:
- `skill_name`: label used in logs/results
- `id_col`: use identifier column
- `ignore_items`: columns to be dropped before simulation (Optional)

WS1:
- `input_path`: path to CSV
- `overall_col`: ground-truth overall score
- `item_cols`: question item columns

WS2:
- Extends WS1 by separating past/current session columns:
  - `past_overall_col`, `past_item_cols`
  - `current_overall_col`, `current_item_cols`

---
`model`
Defines model used internally:
- `item_model`: used for predicting remaining items
- `overall_model`: used for predicting overall score

Currently supports:
- `logistic_regression` (via scikit-learn)
Model parameters are passed directly to scikit-learn.

---
`thresholds`

Controls the adaptive behavior:
- `RC`: threshold for complementing remaining items
- `RI`: confidence thresholds for accepting overall prediction

---
`results`
Controls result file outputs:
- `output_dir`: base directory
- `timestamped`: whether to create timestamped subfolder
- `filename_suffix`: suffix added to output filenames
- `save_fold_results`: whether to export per-fold results

---
`logging`
Controls detailed use-level logs:
- `save_logs`: whether to save logs
- `log_dir`: base directory for logs
- `timestamped`: whether to timestamp directories

## Usage
### Run simulation (cross-validation)
```bash
python scripts/run_sim.py --config configs/config.yaml
```
By default, results and logs will be saved under:
- `outputs/results/`
- `outputs/logs/`

## Testing
Run all unit and integration tests:
```bash
pytest -s
```
Tests are located in:
```bash
tests/
```

## License
MIT License