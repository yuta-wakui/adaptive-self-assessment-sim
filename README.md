# adaptive-self-assessment-sim

## Overview
This repository provides a simulation framework for a **machine-learning-based adaptive self-assessment system**.  
The system dynamically selects or complements checklist items based on response patterns, aiming to **achieve the same assessment accuracy with fewer items** compared to using the full checklist.

The framework supports:
- **WS1 — Single-session self-assessment**
- **WS2 — Two-session self-assessment (previous + current session)**

## Motivation
Inter-subjective assessment using rubrics and checklists can promote self-understanding and continuous learner growth.  
However, a major problem is **assessment fatigue caused by the large number of evaluation items**, which can decrease motivation and result quality.

To address this issue, this project simulates an **adaptive self-assessment mechanism** that:
- presents items sequentially based on the learner’s response history(currently random selection is implemented),
- complements unanswered items using machine learning when reliability is sufficient,
- and, once all checklist items are answered or complemented, predict a final rubric-based overall score using the completed set of items.

The objective is to **reduce the number of required items without degrading assessment accuracy**.

## Adaptive-self-assessment Algorithm
The system sequentially queries items and uses machine-learning-based completion to reduce the number of required responses while maintaining assessment accuracy. The decision process is controlled by two confidence thresholds:

- **Rc: threshold for complementing remaining items**
- **Ri: threshold for final score estimation**

Pseudo-code below summarizes the adaptive evaluation process:

```pseudo
Inputs:
    I      : full set of checklist items
    Pra    : previous overall score (WS2 only, else null)
    Pca    : previous checklist results (WS2 only, else null)
    Rc     : confidence threshold for item-level completion
    Ri     : confidence threshold for overall score prediction

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

    # after all checklist scores are obtained (answer or completion), predict overall score using Ca, Pra, Pca
    (pred_R, conf_R) ← overall_estimator.predict(Ca, Pra, Pca)
    
    if conf_R >= Ri:
        # if confidence is insufficient, additional human evaluation is required
        pred_R = request_manual_rating()

Return pred_R, Ca
```

## Requirement
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
### Install dependencies
install all required dependencies:
```bash
pip install -r requirements.txt
```
Or install via the package configuration:
```bash
pip install -e .
```
## Usage
### Simulation
#### Command-line Arguments
The simulation scripts accept the following main arguments:

| Argument     | Meaning                                           | Example                                                 |
| ------------ | ------------------------------------------------- | ------------------------------------------------------- |
| `--data_dir` | Directory containing the dataset (WS1 or WS2)     | `data/sample/ws1`                                       |
| `--rc`       | Confidence threshold for item-level completion    | 0.80                                                  |
| `--ri`       | Confidence threshold for overall score estimation | 0.70                                                  |
| `--k`        | Number of cross-validation folds                  | 5                                                    |
| `--output`   | Path to save simulation results                   | `outputs/results_csv/ws1/sim_results/rc0p80_ri0p70.csv` |

#### WS1 Simulation (Single-session)
Run adaptive simulation using single-session datasets:
```bash
python scripts/run_ws1_sim.py \
  --data_dir data/sample/ws1 \
  --rc 0.80 \
  --ri 0.70 \
  --k 5 \
  --output outputs/results_csv/ws1/sim_results/ws1_results_rc0p80_ri0p70_date.csv
```
#### WS2 Simulation (Two-session)
Run adaptive simulation using two-session datasets:
```bash
python scripts/run_ws2_sim.py \
  --data_dir data/sample/ws2 \
  --rc 0.80 \
  --ri 0.70 \
  --k 5 \
  --output outputs/results_csv/ws2/sim_results/ws2_results_rc0p80_ri0p70_date.csv
```
### Threshold Comparison (RC × RI)
Automatically run simulations for multiple confidence thresholds.

#### Command-line Arguments
The simulation scripts accept the following main arguments:

| Argument     | Meaning                                           | Example                                                 |
| ------------ | ------------------------------------------------- | ------------------------------------------------------- |
| `--data_dir` | Directory containing the dataset (WS1 or WS2)     | `data/sample/ws1`                                       |
| `--rc_values`       | List of confidence thresholds for item-level completion    | 0.7 0.8 0.9                                    |
| `--ri_values`       | List of confidence thresholds for overall score estimation | 0.6 0.7 0.8                                                 |
| `--k`        | Number of cross-validation folds                  | 5                                                   |
| `--output`   | Path to save simulation results                   | outputs/results_csv/ws1/cmp_thresholds/cmp_results.csv |

#### WS1
```bash
python scripts/compare_thresholds_ws1.py \
  --data_dir data/sample/ws1 \
  --rc_values 0.7 0.8 0.9 \
  --ri_values 0.6 0.7 0.8 \
  --k 5 \
  --output outputs/results_csv/ws1/cmp_thresholds/ws1_cmp_rc_ri_grid.csv
```
#### WS2
```bash
python scripts/compare_thresholds_ws2.py \
  --data_dir data/sample/ws2 \
  --rc_values 0.7 0.8 0.9 \
  --ri_values 0.6 0.7 0.8 \
  --k 5 \
  --output outputs/results_csv/ws2/cmp_thresholds/ws2_cmp_rc_ri_grid.csv
```
## Testing
Rim all unit and integration tests:
```bash
pytest -s
```
Tests are located in:
```bash
tests/
```

## License
MIT License