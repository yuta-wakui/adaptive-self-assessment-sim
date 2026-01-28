# adaptive-self-assessment-sim

## Overview
This repository provides a simulation framework for a **machine-learning-based adaptive self-assessment system**.  
The system dynamically selects or complements checklist items based on response patterns and previous assessment results, aiming to **achieve the same assessment accuracy with fewer items** compared to using the full checklist.

The framework supports:
- **WS1 — Single-session self-assessment**
- **WS2 — Two-session self-assessment (previous + current session)**

## Motivation
Rubrics and checklists can support self-understanding and learner growth.
However, a major challenge is **assessment fatigue caused by a large number of items**, which can reduce motivation and response quality.

This project simulates an **adaptive self-assessment mechanism** that:
- presents items sequentially (currently random selection),
- complements unanswered items using machine learning when reliability is sufficient,
- predicts an overall rubric-based score after all checklist items are answered or complemented.

The objective is to **reduce the number of required answers without degrading overall score prediction performance**.

## Adaptive-self-assessment Algorithm
The decision process is controlled by two confidence thresholds:

- **Rc**: threshold for complementing remaining items (item-level completion)
- **Ri**: threshold for evaluating overall score predictions (overall-level confidence)

Pseudo-code below summarizes the process:

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
### Install
simulation development install:
```bash
pip install -e .
```
## Usage
### Configuration
Simulations are controlled by a YAML configuration file.

### Run simulation (cross^validation)
run with a config file:
```bash
python scripts/run_sim.py --config configs/config.yaml
```
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