# adaptive-self-assessment-sim

A simulation framework for **machine-learning–driven adaptive self-assessment**.  
This repository evaluates how dynamic question selection can reduce the number of required checklist items while maintaining assessment accuracy.

---

## Features

- **WS1 (One-shot Assessment)**  
  Uses a single self-assessment dataset to simulate dynamic question selection.

- **WS2 (Two-step Assessment).**  
  Uses prior + current evaluation data to perform more accurate dynamic estimation.

- **Dynamic Question Selection**  
  Questions are selected adaptively based on previously known responses.

- **Adaptive Completion (補完)**  
  Remaining unanswered items are predicted using ML.

- **Evaluation Metrics**
  - Number of answered vs. complemented items  
  - Reduction rate  
  - Accuracy over confidence threshold  
  - Coverage  
  - Execution time  

---

## Installation

```bash
git clone https://github.com/yuta-wakui/adaptive-self-assessment-sim.git
cd adaptive-self-assessment-sim

# (Recommended) Create virtual environment
python -m venv venv
source venv/bin/activate     # Windows: venv\Scripts\activate

# Install package
pip install -e .

# Install dependencies
pip install -r requirements.txt
```

## Project Structure
adaptive-self-assessment-sim/
├── src/adaptive_self_assessment/
│ ├── selector/ # Dynamic question selection logic
│ ├── predictor/ # Item-level & overall score prediction
│ ├── simulation/ # WS1 / WS2 simulation core
│ ├── spec/ # Column & dataset specifications
│ └── ...
├── scripts/
│ ├── run_ws1_sim.py
│ ├── run_ws2_sim.py
│ ├── compare_thresholds_ws1.py
│ ├── compare_thresholds_ws2.py
├── tests/ # pytest tests
├── data/ # Raw & processed datasets (ignored except sample)
├── outputs/ # Result CSVs and logs
└── requirements.txt

---

##  Usage

### WS1 Simulation (One-shot)
```bash
python scripts/run_ws1_sim.py
```

### WS2 Simulation (Two-shot)
```bash
python scripts/run_ws2_sim.py
```
---

## Threshold Comparison (RC × RI)
Run simulations for multiple RC / RI combinations:

### WS1 Threshold Comparison

```bash
python scripts/compare_thresholds_ws1.py
```

### WS2 Threshold Comparison
```bash
python scripts/compare_thresholds_ws2.py
```

Outputs will be automatically generated under:
```bash
outputs/results_csv/ws1/
outputs/results_csv/ws2/
```

## Testing
Run all tests:
```bash
pytest -s
```

## License
MIT License