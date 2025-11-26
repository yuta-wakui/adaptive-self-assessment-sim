# adaptive-self-assessment-sim

## Overview
This repository simulates **a machine-learning-driven adaptive self-assessment system** that presents evaluation items dynamically and aims to achieve the same assessment accuracy with fewer evaluation items compared to using the full-length items.

The framework supports simulations for both:
- **WS1** — Single-session self-assessment (one assessment session)
- **WS2** — Two-session self-assessment (previous + current session)

## Description

## Requirement
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
### WS1 Simulation (Single-session)
Run adaptive simulation using single-session datasets:
```bash
python scripts/run_ws1_sim.py \
  --data_dir data/sample/ws1_sample.csv \
  --rc 0.80 \
  --ri 0.70 \
  --k 5
```
### WS2 Simulation (Two-session)
Run adaptive simulation using two-session datasets:
```bash
python scripts/run_ws2_sim.py \
  --data_dir data/sample/ws1_sample.csv \
  --rc 0.80 \
  --ri 0.70 \
  --k 5
```
### Threshold Comparison (RC × RI)
Automatically run simulations for multiple confidence thresholds:
#### WS1
```bash
python scripts/compare_thresholds_ws1.py
```
#### WS2
```bash
python scripts/compare_thresholds_ws2.py
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