# Data Policy

This project uses self-assessment data collected in an educational setting.
Due to privacy and ethical constraints, **the original datasets are NOT included in this repository**.

## Directory Structure

- `synthetic/`  
  Synthetic datasets generated from the original data.  
  Used for internal development and analysis.  
  Only a subset may be shared as public samples.

- `processed/`  
  Preprocessed versions of the original data used in the experiments.  
  **Not tracked by Git. Not publicly available.**

- `sample/`  
  Publicly available **sample datasets** for reproducing the simulation pipeline
  on a limited subset (e.g., the "information literacy" skill).

## Public Sample Files

Currently included:

- `sample/ws2_1_information_130_processed.csv`  
- `sample/ws2_1_information_1300_processed.csv`  
- `sample/1_syntheticdata_informationliteracy.csv`  

These files are:
- anonymized
- limited to a single skill ("information literacy")
- intended for demonstration and example usage only

For full-scale experiments, please prepare your own datasets that follow
the same column specification as defined in `adaptive_self_assessment/spec.py`.
