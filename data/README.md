# Data Directory

This project uses self-assessment data of soft skills collected in an educational context to run simulations for **Machine Learning-driven Adaptive Self-Assessment**

Due to privacy, ethics, and institutional data-handling policies, **the original datasets and full synthetic datasets used in the experiments are NOT included in this repository**

Only small, safe-to-share sample data is provided for demonstration and usability testing.

## Directory Structure

### **`synthetic/`**
Synthetic datasets generated from the real self-assessment data.

The datasets were created using the statistical analysis software **HAD** by generating multivariate data from the **covariance matrix** of the original datasets.

These datasets replicate statistical properties but are derived from non-public data.
**They are not tracked by Git and not publicly available.**

### **`processed/`**
Preprocessed versions of the synthetic data, used as input for simulation experiments.

Processing steps include:
- Converting continuous synthetic values into discrete integer scales
- Handling missing values
- Cleaning and normalizing column names

These datasets represent the actual inputs used in the experiments, but **are not included in this repository**

### **`sample/`**
Example datasets generated for demonstration purposes.

These samples are small, randomly generated datasets that mimic the structure of the
processed synthetic data. They are safe to publish and can be used to test the
simulation pipeline, verify the expected input format, and explore the library's behavior.