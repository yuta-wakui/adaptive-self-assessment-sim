# Data Specification

This directory contains sample datasets for running simulations.

## Directory structure

```
data
    ├── README.md
    └── sample
        ├── ws1
            └──ws1_data_sample.csv
        └── ws2
            └──ws2_data_sample.csv
```

- `sample/` : Public toy datasets for quick testing.
- Real datasets are not included in this repository.

---

## WS1 Dataset

File: [ws1_data_sample.csv](sample/ws1/ws1_data_sample.csv)


Each row represents one user.

| Column | Type | Description |
|------|------|--------------|
| user_id | int | Unique user identifier |
| overall_score | int | Ground-truth overall evaluation score (1-4) |
| reflection_length | int | Length of free-text reflection (character count) |
| item_1 ~ item_15 | int | Responses to individual evaluation items (0-2) |

---

## WS2 Dataset

File: [ws2_data_sample.csv](sample/ws2/ws2_data_sample.csv)


Each row represents one user with both past and current information.

| Column | Type | Description |
|------|------|--------------|
| user_id | int | Unique user identifier |
| past_overall_score | int | Overall score from past evaluation (1-4) |
| past_reflection_length | int | Past reflection length (character count) |
| past_item_1 ~ past_item_15 | int | Past responses to evaluation items (0-2) |
| current_overall_score | int | Ground-truth current overall score (1-4) |
| current_reflection_length | int | Current reflection length (character count) |
| current_item_1 ~ current_item_15 | int | Current responses to evaluation items (0-2) |

---

## Notes

- These sample datasets are **synthetically generated** and are **not based on real user data**.
- The following information is required for running the simulations:
  - User identifier  
  - Overall evaluation score  
  - Individual item responses  
- Column names do **not** need to match exactly, as they can be customized via the configuration file [configs/config.yaml](../configs/config.yaml).