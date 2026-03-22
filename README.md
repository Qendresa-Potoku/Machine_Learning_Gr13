# Traffic Dataset Preprocessing Pipeline

<table>
  <tr>
    <td width="150" align="center" valign="center">
      <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/e/e1/University_of_Prishtina_logo.svg/1200px-University_of_Prishtina_logo.svg.png" width="120" alt="University Logo" />
    </td>
    <td valign="top">
      <p><strong>Universiteti i PrishtinÃ«s</strong></p>
      <p>Fakulteti i InxhinierisÃ« Elektrike dhe Kompjuterike</p>
      <p>Inxhinieri Kompjuterike dhe Softuerike - Programi Master</p>
      <p><strong>Projekti nga lÃ«nda:</strong> Machine Learning</p>
      <p><strong>StudentÃ«t (Gr. 13):</strong></p>
      <ul>
        <li>Olta Pllana</li>
        <li>QÃ«ndresa Potoku</li>
        <li>Besarta Berisha</li>
      </ul>
    </td>
  </tr>
</table>

---

## Table of Contents

- [Project Overview](#project-overview)
- [Repository Structure](#repository-structure)
- [Dataset Description](#dataset-description)
- [Implemented Modules](#implemented-modules)
- [Technologies Used](#technologies-used)
- [Installation & Setup](#installation--setup)
- [Results](#results)
- [Key Takeaways](#key-takeaways)

---

## Project Overview

This repository implements an end-to-end preprocessing and analysis workflow for a traffic dataset.
The pipeline is designed for machine learning preparation and supports:

- dataset scope selection (full vs sampled)
- feature engineering for temporal and route context
- data cleaning and domain-based filtering
- optional missing-value imputation
- encoding and scaling
- skewness analysis with generated plots
- IQR-based outlier analysis for continuous variables
- export of cleaned dataset and JSON report

The current run configuration in [data_analysis.py](data_analysis.py) uses the regression task with `delay_min` as target and exports results to [outputs/](outputs/).

## Repository Structure

```
Machine_Learning_Gr13/
|
|-- data_analysis.py                    # Main preprocessing pipeline
|-- skewness_utils.py                   # Skewness + histogram/boxplot generation
|-- traffic_dataset.csv                 # Input dataset
|-- README.md                           # Project documentation
|-- outputs/
|   |-- cleaned_dataset_regression.csv  # Final processed dataset
|   |-- cleaned_report_regression.json  # Detailed processing report
|   `-- skewness_plots/                 # Auto-generated skewness/outlier plots
`-- __pycache__/
```

## Dataset Description

The input file [traffic_dataset.csv](traffic_dataset.csv) contains route-level traffic observations.
Based on the pipeline typing groups, the core attributes are:

| Column Group | Columns |
|---|---|
| Numeric | `distance_km`, `duration_normal_min`, `duration_traffic_min`, `delay_min`, `temperature`, `wind` |
| Categorical | `origin`, `destination` |
| Temporal | `timestamp` |
| Binary | `is_weekend`, `rain` |
| Discrete | `hour`, `day_of_week` |

### Target Definition

- Regression mode: target = `delay_min`
- Classification mode (supported in code): target = `traffic_level` where
  - Low: `delay_min < 3`
  - Medium: `3 <= delay_min < 7`
  - High: `delay_min >= 7`

## Implemented Modules

### Main Pipeline: [data_analysis.py](data_analysis.py)

1. `choose_dataset_scope`
- Interactive choice between full dataset and random sample.

2. `analyze_data_types`
- Validates expected type groups and reports missing columns.

3. `feature_engineering`
- Creates features:
  - temporal: `hour`, `day_of_week`, `is_weekend`
  - route: `route`
  - traffic context: `is_rush_hour`
  - weather context: `is_bad_weather`
  - ratio feature: `speed_normal`
  - cyclic encoding: `hour_sin`, `hour_cos`

4. Missing-value strategy
- `suggest_missing_value_strategy` proposes column-wise actions.
- `apply_missing_value_strategy` applies imputation/drop rules when enabled.

5. `clean_data`
- Removes null rows and duplicates.
- Applies domain filters:
  - removes rows with `delay_min <= -5`
  - removes rows with `speed_normal <= 0.05`

6. `encode_features`
- One-hot encodes `route` using `pd.get_dummies`.

7. `drop_unused_columns`
- Drops raw columns no longer needed for modeling and removes leakage-prone column `duration_traffic_min`.

8. `create_target`
- Builds the target according to selected task.

9. `normalize_features`
- Supports `standard`, `minmax`, or no scaling.
- Excludes binary-like columns and target from scaling.

10. `analyze_data_quality` and `profile_completeness`
- Reports missingness, duplicates, completeness score, and per-column completeness.

11. `detect_outliers_iqr`
- IQR outlier counting on continuous numeric features with exclusion rules for prefixes/keywords.

12. `print_full_terminal_report`
- Prints shape, memory usage, column list, and summary statistics.

13. `save_outputs`
- Exports cleaned dataset CSV and detailed JSON report.

### Plot Utility: [skewness_utils.py](skewness_utils.py)

`analyze_skewness_with_graphics`
- computes skewness for numeric features
- ranks columns by absolute skewness
- generates histogram + boxplot per selected column
- calculates IQR outlier counts for plotted columns
- saves plots into [outputs/skewness_plots/](outputs/skewness_plots/)

## Technologies Used

- Python 3
- pandas
- numpy
- scikit-learn
- matplotlib
- pathlib / json / typing (standard library)

## Installation & Setup

### 1. Create and activate a virtual environment (recommended)

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2. Install dependencies

```powershell
pip install pandas numpy scikit-learn matplotlib
```

### 3. Run the pipeline

```powershell
python data_analysis.py
```

At runtime, choose:
- `1` for full dataset
- `2` (default) for sampled processing

## Results

From the latest generated report in [outputs/cleaned_report_regression.json](outputs/cleaned_report_regression.json):

- Task: regression
- Target: `delay_min`
- Original shape: 28,799 rows x 13 columns
- Processed shape: 24,103 rows x 56 columns
- Original memory: 8.42 MB
- Processed memory: 3.38 MB

### Cleaning Summary

- Rows after `dropna + drop_duplicates`: 28,779
- Domain-filtered rows removed: 4,676
  - `delay_min <= -5`: 726
  - `speed_normal <= 0.05`: 3,950

### Skewness Highlights

Most skewed continuous features (absolute skewness):
- `distance_km`: 3.1972
- `speed_normal`: 1.9447
- `duration_normal_min`: 1.2915
- `delay_min`: 1.2659

Generated plots:

| **Distance KM Distribution** | **Speed Normal Distribution** |
| :---: | :---: |
| ![Distance KM Distribution and Outliers](outputs/skewness_plots/distance_km.png) | ![Speed Normal Distribution and Outliers](outputs/skewness_plots/speed_normal.png) |
| *Figure 1: Histogram and boxplot for distance_km.* | *Figure 2: Histogram and boxplot for speed_normal.* |

| **Delay Distribution** | **Duration Normal Distribution** |
| :---: | :---: |
| ![Delay Distribution and Outliers](outputs/skewness_plots/delay_min.png) | ![Duration Normal Distribution and Outliers](outputs/skewness_plots/duration_normal_min.png) |
| *Figure 3: Histogram and boxplot for delay_min.* | *Figure 4: Histogram and boxplot for duration_normal_min.* |

| **Temperature Distribution** | **Wind Distribution** |
| :---: | :---: |
| ![Temperature Distribution and Outliers](outputs/skewness_plots/temperature.png) | ![Wind Distribution and Outliers](outputs/skewness_plots/wind.png) |
| *Figure 5: Histogram and boxplot for temperature.* | *Figure 6: Histogram and boxplot for wind.* |

| **Hour Sin Distribution** | **Hour Cos Distribution** |
| :---: | :---: |
| ![Hour Sin Distribution and Outliers](outputs/skewness_plots/hour_sin.png) | ![Hour Cos Distribution and Outliers](outputs/skewness_plots/hour_cos.png) |
| *Figure 7: Histogram and boxplot for hour_sin.* | *Figure 8: Histogram and boxplot for hour_cos.* |

### Output Files

- Cleaned dataset: [outputs/cleaned_dataset_regression.csv](outputs/cleaned_dataset_regression.csv)
- Detailed report: [outputs/cleaned_report_regression.json](outputs/cleaned_report_regression.json)

## Key Takeaways

- The project demonstrates a complete preprocessing workflow from raw traffic logs to ML-ready features.
- Feature engineering includes temporal, route-based, weather-aware, and cyclic transformations.
- Domain filtering and outlier analysis improve dataset reliability before modeling.
- The pipeline is modular and can be extended for both regression and classification tasks.
- Auto-generated report and plots make results reproducible and easy to inspect.

---

## License


