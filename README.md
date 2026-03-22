# Traffic Dataset Preprocessing Pipeline

<table>
  <tr>
    <td width="150" align="center" valign="center">
      <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/e/e1/University_of_Prishtina_logo.svg/1200px-University_of_Prishtina_logo.svg.png" width="120" alt="University Logo" />
    </td>
    <td valign="top">
      <p><strong>Universiteti i Prishtinës</strong></p>
      <p>Fakulteti i Inxhinierisë Elektrike dhe Kompjuterike</p>
      <p>Inxhinieri Kompjuterike dhe Softuerike - Programi Master</p>
      <p><strong>Projekti nga lënda:</strong> Machine Learning</p>
      <p><strong>Studentët (Gr. 13):</strong></p>
      <ul>
        <li>Olta Pllana</li>
        <li>Qëndresa Potoku</li>
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
- data cleaning with median imputation + deduplication
- IQR outlier counting (reporting only) and percentile winsorization
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

1. **Dataset Scope Selection (choose_dataset_scope)**
- **Functionality:** Lets the user choose full-dataset processing or sampled processing.
- **Logic:** Uses interactive input and reproducible random sampling when sample mode is selected.

![Dataset Scope Selection](ReadMe-Images/Dataset%20Scope%20Selection.png)

2. **Data Type Analysis (analyze_data_types)**
- **Functionality:** Checks expected feature groups (numeric, categorical, temporal, binary, discrete).
- **Logic:** Verifies column presence per group and prints dtype diagnostics for each column.

![Data Type Analysis](ReadMe-Images/Data%20Type%20Analysis.png)

3. **Feature Engineering (feature_engineering)**
- **Functionality:** Creates additional features for time, weather, route context, and cyclic hour encoding.
- **Logic:** Derives hour, day_of_week, is_weekend, route, is_rush_hour, is_bad_weather, speed_normal, hour_sin, and hour_cos.

![Feature Engineering](ReadMe-Images/Feature%20Engineering.png)

4. **Data Cleaning (clean_data)**
- **Functionality:** Handles missing values and duplicates while preserving rows.
- **Logic:** Fills numeric nulls with median values, removes duplicates, reports selected-column IQR counts, and applies 1%-99% winsorization to `delay_min` and `speed_normal`.

![Data Cleaning](ReadMe-Images/Data%20Cleaning.png)

5. **Categorical Encoding (encode_features)**
- **Functionality:** Converts route text into model-ready binary indicators.
- **Logic:** Uses one-hot encoding via pd.get_dummies for the route feature.

![Categorical Encoding](ReadMe-Images/Categorical%20Encoding.png)

6. **Column Pruning (drop_unused_columns)**
- **Functionality:** Drops raw, non-model columns and leakage-prone fields.
- **Logic:** Removes timestamp, origin, destination, route, hour, and duration_traffic_min.

![Column Pruning](ReadMe-Images/Column%20Pruning.png)

7. **Target Creation (create_target)**
- **Functionality:** Defines the prediction target based on selected ML task.
- **Logic:** Uses delay_min for regression or generates traffic_level bins for classification.

![Target Creation](ReadMe-Images/Target%20Creation.png)

8. **Normalization (normalize_features)**
- **Functionality:** Scales continuous numeric features for model compatibility.
- **Logic:** Applies StandardScaler or MinMaxScaler while excluding target and binary-like columns.

![Normalization](ReadMe-Images/Normalization.png)

9. **Quality and Completeness (analyze_data_quality, profile_completeness)**
- **Functionality:** Reports dataset quality after transformations.
- **Logic:** Computes missing values, duplicates, quality score, and completeness metrics.

![Quality and Completeness](ReadMe-Images/Quality%20and%20Completeness.png)

10. **Outlier Detection (detect_outliers_iqr)**
- **Functionality:** Measures outlier counts for continuous features.
- **Logic:** Uses IQR bounds with exclusion rules for encoded, binary, and low-cardinality columns.

![Outlier Detection](ReadMe-Images/Outlier.png)

11. **Terminal Report (print_full_terminal_report)**
- **Functionality:** Generates a full end-of-pipeline textual summary.
- **Logic:** Prints shape changes, memory usage, numeric summaries, and target distribution.

12. **Output Export (save_outputs)**
- **Functionality:** Saves final artifacts for downstream work.
- **Logic:** Writes cleaned dataset CSV and structured JSON report into outputs.

![Output Export](ReadMe-Images/Output%20Export.png)

### Plot Utility: [skewness_utils.py](skewness_utils.py)

`analyze_skewness_with_graphics`
- computes skewness for numeric features
- ranks columns by absolute skewness
- generates histogram + boxplot per selected column
- calculates IQR outlier counts for plotted columns
- saves plots into [outputs/skewness_plots/](outputs/skewness_plots/)

![Skewness Analysis](ReadMe-Images/Skewness.png)

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
- Run mode: full dataset
- Original shape: 32,070 rows x 13 columns
- Processed shape: 32,070 rows x 63 columns

### Cleaning Summary

- Rows before cleaning: 32,070
- Rows after median-imputation + deduplication: 32,070
- Nulls handled by median imputation: 60 -> 0
- Median-imputed columns: `temperature`, `wind`, `rain`
- IQR outlier row removal: disabled (`0` rows removed)
- Winsorization (1%-99%) applied to:
  - `delay_min`: `p01=-5.55`, `p99=9.9062`, clipped values = `641`
  - `speed_normal`: `p01=0.0`, `p99=0.768678`, clipped values = `106`

Selected-column IQR outlier counts (reporting only):
- `delay_min`: 4,328
- `speed_normal`: 6,270
- `distance_km`: 2,046
- `duration_normal_min`: 755

### Skewness Highlights

Most skewed continuous features (absolute skewness):
- `distance_km`: 2.7787
- `delay_min`: 1.1848
- `wind`: 0.7558
- `duration_normal_min`: 0.5822

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
- Outlier handling is non-destructive: IQR is used for analysis/reporting, while winsorization caps extreme values in selected columns.
- The pipeline is modular and can be extended for both regression and classification tasks.
- Auto-generated report and plots make results reproducible and easy to inspect.

---

## License






