# CSE382-ML-Project

City Hotel Booking Demand

## Environment Setup

Run the following command in the project root directory:

```bash
python -m pip install -r requirements.txt
```

## Initial Data Audit and EDA

- **Dataset:** `H2.csv` contains 79,330 rows and 31 columns.
- **Missing Values:** `Country` has 24 missing values (0.0303%), and `Children` has 4 missing values (0.0050%).
- **Duplicate Rows:** The initial audit detected 25,902 fully duplicated rows. Further investigation is required before deciding whether to remove them.
- **LeadTime:** The distribution is right-skewed, with most bookings concentrated at shorter lead times.
- **Cancellation Target:** `IsCanceled` contains 46,228 non-canceled bookings (58.27%) and 33,102 canceled bookings (41.73%).
- **ADR:** The distribution is concentrated mainly between approximately 60 and 150. One value above 1,000 requires further inspection.
- **Potentially Unusual Records:** 167 records have zero adults, children, and babies. These records require investigation.
- **Next Steps:** Investigate duplicates and unusual records, analyze relationships between cancellation and selected features, assess leakage risks, and define preprocessing and data-splitting strategies.

## Task A: ADR Regression (Milestone 1 progress)

### ADR exploration
- Dataset shape: 79,330 rows x 31 columns
- ADR: mean 105.3, median 99.9, 25%-75% range 79.2-126.0, max 5400
- 1,208 rows have ADR = 0: 508 are Complementary, the rest are spread across other
  market segments; all are No Deposit and 1,079 were not cancelled
- 2 rows have ADR > 500 (5400 and 510)

### Cleaning decisions (regression only)
- Remove rows with ADR <= 0: they are not real nightly prices (complimentary or
  unpriced bookings) and would distort MAE/R2
- Remove the ADR = 5400 row: likely a data-entry error
- Keep ADR = 510: plausible high-season price
- These rows are removed only for the regression task, not from the shared data,
  so classification and clustering are unaffected

### Feature selection (regression)
Prediction point: booking time. Only fields known at booking are used.
- Dropped as leakage: IsCanceled, ReservationStatus, ReservationStatusDate
- Dropped because they change after booking: AssignedRoomType, BookingChanges
- Dropped as uncertain (may be reconsidered with justification): DaysInWaitingList,
  RequiredCarParkingSpaces, TotalOfSpecialRequests
- Country, Agent, Company excluded from the first baseline (high cardinality)
- Feature lists are stored in `config.yaml` under `features.regression`

### Fixes to shared code
- `src/preprocessing/preprocessing.py`: replaced `match` statement with `if/elif`
  for Python 3.9 compatibility, and added the missing `return` in `build_preprocessor`
- Exploration scripts now print in English to avoid terminal encoding issues

### Next steps
- Add regression-only row cleaning (ADR <= 0 and ADR > 1000)
- Implement `build_regression_preprocessor`
- Train the Linear Regression baseline and report MAE, RMSE, R2
 Regression: Predicting Hotel ADR (Milestone 1)
## DAY 2
### Regression: Predicting Hotel ADR
**Authors:** Seif and Rovan. The regression work (data, preprocessing, baseline, plots and documentation) was done jointly by both team members.

This part of the project predicts the nightly price (**ADR**, average daily rate) of a hotel booking from the booking features in `data/H2.csv`.

---

## 1. Files

```
config.yaml                        # settings used by the regression scripts
src/regression/
├── data.py                        # load config + data, clean, split train/test
├── preprocessor.py                # preprocessing of the features
├── baseline.py                    # baseline model + 5-fold CV on the train set
└── plots.py                       # EDA plots (saved as PNG)
outputs/
├── audit/lost_data.csv            # rows removed during cleaning
├── regression_baseline_cv.csv     # baseline 5-fold CV results
└── regression/
    ├── adr_distribution.png
    ├── adr_by_month.png
    └── adr_by_room_type.png
```

---

## 2. Setup

Python 3.9+ and:

```
pip install pandas numpy scikit-learn matplotlib pyyaml
```

Run every command from the project root (the folder that contains `src/`), using `-m`.
On Windows use `py -m ...`, elsewhere `python -m ...`.

---

## 3. Configuration (`config.yaml`)

All settings (data path, target column, train/test split and seed, number of CV folds, cleaning options) are in `config.yaml` and read by `load_config()` in `data.py`, so nothing is hard-coded.
## 4. What each step does

### 4.1 Data loading and cleaning (data.py)
- load_config() reads config.yaml.
- load_regression_data(cfg) loads the data, cleans it, splits it into train and test, and returns X_train, X_test, y_train, y_test, log.
- Rows removed during cleaning are written to outputs/audit/lost_data.csv, so every removal can be traced.
- Target: *ADR*.



### 4.2 Preprocessing (preprocessor.py)
Prepares the features for modelling. It is fitted on the train data only, so no information leaks from the test set.

### 4.3 Baseline (baseline.py)
- Trains a baseline regression model on the *train set only*.
- Evaluates it with *5-fold cross-validation*.
- Saves the results to outputs/regression_baseline_cv.csv.

>

### 4.4 EDA plots (plots.py)
All plots use the *train set only* and are saved in outputs/regression/.

| File | Shows | Reading |
|------|-------|---------|
| adr_distribution.png | Histogram of ADR (50 bins) with a dashed line at the median (107.1) | Bell-shaped but right-skewed. Most bookings are between about 70 and 150, with a long tail up to about 300, a few extreme values near 450, and a few bookings near 0. Supports the cleaning decisions. |
| adr_by_month.png | Mean ADR per arrival month | Shows seasonality of the price across the year. |
| adr_by_room_type.png | Mean ADR per reserved room type | Shows how much the price depends on room type. |

---

## 5. How to run


py -m src.regression.plots       # creates the 3 plots
py -m src.regression.baseline    # runs the baseline + 5-fold CV



Plots output:


Saved: outputs/regression\adr_distribution.png
Saved: outputs/regression\adr_by_month.png
Saved: outputs/regression\adr_by_room_type.png


---

## 6. Evaluation rules

- The test set is split once and *kept untouched* until final evaluation.
- Preprocessing and model choices use *5-fold CV on the train set only*.
- Test performance is not checked repeatedly while choosing preprocessing or hyperparameters.
- Reported results come from the *CV numbers only*.
## Regression Models (Milestone 1)

Predicts **ADR** from booking-time features only. Columns that are only known after
the booking (cancellation status, reservation status, assigned room type, booking
changes, etc.) are excluded to avoid target leakage. Exact duplicates are removed
before the train/test split so identical bookings cannot appear in both sets.

### Models
Three model families, each evaluated as an untuned **baseline** and a
hyperparameter-**tuned** configuration:

| Model | Baseline | Tuned |
|-------|----------|-------|
| Linear Regression | `LinearRegression` | `Ridge` (tuned over `alpha`) |
| Random Forest | default settings, 100 trees | random search over trees, depth, leaf size |
| XGBoost | default settings | random search over trees, depth, learning rate, subsample |

Plain Linear Regression has no hyperparameters, so Ridge is used as its tuned version.

### Evaluation
- **5-fold cross-validation on the training set only** (shuffled, fixed seed).
- Preprocessing (imputation, scaling, one-hot encoding) is inside the pipeline, so it is
  fitted on each fold's training part only (no leakage from the validation fold).
- Tuning uses `RandomizedSearchCV` with CV RMSE as the selection criterion.
- Metrics: **MAE, MSE, RMSE, R²**.

### Cross-validation results

| Model | Config | MAE | RMSE | R² |
|-------|--------|-----|------|----|
| Linear Regression | baseline | 15.843 | 22.058 | 0.693 |
| Linear Regression | tuned (Ridge) | 15.843 | 22.057 | 0.693 |
| Random Forest | baseline | 11.111 | 17.947 | 0.797 |
| Random Forest | tuned | 11.183 | 17.898 | 0.798 |
| XGBoost | baseline | 11.489 | 17.600 | 0.804 |
| **XGBoost** | **tuned** | **10.869** | **17.026** | **0.817** |

A Dummy model that predicts the mean gives R² = 0, MAE = 30.9, RMSE = 39.8.

### Findings
- Tree-based models clearly beat Linear Regression (R² about 0.80 vs 0.69), so the
  relationship between ADR and the features is not purely linear.
- Tuning helps XGBoost most. It barely changes Linear Regression and Random Forest.
- For Random Forest, tuning lowered RMSE slightly but raised MAE slightly, because
  models are selected by RMSE.
- **Selected model: tuned XGBoost** (lowest CV RMSE).

### Final test result
The test set is used **once**, only for the selected model:


### How to run
py -m src.regression.models 5 # quick CV run (5 random-search candidates per model)
py -m src.regression.models # full CV run (10 candidates), writes outputs/regression_cv_results.csv
py -m src.regression.models final # evaluates the selected model once on the test set

## TASK B CLASSIFICATION

**Objective:** Predict whether a hotel booking will be canceled using the `IsCanceled` target column.

### Preprocessing

The preprocessing steps were implemented manually in `src/classification/preprocessing.ipynb`.

1. **Feature Selection**
   - Selected 19 features as input variables (`X`).
   - Set `IsCanceled` as the target variable (`y`).

2. **Train-Test Split**
   - Split the dataset into 80% training data and 20% testing data.
   - Used `random_state=42` for reproducibility.
   - Used `stratify=y` to preserve the target class distribution.

3. **Feature Type Identification**
   - Automatically identified numerical and categorical features using `select_dtypes()`.

4. **Missing Values Handling**
   - Used `SimpleImputer(strategy="median")` for numerical features.
   - Used `SimpleImputer(strategy="most_frequent")` for categorical features.
   - Fitted imputers on training data and applied the same transformations to testing data.

5. **Categorical Encoding**
   - Applied `OneHotEncoder(handle_unknown="ignore")` to categorical features.
   - Fitted the encoder on training data and transformed testing data using the same encoder.

6. **Numerical Scaling**
   - Applied `StandardScaler()` to numerical features.
   - Fitted the scaler on training data and used it to transform testing data.

7. **Feature Combination**
   - Combined the scaled numerical features and encoded categorical features using `scipy.sparse.hstack()`.
   - Converted the resulting matrices to CSR format.

8. **Export Preprocessed Data**
   - Converted the resulting feature matrices into DataFrames.
   - Added the `IsCanceled` target column.
   - Exported the training and testing datasets as CSV files:
     - `data/train_preprocessed.csv`
     - `data/test_preprocessed.csv`

### Baseline Data Loading

The preprocessed datasets were imported into `src/classification/baseline.ipynb`:

- Loaded `train_preprocessed.csv` and `test_preprocessed.csv` using Pandas.
- Separated the input features (`X_train`, `X_test`) from the target variable (`y_train`, `y_test`).

**Final Dataset Shapes:**
- Training features: 63,464 rows × 213 features.
- Testing features: 15,866 rows × 213 features.
- Training target: 63,464 values.
- Testing target: 15,866 values.


## Preprocessing

- All data preparation lives in `src/preprocessing/preprocessing.py`, and all settings live in `config.yaml` (paths, seed, split size, cleaning thresholds, and the feature lists per task).

- To get data for a task, call `get_preprocessing_data(Task.<task>)`.
- To get the pipeline for a task, call `build_preprocessor(Task.<task>`.

- Your edits should be one of the two:
  - edit the features in config.yaml
  - change your `build_<task>_preprocessor()` in `preprocessing.py`

## Classification
- Target: IsCanceled. Dropped ReservationStatus and ReservationStatusDate to avoid data leakage.
- Stratified 80/20 split, with imputation, scaling and one-hot encoding in a pipeline.
- Models compared: Logistic Regression and Random Forest.
- Logistic Regression: accuracy = 0.8255, F1 = 0.7775, ROC-AUC = 0.9098.
- Random Forest: accuracy = 0.8901, F1 = 0.8617, ROC-AUC = 0.9592.
- Best model: Random Forest. Full results in outputs/classification/metrics.json.
- Note: duplicate rows were not removed, so scores may be slightly optimistic.
