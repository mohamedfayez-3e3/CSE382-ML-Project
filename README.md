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

