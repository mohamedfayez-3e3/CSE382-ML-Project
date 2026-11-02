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

## Classification Task

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
