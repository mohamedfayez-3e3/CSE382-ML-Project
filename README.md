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
seif step untill now
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