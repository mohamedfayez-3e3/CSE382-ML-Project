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