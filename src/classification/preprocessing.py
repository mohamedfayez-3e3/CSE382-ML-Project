
# prepering the data for classification

import pandas as pd

df = pd.read_csv("../../data/H2.csv")

y = df["IsCanceled"]

features = [
    "LeadTime",
    "ArrivalDateYear",
    "ArrivalDateMonth",
    "ArrivalDateWeekNumber",
    "ArrivalDateDayOfMonth",
    "StaysInWeekendNights",
    "StaysInWeekNights",
    "Adults",
    "Children",
    "Babies",
    "Meal",
    "Country",
    "MarketSegment",
    "DistributionChannel",
    "IsRepeatedGuest",
    "PreviousCancellations",
    "PreviousBookingsNotCanceled",
    "ReservedRoomType",
    "CustomerType"
]


X = df[features].copy()



from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


numeric_features = X_train.select_dtypes(
    include=["number"]
).columns.tolist()

categorical_features = X_train.select_dtypes(
    include=["object", "category", "bool"]
).columns.tolist()



from sklearn.impute import SimpleImputer

num_imputer = SimpleImputer(strategy="median")

X_train[numeric_features] = num_imputer.fit_transform(
    X_train[numeric_features]
)

X_test[numeric_features] = num_imputer.transform(
    X_test[numeric_features]
)



cat_imputer = SimpleImputer(strategy="most_frequent")

X_train[categorical_features] = cat_imputer.fit_transform(
    X_train[categorical_features]
)

X_test[categorical_features] = cat_imputer.transform(
    X_test[categorical_features]
)





from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder(handle_unknown="ignore")

X_train_encoded = encoder.fit_transform(
    X_train[categorical_features]
)

X_test_encoded = encoder.transform(
    X_test[categorical_features]
)




from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(
    X_train[numeric_features]
)

X_test_scaled = scaler.transform(
    X_test[numeric_features]
)





from scipy.sparse import hstack

X_train_final = hstack([
    X_train_scaled,
    X_train_encoded
]).tocsr()

X_test_final = hstack([
    X_test_scaled,
    X_test_encoded
]).tocsr()






encoded_features = encoder.get_feature_names_out(categorical_features)

all_features = list(numeric_features) + list(encoded_features)

X_train_df = pd.DataFrame(
    X_train_final.toarray(),
    columns=all_features
)

X_test_df = pd.DataFrame(
    X_test_final.toarray(),
    columns=all_features
)

train_df = X_train_df.copy()
train_df["IsCanceled"] = y_train.reset_index(drop=True)

test_df = X_test_df.copy()
test_df["IsCanceled"] = y_test.reset_index(drop=True)

train_df.to_csv("../../data/train_preprocessed.csv", index=False)

test_df.to_csv("../../data/test_preprocessed.csv", index=False)





