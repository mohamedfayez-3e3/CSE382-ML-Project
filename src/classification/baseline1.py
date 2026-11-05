import pandas as pd


from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

print("Project Root:", PROJECT_ROOT)
print("Data Folder:", PROJECT_ROOT / "data")
print("Train File Exists:",
      (PROJECT_ROOT / "data" / "train_preprocessed.csv").exists())


train_df = pd.read_csv(PROJECT_ROOT / "data" / "train_preprocessed.csv")

X_train = train_df.drop("IsCanceled", axis=1)
y_train = train_df["IsCanceled"]

test_df = pd.read_csv(PROJECT_ROOT / "data" / "test_preprocessed.csv")

X_test = test_df.drop("IsCanceled", axis=1)
y_test = test_df["IsCanceled"]

train_df.drop("IsCanceled", axis=1)


from sklearn.linear_model import LogisticRegression

logistic_model = LogisticRegression(
    max_iter=1000,
    random_state=42
)

logistic_model.fit(X_train, y_train)

y_pred = logistic_model.predict(X_test)


from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, y_pred)




from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)


import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(6, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Not Canceled", "Canceled"],
    yticklabels=["Not Canceled", "Canceled"]
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Logistic Regression - Confusion Matrix")
plt.show()




error_rate = 1 - accuracy


from sklearn.metrics import precision_score

precision = precision_score(y_test, y_pred)



from sklearn.metrics import recall_score

recall = recall_score(y_test, y_pred)




from sklearn.metrics import f1_score

f1 = f1_score(y_test, y_pred)





print("Accuracy:", accuracy)

print("Error Rate:", error_rate)

print("Precision:", precision)

print("Recall:", recall)

print("F1-Score:", f1)