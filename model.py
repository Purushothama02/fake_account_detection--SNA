import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv("fake_social_media.csv")

print("===== First 5 Rows =====")
print(df.head())

print("\n===== Column Names =====")
print(df.columns)

print("\n===== Dataset Shape =====")
print(df.shape)

df = df.dropna()

target = "is_fake"

X = df.drop(target, axis=1)
y = df[target]

X = pd.get_dummies(X, drop_first=True)

print("\n===== Preprocessing Completed =====")
print(X.head())

print("\n===== Data Types After Encoding =====")
print(X.dtypes)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTrain Shape:", X_train.shape)
print("Test Shape:", X_test.shape)

lr = LogisticRegression(max_iter=2000)

lr.fit(X_train, y_train)

y_pred_lr = lr.predict(X_test)

print("\n===================================")
print("LOGISTIC REGRESSION RESULTS")
print("===================================")

lr_accuracy = accuracy_score(y_test, y_pred_lr)

print("Accuracy:", lr_accuracy)
print("\nClassification Report:")
print(classification_report(y_test, y_pred_lr))

dt = DecisionTreeClassifier(
    random_state=42
)

dt.fit(X_train, y_train)

y_pred_dt = dt.predict(X_test)

print("\n===================================")
print("DECISION TREE RESULTS")
print("===================================")

dt_accuracy = accuracy_score(y_test, y_pred_dt)

print("Accuracy:", dt_accuracy)
print("\nClassification Report:")
print(classification_report(y_test, y_pred_dt))

rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

rf.fit(X_train, y_train)

y_pred_rf = rf.predict(X_test)

print("\n===================================")
print("RANDOM FOREST RESULTS")
print("===================================")

rf_accuracy = accuracy_score(y_test, y_pred_rf)

print("Accuracy:", rf_accuracy)
print("\nClassification Report:")
print(classification_report(y_test, y_pred_rf))

results = pd.DataFrame({
    "Model": [
        "Logistic Regression",
        "Decision Tree",
        "Random Forest"
    ],
    "Accuracy": [
        lr_accuracy,
        dt_accuracy,
        rf_accuracy
    ]
})

print("\n===================================")
print("FINAL MODEL COMPARISON")
print("===================================")

print(results)

plt.figure(figsize=(8, 5))

sns.barplot(
    x="Model",
    y="Accuracy",
    data=results
)

plt.title("Model Accuracy Comparison")
plt.xticks(rotation=15)
plt.tight_layout()
plt.show()

cm = confusion_matrix(y_test, y_pred_rf)

plt.figure(figsize=(6, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d"
)

plt.title("Random Forest Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

best_model = results.loc[
    results["Accuracy"].idxmax(),
    "Model"
]

best_accuracy = results["Accuracy"].max()

print("\n===================================")
print("FINAL CONCLUSION")
print("===================================")

print(f"Best Performing Model: {best_model}")
print(f"Best Accuracy: {best_accuracy:.4f}")

print("""
Conclusion:
Random Forest generally performs best because it handles
nonlinear patterns and feature interactions effectively.

Hence, Random Forest is selected as the final model
for Fake Account Detection in Social Networks.
""")
