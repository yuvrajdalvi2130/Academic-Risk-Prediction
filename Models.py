import pandas as pd
import pickle
import os

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load dataset
df = pd.read_csv("student_dataset.csv")

# Features
X = df[["Attendance","Study_Hours","Previous_Marks","Assignments_Completed"]]

# Targets
y_class = df["Risk_Level"]
y_reg = df["Score"]

# Train Test Split
X_train, X_test, y_train_class, y_test_class = train_test_split(
    X, y_class, test_size=0.2, random_state=42
)

X_train_r, X_test_r, y_train_reg, y_test_reg = train_test_split(
    X, y_reg, test_size=0.2, random_state=42
)

# -------------------------
# Linear Regression (Score Prediction)
# -------------------------

linear_model = LinearRegression()
linear_model.fit(X_train_r, y_train_reg)

print("Linear Regression trained (Score Prediction)")

# -------------------------
# Logistic Regression
# -------------------------

log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train, y_train_class)

log_pred = log_model.predict(X_test)

print("\nLOGISTIC REGRESSION")
print("Accuracy:", accuracy_score(y_test_class, log_pred))
print("Precision:", precision_score(y_test_class, log_pred, average="weighted"))
print("Recall:", recall_score(y_test_class, log_pred, average="weighted"))
print("F1 Score:", f1_score(y_test_class, log_pred, average="weighted"))

# -------------------------
# Decision Tree
# -------------------------

tree_model = DecisionTreeClassifier()
tree_model.fit(X_train, y_train_class)

tree_pred = tree_model.predict(X_test)

print("\nDECISION TREE")
print("Accuracy:", accuracy_score(y_test_class, tree_pred))
print("Precision:", precision_score(y_test_class, tree_pred, average="weighted"))
print("Recall:", recall_score(y_test_class, tree_pred, average="weighted"))
print("F1 Score:", f1_score(y_test_class, tree_pred, average="weighted"))

# -------------------------
# Save Models
# -------------------------

os.makedirs("saved_models", exist_ok=True)

pickle.dump(linear_model, open("saved_models/linear_model.pkl","wb"))
pickle.dump(log_model, open("saved_models/logistic_model.pkl","wb"))
pickle.dump(tree_model, open("saved_models/tree_model.pkl","wb"))

print("\nAll models saved successfully!")