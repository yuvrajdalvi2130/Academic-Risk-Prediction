import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# -------------------------
# Page Title
# -------------------------

st.set_page_config(page_title="Academic Risk Prediction", layout="wide")

st.title("📊 Academic Risk Prediction Dashboard")

# -------------------------
# Load Dataset
# -------------------------

df = pd.read_csv("student_dataset.csv")

# -------------------------
# Load Models
# -------------------------

linear_model = pickle.load(open("saved_models/linear_model.pkl","rb"))
log_model = pickle.load(open("saved_models/logistic_model.pkl","rb"))
tree_model = pickle.load(open("saved_models/tree_model.pkl","rb"))

# -------------------------
# Tabs for Dashboard
# -------------------------

tab1, tab2, tab3, tab4 = st.tabs([
    "Dataset Overview",
    "Visualizations",
    "Prediction",
    "Model Evaluation"
])

# =====================================================
# TAB 1 — Dataset Overview
# =====================================================

with tab1:

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    st.subheader("Dataset Shape")
    st.write("Rows:", df.shape[0])
    st.write("Columns:", df.shape[1])

    st.subheader("Risk Level Distribution")
    st.bar_chart(df["Risk_Level"].value_counts())

# =====================================================
# TAB 2 — Visualizations
# =====================================================

with tab2:

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("Study Hours vs Previous Marks")

        fig, ax = plt.subplots()
        ax.scatter(df["Study_Hours"], df["Previous_Marks"])
        ax.set_xlabel("Study Hours")
        ax.set_ylabel("Previous Marks")

        st.pyplot(fig)

    with col2:

        st.subheader("Marks Distribution")

        fig, ax = plt.subplots()
        ax.hist(df["Previous_Marks"], bins=20)

        ax.set_xlabel("Marks")
        ax.set_ylabel("Frequency")

        st.pyplot(fig)

    st.subheader("Correlation Heatmap")

    fig, ax = plt.subplots()

    sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm", ax=ax)

    st.pyplot(fig)

    st.subheader("Risk Level Percentage")

    risk_counts = df["Risk_Level"].value_counts()

    fig, ax = plt.subplots()

    ax.pie(risk_counts, labels=risk_counts.index, autopct="%1.1f%%")

    st.pyplot(fig)

# =====================================================
# TAB 3 — Prediction
# =====================================================

with tab3:

    st.subheader("Enter Student Details")

    col1, col2 = st.columns(2)

    with col1:

        attendance = st.slider("Attendance (%)",0,100)
        study_hours = st.slider("Study Hours per Day",0,10)

    with col2:

        marks = st.slider("Previous Marks",0,100)
        assignments = st.slider("Assignments Completed",0,5)

    model_option = st.selectbox(
        "Select Model",
        ["Linear Regression","Logistic Regression","Decision Tree"]
    )

    input_data = [[attendance,study_hours,marks,assignments]]

    if st.button("Predict"):

        if model_option == "Linear Regression":

            prediction = linear_model.predict(input_data)

            st.success(f"Predicted Score: {prediction[0]:.2f}")

        elif model_option == "Logistic Regression":

            prediction = log_model.predict(input_data)

            st.success(f"Predicted Risk Level: {prediction[0]}")

        elif model_option == "Decision Tree":

            prediction = tree_model.predict(input_data)

            st.success(f"Predicted Risk Level: {prediction[0]}")

# =====================================================
# TAB 4 — Model Evaluation
# =====================================================

with tab4:

    X = df[["Attendance","Study_Hours","Previous_Marks","Assignments_Completed"]]
    y = df["Risk_Level"]

    y_pred = log_model.predict(X)

    accuracy = accuracy_score(y,y_pred)
    precision = precision_score(y,y_pred,average="weighted")
    recall = recall_score(y,y_pred,average="weighted")
    f1 = f1_score(y,y_pred,average="weighted")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Accuracy", round(accuracy,3))
    col2.metric("Precision", round(precision,3))
    col3.metric("Recall", round(recall,3))
    col4.metric("F1 Score", round(f1,3))

    st.subheader("Confusion Matrix")

    cm = confusion_matrix(y,y_pred)

    fig, ax = plt.subplots()

    sns.heatmap(cm,annot=True,fmt="d",cmap="Blues",ax=ax)

    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

    st.pyplot(fig)