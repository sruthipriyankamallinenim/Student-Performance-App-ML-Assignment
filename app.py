
import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)

st.set_page_config(
    page_title="Student Performance Prediction",
    page_icon="🎓",
    layout="wide"
)

st.title("🎓 Student Performance Classification")
st.write(
    "Compare five machine learning models for predicting student GradeClass."
)

# Model file paths
model_files = {
    "Logistic Regression": "model/logistic_regression.pkl",
    "Decision Tree": "model/decision_tree.pkl",
    "KNN": "model/knn.pkl",
    "Naive Bayes": "model/naive_bayes.pkl",
    "Random Forest": "model/random_forest.pkl"
}

# Load scaler
with open("model/scaler.pkl", "rb") as file:
    scaler = pickle.load(file)

# Sidebar
st.sidebar.header("Configuration")

selected_model = st.sidebar.selectbox(
    "Select a Machine Learning Model",
    list(model_files.keys())
)

uploaded_file = st.sidebar.file_uploader(
    "Upload Test Data CSV",
    type=["csv"]
)

if uploaded_file is None:
    st.info("Please upload test_data.csv to begin.")
    st.stop()

# Read uploaded data
data = pd.read_csv(uploaded_file, sep=',')

st.subheader("Uploaded Test Dataset")
st.write(f"Number of records: {len(data)}")

st.dataframe(data.head(10), use_container_width=True)

# Check target column
if "GradeClass" not in data.columns:
    st.error("The uploaded CSV must contain the GradeClass column.")
    st.stop()

# Separate target and features
y_test = data["GradeClass"].astype(int)

X_test = data.drop(columns=["GradeClass"])

# Remove unnecessary columns if present
for column in ["StudentID", "GPA"]:
    if column in X_test.columns:
        X_test = X_test.drop(columns=[column])

# Scale data
X_test_scaled = scaler.transform(X_test)

# Load selected model
with open(model_files[selected_model], "rb") as file:
    model = pickle.load(file)

# Predictions
y_pred = model.predict(X_test_scaled)
y_prob = model.predict_proba(X_test_scaled)

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)

auc = roc_auc_score(
    y_test,
    y_prob,
    multi_class="ovr",
    average="weighted"
)

precision = precision_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)

mcc = matthews_corrcoef(y_test, y_pred)

# Display metrics
st.subheader(f"Performance: {selected_model}")

col1, col2, col3 = st.columns(3)

col1.metric("Accuracy", f"{accuracy:.4f}")
col2.metric("AUC", f"{auc:.4f}")
col3.metric("Precision", f"{precision:.4f}")

col4, col5, col6 = st.columns(3)

col4.metric("Recall", f"{recall:.4f}")
col5.metric("F1 Score", f"{f1:.4f}")
col6.metric("MCC", f"{mcc:.4f}")

# Confusion Matrix
st.subheader("Confusion Matrix")

cm = confusion_matrix(y_test, y_pred)

fig, ax = plt.subplots()

ax.imshow(cm)

ax.set_xlabel("Predicted Class")
ax.set_ylabel("Actual Class")
ax.set_title(f"Confusion Matrix - {selected_model}")

classes = sorted(y_test.unique())

ax.set_xticks(range(len(classes)))
ax.set_yticks(range(len(classes)))

ax.set_xticklabels(classes)
ax.set_yticklabels(classes)

for i in range(len(classes)):
    for j in range(len(classes)):
        ax.text(
            j,
            i,
            str(cm[i, j]),
            ha="center",
            va="center"
        )

st.pyplot(fig)

# Classification Report
st.subheader("Classification Report")

report = classification_report(
    y_test,
    y_pred,
    output_dict=True,
    zero_division=0
)

report_df = pd.DataFrame(report).transpose()

st.dataframe(report_df, use_container_width=True)
