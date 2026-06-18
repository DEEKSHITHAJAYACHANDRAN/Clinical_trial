import streamlit as st
import pandas as pd
import numpy as np
import joblib
import re
import nltk
import matplotlib.pyplot as plt
import seaborn as sns

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.metrics import confusion_matrix

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Clinical Trial Disease Classification",
    layout="wide"
)

# =========================
# NLTK DOWNLOADS
# =========================
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("omw-1.4")

# =========================
# LOAD MODEL FILES
# =========================
@st.cache_resource
def load_model_files():
    tfidf = joblib.load("tfidf_vectorizer.pkl")
    model = joblib.load("disease_classifier_svm.pkl")
    return tfidf, model

tfidf, model = load_model_files()

# =========================
# LOAD DATA
# =========================
@st.cache_data
def load_data():
    data = pd.read_csv(
        r"D:\GUVI\Projects\Clinical Trial Disease_project5\clinical_trials_raw_patient2trial_conditions.csv"
    )

    clinical_nlp = data[["brief_summary", "source_condition_query"]].copy()
    clinical_nlp.columns = ["summary", "disease_category"]

    clinical_nlp = clinical_nlp.dropna()
    clinical_nlp = clinical_nlp.drop_duplicates()

    clinical_nlp["summary_length"] = clinical_nlp["summary"].str.len()

    return clinical_nlp

clinical_nlp = load_data()

# =========================
# TEXT CLEANING FUNCTION
# =========================
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    words = nltk.word_tokenize(text)
    words = [word for word in words if word not in stop_words]
    words = [lemmatizer.lemmatize(word) for word in words]
    return " ".join(words)

def predict_disease(summary):
    cleaned = clean_text(summary)
    vectorized = tfidf.transform([cleaned])
    prediction = model.predict(vectorized)
    return prediction[0]

# =========================
# SIDEBAR
# =========================
st.sidebar.title("Navigation")

page = st.sidebar.selectbox(
    "Select Page",
    [
        
        "Dataset Overview",
        "EDA Visualizations",
        "Model Performance",
        "Disease Prediction"
    ]
)


# =========================
# PAGE 2: DATASET OVERVIEW
# =========================
if page == "Dataset Overview":
    st.title("Dataset Overview")

    st.subheader("Dataset Shape")
    st.write(clinical_nlp.shape)

    st.subheader("Dataset Columns")
    st.write(clinical_nlp.columns.tolist())

    st.subheader("First 5 Records")
    st.dataframe(clinical_nlp.head())

    st.subheader("Missing Values")
    st.write(clinical_nlp.isnull().sum())

    st.subheader("Duplicate Records")
    st.write(clinical_nlp.duplicated().sum())

    st.subheader("Disease Category Counts")
    st.dataframe(clinical_nlp["disease_category"].value_counts())

    st.subheader("Disease Category Percentage")
    disease_percent = round(
        clinical_nlp["disease_category"].value_counts(normalize=True) * 100, 2
    )
    st.dataframe(disease_percent)

    st.subheader("Summary Length Statistics")
    st.write(clinical_nlp["summary_length"].describe())

# =========================
# PAGE 3: EDA VISUALIZATIONS
# =========================
elif page == "EDA Visualizations":
    st.title("Exploratory Data Analysis")

    st.subheader("Disease Category Distribution")

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.countplot(
        y="disease_category",
        data=clinical_nlp,
        order=clinical_nlp["disease_category"].value_counts().index,
        ax=ax
    )
    ax.set_title("Disease Category Distribution")
    ax.set_xlabel("Number of Clinical Trials")
    ax.set_ylabel("Disease Category")
    st.pyplot(fig)

    st.subheader("Summary Length Distribution")

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(clinical_nlp["summary_length"], bins=50, kde=True, ax=ax)
    ax.set_title("Distribution of Summary Length")
    ax.set_xlabel("Summary Length")
    ax.set_ylabel("Frequency")
    st.pyplot(fig)

    st.subheader("Average Summary Length by Disease Category")

    avg_len = clinical_nlp.groupby("disease_category")["summary_length"].mean().sort_values()

    fig, ax = plt.subplots(figsize=(10, 6))
    avg_len.plot(kind="barh", ax=ax)
    ax.set_title("Average Summary Length by Disease Category")
    ax.set_xlabel("Average Summary Length")
    ax.set_ylabel("Disease Category")
    st.pyplot(fig)

# =========================
# PAGE 4: MODEL PERFORMANCE
# =========================
elif page == "Model Performance":
    st.title("Model Performance")

    st.subheader("Model Accuracy Comparison")

    model_results = pd.DataFrame({
        "Model": [
            "Multinomial Naive Bayes",
            "Logistic Regression",
            "Linear SVM"
        ],
        "Accuracy": [
            0.8966,
            0.9447,
            0.9471
        ]
    })

    st.dataframe(model_results)

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(
        x="Accuracy",
        y="Model",
        data=model_results,
        ax=ax
    )
    ax.set_title("Model Accuracy Comparison")
    ax.set_xlim(0, 1)
    st.pyplot(fig)

    st.subheader("Best Model")
    st.success("Linear SVM was selected as the final model with 94.71% accuracy.")

    st.subheader("Linear SVM Classification Report")

    report_data = {
        "Disease Category": [
            "anxiety",
            "breast cancer",
            "chronic obstructive pulmonary disease",
            "covid-19",
            "glaucoma",
            "rheumatoid arthritis",
            "sickle cell anemia",
            "type 2 diabetes"
        ],
        "Precision": [0.90, 0.96, 0.93, 0.97, 0.96, 0.93, 0.96, 0.95],
        "Recall": [0.95, 0.97, 0.89, 0.94, 0.94, 0.91, 0.89, 0.96],
        "F1-Score": [0.92, 0.97, 0.91, 0.96, 0.95, 0.92, 0.92, 0.96],
        "Support": [1852, 3253, 1228, 2021, 430, 724, 227, 2280]
    }

    report_df = pd.DataFrame(report_data)
    st.dataframe(report_df)

    st.subheader("Confusion Matrix - Linear SVM")

    cm = np.array([
        [1757, 30, 24, 20, 3, 6, 1, 11],
        [40, 3168, 8, 7, 2, 6, 2, 20],
        [51, 27, 1094, 11, 3, 13, 2, 27],
        [51, 20, 22, 1903, 1, 4, 1, 19],
        [4, 4, 2, 2, 406, 8, 0, 4],
        [10, 12, 7, 4, 2, 660, 1, 28],
        [6, 3, 9, 1, 1, 2, 201, 4],
        [30, 20, 16, 8, 5, 10, 1, 2190]
    ])

    labels = [
        "anxiety",
        "breast cancer",
        "COPD",
        "covid-19",
        "glaucoma",
        "rheumatoid arthritis",
        "sickle cell anemia",
        "type 2 diabetes"
    ]

    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        ax=ax
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix - Linear SVM")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    st.pyplot(fig)

# =========================
# PAGE 5: DISEASE PREDICTION
# =========================
elif page == "Disease Prediction":
    st.title("Clinical Trial Disease Prediction")

    st.write("""
    Enter a clinical trial summary below. The model will predict the disease category.
    """)

    user_input = st.text_area(
        "Enter Clinical Trial Summary",
        height=200,
        placeholder="Example: This study evaluates insulin therapy and blood glucose control in patients with type 2 diabetes."
    )

    if st.button("Predict Disease"):
        if user_input.strip() == "":
            st.warning("Please enter a clinical trial summary.")
        else:
            result = predict_disease(user_input)
            st.success(f"Predicted Disease Category: {result}")

            st.subheader("Cleaned Input Text")
            st.write(clean_text(user_input))