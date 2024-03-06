# Required Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import streamlit as st

# Load the dataset
# file = r"C:\Users\Admin\Desktop\NextHikes\E-commerce-Project\Data\train_data.csv"
df = pd.read_csv(r"C:\Users\Admin\Desktop\NextHikes\E-commerce-Project\Data\train_data.csv")

# Preprocessing
# Assuming 'reviews.text' is the column containing review text
X = df['reviews.text']
y = df['sentiment']

# Splitting the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Encode target variable
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

# TF-IDF Vectorization
tfidf_vectorizer = TfidfVectorizer(max_features=1000)
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Get feature names
feature_names = tfidf_vectorizer.get_feature_names_out()

# Support Vector Machine (SVM)
svm_model = SVC(kernel='linear')
svm_model.fit(X_train_tfidf, y_train)
svm_pred = svm_model.predict(X_test_tfidf)
svm_accuracy = accuracy_score(y_test, svm_pred)

# Neural Network
nn_model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000)
nn_model.fit(X_train_tfidf, y_train)
nn_pred = nn_model.predict(X_test_tfidf)
nn_accuracy = accuracy_score(y_test, nn_pred)

# XGBoost
xgb_model = XGBClassifier()
xgb_model.fit(X_train_tfidf, y_train_encoded)
xgb_pred = xgb_model.predict(X_test_tfidf)
xgb_accuracy = accuracy_score(y_test_encoded, xgb_pred)

# Random Forest (Ensemble)
rf_model = RandomForestClassifier()
rf_model.fit(X_train_tfidf, y_train)
rf_pred = rf_model.predict(X_test_tfidf)
rf_accuracy = accuracy_score(y_test, rf_pred)

# LSTM (Deep Learning)
# Note: LSTM implementation requires a different approach and is not feasible to demonstrate here due to space limitations.

# Topic Modeling (Latent Dirichlet Allocation)
lda_model = LatentDirichletAllocation(n_components=5, random_state=42)
lda_model.fit(X_train_tfidf)

# Display the top words for each topic
for idx, topic in enumerate(lda_model.components_):
    st.write(f"Topic {idx}:", " ".join([feature_names[i] for i in topic.argsort()[:-10 - 1:-1]]))

# Model Comparison
st.write("Model Comparison:")
st.write("SVM Accuracy:", svm_accuracy)
st.write("Neural Network Accuracy:", nn_accuracy)
st.write("XGBoost Accuracy:", xgb_accuracy)
st.write("Random Forest Accuracy:", rf_accuracy)

# Streamlit Deployment
st.title("Sentiment Analysis with Machine Learning")
text_input = st.text_input("Enter a review:")
if text_input:
    text_input_tfidf = tfidf_vectorizer.transform([text_input])
    prediction = svm_model.predict(text_input_tfidf)
    st.write("Predicted Sentiment:", prediction)
