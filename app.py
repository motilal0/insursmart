import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import accuracy_score, classification_report
import shap
import matplotlib.pyplot as plt

# Load your dataset
data = pd.read_csv("your_dataset.csv")  # Replace with your actual data file
X = data.drop("target", axis=1)
y = data["target"]

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define numerical and categorical features
numerical_features = X.select_dtypes(include=['int64', 'float64']).columns
categorical_features = X.select_dtypes(include=['object']).columns

# Preprocessor for pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ]
)

# Model pipeline
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])

# Train the model
pipeline.fit(X_train, y_train)

# Evaluate the model
y_pred = pipeline.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

# Display model evaluation in Streamlit
st.write("Model Accuracy:", accuracy)
st.write("Classification Report:")
st.text(report)

# Extract trained model from the pipeline for SHAP explanations
best_model = pipeline.named_steps["classifier"]

# Transform X_test for SHAP without the classifier
X_test_transformed = preprocessor.transform(X_test)

# Initialize SHAP Explainer
try:
    explainer = shap.TreeExplainer(best_model)
    shap_values = explainer.shap_values(X_test_transformed)
except shap.utils._exceptions.ExplainerError:
    # Fallback to KernelExplainer if TreeExplainer raises errors
    explainer = shap.KernelExplainer(best_model.predict, X_test_transformed)
    shap_values = explainer.shap_values(X_test_transformed)

# Visualize SHAP Waterfall Plot for a single observation
st.subheader("SHAP Explanation for a Sample Prediction")
fig, ax = plt.subplots()
shap.plots.waterfall(shap_values[0][0], feature_names=X_test.columns, show=False)
st.pyplot(fig)

# Display SHAP Summary Plot for Global Feature Importance
st.subheader("Global Feature Importance (SHAP Summary Plot)")
fig, ax = plt.subplots()
shap.summary_plot(shap_values[1], X_test_transformed, feature_names=X_test.columns, show=False)
st.pyplot(fig)
