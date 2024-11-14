import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap  # SHAP library for interpretability
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import OneHotEncoder

# Load the dataset
data = pd.read_csv('approved_data.csv')
st.write("Dataset loaded successfully!")

# Function to evaluate models
def evaluate_models(models, X_train, y_train, X_test, y_test):
    results = {}
    for name, model in models.items():
        pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                   ('classifier', model)])
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        results[name] = accuracy
    return results

# Streamlit interface
st.title('Life Insurance Underwriting')

# Create tabs
tab1, tab2, tab3 = st.tabs(["EDA", "Modeling", "Scoring"])

# Separating features and target variable
X = data.drop(columns=['Approved'])
y = data['Approved']

# Identify categorical and numerical columns
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns
categorical_cols = X.select_dtypes(include=['object']).columns

# Preprocessing pipelines for numerical and categorical features
numerical_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combine preprocessing pipelines
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_pipeline, numerical_cols),
        ('cat', categorical_pipeline, categorical_cols)
    ]
)

# Define the models to evaluate
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(),
    'Gradient Boosting': GradientBoostingClassifier(),
    'Support Vector Machine': SVC(),
    'K-Nearest Neighbors': KNeighborsClassifier(),
    'Naive Bayes': GaussianNB()
}

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Evaluate the models
model_accuracies = evaluate_models(models, X_train, y_train, X_test, y_test)

# Convert results to DataFrame
results_df = pd.DataFrame.from_dict(model_accuracies, orient='index', columns=['Accuracy'])
results_df = results_df.sort_values(by='Accuracy', ascending=False)

# Display the accuracies
st.subheader("Model Accuracies")
st.write(results_df)

# Highlight the best model dynamically
best_model_name = results_df.index[0]
best_model_accuracy = results_df.iloc[0, 0]
st.subheader(f"Best Model: {best_model_name}")
st.write(f"Accuracy: {best_model_accuracy}")

# Select and train the best model
best_model = models[best_model_name]
pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', best_model)])
pipeline.fit(X_train, y_train)

# Variable Importance for Tree-Based Models
if best_model_name in ['Random Forest', 'Gradient Boosting']:
    st.subheader(f"Feature Importance for {best_model_name}")
    feature_importances = best_model.feature_importances_
    feature_names = preprocessor.get_feature_names_out()
    feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances})
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

    # Plot feature importance
    fig, ax = plt.subplots()
    sns.barplot(data=feature_importance_df, x='Importance', y='Feature', ax=ax)
    ax.set_title(f"Feature Importance for {best_model_name}")
    st.pyplot(fig)

# SHAP Explanation for the Best Model
st.subheader("SHAP Explanation")
explainer = shap.Explainer(pipeline.named_steps["classifier"], pipeline.transform(X_test))
shap_values = explainer(pipeline.transform(X_test))

# SHAP summary plot
st.write("SHAP Summary Plot")
fig, ax = plt.subplots()
shap.summary_plot(shap_values, features=pipeline.transform(X_test), feature_names=feature_names, show=False)
st.pyplot(fig)

# SHAP dependence plot for the most important feature
st.write("SHAP Dependence Plot for the Top Feature")
top_feature = feature_importance_df.iloc[0]['Feature']
fig, ax = plt.subplots()
shap.dependence_plot(top_feature, shap_values.values, pipeline.transform(X_test), feature_names=feature_names, show=False)
st.pyplot(fig)
