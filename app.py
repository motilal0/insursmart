import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import prince  # For Correspondence Analysis
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
from sklearn.decomposition import PCA
from adjustText import adjust_text
import shap  # SHAP for explainability
import eli5  # For variable importance visualization
from eli5.sklearn import PermutationImportance

# Load the dataset
data = pd.read_csv('approved_data.csv')
st.write("Dataset loaded successfully!")

# Function to evaluate models
def evaluate_models(models, X_train, y_train, X_test, y_test):
    results = {}
    shap_values_dict = {}
    for name in models.keys():
        model = models[name]
        pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                   ('classifier', model)])
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        results[name] = accuracy
        
        # SHAP explanation
        shap_values_dict[name] = shap_explain(pipeline, X_train)
        
    return results, shap_values_dict

# SHAP explanation function
def shap_explain(pipeline, X_train):
    # Extract the classifier from the pipeline
    model = pipeline.named_steps['classifier']
    
    # Create SHAP explainer using the model and the data
    explainer = shap.Explainer(model, X_train)
    
    # Get SHAP values
    shap_values = explainer(X_train)
    
    return shap_values

# Streamlit interface
st.title('Life Insurance Underwriting')

# Create tabs
tab1, tab2, tab3 = st.tabs(["EDA", "Modeling", "Scoring"])

with tab1:
    st.header("Exploratory Data Analysis (EDA)")

    # Summary statistics
    st.subheader("Summary Statistics")
    st.write(data.describe())

    # Missing values
    st.subheader("Missing Values")
    st.write(data.isnull().sum())

    # Histograms for categorical features with percentages
    st.subheader("Histograms for Smoking Status, Medical History, and Alcohol Consumption")
    categorical_columns = ['Smoking Status', 'Medical History', 'Alcohol Consumption']

    for column in categorical_columns:
        fig, ax = plt.subplots()
        counts = data[column].value_counts(normalize=True) * 100  # Calculate percentage
        sns.barplot(x=counts.index, y=counts.values, ax=ax)
        ax.set_title(f'Distribution of {column}')
        ax.set_xlabel(column)
        ax.set_ylabel("Percentage")

        # Annotate bars with percentages
        for p in ax.patches:
            ax.annotate(f'{p.get_height():.2f}%', (p.get_x() + p.get_width() / 2., p.get_height()), 
                        ha='center', va='center', fontsize=11, color='black', xytext=(0, 10),
                        textcoords='offset points')

        # Set custom x-axis labels for Smoking Status
        if column == 'Smoking Status':
            ax.set_xticks([0, 1])
            ax.set_xticklabels(['Non-Smoker (0)', 'Smoker (1)'])

        # Set custom x-axis labels for Medical History
        if column == 'Medical History':
            ax.set_xticks([0, 1, 2, 3])
            ax.set_xticklabels(['No Disease (0)', 'Diabetes (1)', 'Hypertension (2)', 'Heart Disease (3)'])

        # Set custom x-axis labels for Alcohol Consumption
        if column == 'Alcohol Consumption':
            ax.set_xticks([0, 1, 2, 3])
            ax.set_xticklabels(['Never (0)', 'Low (1)', 'Moderate (2)', 'High (3)'])

        st.pyplot(fig)

    # Detailed Correspondence Analysis
    st.subheader("Detailed Correspondence Analysis")
    # Define labels for correspondence analysis (same as before)

    # Correspondence Analysis code (same as before)

    # Target distribution
    st.subheader("Approval & Rejection")
    approval_rate = data['Approved'].mean() * 100
    st.write(f"Approval Rate: {approval_rate:.2f}%")
    
    fig, ax = plt.subplots()
    sns.countplot(x='Approved', data=data, ax=ax)
    ax.set_xlabel("Approved")
    ax.set_ylabel("Count")
    st.pyplot(fig)


with tab2:
    st.header("Modeling")

    # Separating features and target variable
    X = data.drop(columns=['Approved'])
    y = data['Approved']

    # Identify categorical and numerical columns
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_cols = X.select_dtypes(include=['object']).columns

    # Preprocessing pipelines for numerical and categorical features
    numerical_pipeline = Pipeline(steps=[('imputer', SimpleImputer(strategy='median')),
                                         ('scaler', StandardScaler())])

    categorical_pipeline = Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')),
                                           ('onehot', OneHotEncoder(handle_unknown='ignore'))])

    # Combine preprocessing pipelines
    preprocessor = ColumnTransformer(
        transformers=[('num', numerical_pipeline, numerical_cols),
                      ('cat', categorical_pipeline, categorical_cols)])

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
    model_accuracies, shap_values_dict = evaluate_models(models, X_train, y_train, X_test, y_test)

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

    # Show SHAP summary plot for the best model
    st.subheader("SHAP Summary Plot for Best Model")
    shap_values = shap_values_dict[best_model_name]
    shap.summary_plot(shap_values, X_train)

    # Show Variable Importance for Best Model
    st.subheader("Variable Importance for Best Model")
    best_model = models[best_model_name]
    best_model.fit(X_train, y_train)
    
    # Using Permutation Importance
    perm = PermutationImportance(best_model, random_state=42).fit(X_train, y_train)
    eli5.show_weights(perm, feature_names=X_train.columns.tolist())


with tab3:
    st.header("Scoring")

    # File uploader for custom data
    uploaded_file = st.file_uploader("Upload your dataset for scoring", type="csv")
    
    if uploaded_file is not None:
        # Load the uploaded data
        custom_data = pd.read_csv(uploaded_file)
        st.write("Uploaded data:")
        st.write(custom_data.head())

        # Define approval criteria
        def approve_application(row):
            if (18 <= row['Age'] <= 65 and
                row['Smoking Status'] == 0 and
                18.5 <= row['BMI'] <= 24.9 and
                row['Medical History'] == 0 and
                row['Alcohol Consumption'] <= 2 and
                row['Family History of Disease'] == 0 and
                row['Occupation'] != 'High Risk'):
                return 1
            else:
                return 0

        # Apply criteria to create 'Approved' column
        custom_data['Approved'] = custom_data.apply(approve_application, axis=1)

        # Count of approved and rejected forms
        approved_count = custom_data['Approved'].sum()
        rejected_count = len(custom_data) - approved_count
        approval_rate = (approved_count / len(custom_data)) * 100

        st.write(f"Number of Approved Forms: {approved_count}")
        st.write(f"Number of Rejected Forms: {rejected_count}")
        st.write(f"Approval Rate: {approval_rate:.2f}%")

        # Display the scored data
        st.subheader("Scored Data")
        st.write(custom_data)

        # Download the scored data
        csv = custom_data.to_csv(index=False).encode()
        st.download_button("Download Scored Data", data=csv, file_name='scored_data.csv', mime='text/csv')
