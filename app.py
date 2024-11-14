import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

# Load the dataset
data = pd.read_csv('approved_data.csv')
st.write("Dataset loaded successfully!")

# Function to evaluate models
def evaluate_models(models, X_train, y_train, X_test, y_test):
    results = {}
    for name, model in models.items():
        pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', model)])
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        results[name] = accuracy
    return results

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

    # Histograms for categorical features
    st.subheader("Histograms for Selected Categorical Features")
    categorical_columns = ['Smoking Status', 'Medical History', 'Alcohol Consumption']
    for column in categorical_columns:
        fig, ax = plt.subplots()
        counts = data[column].value_counts(normalize=True) * 100
        sns.barplot(x=counts.index, y=counts.values, ax=ax)
        ax.set_title(f'Distribution of {column}')
        ax.set_xlabel(column)
        ax.set_ylabel("Percentage")
        for p in ax.patches:
            ax.annotate(f'{p.get_height():.2f}%', (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', 
                        va='center', fontsize=11, color='black', xytext=(0, 10), textcoords='offset points')
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

    # Select and train the best model (Random Forest used here for SHAP)
    best_model_name = 'Random Forest'
    best_model = models[best_model_name]
    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', best_model)])
    pipeline.fit(X_train, y_train)
    st.subheader(f"Best Model Selected: {best_model_name}")

    # Applying SHAP to explain the best model
    st.subheader("SHAP Explanation for a Sample Prediction")
    
    # Transform X_test with preprocessing
    X_test_transformed = pipeline.named_steps['preprocessor'].transform(X_test)
    
    # Fit SHAP explainer on the Random Forest model
    explainer = shap.Explainer(pipeline.named_steps['classifier'], X_test_transformed)
    
    # Generate SHAP values
    shap_values = explainer(X_test_transformed)
    
    # Select a single instance for SHAP explanation (first row of X_test)
    single_obs = X_test_transformed[0].reshape(1, -1)
    shap_values_single = explainer(single_obs)
    
    # Waterfall plot for a single observation
    fig, ax = plt.subplots()
    shap.plots.waterfall(shap_values_single[0], show=False)
    st.pyplot(fig)

    # Summary plot for SHAP values across all features
    fig, ax = plt.subplots()
    shap.summary_plot(shap_values, X_test_transformed, show=False)
    st.pyplot(fig)

with tab3:
    st.header("Scoring")

    # File uploader for custom data
    uploaded_file = st.file_uploader("Upload your dataset for scoring", type="csv")
    
    if uploaded_file is not None:
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
        csv = custom_data.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Scored Data",
            data=csv,
            file_name='scored_data.csv',
            mime='text/csv',
        )
