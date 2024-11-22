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

# Load the dataset
data = pd.read_csv('approved_data.csv')
st.write("Dataset loaded successfully!")

# Function to evaluate models
def evaluate_models(models, X_train, y_train, X_test, y_test):
    results = {}
    for name in models.keys():
        model = models[name]
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

    original_labels = {
        'Gender': ['Gender_Male', 'Gender_Female', 'Gender_Other'],
        'Smoking Status': ['Smoking_Status_Smoker', 'Smoking_Status_Non-Smoker'],
        'Medical History': ['Medical_History_None', 'Medical_History_Diabetes', 'Medical_History_Hypertension', 'Medical_History_Heart Disease'],
        'Occupation': ['Occupation_Engineer', 'Occupation_Teacher', 'Occupation_Doctor', 'Occupation_Lawyer', 'Occupation_Artist', 'Occupation_Business_owner', 'Occupation_Clerk', 'Occupation_Self-Employed', 'Occupation_Other'],
        'Family History of Disease': ['Family_History_of_Disease_None', 'Family_History_of_Disease_Diabetes', 'Family_History_of_Disease_Hypertension', 'Family_History_of_Disease_Heart Disease', 'Family_History_of_Disease_Cancer'],
        'Physical Activity Level': ['Physical_Activity_Level_Low', 'Physical_Activity_Level_Moderate', 'Physical_Activity_Level_High', 'Physical_Activity_Level_Very High'],
        'Alcohol Consumption': ['Alcohol_Consumption_None', 'Alcohol_Consumption_Low', 'Alcohol_Consumption_Moderate', 'Alcohol_Consumption_High'],
        'Premium Payment Frequency': ['Premium_Payment_Frequency_Monthly', 'Premium_Payment_Frequency_Quarterly', 'Premium_Payment_Frequency_Annually'],
        'Term Length': ['Term_Length_5', 'Term_Length_10', 'Term_Length_15', 'Term_Length_20', 'Term_Length_25', 'Term_Length_30', 'Term_Length_35']
    }

    # Initialize a new plot for the full detailed correspondence analysis with original labels
    plt.figure(figsize=(20, 18))  # Increased size for better visibility
 
    # Collect all annotations for adjustment
    texts = []

    # Iterate over each categorical variable and its encoded values
    for col, levels in original_labels.items():
        onehot_encoded = pd.get_dummies(data[col])  # One-hot encode each variable
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(onehot_encoded)  # Apply PCA to the one-hot encoded data

        # Plot each level with its original name
        for i, level in enumerate(levels):
            x, y = pca_result[i, 0], pca_result[i, 1]
            plt.scatter(x, y, label=f"{col}: {level}", s=100)  # Larger marker size
            
            # Store the annotation for later adjustment
            texts.append(plt.text(x, y, f"{level}", fontsize=12, ha='right', va='bottom'))

    # Adjust the text to prevent overlaps
    adjust_text(texts, 
                expand_points=(1.2, 1.2),  # How much to move the labels around the points
                arrowprops=dict(arrowstyle='-', color='grey'))  # Optional: Add arrows to point to original location

    # Make the labels bold where they overlap
    for text in texts:
        for other_text in texts:
            if text == other_text:
                continue
            if np.hypot(text.get_position()[0] - other_text.get_position()[0],
                        text.get_position()[1] - other_text.get_position()[1]) < 0.05:  # Adjust threshold as needed
                text.set_fontweight('bold')
                other_text.set_fontweight('bold')

    # Finalize the plot
    plt.title('Correspondence Analysis of Categorical Variables with Original Labels', fontsize=16)
    plt.xlabel(f'Dim 1 ({pca.explained_variance_ratio_[0] * 100:.2f}%)', fontsize=14)
    plt.ylabel(f'Dim 2 ({pca.explained_variance_ratio_[1] * 100:.2f}%)', fontsize=14)
    plt.axhline(0, color='grey', lw=1)
    plt.axvline(0, color='grey', lw=1)
    plt.grid(True)

    # Position the legend at the bottom of the plot
    plt.legend(bbox_to_anchor=(0.5, -0.1), loc='upper center', fontsize='large', ncol=3)

    plt.tight_layout()  # Adjust layout to fit everything nicely
    st.pyplot(plt)

    # Target distribution
    st.subheader("Approval & Rejection")
    
    # Calculate approval rate
    approval_rate = data['Approved'].mean() * 100
    st.write(f"Approval Rate: {approval_rate:.2f}%")
    
    fig, ax = plt.subplots()
    sns.countplot(x='Approved', data=data, ax=ax)
    ax.set_xlabel("Approved")
    ax.set_ylabel("Count")
    st.pyplot(fig)


with tab2:
    st.header("Modeling")

    # Define models to try
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000),
        'Decision Tree': DecisionTreeClassifier(),
        'Random Forest': RandomForestClassifier(),
        'Gradient Boosting': GradientBoostingClassifier(),
        'Support Vector Machine': SVC(),
        'K-Nearest Neighbors': KNeighborsClassifier(),
        'Naive Bayes': GaussianNB()
    }

    # Load and prepare data
    data = pd.read_csv('approved_data.csv')  # Assuming this is the correct file path
    X = data.drop(columns=['Approved', 'CustomerID'], errors='ignore')
    y = data['Approved']

    # Split the data into training and test sets
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Preprocessing: You might be using a ColumnTransformer or similar
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import StandardScaler

    # Example preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), X.select_dtypes(include=['float64', 'int64']).columns)
        ]
    )

    # Fit the preprocessor to the training data
    preprocessor.fit(X_train)

    results = []

    # Train models and calculate accuracy
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        if accuracy < 1.0:  # Only consider models with accuracy < 1
            results.append({
                'Model': name,
                'Accuracy': accuracy
            })

    # Sort results by accuracy in descending order
    if results:
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values(by='Accuracy', ascending=False)  # Sort in descending order
        st.subheader("Model Accuracy Table")
        st.write(results_df)

        # Store the best model and preprocessor in session state
        best_model_data = max(results, key=lambda x: x['Accuracy'])
        best_model = models[best_model_data['Model']]  # Get the actual model object
        st.session_state.best_model = best_model
        st.session_state.preprocessor = preprocessor  # Store preprocessor as well

        st.write(f"Best Model Selected: {best_model_data['Model']} with Accuracy: {best_model_data['Accuracy']:.3f}")
    else:
        st.error("All models have achieved perfect accuracy (1.0) and were excluded from the results.")


    # Feature importance for tree-based models (RandomForest, DecisionTree, etc.)
    if hasattr(best_model, 'feature_importances_'):
        importance = best_model.feature_importances_
        feature_names = X.columns
        feature_importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importance
        }).sort_values(by='Importance', ascending=False)

        st.subheader("Feature Importance")
        st.write(feature_importance_df)

        # Plot feature importance
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x='Importance', y='Feature', data=feature_importance_df, ax=ax)
        st.pyplot(fig)


with tab3:
    st.header("Scoring")

    # Ensure the best model and preprocessor are available
    if 'best_model' not in st.session_state or 'preprocessor' not in st.session_state:
        st.error("No trained model or preprocessor found. Please train a model first.")
    else:
        best_model = st.session_state.best_model
        preprocessor = st.session_state.preprocessor

        # File uploader for custom data
        uploaded_file = st.file_uploader("Upload your dataset for scoring", type="csv")

        if uploaded_file is not None:
            # Load the uploaded data
            custom_data = pd.read_csv(uploaded_file)
            st.write("Uploaded data:")
            st.write(custom_data.head())

            # Remove 'CustomerID' from scoring data if it's present
            custom_data_clean = custom_data.drop(columns=['CustomerID'], errors='ignore')

            # Apply the same preprocessing used during training
            X_scoring = preprocessor.transform(custom_data_clean)

            # Get predictions and probabilities
            y_pred = best_model.predict(X_scoring)
            y_pred_proba = best_model.predict_proba(X_scoring)[:, 1]  # Assuming binary classification, get probability for the positive class

            # Add predictions and probabilities to the original data
            custom_data['Prediction'] = y_pred
            custom_data['Probability'] = y_pred_proba

            # Reorder columns to have Probability as the second last column
            columns_order = [col for col in custom_data.columns if col != 'Probability'] + ['Probability']
            custom_data = custom_data[columns_order]

            # Display the scored data
            st.subheader("Scored Data")
            st.write(custom_data)

            # Optionally, you can export the scored data as a CSV
            st.download_button(
                label="Download Scored Data",
                data=custom_data.to_csv(index=False),
                file_name='scored_data.csv',
                mime='text/csv'
            )
