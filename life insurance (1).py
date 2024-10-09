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

    # Detailed Correspondence Analysis with Original Labels
    st.subheader("Detailed Correspondence Analysis with Original Labels")

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

    # Highlight the best model dynamically
    best_model_name = results_df.index[0]
    best_model_accuracy = results_df.iloc[0, 0]
    st.subheader(f"Best Model: {best_model_name}")
    st.write(f"Accuracy: {best_model_accuracy}")

    # Select and train the best model
    best_model = models[best_model_name]
    best_model.fit(X_train, y_train)

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
            # Example criteria (adjust based on actual underwriting policies)
            if (18 <= row['Age'] <= 65 and
                row['Smoking Status'] == 0 and  # Assuming 0 is Non-Smoker
                18.5 <= row['BMI'] <= 24.9 and  # Healthy BMI range
                row['Medical History'] == 0 and  # Assuming 0 is No significant medical history
                row['Alcohol Consumption'] <= 2 and  # Assuming 2 or less is moderate/no consumption
                row['Family History of Disease'] == 0 and  # Assuming 0 is No significant family history
                row['Occupation'] != 'High Risk'):  # Assuming 'High Risk' occupation is identified
                return 1
            else:
                return 0

        # Apply criteria to create 'Approved' column
        custom_data['Approved'] = custom_data.apply(approve_application, axis=1)

        # Display the updated data with the 'Approved' column
        st.write("Scored data with 'Approved' column:")
        st.write(custom_data.head())

        # Calculate the approval rate
        approval_rate_calculated = custom_data['Approved'].mean() * 100
        st.write(f"Approval Rate: {approval_rate_calculated:.2f}%")
