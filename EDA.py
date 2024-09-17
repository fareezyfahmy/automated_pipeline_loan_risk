import os
import glob
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

# Step 0: Function to get the latest file from a folder
def get_latest_file(folder_path, file_extension='*.csv'):
    """Get the latest file from a folder."""
    list_of_files = glob.glob(os.path.join(folder_path, file_extension))
    latest_file = max(list_of_files, key=os.path.getctime)
    return latest_file

# Specify the folder where the new monthly datasets are stored
data_folder = 'C:\\Users\\muhdf\\Documents\\WORK\\MONEYLION\\predict_loan\\data\\data'
latest_file = get_latest_file(data_folder)
print(f"Loading data from: {latest_file}")

# Load the latest loan dataset
loan_data = pd.read_csv(latest_file)

# Step 1: General Overview
print("Data Overview:\n")
print(loan_data.info())  # Basic information about the dataset (columns, types, null values)
print("\nSummary Statistics:\n")
print(loan_data.describe())  # Summary statistics for numerical columns

# Step 2: Missing Data Analysis
print("\nMissing Data:\n")
missing_data = loan_data.isnull().sum()
missing_data = missing_data[missing_data > 0]
print(missing_data)

# Visualize missing data
plt.figure(figsize=(10, 6))
sns.heatmap(loan_data.isnull(), cbar=False, cmap='viridis')
plt.title('Missing Data Heatmap')
plt.show()

# Step 3: Target Variable Analysis (loanStatusBinary)
# Function to convert loanStatus to a binary outcome with null handling
def loan_status_to_binary(status):
    if isinstance(status, str):
        if status in ['Returned Item', 'Rejected', 'Withdrawn Application'] or 'void' in status:
            return 1  # Default (risky)
        else:
            return 0  # Non-default (safe)
    return 0  # Treat NaN as non-default

# Apply the function to transform the loanStatus column
loan_data['loanStatusBinary'] = loan_data['loanStatus'].apply(loan_status_to_binary)

# Plot distribution of loanStatusBinary
plt.figure(figsize=(6, 4))
sns.countplot(x='loanStatusBinary', data=loan_data)
plt.title('Distribution of Loan Status (Default vs. Non-default)')
plt.xlabel('Loan Status (0 = Safe, 1 = Risky)')
plt.ylabel('Count')
plt.show()

# Encode all categorical features using Label Encoding
label_encoder = LabelEncoder()
for column in loan_data.columns:
    if loan_data[column].dtype == 'object':
        loan_data[column] = label_encoder.fit_transform(loan_data[column].astype(str))

# Step 4: Correlation Analysis
# Calculate the Pearson correlation coefficient
correlation_matrix = loan_data.corr()

# Extract the correlation with the target variable and sort it
target_correlation = correlation_matrix['loanStatusBinary'].sort_values(ascending=False)
print(f"\nCorrelation with Loan Status Binary:\n{target_correlation}")

# Step 5: Feature Distributions
# Function to output top features influencing the target
def get_influential_features(corr_matrix, target, top_n=5):
    """
    Outputs top n features that have the highest correlation with the target variable.
    Excludes features that may cause data leakage.
    """
    excluded_features = ['loanStatus', 'loanStatusBinary']
    influential_features = corr_matrix[target].drop(labels=excluded_features, errors='ignore')
    influential_features = influential_features.abs().sort_values(ascending=False).head(top_n)
    return influential_features

# Get top 5 features influencing loanStatusBinary
top_5_features = get_influential_features(correlation_matrix, 'loanStatusBinary', top_n=5)
top_5_feature_names = top_5_features.index.tolist()

# Plot distributions for top 5 features
for feature in top_5_feature_names:
    plt.figure(figsize=(6, 4))
    sns.histplot(loan_data[feature], kde=True, bins=30)
    plt.title(f'Distribution of {feature}')
    plt.show()

# Step 6: Correlation Heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm')
plt.title('Correlation Heatmap for Loan Data')
plt.show()

# Step 7: Distribution of Numerical Features by Target Variable (loanStatusBinary)
for feature in top_5_feature_names:
    plt.figure(figsize=(6, 4))
    sns.boxplot(x='loanStatusBinary', y=feature, data=loan_data)
    plt.title(f'Distribution of {feature} by Loan Status')
    plt.show()
