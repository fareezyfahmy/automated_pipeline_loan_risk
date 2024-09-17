import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder

def missing_data_analysis(data):
    """Analyze and visualize missing data."""
    missing_data = data.isnull().sum()
    missing_data = missing_data[missing_data > 0]
    print(missing_data)

    plt.figure(figsize=(10, 6))
    sns.heatmap(data.isnull(), cbar=False, cmap='viridis')
    plt.title('Missing Data Heatmap')
    plt.show()

    return missing_data

def loan_status_to_binary(status):
    """Convert loanStatus to binary outcome."""
    if isinstance(status, str):
        if status in ['Returned Item', 'Rejected', 'Withdrawn Application'] or 'void' in status:
            return 1  # Default (risky)
        else:
            return 0  # Non-default (safe)
    return 0  # Treat NaN as non-default

def preprocess_data(loan_data):
    """Preprocess loan data by handling missing values and encoding categorical features."""
    # Convert loanStatus to a binary variable
    loan_data['loanStatusBinary'] = loan_data['loanStatus'].apply(loan_status_to_binary)

    # Label encode categorical features
    label_encoder = LabelEncoder()
    for column in loan_data.columns:
        if loan_data[column].dtype == 'object':
            loan_data[column] = label_encoder.fit_transform(loan_data[column].astype(str))

    # Handle missing values by filling them with zeros (you can modify this strategy if necessary)
    loan_data = loan_data.fillna(0)

    return loan_data

def get_influential_features(corr_matrix, target, top_n=5):
    """Outputs top n features that have the highest correlation with the target variable."""
    excluded_features = ['loanStatus', 'loanStatusBinary']
    influential_features = corr_matrix[target].drop(labels=excluded_features, errors='ignore')
    influential_features = influential_features.abs().sort_values(ascending=False).head(top_n)
    print(influential_features)
    return influential_features

# Example usage:
# missing_data_analysis(loan_data)
# loan_data = preprocess_data(loan_data)