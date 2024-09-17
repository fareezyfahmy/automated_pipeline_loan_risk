from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import cross_val_score
import numpy as np

def evaluate_model(clf, X, y, X_test, y_test):
    """
    Evaluate the model using ROC AUC, Precision, Recall, F1 Score, and Confusion Matrix.
    Perform 5-fold cross-validation on the model and calculate AUC score.

    Parameters:
    clf: trained classifier
    X: features
    y: labels
    X_test: test features
    y_test: true labels for test set
    
    Returns:
    Prints out evaluation metrics, cross-validated AUC scores and the average AUC score
    """
    # Predict labels and probabilities
    y_pred = clf.predict(X_test)
    y_pred_prob = clf.predict_proba(X_test)[:, 1]
    
    # Evaluate the model using various metrics
    roc_auc = roc_auc_score(y_test, y_pred_prob)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)

    # Perform 5-fold cross-validation
    cv_scores = cross_val_score(clf, X, y, cv=5, scoring='roc_auc')
    
    # Print the metrics
    print(f'ROC AUC: {roc_auc:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1 Score: {f1:.4f}')
    print(f'Confusion Matrix:\n{conf_matrix}')

    # Print cross-validated scores
    print(f'Cross-validated AUC scores: {cv_scores}')
    print(f'Average AUC score: {cv_scores.mean():.4f}')

def calculate_psi(expected, actual, buckets=10):
    """Calculate the Population Stability Index (PSI) between two distributions."""
    
    def scale_range_values(series, range_min=0, range_max=1):
        """Scale feature values to the specified range."""
        return (series - series.min()) / (series.max() - series.min()) * (range_max - range_min) + range_min

    # Scale both datasets to the same range
    expected_scaled = scale_range_values(expected)
    actual_scaled = scale_range_values(actual)

    # Create bins and calculate the distribution within the bins
    expected_counts, bin_edges = np.histogram(expected_scaled, bins=buckets)
    actual_counts, _ = np.histogram(actual_scaled, bins=bin_edges)

    # Avoid division by zero
    expected_counts = np.where(expected_counts == 0, 0.0001, expected_counts)
    actual_counts = np.where(actual_counts == 0, 0.0001, actual_counts)

    expected_dist = expected_counts / len(expected)
    actual_dist = actual_counts / len(actual)

    # Calculate PSI for each bucket
    psi_values = (expected_dist - actual_dist) * np.log(expected_dist / actual_dist)

    # Return the total PSI
    return np.sum(psi_values)

def detect_feature_drift(train_data, new_data, threshold=0.1):
    """Detect drift across all features between training and new data."""
    drifted_features = []

    for column in train_data.columns:
        if train_data[column].dtype in [np.float64, np.int64]:  # Only apply to numeric columns
            psi_value = calculate_psi(train_data[column], new_data[column])
            if psi_value > threshold:
                drifted_features.append((column, psi_value))

    return drifted_features
