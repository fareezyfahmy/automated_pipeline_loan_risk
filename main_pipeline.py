import logging
from data_ingestion import load_data
from preprocessing import preprocess_data, get_influential_features
from model_training import train_model, retrain_model_with_tuning, save_model, load_model
from model_evaluation import evaluate_model, detect_feature_drift
from sklearn.model_selection import train_test_split
from datetime import datetime
import os


# Set up logging
logging.basicConfig(filename='pipeline.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

try:
    # Log start of the pipeline
    logging.info("Pipeline execution started.")

    # Load data
    folder_path = 'C:\\Users\\muhdf\\Documents\\WORK\\MONEYLION\\predict_loan\\data\\data'
    loan_data = load_data(folder_path)
    logging.info("Data loaded successfully from the latest file.")

    # Preprocess data
    loan_data = preprocess_data(loan_data)
    logging.info("Data preprocessed successfully.")

    # Feature selection
    correlation_matrix = loan_data.corr()
    top_5_features = get_influential_features(correlation_matrix, 'loanStatusBinary', top_n=5)
    top_5_feature_names = top_5_features.index.tolist()

    # Train-test split
    X = loan_data[top_5_feature_names]
    y = loan_data['loanStatusBinary']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    logging.info("Data split into train and test sets.")

    # Load or train the model
    model_filename = f'lightgbm_model_{datetime.now().strftime("%Y%m%d")}.pkl'
    clf = load_model(model_filename)
    if clf is None:
        clf = train_model(X_train, y_train)
        logging.info("No pre-trained model found. Training a new model.")
    else:
        logging.info("Pre-trained model found and fine-tuned.")    
    
    # Load previous training data (for drift detection)
    previous_data_path = 'C:\\Users\\muhdf\\Documents\\WORK\\MONEYLION\\predict_loan\\data\\data'
    previous_data = load_data(previous_data_path, n=2)  # Load the second latest dataset
    previous_data = preprocess_data(previous_data)

    # Detect feature drift between previous and current data
    drifted_features = detect_feature_drift(previous_data[top_5_feature_names], loan_data[top_5_feature_names])

    if drifted_features:
        logging.info(f"Drift detected in the following features: {drifted_features}")
        print(f"Drift detected in the following features: {drifted_features}")
        # Retrain model if drift is detected
        clf = retrain_model_with_tuning(clf, X_train, y_train)
        logging.info("Model retrained due to feature drift.")
    else:
        logging.info("No drift detected.")
        print("No drift detected, proceeding with the existing model.")    
    
    evaluate_model(clf, X, y, X_test, y_test)
    logging.info("Model evaluation completed.")

    # Save the model with versioning
    save_model(clf, model_filename)
    logging.info(f"Model saved as {model_filename}.")

    # Log end of the pipeline
    logging.info("Pipeline execution finished successfully.")

except Exception as e:
    logging.error(f"Error occurred during pipeline execution: {str(e)}")
    raise