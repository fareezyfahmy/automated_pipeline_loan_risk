import joblib

def save_deployment_model(clf, model_path='lightgbm_pretrained_model.pkl'):
    """Save the trained model for deployment."""
    joblib.dump(clf, model_path)
    print(f"Model saved for deployment at {model_path}")

def load_deployment_model(model_path='lightgbm_pretrained_model.pkl'):
    """Load the model for deployment."""
    try:
        clf = joblib.load(model_path)
        print(f"Model loaded from {model_path}")
        return clf
    except FileNotFoundError:
        print("Model not found.")
        return None