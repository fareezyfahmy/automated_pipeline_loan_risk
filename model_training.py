from lightgbm import LGBMClassifier
from sklearn.model_selection import RandomizedSearchCV
import joblib

def train_model(X_train, y_train):
    """Train a new LightGBM model."""
    clf = LGBMClassifier(
        objective='binary',
        num_leaves=31,
        learning_rate=0.05,
        n_estimators=100,
        reg_alpha=1.0,
        reg_lambda=1.0,
        max_depth=5,
        min_child_samples=50,
        feature_fraction=0.8,
        bagging_fraction=0.8,
        bagging_freq=5
    )
    clf.fit(X_train, y_train)
    return clf

def retrain_model_with_tuning(clf, X_train, y_train):
    """Perform hyperparameter tuning and retrain the model."""
    param_dist = {
        'num_leaves': [31, 50, 70],
        'learning_rate': [0.05, 0.01, 0.1],
        'n_estimators': [100, 200, 300],
        'reg_alpha': [0.0, 1.0, 5.0],
        'reg_lambda': [0.0, 1.0, 5.0],
        'max_depth': [5, 7, 10],
        'min_child_samples': [20, 50, 100],
        'feature_fraction': [0.7, 0.8, 0.9]
    }

    random_search = RandomizedSearchCV(
        clf, 
        param_distributions=param_dist, 
        n_iter=10, 
        scoring='roc_auc', 
        cv=5, 
        verbose=1, 
        random_state=42
    )

    random_search.fit(X_train, y_train)
    clf = random_search.best_estimator_
    return clf

def save_model(clf, filename='lightgbm_pretrained_model.pkl'):
    """Save the trained model to a file."""
    joblib.dump(clf, filename)
    print(f"Model saved to {filename}")

def load_model(filename='lightgbm_pretrained_model.pkl'):
    """Load a pre-trained model."""
    try:
        clf = joblib.load(filename)
        print("Pre-trained model loaded successfully!")
        return clf
    except FileNotFoundError:
        print("No pre-trained model found.")
        return None

# Example usage:
# clf = train_model(X_train, y_train)
# save_model(clf)
# clf = load_model()