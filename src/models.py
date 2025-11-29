"""
Machine learning models for DDoS detection.
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import joblib
from pathlib import Path
import logging
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DDoSDetector:
    """Wrapper class for DDoS detection models."""
    
    def __init__(self, model_type: str = 'random_forest'):
        """
        Initialize detector with specified model type.
        
        Args:
            model_type: Type of model ('random_forest', 'svm', 'neural_network')
        """
        self.model_type = model_type
        self.model = self._create_model(model_type)
        self.training_time = None
        
    def _create_model(self, model_type: str):
        """Create the specified model."""
        if model_type == 'random_forest':
            return RandomForestClassifier(
                n_estimators=100,
                max_depth=20,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1,
                verbose=1
            )
        elif model_type == 'svm':
            # Use SGDClassifier for large datasets (much faster than SVC)
            return SGDClassifier(
                loss='hinge',
                penalty='l2',
                alpha=0.0001,
                max_iter=1000,
                random_state=42,
                n_jobs=-1,
                verbose=1
            )
        elif model_type == 'neural_network':
            return MLPClassifier(
                hidden_layer_sizes=(128, 64, 32),
                activation='relu',
                solver='adam',
                alpha=0.0001,
                batch_size=256,
                learning_rate='adaptive',
                max_iter=200,
                random_state=42,
                verbose=True,
                early_stopping=True,
                validation_fraction=0.1
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def train(self, X_train, y_train):
        """
        Train the model.
        
        Args:
            X_train: Training features
            y_train: Training labels
        """
        logger.info(f"Training {self.model_type} model...")
        start_time = time.time()
        
        self.model.fit(X_train, y_train)
        
        self.training_time = time.time() - start_time
        logger.info(f"Training completed in {self.training_time:.2f} seconds")
    
    def predict(self, X):
        """Make predictions."""
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """Get prediction probabilities."""
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)
        else:
            # For models without predict_proba (like SGDClassifier)
            if hasattr(self.model, 'decision_function'):
                return self.model.decision_function(X)
            else:
                raise AttributeError(f"{self.model_type} does not support probability predictions")
    
    def save_model(self, save_path: str):
        """Save the trained model."""
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        joblib.dump(self.model, save_path)
        logger.info(f"Model saved to {save_path}")
    
    def load_model(self, load_path: str):
        """Load a trained model."""
        self.model = joblib.load(load_path)
        logger.info(f"Model loaded from {load_path}")


def train_random_forest(X_train, y_train, tune_hyperparameters: bool = False):
    """
    Train a Random Forest classifier.
    
    Args:
        X_train: Training features
        y_train: Training labels
        tune_hyperparameters: Whether to perform hyperparameter tuning
        
    Returns:
        Trained model
    """
    logger.info("Training Random Forest...")
    
    if tune_hyperparameters:
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [10, 20, 30, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        
        rf = RandomForestClassifier(random_state=42, n_jobs=-1)
        grid_search = GridSearchCV(rf, param_grid, cv=3, scoring='f1_weighted', n_jobs=-1, verbose=2)
        grid_search.fit(X_train, y_train)
        
        logger.info(f"Best parameters: {grid_search.best_params_}")
        return grid_search.best_estimator_
    else:
        detector = DDoSDetector('random_forest')
        detector.train(X_train, y_train)
        return detector.model


def train_svm(X_train, y_train):
    """
    Train an SVM classifier (using SGDClassifier for efficiency).
    
    Args:
        X_train: Training features
        y_train: Training labels
        
    Returns:
        Trained model
    """
    logger.info("Training SVM (SGDClassifier)...")
    detector = DDoSDetector('svm')
    detector.train(X_train, y_train)
    return detector.model


def train_neural_network(X_train, y_train):
    """
    Train a Neural Network classifier.
    
    Args:
        X_train: Training features
        y_train: Training labels
        
    Returns:
        Trained model
    """
    logger.info("Training Neural Network...")
    detector = DDoSDetector('neural_network')
    detector.train(X_train, y_train)
    return detector.model


if __name__ == "__main__":
    # Example usage
    from data_loader import load_dataset
    from preprocessing import DataPreprocessor
    
    # Load and preprocess data
    df = load_dataset("data/raw/cicddos2019_dataset.csv", sample_size=10000)
    preprocessor = DataPreprocessor()
    X_train, _, X_test, y_train, _, y_test = preprocessor.prepare_data(df)
    
    # Train models
    rf_model = train_random_forest(X_train, y_train)
    svm_model = train_svm(X_train, y_train)
    nn_model = train_neural_network(X_train, y_train)
    
    # Save models
    joblib.dump(rf_model, "models/saved_models/random_forest.pkl")
    joblib.dump(svm_model, "models/saved_models/svm.pkl")
    joblib.dump(nn_model, "models/saved_models/neural_network.pkl")
    
    print("All models trained and saved!")
