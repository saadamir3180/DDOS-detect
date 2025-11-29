"""
Prediction interface for trained models.
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DDoSPredictor:
    """Interface for making predictions with trained models."""
    
    def __init__(self, model_path: str, preprocessor_dir: str = "models/saved_models"):
        """
        Initialize predictor.
        
        Args:
            model_path: Path to the trained model
            preprocessor_dir: Directory containing preprocessor objects
        """
        self.model_path = Path(model_path)
        self.preprocessor_dir = Path(preprocessor_dir)
        
        # Load model and preprocessors
        self.load_pipeline()
    
    def load_pipeline(self):
        """Load model and all preprocessing objects."""
        logger.info("Loading prediction pipeline...")
        
        # Load model
        self.model = joblib.load(self.model_path)
        logger.info(f"Model loaded from {self.model_path}")
        
        # Load preprocessors
        try:
            self.scaler = joblib.load(self.preprocessor_dir / "scaler.pkl")
            self.label_encoder = joblib.load(self.preprocessor_dir / "label_encoder.pkl")
            self.feature_columns = joblib.load(self.preprocessor_dir / "feature_columns.pkl")
            logger.info("Preprocessors loaded successfully")
        except FileNotFoundError as e:
            logger.warning(f"Preprocessor not found: {e}")
            self.scaler = None
            self.label_encoder = None
            self.feature_columns = None
    
    def preprocess_input(self, X: pd.DataFrame) -> np.ndarray:
        """
        Preprocess input data for prediction.
        
        Args:
            X: Input DataFrame
            
        Returns:
            Preprocessed feature array
        """
        # Ensure correct columns
        if self.feature_columns:
            # Check if all required columns are present
            missing_cols = set(self.feature_columns) - set(X.columns)
            if missing_cols:
                logger.warning(f"Missing columns: {missing_cols}")
                # Add missing columns with zeros
                for col in missing_cols:
                    X[col] = 0
            
            # Select and reorder columns
            X = X[self.feature_columns]
        
        # Scale features
        if self.scaler:
            X_scaled = self.scaler.transform(X)
        else:
            X_scaled = X.values
        
        return X_scaled
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions on input data.
        
        Args:
            X: Input DataFrame
            
        Returns:
            Predicted labels
        """
        X_processed = self.preprocess_input(X)
        predictions = self.model.predict(X_processed)
        
        # Decode labels if encoder is available
        if self.label_encoder:
            predictions = self.label_encoder.inverse_transform(predictions)
        
        return predictions
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Get prediction probabilities.
        
        Args:
            X: Input DataFrame
            
        Returns:
            Prediction probabilities
        """
        X_processed = self.preprocess_input(X)
        
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X_processed)
        elif hasattr(self.model, 'decision_function'):
            return self.model.decision_function(X_processed)
        else:
            raise AttributeError("Model does not support probability predictions")
    
    def predict_single(self, traffic_data: dict) -> tuple:
        """
        Predict on a single traffic sample.
        
        Args:
            traffic_data: Dictionary with feature values
            
        Returns:
            Tuple of (prediction, confidence)
        """
        # Convert to DataFrame
        df = pd.DataFrame([traffic_data])
        
        # Get prediction
        prediction = self.predict(df)[0]
        
        # Get confidence
        try:
            proba = self.predict_proba(df)
            if len(proba.shape) > 1:
                confidence = np.max(proba[0])
            else:
                confidence = abs(proba[0])
        except:
            confidence = None
        
        return prediction, confidence
    
    def batch_predict(self, csv_path: str, output_path: str = None) -> pd.DataFrame:
        """
        Make predictions on a CSV file.
        
        Args:
            csv_path: Path to input CSV
            output_path: Path to save predictions (optional)
            
        Returns:
            DataFrame with predictions
        """
        logger.info(f"Making predictions on {csv_path}")
        
        # Load data
        df = pd.read_csv(csv_path)
        
        # Make predictions
        predictions = self.predict(df)
        
        # Add predictions to DataFrame
        df['Predicted_Label'] = predictions
        
        # Add confidence if available
        try:
            proba = self.predict_proba(df)
            if len(proba.shape) > 1:
                df['Confidence'] = np.max(proba, axis=1)
            else:
                df['Confidence'] = np.abs(proba)
        except:
            logger.warning("Could not calculate confidence scores")
        
        # Save if output path provided
        if output_path:
            df.to_csv(output_path, index=False)
            logger.info(f"Predictions saved to {output_path}")
        
        return df


if __name__ == "__main__":
    # Example usage
    predictor = DDoSPredictor(
        model_path="models/saved_models/random_forest.pkl",
        preprocessor_dir="models/saved_models"
    )
    
    # Batch prediction
    results = predictor.batch_predict(
        csv_path="data/samples/test_traffic.csv",
        output_path="data/samples/predictions.csv"
    )
    
    print(f"Predictions:\n{results[['Predicted_Label', 'Confidence']].head()}")
