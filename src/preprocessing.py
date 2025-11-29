"""
Data preprocessing utilities for DDoS detection.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import joblib
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataPreprocessor:
    """Handles all preprocessing steps for the DDoS dataset."""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_columns = None
        self.target_column = 'Label'  # Adjust based on actual dataset
        
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean the dataset by handling missing values and duplicates.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Cleaned DataFrame
        """
        logger.info("Cleaning data...")
        initial_rows = len(df)
        
        # Remove duplicates
        df = df.drop_duplicates()
        logger.info(f"Removed {initial_rows - len(df)} duplicate rows")
        
        # Handle missing values
        missing_before = df.isnull().sum().sum()
        
        # For numeric columns, fill with median
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].isnull().any():
                df[col].fillna(df[col].median(), inplace=True)
        
        # For categorical columns, fill with mode
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if df[col].isnull().any():
                df[col].fillna(df[col].mode()[0], inplace=True)
        
        missing_after = df.isnull().sum().sum()
        logger.info(f"Filled {missing_before - missing_after} missing values")
        
        # Remove infinite values
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.dropna()
        
        return df
    
    def encode_labels(self, df: pd.DataFrame, target_col: str = 'Label') -> pd.DataFrame:
        """
        Encode target labels to numeric values.
        
        Args:
            df: Input DataFrame
            target_col: Name of the target column
            
        Returns:
            DataFrame with encoded labels
        """
        logger.info(f"Encoding labels in column: {target_col}")
        
        if target_col in df.columns:
            df[target_col] = self.label_encoder.fit_transform(df[target_col])
            logger.info(f"Label classes: {self.label_encoder.classes_}")
        else:
            logger.warning(f"Target column '{target_col}' not found")
        
        return df
    
    def scale_features(self, X: pd.DataFrame, fit: bool = True) -> np.ndarray:
        """
        Scale numeric features using StandardScaler.
        
        Args:
            X: Feature DataFrame
            fit: Whether to fit the scaler (True for training, False for test)
            
        Returns:
            Scaled feature array
        """
        logger.info("Scaling features...")
        
        if fit:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = self.scaler.transform(X)
        
        return X_scaled
    
    def prepare_data(self, df: pd.DataFrame, target_col: str = 'Label', 
                     test_size: float = 0.2, val_size: float = 0.1, 
                     random_state: int = 42):
        """
        Complete preprocessing pipeline.
        
        Args:
            df: Input DataFrame
            target_col: Name of target column
            test_size: Proportion of data for testing
            val_size: Proportion of training data for validation
            random_state: Random seed
            
        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        logger.info("Starting preprocessing pipeline...")
        
        # Clean data
        df = self.clean_data(df)
        
        # Separate features and target
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in dataset")
        
        # Encode labels
        df = self.encode_labels(df, target_col)
        
        # Identify columns to drop (target and any other label-like columns)
        columns_to_drop = [target_col]
        
        # Drop 'Class' column if it exists and is different from target_col
        if 'Class' in df.columns and 'Class' != target_col:
            columns_to_drop.append('Class')
            logger.info("Dropping 'Class' column as it's not the target")
        
        # Drop any other non-numeric columns that aren't features
        non_numeric_cols = df.select_dtypes(include=['object']).columns.tolist()
        for col in non_numeric_cols:
            if col not in columns_to_drop and col != target_col:
                columns_to_drop.append(col)
                logger.info(f"Dropping non-numeric column: {col}")
        
        # Split features and target
        X = df.drop(columns=columns_to_drop)
        y = df[target_col]
        
        logger.info(f"Features shape: {X.shape}")
        logger.info(f"Target shape: {y.shape}")
        
        # Store feature columns
        self.feature_columns = X.columns.tolist()
        
        # Check class distribution for stratification
        class_counts = y.value_counts()
        min_class_count = class_counts.min()
        
        # Remove classes with too few samples (less than 2)
        if min_class_count < 2:
            logger.warning(f"Found classes with only {min_class_count} sample(s). Removing rare classes...")
            valid_classes = class_counts[class_counts >= 2].index
            mask = y.isin(valid_classes)
            X = X[mask]
            y = y[mask]
            logger.info(f"Removed {(~mask).sum()} samples from rare classes")
            logger.info(f"Remaining samples: {len(y)}")
        
        # Determine if we can use stratification
        class_counts = y.value_counts()
        min_class_count = class_counts.min()
        use_stratify = min_class_count >= 2
        
        if not use_stratify:
            logger.warning("Cannot use stratified split due to class imbalance. Using random split.")
        
        # Split into train and test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, 
            stratify=y if use_stratify else None
        )
        
        # Check if validation split is possible with stratification
        val_class_counts = y_train.value_counts()
        val_min_count = val_class_counts.min()
        use_val_stratify = val_min_count >= 2
        
        # Split train into train and validation
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=val_size, random_state=random_state, 
            stratify=y_train if use_val_stratify else None
        )
        
        # Scale features
        X_train_scaled = self.scale_features(X_train, fit=True)
        X_val_scaled = self.scale_features(X_val, fit=False)
        X_test_scaled = self.scale_features(X_test, fit=False)
        
        logger.info(f"Train set: {X_train_scaled.shape}")
        logger.info(f"Validation set: {X_val_scaled.shape}")
        logger.info(f"Test set: {X_test_scaled.shape}")
        
        return X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test
    
    def save_preprocessor(self, save_dir: str = "models/saved_models"):
        """Save the preprocessor objects."""
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        joblib.dump(self.scaler, save_path / "scaler.pkl")
        joblib.dump(self.label_encoder, save_path / "label_encoder.pkl")
        joblib.dump(self.feature_columns, save_path / "feature_columns.pkl")
        
        logger.info(f"Preprocessor saved to {save_dir}")
    
    def load_preprocessor(self, load_dir: str = "models/saved_models"):
        """Load the preprocessor objects."""
        load_path = Path(load_dir)
        
        self.scaler = joblib.load(load_path / "scaler.pkl")
        self.label_encoder = joblib.load(load_path / "label_encoder.pkl")
        self.feature_columns = joblib.load(load_path / "feature_columns.pkl")
        
        logger.info(f"Preprocessor loaded from {load_dir}")


if __name__ == "__main__":
    # Example usage
    from data_loader import load_dataset
    
    # Load sample data
    df = load_dataset("data/raw/cicddos2019_dataset.csv", sample_size=1000)
    
    # Preprocess
    preprocessor = DataPreprocessor()
    X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.prepare_data(df)
    
    print(f"Preprocessing complete!")
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Test samples: {len(X_test)}")
