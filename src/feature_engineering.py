"""
Feature engineering and selection utilities.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Handles feature selection and engineering."""
    
    def __init__(self):
        self.feature_importance = None
        self.selected_features = None
    
    def calculate_feature_importance(self, X, y, n_estimators: int = 100):
        """
        Calculate feature importance using Random Forest.
        
        Args:
            X: Feature array or DataFrame
            y: Target array
            n_estimators: Number of trees in Random Forest
            
        Returns:
            DataFrame with feature importance scores
        """
        logger.info("Calculating feature importance...")
        
        # Train a Random Forest to get feature importance
        rf = RandomForestClassifier(n_estimators=n_estimators, random_state=42, n_jobs=-1)
        rf.fit(X, y)
        
        # Get feature importance
        if isinstance(X, pd.DataFrame):
            feature_names = X.columns
        else:
            feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': rf.feature_importances_
        }).sort_values('importance', ascending=False)
        
        self.feature_importance = importance_df
        logger.info(f"Top 5 features: {importance_df.head()['feature'].tolist()}")
        
        return importance_df
    
    def select_top_features(self, X, importance_df: pd.DataFrame, 
                           n_features: int = None, threshold: float = None):
        """
        Select top features based on importance.
        
        Args:
            X: Feature array or DataFrame
            importance_df: DataFrame with feature importance
            n_features: Number of top features to select
            threshold: Importance threshold (alternative to n_features)
            
        Returns:
            Selected features array and feature names
        """
        if n_features:
            selected = importance_df.head(n_features)['feature'].tolist()
        elif threshold:
            selected = importance_df[importance_df['importance'] >= threshold]['feature'].tolist()
        else:
            # Default: select features that contribute to 95% cumulative importance
            cumsum = importance_df['importance'].cumsum()
            n_features = (cumsum <= 0.95).sum() + 1
            selected = importance_df.head(n_features)['feature'].tolist()
        
        self.selected_features = selected
        logger.info(f"Selected {len(selected)} features")
        
        if isinstance(X, pd.DataFrame):
            return X[selected], selected
        else:
            # Assuming feature names match column indices
            feature_indices = [i for i, f in enumerate(importance_df['feature']) if f in selected]
            return X[:, feature_indices], selected
    
    def plot_feature_importance(self, importance_df: pd.DataFrame = None, 
                                top_n: int = 20, save_path: str = None):
        """
        Plot feature importance.
        
        Args:
            importance_df: DataFrame with feature importance
            top_n: Number of top features to plot
            save_path: Path to save the plot
        """
        if importance_df is None:
            importance_df = self.feature_importance
        
        if importance_df is None:
            raise ValueError("No feature importance data available")
        
        plt.figure(figsize=(10, 8))
        top_features = importance_df.head(top_n)
        
        sns.barplot(data=top_features, x='importance', y='feature', palette='viridis')
        plt.title(f'Top {top_n} Feature Importance')
        plt.xlabel('Importance Score')
        plt.ylabel('Feature')
        plt.tight_layout()
        
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Feature importance plot saved to {save_path}")
        
        plt.close()
    
    def apply_pca(self, X, n_components: float = 0.95):
        """
        Apply PCA for dimensionality reduction.
        
        Args:
            X: Feature array
            n_components: Number of components or variance to retain
            
        Returns:
            Transformed features and PCA object
        """
        logger.info(f"Applying PCA with {n_components} variance retention...")
        
        pca = PCA(n_components=n_components, random_state=42)
        X_pca = pca.fit_transform(X)
        
        logger.info(f"Reduced from {X.shape[1]} to {X_pca.shape[1]} components")
        logger.info(f"Explained variance: {pca.explained_variance_ratio_.sum():.4f}")
        
        return X_pca, pca
    
    def create_feature_report(self, X, y, save_dir: str = "models/results"):
        """
        Create a comprehensive feature analysis report.
        
        Args:
            X: Feature array or DataFrame
            y: Target array
            save_dir: Directory to save results
        """
        logger.info("Creating feature analysis report...")
        
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Calculate importance
        importance_df = self.calculate_feature_importance(X, y)
        
        # Save importance to CSV
        importance_df.to_csv(save_path / "feature_importance.csv", index=False)
        
        # Plot importance
        self.plot_feature_importance(importance_df, save_path=save_path / "feature_importance.png")
        
        logger.info(f"Feature report saved to {save_dir}")
        
        return importance_df


if __name__ == "__main__":
    # Example usage
    from data_loader import load_dataset
    from preprocessing import DataPreprocessor
    
    # Load and preprocess data
    df = load_dataset("data/raw/cicddos2019_dataset.csv", sample_size=5000)
    preprocessor = DataPreprocessor()
    X_train, _, _, y_train, _, _ = preprocessor.prepare_data(df)
    
    # Feature engineering
    engineer = FeatureEngineer()
    importance_df = engineer.create_feature_report(X_train, y_train)
    
    # Select top features
    X_selected, selected_features = engineer.select_top_features(X_train, importance_df, n_features=30)
    print(f"Selected features: {selected_features}")
