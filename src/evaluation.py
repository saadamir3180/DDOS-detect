"""
Model evaluation utilities and metrics visualization.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc,
    roc_auc_score
)
from pathlib import Path
import logging
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Comprehensive model evaluation and visualization."""
    
    def __init__(self, model_name: str = "Model"):
        self.model_name = model_name
        self.metrics = {}
    
    def calculate_metrics(self, y_true, y_pred, y_pred_proba=None):
        """
        Calculate all evaluation metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Prediction probabilities (optional)
            
        Returns:
            Dictionary of metrics
        """
        logger.info(f"Calculating metrics for {self.model_name}...")
        
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'f1_score': f1_score(y_true, y_pred, average='weighted', zero_division=0)
        }
        
        # Add AUC if probabilities are provided
        if y_pred_proba is not None:
            try:
                # For binary classification
                if len(np.unique(y_true)) == 2:
                    if len(y_pred_proba.shape) > 1:
                        metrics['auc'] = roc_auc_score(y_true, y_pred_proba[:, 1])
                    else:
                        metrics['auc'] = roc_auc_score(y_true, y_pred_proba)
                else:
                    # For multi-class
                    metrics['auc'] = roc_auc_score(y_true, y_pred_proba, 
                                                   multi_class='ovr', average='weighted')
            except Exception as e:
                logger.warning(f"Could not calculate AUC: {e}")
        
        self.metrics = metrics
        
        # Log metrics
        logger.info(f"Metrics for {self.model_name}:")
        for metric, value in metrics.items():
            logger.info(f"  {metric}: {value:.4f}")
        
        return metrics
    
    def plot_confusion_matrix(self, y_true, y_pred, labels=None, save_path=None):
        """
        Plot confusion matrix.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            labels: Class labels
            save_path: Path to save the plot
        """
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=labels, yticklabels=labels)
        plt.title(f'Confusion Matrix - {self.model_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Confusion matrix saved to {save_path}")
        
        plt.close()
    
    def plot_roc_curve(self, y_true, y_pred_proba, save_path=None):
        """
        Plot ROC curve (for binary classification).
        
        Args:
            y_true: True labels
            y_pred_proba: Prediction probabilities
            save_path: Path to save the plot
        """
        if len(np.unique(y_true)) != 2:
            logger.warning("ROC curve is only for binary classification")
            return
        
        # Get probabilities for positive class
        if len(y_pred_proba.shape) > 1:
            y_scores = y_pred_proba[:, 1]
        else:
            y_scores = y_pred_proba
        
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {self.model_name}')
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"ROC curve saved to {save_path}")
        
        plt.close()
    
    def generate_classification_report(self, y_true, y_pred, labels=None, save_path=None):
        """
        Generate detailed classification report.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            labels: Class labels
            save_path: Path to save the report
            
        Returns:
            Classification report as string
        """
        report = classification_report(y_true, y_pred, target_names=labels)
        
        logger.info(f"Classification Report for {self.model_name}:\n{report}")
        
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            with open(save_path, 'w') as f:
                f.write(f"Classification Report - {self.model_name}\n")
                f.write("=" * 60 + "\n\n")
                f.write(report)
            logger.info(f"Classification report saved to {save_path}")
        
        return report
    
    def save_metrics(self, save_path: str):
        """Save metrics to JSON file."""
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(save_path, 'w') as f:
            json.dump(self.metrics, f, indent=4)
        
        logger.info(f"Metrics saved to {save_path}")


def compare_models(models_results: dict, save_path: str = None):
    """
    Compare multiple models side by side.
    
    Args:
        models_results: Dictionary with model names as keys and metrics as values
        save_path: Path to save the comparison plot
    """
    logger.info("Comparing models...")
    
    # Create DataFrame for comparison
    df = pd.DataFrame(models_results).T
    
    # Plot comparison
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    metrics = ['accuracy', 'precision', 'recall', 'f1_score']
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx // 2, idx % 2]
        if metric in df.columns:
            df[metric].plot(kind='bar', ax=ax, color='skyblue', edgecolor='black')
            ax.set_title(f'{metric.replace("_", " ").title()} Comparison')
            ax.set_ylabel('Score')
            ax.set_ylim([0, 1])
            ax.grid(axis='y', alpha=0.3)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Model comparison saved to {save_path}")
    
    plt.close()
    
    # Print comparison table
    print("\nModel Comparison:")
    print("=" * 80)
    print(df.to_string())
    print("=" * 80)
    
    return df


if __name__ == "__main__":
    # Example usage
    from data_loader import load_dataset
    from preprocessing import DataPreprocessor
    from models import train_random_forest
    
    # Load and preprocess data
    df = load_dataset("data/raw/cicddos2019_dataset.csv", sample_size=5000)
    preprocessor = DataPreprocessor()
    X_train, _, X_test, y_train, _, y_test = preprocessor.prepare_data(df)
    
    # Train model
    model = train_random_forest(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    
    evaluator = ModelEvaluator("Random Forest")
    metrics = evaluator.calculate_metrics(y_test, y_pred, y_pred_proba)
    evaluator.plot_confusion_matrix(y_test, y_pred, save_path="models/results/confusion_matrix.png")
    evaluator.generate_classification_report(y_test, y_pred, save_path="models/results/classification_report.txt")
