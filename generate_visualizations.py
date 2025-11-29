"""
Generate visualizations and performance reports for all trained models.
Run this after training your models.
"""

import sys
sys.path.insert(0, 'src')

from src.data_loader import load_dataset
from src.preprocessing import DataPreprocessor
from src.evaluation import ModelEvaluator, compare_models
import joblib
import pandas as pd

print("="*60)
print("Generating Performance Visualizations")
print("="*60)

# Load test data
print("\n1. Loading dataset...")
df = load_dataset("data/raw/cicddos2019_dataset.csv", sample_size=50000)  # Use subset for speed

print("2. Preprocessing...")
preprocessor = DataPreprocessor()
preprocessor.load_preprocessor("models/saved_models")
X_train, _, X_test, y_train, _, y_test = preprocessor.prepare_data(df)

# Load all models
print("\n3. Loading trained models...")
models = {
    'Random Forest': joblib.load('models/saved_models/random_forest.pkl'),
    'SVM': joblib.load('models/saved_models/svm.pkl'),
    'Neural Network': joblib.load('models/saved_models/neural_network.pkl')
}

# Evaluate each model
print("\n4. Evaluating models and generating visualizations...")
results = {}

for name, model in models.items():
    print(f"\n   Evaluating {name}...")
    
    # Predictions
    y_pred = model.predict(X_test)
    
    # Create evaluator
    evaluator = ModelEvaluator(name)
    
    # Calculate metrics
    try:
        y_pred_proba = model.predict_proba(X_test)
    except:
        y_pred_proba = None
    
    metrics = evaluator.calculate_metrics(y_test, y_pred, y_pred_proba)
    results[name] = metrics
    
    # Generate confusion matrix
    evaluator.plot_confusion_matrix(
        y_test, y_pred,
        labels=preprocessor.label_encoder.classes_,
        save_path=f'models/results/{name.replace(" ", "_")}_confusion_matrix.png'
    )
    
    # Generate classification report
    evaluator.generate_classification_report(
        y_test, y_pred,
        labels=preprocessor.label_encoder.classes_,
        save_path=f'models/results/{name.replace(" ", "_")}_classification_report.txt'
    )
    
    # Save metrics
    evaluator.save_metrics(f'models/results/{name.replace(" ", "_")}_metrics.json')

# Compare all models
print("\n5. Creating model comparison chart...")
comparison_df = compare_models(results, save_path='models/results/model_comparison.png')

# Save comparison to CSV
comparison_df.to_csv('models/results/model_comparison.csv')

print("\n" + "="*60)
print("✅ All visualizations generated successfully!")
print("="*60)
print("\nGenerated files:")
print("  - models/results/*_confusion_matrix.png")
print("  - models/results/*_classification_report.txt")
print("  - models/results/*_metrics.json")
print("  - models/results/model_comparison.png")
print("  - models/results/model_comparison.csv")
print("\nCheck the models/results/ folder for all outputs!")
