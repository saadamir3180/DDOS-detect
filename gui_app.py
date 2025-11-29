"""
GUI Application for DDoS Detection System
A user-friendly interface for detecting DDoS attacks using trained ML models.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
from pathlib import Path
import pandas as pd
import threading
import sys
import os

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.predict import DDoSPredictor
from src.data_loader import load_dataset
from src.preprocessing import DataPreprocessor
from src.models import train_random_forest, train_svm, train_neural_network
from src.evaluation import ModelEvaluator
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DDoSDetectionGUI:
    """Main GUI application for DDoS detection."""
    
    def __init__(self, root):
        self.root = root
        self.root.title("DDoS Attack Detection System")
        self.root.geometry("900x700")
        self.root.resizable(True, True)
        
        # Variables
        self.dataset_path = tk.StringVar()
        self.model_path = tk.StringVar()
        self.predictor = None
        
        # Setup UI
        self.setup_ui()
        
    def setup_ui(self):
        """Setup the user interface."""
        # Create notebook (tabs)
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Create tabs
        self.create_training_tab()
        self.create_prediction_tab()
        self.create_batch_prediction_tab()
        self.create_about_tab()
        
    def create_training_tab(self):
        """Create the model training tab."""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="Train Model")
        
        # Title
        title = ttk.Label(tab, text="Train DDoS Detection Model", 
                         font=('Arial', 16, 'bold'))
        title.pack(pady=10)
        
        # Dataset selection
        dataset_frame = ttk.LabelFrame(tab, text="Dataset", padding=10)
        dataset_frame.pack(fill='x', padx=20, pady=10)
        
        ttk.Label(dataset_frame, text="Dataset Path:").grid(row=0, column=0, sticky='w', pady=5)
        ttk.Entry(dataset_frame, textvariable=self.dataset_path, width=50).grid(row=0, column=1, padx=5)
        ttk.Button(dataset_frame, text="Browse", command=self.browse_dataset).grid(row=0, column=2)
        
        # Sample size
        ttk.Label(dataset_frame, text="Sample Size (optional):").grid(row=1, column=0, sticky='w', pady=5)
        self.sample_size_var = tk.StringVar(value="10000")
        ttk.Entry(dataset_frame, textvariable=self.sample_size_var, width=20).grid(row=1, column=1, sticky='w', padx=5)
        
        # Model selection
        model_frame = ttk.LabelFrame(tab, text="Model Configuration", padding=10)
        model_frame.pack(fill='x', padx=20, pady=10)
        
        ttk.Label(model_frame, text="Select Model:").grid(row=0, column=0, sticky='w', pady=5)
        self.model_type_var = tk.StringVar(value="random_forest")
        models = [("Random Forest", "random_forest"), 
                 ("SVM (SGD)", "svm"), 
                 ("Neural Network", "neural_network")]
        
        for idx, (text, value) in enumerate(models):
            ttk.Radiobutton(model_frame, text=text, variable=self.model_type_var, 
                           value=value).grid(row=0, column=idx+1, padx=10)
        
        # Train button
        ttk.Button(tab, text="Start Training", command=self.train_model, 
                  style='Accent.TButton').pack(pady=20)
        
        # Progress and log
        log_frame = ttk.LabelFrame(tab, text="Training Log", padding=10)
        log_frame.pack(fill='both', expand=True, padx=20, pady=10)
        
        self.training_log = scrolledtext.ScrolledText(log_frame, height=15, wrap=tk.WORD)
        self.training_log.pack(fill='both', expand=True)
        
    def create_prediction_tab(self):
        """Create the single prediction tab."""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="Single Prediction")
        
        # Title
        title = ttk.Label(tab, text="Predict DDoS Attack", 
                         font=('Arial', 16, 'bold'))
        title.pack(pady=10)
        
        # Model selection
        model_frame = ttk.LabelFrame(tab, text="Load Model", padding=10)
        model_frame.pack(fill='x', padx=20, pady=10)
        
        ttk.Label(model_frame, text="Model Path:").grid(row=0, column=0, sticky='w', pady=5)
        ttk.Entry(model_frame, textvariable=self.model_path, width=50).grid(row=0, column=1, padx=5)
        ttk.Button(model_frame, text="Browse", command=self.browse_model).grid(row=0, column=2)
        ttk.Button(model_frame, text="Load Model", command=self.load_model).grid(row=1, column=1, pady=10)
        
        # Input file
        input_frame = ttk.LabelFrame(tab, text="Input Traffic Data", padding=10)
        input_frame.pack(fill='x', padx=20, pady=10)
        
        self.input_file_var = tk.StringVar()
        ttk.Label(input_frame, text="CSV File:").grid(row=0, column=0, sticky='w', pady=5)
        ttk.Entry(input_frame, textvariable=self.input_file_var, width=50).grid(row=0, column=1, padx=5)
        ttk.Button(input_frame, text="Browse", command=self.browse_input_file).grid(row=0, column=2)
        
        # Predict button
        ttk.Button(tab, text="Predict", command=self.make_prediction, 
                  style='Accent.TButton').pack(pady=20)
        
        # Results
        result_frame = ttk.LabelFrame(tab, text="Prediction Results", padding=10)
        result_frame.pack(fill='both', expand=True, padx=20, pady=10)
        
        self.prediction_result = scrolledtext.ScrolledText(result_frame, height=15, wrap=tk.WORD)
        self.prediction_result.pack(fill='both', expand=True)
        
    def create_batch_prediction_tab(self):
        """Create the batch prediction tab."""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="Batch Prediction")
        
        # Title
        title = ttk.Label(tab, text="Batch DDoS Detection", 
                         font=('Arial', 16, 'bold'))
        title.pack(pady=10)
        
        # Instructions
        ttk.Label(tab, text="Load a CSV file with network traffic data to detect DDoS attacks in bulk.",
                 wraplength=600).pack(pady=5)
        
        # File selection
        file_frame = ttk.LabelFrame(tab, text="Files", padding=10)
        file_frame.pack(fill='x', padx=20, pady=10)
        
        self.batch_input_var = tk.StringVar()
        self.batch_output_var = tk.StringVar()
        
        ttk.Label(file_frame, text="Input CSV:").grid(row=0, column=0, sticky='w', pady=5)
        ttk.Entry(file_frame, textvariable=self.batch_input_var, width=45).grid(row=0, column=1, padx=5)
        ttk.Button(file_frame, text="Browse", command=self.browse_batch_input).grid(row=0, column=2)
        
        ttk.Label(file_frame, text="Output CSV:").grid(row=1, column=0, sticky='w', pady=5)
        ttk.Entry(file_frame, textvariable=self.batch_output_var, width=45).grid(row=1, column=1, padx=5)
        ttk.Button(file_frame, text="Browse", command=self.browse_batch_output).grid(row=1, column=2)
        
        # Process button
        ttk.Button(tab, text="Process Batch", command=self.batch_predict, 
                  style='Accent.TButton').pack(pady=20)
        
        # Results
        result_frame = ttk.LabelFrame(tab, text="Batch Results", padding=10)
        result_frame.pack(fill='both', expand=True, padx=20, pady=10)
        
        self.batch_result = scrolledtext.ScrolledText(result_frame, height=15, wrap=tk.WORD)
        self.batch_result.pack(fill='both', expand=True)
        
    def create_about_tab(self):
        """Create the about tab."""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="About")
        
        about_text = """
        DDoS Attack Detection System
        ═══════════════════════════════════════════════════════
        
        A machine learning-based system for detecting and classifying
        Distributed Denial of Service (DDoS) attacks.
        
        Team Members:
        • Saad Amir (22L-7978)
        • Asad Shafiq (22L-7932)
        • Usman Khurshid (22L-7877)
        
        Features:
        • Multiple ML models (Random Forest, SVM, Neural Network)
        • Real-time DDoS detection
        • Batch processing capabilities
        • Comprehensive performance metrics
        
        Dataset:
        CICDDoS2019 from Mendeley Data
        
        How to Use:
        1. Train Model: Load dataset and train a model
        2. Single Prediction: Load model and predict on single samples
        3. Batch Prediction: Process multiple samples at once
        
        For more information, see README.md
        
        Version: 1.0.0
        License: CC BY 4.0
        """
        
        text_widget = tk.Text(tab, wrap=tk.WORD, font=('Arial', 10))
        text_widget.pack(fill='both', expand=True, padx=20, pady=20)
        text_widget.insert('1.0', about_text)
        text_widget.config(state='disabled')
    
    # Callback functions
    def browse_dataset(self):
        """Browse for dataset file."""
        filename = filedialog.askopenfilename(
            title="Select Dataset",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if filename:
            self.dataset_path.set(filename)
    
    def browse_model(self):
        """Browse for model file."""
        filename = filedialog.askopenfilename(
            title="Select Model",
            filetypes=[("Pickle files", "*.pkl"), ("All files", "*.*")]
        )
        if filename:
            self.model_path.set(filename)
    
    def browse_input_file(self):
        """Browse for input file."""
        filename = filedialog.askopenfilename(
            title="Select Input File",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if filename:
            self.input_file_var.set(filename)
    
    def browse_batch_input(self):
        """Browse for batch input file."""
        filename = filedialog.askopenfilename(
            title="Select Input CSV",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if filename:
            self.batch_input_var.set(filename)
    
    def browse_batch_output(self):
        """Browse for batch output file."""
        filename = filedialog.asksaveasfilename(
            title="Save Output As",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if filename:
            self.batch_output_var.set(filename)
    
    def log_message(self, message):
        """Add message to training log."""
        self.training_log.insert(tk.END, message + "\n")
        self.training_log.see(tk.END)
        self.root.update()
    
    def train_model(self):
        """Train the selected model."""
        if not self.dataset_path.get():
            messagebox.showerror("Error", "Please select a dataset file")
            return
        
        # Run training in a separate thread
        thread = threading.Thread(target=self._train_model_thread)
        thread.daemon = True
        thread.start()
    
    def _train_model_thread(self):
        """Training thread."""
        try:
            self.log_message("="*60)
            self.log_message("Starting model training...")
            self.log_message(f"Dataset: {self.dataset_path.get()}")
            self.log_message(f"Model: {self.model_type_var.get()}")
            
            # Load dataset
            sample_size = self.sample_size_var.get()
            sample_size = int(sample_size) if sample_size else None
            
            self.log_message(f"\nLoading dataset (sample size: {sample_size})...")
            df = load_dataset(self.dataset_path.get(), sample_size=sample_size)
            self.log_message(f"Loaded {len(df)} samples")
            
            # Preprocess
            self.log_message("\nPreprocessing data...")
            preprocessor = DataPreprocessor()
            X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.prepare_data(df)
            preprocessor.save_preprocessor()
            self.log_message("Preprocessing complete!")
            
            # Train model
            self.log_message(f"\nTraining {self.model_type_var.get()} model...")
            model_type = self.model_type_var.get()
            
            if model_type == 'random_forest':
                from src.models import train_random_forest
                model = train_random_forest(X_train, y_train)
            elif model_type == 'svm':
                from src.models import train_svm
                model = train_svm(X_train, y_train)
            else:
                from src.models import train_neural_network
                model = train_neural_network(X_train, y_train)
            
            # Save model
            import joblib
            model_save_path = f"models/saved_models/{model_type}.pkl"
            Path(model_save_path).parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(model, model_save_path)
            self.log_message(f"\nModel saved to: {model_save_path}")
            
            # Evaluate
            self.log_message("\nEvaluating model...")
            y_pred = model.predict(X_test)
            
            from src.evaluation import ModelEvaluator
            evaluator = ModelEvaluator(model_type)
            metrics = evaluator.calculate_metrics(y_test, y_pred)
            
            self.log_message("\n" + "="*60)
            self.log_message("TRAINING COMPLETE!")
            self.log_message("="*60)
            self.log_message(f"Accuracy:  {metrics['accuracy']:.4f}")
            self.log_message(f"Precision: {metrics['precision']:.4f}")
            self.log_message(f"Recall:    {metrics['recall']:.4f}")
            self.log_message(f"F1-Score:  {metrics['f1_score']:.4f}")
            self.log_message("="*60)
            
            messagebox.showinfo("Success", "Model training completed successfully!")
            
        except Exception as e:
            self.log_message(f"\nERROR: {str(e)}")
            messagebox.showerror("Error", f"Training failed: {str(e)}")
    
    def load_model(self):
        """Load a trained model."""
        if not self.model_path.get():
            messagebox.showerror("Error", "Please select a model file")
            return
        
        try:
            self.predictor = DDoSPredictor(self.model_path.get())
            messagebox.showinfo("Success", "Model loaded successfully!")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model: {str(e)}")
    
    def make_prediction(self):
        """Make a prediction on input file."""
        if not self.predictor:
            messagebox.showerror("Error", "Please load a model first")
            return
        
        if not self.input_file_var.get():
            messagebox.showerror("Error", "Please select an input file")
            return
        
        try:
            # Load input
            df = pd.read_csv(self.input_file_var.get())
            
            # Make prediction
            predictions = self.predictor.predict(df)
            
            # Display results
            self.prediction_result.delete('1.0', tk.END)
            self.prediction_result.insert(tk.END, "Prediction Results\n")
            self.prediction_result.insert(tk.END, "="*60 + "\n\n")
            
            # Show first 10 predictions
            for i, pred in enumerate(predictions[:10]):
                self.prediction_result.insert(tk.END, f"Sample {i+1}: {pred}\n")
            
            if len(predictions) > 10:
                self.prediction_result.insert(tk.END, f"\n... and {len(predictions)-10} more\n")
            
            # Summary
            unique, counts = pd.Series(predictions).value_counts().items(), pd.Series(predictions).value_counts().values
            self.prediction_result.insert(tk.END, "\n" + "="*60 + "\n")
            self.prediction_result.insert(tk.END, "Summary:\n")
            for label, count in zip(*[list(x) for x in [unique, counts]]):
                self.prediction_result.insert(tk.END, f"{label}: {count}\n")
            
        except Exception as e:
            messagebox.showerror("Error", f"Prediction failed: {str(e)}")
    
    def batch_predict(self):
        """Perform batch prediction."""
        if not self.predictor:
            messagebox.showerror("Error", "Please load a model first (in Single Prediction tab)")
            return
        
        if not self.batch_input_var.get():
            messagebox.showerror("Error", "Please select an input file")
            return
        
        if not self.batch_output_var.get():
            messagebox.showerror("Error", "Please select an output file")
            return
        
        try:
            self.batch_result.delete('1.0', tk.END)
            self.batch_result.insert(tk.END, "Processing batch predictions...\n")
            
            # Make predictions
            results = self.predictor.batch_predict(
                self.batch_input_var.get(),
                self.batch_output_var.get()
            )
            
            # Display results
            self.batch_result.insert(tk.END, f"\nProcessed {len(results)} samples\n")
            self.batch_result.insert(tk.END, f"Results saved to: {self.batch_output_var.get()}\n\n")
            
            # Summary
            self.batch_result.insert(tk.END, "="*60 + "\n")
            self.batch_result.insert(tk.END, "Summary:\n")
            summary = results['Predicted_Label'].value_counts()
            for label, count in summary.items():
                self.batch_result.insert(tk.END, f"{label}: {count}\n")
            
            messagebox.showinfo("Success", "Batch prediction completed!")
            
        except Exception as e:
            messagebox.showerror("Error", f"Batch prediction failed: {str(e)}")


def main():
    """Main entry point."""
    root = tk.Tk()
    
    # Set style
    style = ttk.Style()
    style.theme_use('clam')
    
    app = DDoSDetectionGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
