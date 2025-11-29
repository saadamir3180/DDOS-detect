# Quick Start Guide - DDoS Detection System

## 🚀 Getting Started in 5 Minutes

### Step 1: Download the Dataset

1. Go to the [CICDDoS2019 Dataset on Mendeley](https://data.mendeley.com/datasets/jxpfjc64kr/1)
2. Click "Download All" (147 MB)
3. Extract the CSV file to `data/raw/` folder
4. Rename it to `cicddos2019_dataset.csv` (or update the path in code)

### Step 2: Install Additional Dependencies (Optional)

```bash
py -3.12 -m pip install jupyter notebook ipykernel tqdm
```

### Step 3: Choose Your Workflow

#### Option A: GUI Application (Easiest) ⭐

```bash
py -3.12 gui_app.py
```

The GUI has 3 main tabs:
1. **Train Model** - Load dataset and train a model
2. **Single Prediction** - Load model and predict on samples
3. **Batch Prediction** - Process multiple samples at once

#### Option B: Jupyter Notebooks (For Experimentation)

```bash
jupyter notebook
```

Then open:
1. `notebooks/01_eda.ipynb` - Explore the dataset
2. `notebooks/02_preprocessing.ipynb` - Preprocess data (create this)
3. `notebooks/03_model_training.ipynb` - Train models (create this)

#### Option C: Python Scripts (For Automation)

```python
# Train a model
from src.data_loader import load_dataset
from src.preprocessing import DataPreprocessor
from src.models import train_random_forest
import joblib

# Load data
df = load_dataset("data/raw/cicddos2019_dataset.csv", sample_size=10000)

# Preprocess
preprocessor = DataPreprocessor()
X_train, _, X_test, y_train, _, y_test = preprocessor.prepare_data(df)
preprocessor.save_preprocessor()

# Train
model = train_random_forest(X_train, y_train)
joblib.dump(model, "models/saved_models/random_forest.pkl")

# Evaluate
from src.evaluation import ModelEvaluator
y_pred = model.predict(X_test)
evaluator = ModelEvaluator("Random Forest")
metrics = evaluator.calculate_metrics(y_test, y_pred)
print(metrics)
```

## 📊 Recommended Workflow for Assignment

### Day 1-2: Data Exploration
1. Download dataset
2. Run `notebooks/01_eda.ipynb`
3. Understand class distribution and features

### Day 3-4: Preprocessing & Feature Engineering
1. Create preprocessing pipeline
2. Perform feature selection
3. Save preprocessed data

### Day 5-6: Model Training
1. Train Random Forest (fastest, good baseline)
2. Train SVM (good performance)
3. Train Neural Network (best accuracy potential)
4. Compare results

### Day 7: Evaluation
1. Calculate metrics for all models
2. Generate confusion matrices
3. Create comparison visualizations
4. Identify best model

### Day 8: Prototype
1. Test GUI application
2. Verify batch prediction works
3. Create sample demonstrations

### Day 9: Documentation
1. Fill in `report.md` with your results
2. Document preprocessing steps
3. Add screenshots to README

### Day 10: Presentation
1. Prepare slides
2. Demo the GUI
3. Show performance metrics

## 🎯 Tips for Success

### Start Small
- Use `sample_size=10000` initially to test your pipeline
- Once everything works, train on full dataset

### Save Everything
- Models are saved to `models/saved_models/`
- Results are saved to `models/results/`
- Always save preprocessors with models

### Use the GUI
- The GUI makes it easy to experiment without coding
- Great for demonstrations and presentations

### Monitor Training
- Random Forest: ~2-5 minutes on 10k samples
- SVM: ~1-3 minutes on 10k samples
- Neural Network: ~5-10 minutes on 10k samples

### Expected Performance
- Accuracy: 95%+ is achievable
- F1-Score: 90%+ is good
- Random Forest usually performs best

## 🐛 Troubleshooting

### "Dataset not found"
- Make sure the CSV is in `data/raw/`
- Check the filename matches in your code

### "Out of memory"
- Reduce `sample_size` parameter
- Use chunked loading for large datasets

### "Model takes too long"
- Start with Random Forest (fastest)
- Use smaller sample for testing
- Reduce hyperparameters (e.g., n_estimators=50)

### GUI doesn't open
- Make sure you're using Python 3.12: `py -3.12 gui_app.py`
- Check tkinter is installed (usually comes with Python)

## 📝 What to Submit

1. **Source Code** - All files in `src/` and root directory
2. **Trained Models** - Best performing model (`.pkl` file)
3. **Report** - Completed `report.md` with results
4. **Presentation** - Slides summarizing findings
5. **Screenshots** - GUI in action, confusion matrices, etc.

## 🎓 Grading Criteria Checklist

- [ ] Multiple models trained (RF, SVM, NN)
- [ ] Performance metrics calculated (accuracy, precision, recall, F1)
- [ ] Preprocessing documented
- [ ] Feature importance analysis
- [ ] Working prototype (GUI or CLI)
- [ ] Comprehensive report
- [ ] Presentation prepared

## 💡 Extra Credit Ideas

1. **Hyperparameter Tuning** - Use GridSearchCV for optimal parameters
2. **Deep Learning** - Try a deeper neural network with Keras
3. **Real-time Detection** - Add live traffic monitoring
4. **Visualization Dashboard** - Create interactive plots with Plotly
5. **Model Ensemble** - Combine multiple models for better accuracy

## 📚 Useful Resources

- [Scikit-learn Documentation](https://scikit-learn.org/)
- [Pandas Documentation](https://pandas.pydata.org/)
- [CICDDoS2019 Paper](https://www.unb.ca/cic/datasets/ddos-2019.html)

---

**Good luck with your project! 🚀**

For questions or issues, check the README.md or review the implementation plan.
