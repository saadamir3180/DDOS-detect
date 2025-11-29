# DDoS Attack Detection and Classification Using Machine Learning

**Team Members:**
- Usman Khurshid (22L-7877)
- Saad Amir (22L-7978)
- Asad Shafiq (22L-7932)


## Project Overview

This project develops a machine learning-based DDoS detection system that automatically classifies network traffic into normal and attack categories using the CICDDoS2019 dataset.

## Features

- Multiple ML models: Random Forest, SVM, Neural Network
- Comprehensive data preprocessing and feature engineering
- GUI interface for easy DDoS detection
- Detailed performance metrics and visualizations
- Feature importance analysis

## Installation

### Prerequisites
- Python 3.12
- pip package manager

### Setup

1. Install dependencies:
```bash
py -3.12 -m pip install -r requirements.txt
```

2. Download the CICDDoS2019 dataset from [Mendeley Data](https://data.mendeley.com/datasets/jxpfjc64kr/1) and place it in `data/raw/`

## Project Structure

```
.
├── data/
│   ├── raw/              # Original dataset
│   ├── processed/        # Preprocessed data
│   └── samples/          # Sample data for testing
├── models/
│   ├── saved_models/     # Trained model files
│   └── results/          # Performance metrics and plots
├── notebooks/
│   ├── 01_eda.ipynb                    # Exploratory Data Analysis
│   ├── 02_preprocessing.ipynb          # Data preprocessing
│   └── 03_model_training.ipynb         # Model training and evaluation
├── src/
│   ├── data_loader.py                  # Dataset loading utilities
│   ├── preprocessing.py                # Preprocessing functions
│   ├── feature_engineering.py          # Feature selection
│   ├── models.py                       # Model implementations
│   ├── evaluation.py                   # Evaluation metrics
│   └── predict.py                      # Prediction interface
├── gui_app.py                          # GUI application
└── requirements.txt                     # Dependencies
```

## Usage

### GUI Application

Run the GUI for easy DDoS detection:
```bash
py -3.12 gui_app.py
```

### Jupyter Notebooks

For experimentation and analysis:
```bash
jupyter notebook
```

Then open notebooks in the `notebooks/` directory.

## Model Performance

Results will be documented in `models/results/` after training.

## License

CC BY 4.0 (following CICDDoS2019 dataset license)

## Dataset Citation

Tabuada, Md Alamir; Uddin, Md Ashraf (2023), "CIC-DDoS2019 Dataset", Mendeley Data, V1, doi: 10.17632/jxpfjc64kr.1
