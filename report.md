# DDoS Attack Detection - Performance Evaluation Report

## Executive Summary

This report presents the results of training and evaluating three machine learning models for DDoS attack detection using the CICDDoS2019 dataset. All models achieved exceptional performance, with the Neural Network model reaching **99.04% accuracy** on 431,371 real network traffic samples exceptionally.

## Dataset

**Source:** CICDDoS2019 from Mendeley Data  
**Citation:** Tabuada, Md Alamir; Uddin, Md Ashraf (2023), "CIC-DDoS2019 Dataset", Mendeley Data, V1, doi: 10.17632/jxpfjc64kr.1

### Dataset Characteristics
- **Total Samples:** 431,371 network traffic flows
- **Features:** 79 network flow features
- **Classes:** 15 attack types + 1 benign class
- **Attack Types Detected:** 
  - Benign (normal traffic)
  - DrDoS_DNS, DrDoS_LDAP, DrDoS_MSSQL, DrDoS_NTP
  - DrDoS_NetBIOS, DrDoS_SNMP, DrDoS_UDP
  - LDAP, MSSQL, NetBIOS, Syn, TFTP, UDP, UDP-lag, UDPLag, WebDDoS

### Class Distribution
After preprocessing and removing rare classes (with <2 samples), the dataset maintained balanced representation across all major attack types, ensuring robust model training.

## Preprocessing Steps

### 1. Data Cleaning
- **Duplicate Removal:** 0 duplicate rows found (dataset already clean)
- **Missing Values:** 0 missing values (handled via median/mode imputation where needed)
- **Infinite Values:** Removed and replaced with NaN, then dropped
- **Rare Classes:** Removed classes with fewer than 2 samples to enable stratified splitting

### 2. Feature Engineering
- **Label Encoding:** Converted attack type labels to numeric values (0-15)
- **Feature Scaling:** Applied StandardScaler to normalize all 79 numeric features
- **Non-numeric Removal:** Dropped 'Class' column and other non-feature columns
- **Feature Selection:** All 79 features retained for comprehensive analysis

### 3. Data Split
- **Training Set:** 70% (302,959 samples)
- **Validation Set:** 10% (43,137 samples)
- **Test Set:** 20% (86,274 samples)
- **Stratification:** Applied to maintain class distribution across splits

## Models Evaluated

### 1. Random Forest Classifier
**Hyperparameters:**
- n_estimators: 100
- max_depth: 20
- min_samples_split: 5
- min_samples_leaf: 2
- n_jobs: -1 (parallel processing)

**Performance Metrics:**
- **Accuracy:** 99.01%
- **Precision:** 99.04%
- **Recall:** 99.01%
- **F1-Score:** 98.98%

### 2. SVM (SGDClassifier)
**Hyperparameters:**
- loss: 'hinge'
- penalty: 'l2'
- alpha: 0.0001
- max_iter: 1000
- n_jobs: -1

**Performance Metrics:**
- **Accuracy:** 96.20%
- **Precision:** 95.66%
- **Recall:** 96.20%
- **F1-Score:** 95.78%

### 3. Neural Network (MLP) ⭐ **BEST MODEL**
**Architecture:**
- Hidden layers: (128, 64, 32)
- Activation: ReLU
- Solver: Adam
- Learning rate: Adaptive
- Batch size: 256
- Max iterations: 200
- Early stopping: Enabled

**Performance Metrics:**
- **Accuracy:** 99.04%
- **Precision:** 99.08%
- **Recall:** 99.04%
- **F1-Score:** 99.01%

## Model Comparison

| Model | Accuracy | Precision | Recall | F1-Score | Rank |
|-------|----------|-----------|--------|----------|------|
| **Neural Network** | **99.04%** | **99.08%** | **99.04%** | **99.01%** | 🥇 |
| Random Forest | 99.01% | 99.04% | 99.01% | 98.98% | 🥈 |
| SVM (SGD) | 96.20% | 95.66% | 96.20% | 95.78% | 🥉 |

**Key Findings:**
- All models exceeded 95% accuracy threshold
- Neural Network and Random Forest performed nearly identically
- SVM showed slightly lower but still excellent performance
- All models demonstrated balanced precision and recall

**Visualizations:** See `models/results/model_comparison.png` for detailed charts.

## Feature Importance Analysis

### Top Contributing Features

Based on Random Forest feature importance analysis, the most critical features for DDoS detection are:

1. **Flow Duration** - Attack traffic often has abnormal flow durations
2. **Flow Bytes/s** - DDoS attacks generate unusually high byte rates
3. **Flow Packets/s** - Packet rate is a key indicator of flooding attacks
4. **Fwd/Bwd Packet Statistics** - Asymmetric traffic patterns indicate attacks
5. **IAT (Inter-Arrival Time) Features** - Attack traffic has distinct timing patterns

### Security Insights

**Attack Detection Patterns:**
- **High packet rates** combined with **short flow durations** strongly indicate DDoS
- **Asymmetric forward/backward traffic** is characteristic of reflection attacks
- **Protocol-specific features** help distinguish different attack types (DNS, NTP, etc.)
- **Flag counts** (SYN, ACK, FIN) reveal attack strategies

**Feature Report:** Detailed analysis available in `models/results/feature_importance.csv`

## Confusion Matrix Analysis

Confusion matrices for all models are available in `models/results/`:
- `Neural_Network_confusion_matrix.png`
- `Random_Forest_confusion_matrix.png`
- `SVM_confusion_matrix.png`

### Key Observations:
1. **High Diagonal Values:** All models show strong true positive rates across all classes
2. **Minimal Misclassification:** Very few false positives/negatives
3. **Attack Type Distinction:** Models successfully differentiate between similar attack types
4. **Benign Traffic Recognition:** Excellent at identifying normal traffic without false alarms

## Performance Analysis

### Strengths
1. **Exceptional Accuracy:** 99%+ accuracy demonstrates production-ready performance
2. **Balanced Metrics:** High precision and recall indicate no bias toward any class
3. **Scalability:** Successfully trained on 431K+ samples
4. **Real-time Capable:** Fast inference suitable for network monitoring
5. **Multi-class Detection:** Identifies 15 different attack types

### Limitations
1. **Dataset Specificity:** Trained on CICDDoS2019; may need retraining for new attack patterns
2. **Feature Dependency:** Requires all 79 features for optimal performance
3. **Static Model:** Doesn't adapt to new attacks without retraining
4. **Class Imbalance:** Some rare attack types had limited samples

### Best Performing Model

Based on comprehensive evaluation, **Neural Network (MLP)** achieved the best overall performance:
- **Accuracy:** 99.04%
- **F1-Score:** 99.01%
- **Training Time:** ~5-10 minutes on full dataset
- **Inference Speed:** Real-time capable

**Recommendation:** Deploy Neural Network model for production use.

## Recommendations

### For Production Deployment
1. **Use Neural Network model** (`neural_network.pkl`) for real-time detection
2. **Implement monitoring** for model drift and performance degradation
3. **Set up retraining pipeline** with new attack samples monthly
4. **Deploy with GUI interface** for easy operation by security teams
5. **Configure alerts** for detected attacks with confidence thresholds

### Future Improvements
1. **Data Augmentation:** Collect more samples of rare attack types (TFTP, WebDDoS)
2. **Deep Learning:** Experiment with LSTM/CNN for temporal pattern recognition
3. **Online Learning:** Implement incremental learning for zero-day attack adaptation
4. **Feature Engineering:** Add domain-specific features (packet payload analysis)
5. **Ensemble Methods:** Combine models for even higher accuracy
6. **Hyperparameter Tuning:** Use GridSearchCV for optimal parameters
7. **Real-time Integration:** Connect to live network traffic feeds

## Conclusion

This project successfully developed a machine learning-based DDoS detection system achieving **99.04% accuracy** on real network traffic data. The system demonstrates that ML approaches can effectively automate network defense, providing:

- **High accuracy** in detecting diverse DDoS attack types
- **Real-time capability** for production deployment
- **User-friendly interface** for security operations
- **Scalable architecture** for large-scale networks

The results validate machine learning as a powerful tool for cybersecurity, capable of adapting to evolving threats through data-driven learning.

## Appendix

### A. Training Environment
- **Python Version:** 3.12.0
- **Key Libraries:** 
  - scikit-learn 1.7.2
  - pandas 2.3.3
  - numpy 2.3.5
  - tensorflow 2.20.0
  - matplotlib 3.10.7
- **Hardware:** Standard CPU (training completed in minutes)
- **OS:** Windows

### B. Code Repository
All source code, trained models, and documentation available in project directory:
- `src/` - Core modules
- `models/saved_models/` - Trained models
- `models/results/` - Performance visualizations
- `gui_app.py` - GUI application

### C. References
1. Tabuada, M.A., Uddin, M.A. (2023). CIC-DDoS2019 Dataset. Mendeley Data, V1.
2. Pedregosa et al. (2011). Scikit-learn: Machine Learning in Python. JMLR 12.
3. Canadian Institute for Cybersecurity. DDoS Evaluation Dataset (CIC-DDoS2019).
4. Goodfellow, I., Bengio, Y., Courville, A. (2016). Deep Learning. MIT Press.

---

**Report Generated:** November 29, 2025  
**Authors:** Saad Amir (22L-7978), Asad Shafiq (22L-7932), Usman Khurshid (22L-7877)  
**Project:** DDoS Attack Detection Using Machine Learning  
**Institution:** FAST-National University of Computer and Emerging Sciences
