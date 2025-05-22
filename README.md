# Network Intrusion Detection System (NIDS) using Machine Learning

This project implements a **Network Intrusion Detection System (NIDS)** that leverages multiple machine learning algorithms to detect anomalies and attacks in network traffic. By combining traditional classifiers and deep learning methods in an ensemble framework, the system enhances the accuracy of detecting intrusions. It also features a real-time web-based dashboard to monitor predictions dynamically.

---

## Overview

Modern networks are vulnerable to a wide range of cyber threats. Traditional systems like firewalls or signature-based intrusion detectors fail to catch unknown or zero-day attacks. This project addresses the issue by building a **machine learning-based NIDS** that learns to distinguish between benign and malicious traffic using historical data.

Key motivations include:

* Automating intrusion detection through data-driven models.
* Increasing accuracy using ensemble techniques.
* Providing real-time monitoring to network administrators.

The approach combines several base models under a **stacking meta-model** and is trained on labeled network traffic data. The final solution offers both offline model evaluation and online deployment with prediction visualization.

---

## Key Features

* **Ensemble Learning**: Combines XGBoost, LightGBM, CatBoost, SVM, MLP, ANN, and GRU under a stacking classifier.
* **Data Preprocessing Pipeline**: Includes outlier removal (IQR), label encoding, scaling, and class balancing (SMOTE).
* **Real-time Detection**: Integrated Flask application generates predictions on live data.
* **Visualization Dashboard**: HTML + Plotly-based interface to visualize predicted class labels over time.
* **Extensive Evaluation**: ROC curves, classification reports, confusion matrices for performance insights.

---

## Dataset

![Dataset](network_intrusion.csv)

The model is trained on a labeled dataset with 41+ network traffic features and a target class:

* **Features**: Include `duration`, `protocol_type`, `service`, `src_bytes`, `dst_bytes`, `error rates`, etc.
* **Labels**: Binary classification with 0 for normal traffic and 1 for anomalies or attacks.

Data handling steps include:

* Filling missing values with column means.
* Removing outliers using Interquartile Range (IQR) filtering.
* Encoding categorical variables with LabelEncoder.
* Feature scaling using StandardScaler.
* Applying SMOTE to balance class distributions.

---

## System Architecture

### High-Level Components

1. **Data Source**: Captured from a real or simulated network environment.
2. **Preprocessing**: Cleans, transforms, and balances the dataset.
3. **Base Models**: Multiple ML models trained independently.
4. **Stacking Ensemble**: Combines base models into a meta-classifier.
5. **Monitoring Interface**: Flask server with a real-time visualization interface.

---

## Algorithms Used

### 1. **XGBoost**

* Efficient gradient boosting tree algorithm
* Parameters: `learning_rate`, `max_depth`, `n_estimators`

### 2. **LightGBM**

* Leaf-wise boosting for speed and memory efficiency
* Handles large datasets effectively

### 3. **CatBoost**

* Efficient categorical feature handling
* No need for manual encoding

### 4. **SVM**

* Classifies data by finding optimal separating hyperplanes
* Effective in high-dimensional spaces

### 5. **MLP (Multi-Layer Perceptron)**

* Fully connected neural network
* Useful for structured/tabular data

### 6. **ANN (Artificial Neural Network)**

* Deep learning model using dense layers
* Designed with 64-32-1 neuron architecture

### 7. **GRU (Gated Recurrent Unit)**

* Recurrent neural network for sequential patterns
* Learns time-series dependencies in traffic

### 8. **Stacking Classifier**

* Combines all the above base models
* Meta-model: XGBoost classifier

---

## Results

### Individual Model Accuracy

| Model    | Accuracy |
| -------- | -------- |
| XGBoost  | 82.53%   |
| LightGBM | 83.80%   |
| CatBoost | 83.86%   |
| SVM      | 77.59%   |
| MLP      | 71.27%   |
| ANN      | 67.35%   |
| GRU      | 78.07%   |

### Ensemble Model

* **Stacking Accuracy**: 83.92%
* **Cross-Validated Accuracy**: 84.43%

---

## Performance Metrics

**Classification Report**:

* **Precision** (Attack): 0.84
* **Recall** (Attack): 1.00
* **F1-Score** (Attack): 0.91

**Observations**:

* High **recall** means very few attacks are missed.
* High **false positives** indicate some normal traffic is misclassified.

---

## Installation & Usage

### Prerequisites

* Python 3.x
* Flask
* Scikit-learn, XGBoost, LightGBM, CatBoost, Keras, etc.

### Run ML Pipeline

```bash
python main.py
```

### Launch Real-Time Web Interface

```bash
python app.py
```

Visit: [http://localhost:5000](http://localhost:5000)

---

## Web Interface

* Developed using **Flask**, **HTML**, **JavaScript**, and **Plotly**
* Shows real-time predictions from the trained stacking model
* Features a dynamic chart with time vs prediction (normal vs attack)
* Endpoint `/predict` returns predictions on test samples generated on-the-fly

---
