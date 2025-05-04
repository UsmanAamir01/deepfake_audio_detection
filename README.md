# Urdu Deepfake Audio Detection

[![Streamlit](https://img.shields.io/badge/Streamlit-App-orange)](#streamlit-demo)  
[![Python](https://img.shields.io/badge/python-3.8%2B-blue)](#requirements)  
[![License](https://img.shields.io/badge/license-MIT-green)](#license)

## 📌 Project Overview

This repository contains the implementation for **Urdu Deepfake Audio Detection**. Using audio features extracted from the CSALT Deepfake Detection Dataset (Urdu), we train and evaluate several machine learning models:

- Logistic Regression  
- Support Vector Machine (SVM)  
- Perceptron  
- Deep Neural Network (DNN)

A Streamlit app is included for interactive classification of new audio samples.

---

## 📁 Repository Structure

```
urdu-deepfake-detection/
├── data/                     # Raw and preprocessed audio data
├── models/                   # Trained models and scalers
│   ├── dnn_deepfake.pt
│   ├── perceptron_deepfake_scaler.pkl
│   └── svm_deepfake_scaler.pkl
├── notebooks/                # Jupyter notebooks for experiments
│   ├── deepfake_dnn.ipynb
│   ├── deepfake_svm.ipynb
│   └── defect_dnn_updated.ipynb
├── results/                  # Evaluation logs and metrics
│   ├── deepfake_evaluation.txt
│   └── defect_evaluation.txt
├── visualizations/           # Generated plots and PNGs
├── streamlit_app/            # Streamlit app for live demo
│   ├── app.py
│   └── utils/                # Helper scripts
├── model_comparison.csv      # Comparison of model metrics
├── model_rankings.csv        # Ranked model performance
├── report.md                 # Project report and insights
├── requirements.txt          # Python dependencies
└── README.md                 # You are here
```

---

## 🚀 Installation

1. **Clone the Repository**
   ```bash
   git clone https://github.com/UsmanAamir01/deepfake_audio_detection.git
   ```

2. **Create and Activate a Virtual Environment**
   ```bash
   python3 -m venv venv
   source venv/bin/activate       # On Linux/Mac
   venv\Scripts\activate          # On Windows
   ```

3. **Install Dependencies**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

---

## ⚙️ Usage

### 1. Run Exploratory Notebooks

Explore training and evaluation steps:
```bash
jupyter notebook notebooks/deepfake_dnn.ipynb
```

### 2. View Results

- CSV summaries: `model_comparison.csv`, `model_rankings.csv`
- Raw logs: `results/deepfake_evaluation.txt`, `results/defect_evaluation.txt`
- Visuals: Check the `visualizations/` directory

### 3. Launch Streamlit Demo

Classify audio samples interactively:
```bash
cd streamlit_app
streamlit run app.py
```

---

## 📄 Report

See `report.md` for detailed discussions on the dataset, features, model design, evaluation methodology, and findings.

---

## 📦 Requirements

Key packages (see `requirements.txt` for full list):

- `streamlit >= 1.22.0`
- `numpy`, `pandas`, `matplotlib`, `seaborn`
- `scikit-learn`, `torch`
- `librosa`, `soundfile`

---

## 🤝 Contributing

We welcome contributions!

1. Fork the repository  
2. Create your feature branch:  
   ```bash
   git checkout -b feature/your-feature
   ```
3. Commit your changes:  
   ```bash
   git commit -m "Add your feature"
   ```
4. Push to GitHub:  
   ```bash
   git push origin feature/your-feature
   ```
5. Open a Pull Request

For significant changes or bug reports, please open an issue first.

---

## 📜 License

This project is licensed under the [MIT License](LICENSE).

---
