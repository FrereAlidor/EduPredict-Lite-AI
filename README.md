
# AI-Powered Early Warning System for Resource-Constrained Schools

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-orange.svg)](https://scikit-learn.org/)

> Lightweight machine learning framework for predicting student performance in secondary schools with limited resources.

![Framework Architecture](archiIEEE_Lyon.png)

---

## 🎯 Overview

An offline-capable, CPU-only AI system designed for resource-constrained secondary schools to identify at-risk students early and enable timely interventions.

### Key Features

- ✅ **CPU-Only** - No GPU required
- ✅ **Fast Training** - Under 2 seconds
- ✅ **Small Dataset** - Works with 300-400 students
- ✅ **Offline Ready** - No internet needed
- ✅ **Low Memory** - <100 MB RAM
- ✅ **Easy to Use** - 2-4 hour teacher training

### Performance

| Model | Accuracy | Precision | Recall | F1-Score | Training Time |
|-------|----------|-----------|--------|----------|---------------|
| **Logistic Regression** | **68.35%** | **72.58%** | **84.91%** | **0.78** | **0.08s** |
| Random Forest | 65.82% | 69.70% | 86.79% | 0.77 | 0.42s |
| Naive Bayes | 65.82% | 70.31% | 84.91% | 0.77 | 0.02s |

**High recall (84.91%)** ensures most at-risk students are identified.

---

## 🚀 Quick Start

### Installation
```bash
git clone https://github.com/yourusername/student-performance-prediction.git
cd student-performance-prediction
pip install -r requirements.txt
```

### Basic Usage
```python
from src.model import StudentPerformancePredictor

# Initialize and train
predictor = StudentPerformancePredictor(model_type='logistic_regression')
predictor.train(X_train, y_train)

# Predict risk levels
risk_categories = predictor.predict_risk(X_test)
# High Risk: <30%, Medium: 30-60%, Low: >60%
```

### Run Full Pipeline
```bash
python main.py --dataset data/student-mat.csv --output results/
```

---

## 📊 Dataset

**UCI Student Performance Dataset** - Portuguese secondary schools
- 395 students
- 33 features (demographics, behavior, academic history)
- Binary target: Pass (≥10/20) or Fail (<10/20)

---

## 🧪 Methodology
```
Data Preparation → Statistical Validation → ML Training → Evaluation → Risk System
```

**5 Algorithms Evaluated:**
1. Logistic Regression (Best)
2. Decision Tree
3. Random Forest
4. Gradient Boosting
5. Naive Bayes

**Statistical Tests:** Normality, Chi-Square, Correlation, T-Tests

---

## 📈 Results

### Key Findings

✅ **68.35% accuracy** with offline data only  
✅ **Past failures** (r = -0.360) and **higher education aspiration** (χ² = 8.353) are strongest predictors  
✅ **27.8% of students** flagged for intervention  
✅ **84.91% recall** ensures early identification  

### Risk Distribution

- **High Risk (15.2%):** 12 students → Intensive intervention
- **Medium Risk (12.7%):** 10 students → Moderate support
- **Low Risk (72.2%):** 57 students → Standard monitoring

---

## 📁 Project Structure
```
├── data/                    # Dataset and templates
├── notebooks/              # Jupyter analysis notebook
├── src/                    # Source code
│   ├── preprocessing.py
│   ├── model.py
│   └── evaluation.py
├── results/                # Figures and reports
├── main.py                 # Main script
├── requirements.txt
└── README.md
```

---

## 🛠️ Dependencies
```txt
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
seaborn>=0.11.0
scipy>=1.7.0
```

---

## 🏫 For Schools

### Hardware Requirements
- Basic laptop (Intel Core i5, 8GB RAM)
- No GPU needed
- Offline operation

### Deployment Steps
1. Collect student data (use template)
2. Run training script
3. Generate risk predictions
4. Implement interventions

### Teacher Training (2-4 hours)
- Data entry basics
- Understanding risk categories
- Interpreting predictions
- Linking to interventions

---

## 📝 Citation
```bibtex
@inproceedings{yourname2025student,
  title={AI-Powered Early Warning Systems for Resource-Constrained Schools},
  author={Your Name},
  booktitle={IEEE Conference},
  year={2025}
}
```

---

## 🤝 Contributing

Contributions welcome! Please open issues or submit pull requests.

---

## 📧 Contact

**[Your Name]**  
Email: your.email@institution.edu  
Issues: [GitHub Issues](https://github.com/yourusername/student-performance-prediction/issues)

---

## 📄 License

MIT License - Free for educational and commercial use.

---

## 🌍 Impact

Designed for rural schools, developing countries, and resource-limited educational contexts to democratize access to AI-powered student support systems.

---

<div align="center">

**Made with ❤️ for educational equity**

⭐ Star this repo if it helps your research or school!

</div>
