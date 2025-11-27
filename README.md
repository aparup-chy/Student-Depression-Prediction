# 🧠 Student Depression Prediction Using Machine Learning

A comprehensive implementation of multiple machine learning algorithms for predicting depression among students through demographic, academic, and lifestyle factor analysis.

**Course:** CSE422 - Artificial Intelligence  
**Institution:** BRAC University  
**Semester:** Fall 2024  

---

## 👥 Team Members

- **Aparup Chowdhury**
- **Nabid Hasan Omi**

---

## 📋 Table of Contents
- [Project Overview](#-project-overview)
- [Key Objectives](#-key-objectives)
- [Dataset Description](#-dataset-description)
- [Methodology](#-methodology)
- [Models Implemented](#-models-implemented)
- [Results & Analysis](#-results--analysis)
- [Key Findings](#-key-findings)
- [Installation & Usage](#-installation--usage)
- [Future Work](#-future-work)
- [References](#-references)

---

## 🎯 Project Overview

This project implements and evaluates machine learning algorithms to predict depression among students based on a dataset of 27,901 individuals. The system analyzes 18 features encompassing demographic information, academic performance, lifestyle choices, and mental health indicators to enable early detection and intervention.

### Problem Statement

Mental health challenges significantly impact student performance and well-being. This project demonstrates how machine learning can transform mental health assessment from reactive to proactive approaches, enabling:

- Timely interventions by educational institutions
- Evidence-based resource allocation
- Data-driven policy making for student welfare
- Personalized support systems for at-risk individuals

---

## 🔑 Key Objectives

1. **Identify** significant predictors of depression through correlation analysis
2. **Develop** optimal preprocessing pipelines for mixed data types  
3. **Implement** multiple classification algorithms with systematic evaluation
4. **Compare** algorithmic approaches to determine optimal solutions
5. **Create** interpretable models suitable for practical deployment

---

## 📊 Dataset Description

### Specifications

- **Source**: [Kaggle Student Depression Dataset](https://www.kaggle.com/datasets/hopesb/student-depression-dataset)
- **Sample Size**: 27,901 student records
- **Features**: 18 (17 input + 1 target)
- **Target Distribution**:
  - Depression (Yes): 15,209 (58.6%)
  - No Depression: 10,739 (41.4%)

### Feature Categories

**Numerical Features (7)**
- Age, CGPA, Academic Pressure, Work Pressure
- Study Satisfaction, Work/Study Hours, Financial Stress

**Categorical Features (10)**
- Gender, City, Profession, Degree
- Sleep Duration, Dietary Habits
- Family Mental History, Suicidal Thoughts

### Key Correlations with Depression

| Feature | Correlation | Significance |
|---------|------------|--------------|
| **Suicidal Thoughts** | 0.53 | Strongest predictor |
| **Academic Pressure** | 0.46 | Major contributor |
| **Financial Stress** | 0.35 | Significant impact |
| Work Pressure | 0.31 | Notable influence |
| Sleep Duration | 0.18 | Moderate effect |

---

## 🔬 Methodology

### Data Preprocessing Pipeline

1. **Data Cleaning**
   - Missing value imputation (mean for numerical, mode for categorical)
   - Outlier detection and treatment using IQR method
   - Removal of records with >40% missing data

2. **Feature Engineering**
   - Label encoding for binary variables
   - One-hot encoding for multi-class categories
   - StandardScaler normalization for numerical features

3. **Data Splitting**
   - Stratified split maintaining class distribution
   - 70% training (19,531 samples)
   - 30% testing (8,370 samples)

---

## 🏗️ Models Implemented

### 1. Logistic Regression
- **Type**: Linear classifier with probabilistic predictions
- **Accuracy**: 80%
- **Strengths**: Simple, interpretable, robust to outliers
- **Best For**: Linear decision boundaries

### 2. K-Nearest Neighbors (KNN)
- **Type**: Instance-based, non-parametric classifier
- **Accuracy**: 78%
- **Strengths**: No training phase, captures local patterns
- **Limitations**: Sensitive to feature scaling

### 3. Decision Tree
- **Type**: Hierarchical, rule-based classifier
- **Accuracy**: 71%
- **Strengths**: Highly interpretable, handles non-linear patterns
- **Limitations**: Prone to overfitting

### 4. Naive Bayes
- **Type**: Probabilistic classifier based on Bayes' theorem
- **Accuracy**: 75%
- **Strengths**: Fast, works well with high-dimensional data
- **Limitations**: Assumes feature independence

---

## 📈 Results & Analysis

### Performance Summary

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|---------|----------|
| **Logistic Regression** ✅ | **80%** | **0.85** | **0.90** | **0.87** |
| K-Nearest Neighbors | 78% | 0.80 | 0.85 | 0.82 |
| Naive Bayes | 75% | 0.78 | 0.82 | 0.80 |
| Decision Tree | 71% | 0.75 | 0.80 | 0.77 |

### Confusion Matrix - Best Model (Logistic Regression)

```
                Predicted
              No Dep    Depression
Actual    ┌─────────┬────────────┐
No Dep    │  2,943  │    880     │
          ├─────────┼────────────┤
Depression│   743   │   3,805    │
          └─────────┴────────────┘

True Positive Rate (Sensitivity): 90%
True Negative Rate (Specificity): 77%
```

### Model Selection Rationale

Logistic Regression emerged as the optimal model due to:
- Highest overall accuracy (80%)
- Best recall (90%) - crucial for identifying at-risk students
- Good precision (85%) - minimizes false alarms
- Model interpretability for clinical insights
- Computational efficiency for real-time predictions

---

## 🔍 Key Findings

### 1. Dominant Predictors
- **Mental health indicators** (suicidal thoughts) are the strongest predictors
- **Stress factors** (academic, financial) show high correlation with depression
- **Lifestyle factors** (sleep, diet) have moderate but notable impact
- **Demographics** (gender, city) show minimal predictive value

### 2. Feature Interactions
- Strong correlation between academic and work pressure (0.77)
- Job satisfaction inversely related to work pressure
- Combined stress factors amplify depression risk

### 3. Clinical Implications
- High recall (90%) ensures most at-risk students are identified
- Good precision (85%) reduces unnecessary interventions
- Model can serve as a screening tool, not diagnostic instrument

---

## 💻 Installation & Usage

### Prerequisites
```bash
Python >= 3.7
pip install pandas numpy scikit-learn matplotlib seaborn jupyter
```

### Quick Start
1. Clone the repository
```bash
git clone https://github.com/yourusername/student-depression-prediction.git
cd student-depression-prediction
```

2. Run the Jupyter notebook
```bash
jupyter notebook CSE422_Project_Student_Depression_Prediction.ipynb
```

3. Execute all cells to reproduce results

### Files Structure
```
├── CSE422_Project_Student_Depression_Prediction.ipynb  # Main implementation
├── Student_Depression_Dataset_Updated.csv              # Dataset
├── 22101229_Aparup_Chowdhury_CSE422_04_Report_Fall2024.pdf  # Report
└── README.md                                           # Documentation
```

---

## 🔮 Future Work

### Model Enhancements
- **Ensemble Methods**: Random Forest, XGBoost for improved accuracy
- **Deep Learning**: Neural networks for complex pattern recognition
- **Hybrid Approaches**: Combining multiple models for robust predictions

### Feature Engineering
- Temporal patterns from longitudinal data
- Text analysis from student feedback
- Integration with academic performance trends

### Deployment Strategy
- Web application for real-time assessment
- Mobile app for student self-monitoring
- API for integration with campus health systems

### Extended Analysis
- Multi-class severity prediction (mild/moderate/severe)
- Explainable AI techniques (SHAP, LIME) for model interpretation
- Causal analysis for intervention planning

---

## 📚 References

1. World Health Organization (2023). *Depression and Other Common Mental Disorders*
2. Kaggle Dataset: [Student Depression Dataset](https://www.kaggle.com/datasets/hopesb/student-depression-dataset)
3. Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning*
4. Scikit-learn Documentation: [Machine Learning in Python](https://scikit-learn.org/)

---

## 🙏 Acknowledgments

- **Dataset Contributors**: Kaggle community
- **Course Instructor**: CSE422 Faculty, BRAC University
- **Open Source Community**: Scikit-learn, Pandas, NumPy developers
- **Institution**: BRAC University for resources and support

---

## ⚠️ Important Notice

**This project is for educational and research purposes only.**

This model should NOT be used as a substitute for professional mental health diagnosis or treatment. If you or someone you know is experiencing depression:

- Seek help from qualified mental health professionals
- Contact university counseling services
- Remember: Seeking help is a sign of strength

---

## 📧 Contact

For questions or collaboration opportunities, please reach out via email us.

**Project Team**
- Aparup Chowdhury: aparup.chowdhury@g.bracu.ac.bd
- Nabid Hasan Omi: nabid.hasan.omi@g.bracu.ac.bd



---

*This project demonstrates the application of artificial intelligence in mental health assessment, emphasizing the importance of data-driven approaches while recognizing the irreplaceable value of professional mental health care.*
