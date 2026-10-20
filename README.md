
# 🏥 Delivery Type Prediction

A Machine Learning project that classifies delivery outcomes as **spontaneous vaginal delivery** or **emergency cesarean delivery** using selected maternal, pregnancy-related, and clinical features.

The project covers data cleaning, exploratory data analysis (EDA), feature engineering, model comparison, hyperparameter tuning, and a voting-ensemble classifier.

> **Medical Disclaimer:** This project is for educational and research purposes only. It is not clinically validated and must not be used to make decisions about patient care or delivery method.

## 📌 Project Overview

The notebook builds a binary classification workflow using historical maternity data.

The two target classes are:

- **Spontaneous Vaginal Delivery (0)**
- **Emergency Cesarean Delivery (1)**

Elective cesarean and assisted vaginal deliveries are excluded from the final dataset.

## 📊 Dataset

The final dataset contains **16,651 records**.

| Delivery Outcome | Number of Records |
|---|---:|
| Spontaneous Vaginal Delivery | 12,639 |
| Emergency Cesarean Delivery | 4,012 |
| **Total** | **16,651** |

The dataset is derived from historical maternity records. The source data is loaded from an Excel file in the notebook.

> Note: Ensure that you have permission to share the dataset before publishing it. Do not upload confidential patient information.

## 🎯 Project Objectives

- Clean and preprocess clinical data.
- Handle missing and invalid values.
- Perform feature engineering.
- Encode categorical and clinical text features.
- Train and compare multiple machine learning classifiers.
- Tune model hyperparameters using GridSearchCV.
- Build a voting ensemble.
- Evaluate classification performance.
- Save the final model using Joblib.

## 🧹 Data Preprocessing

The notebook includes the following steps:

1. Dataset inspection and exploratory data analysis.
2. Filtering delivery outcomes for binary classification.
3. Identifying missing values and invalid clinical measurements.
4. Handling missing values using imputation techniques, including KNN imputation.
5. Encoding categorical and clinical text features using MultiLabelBinarizer.
6. Feature selection and engineering.
7. Stratified train-test split using an 80/20 ratio.

### Mean Arterial Pressure (MAP)

Mean Arterial Pressure is calculated from systolic and diastolic blood pressure:

`MAP = (SBP + 2 × DBP) / 3`

## 🩺 Features

The final model uses 12 selected features:

- Fetal compromise indication
- Previous uterine surgery indication
- Number of previous cesarean sections
- Twins or more risk factor
- Maternal height
- Parity
- Gestational age
- Multiple pregnancy indication
- Antepartum hemorrhage indication
- Preeclampsia indication
- Systolic blood pressure
- Mean Arterial Pressure (MAP)

## 🤖 Machine Learning Models

The notebook evaluates the following classifiers:

- Logistic Regression
- Decision Tree
- Random Forest
- XGBoost
- CatBoost

Hyperparameter tuning is performed using **GridSearchCV with 5-fold cross-validation**, with F1-score used for model selection.

The tuned models are combined in a **Voting Classifier** to produce the final ensemble prediction.

## 📈 Model Evaluation

The following results are reported in the notebook's classification reports.

| Model | Accuracy | Class 1 Precision | Class 1 Recall | Class 1 F1-score |
|---|---:|---:|---:|---:|
| Decision Tree | 89% | 73% | 87% | 79% |
| Logistic Regression | 86% | 80% | 58% | 67% |
| Random Forest | 91% | 78% | 87% | 83% |
| XGBoost | 91% | 78% | 89% | 83% |
| CatBoost | 92% | 80% | 88% | 84% |
| **Voting Ensemble** | **93%** | **85%** | **88%** | **86%** |

### Final Voting Ensemble

- **Accuracy:** 93%
- **Class 0 F1-score:** 96%
- **Class 1 F1-score:** 86%
- **Macro-average F1-score:** 91%
- **Weighted-average F1-score:** 93%

Class 1 represents emergency cesarean delivery.

These results come from the notebook's test split and have not been independently validated. They do not establish clinical safety or real-world performance.

## ⚠️ Important Limitation: Potential Target Leakage

Some selected features are derived from `PrimaryIndicationforCaesarean`. These indications may only become available when a cesarean decision is being made.

Using them could introduce **target leakage** and make the model unsuitable for predicting delivery type in advance.

Before interpreting this as a prospective prediction model, verify when every feature becomes available and evaluate a version that uses only information available at the intended prediction time.

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- XGBoost
- CatBoost
- Joblib
- Jupyter Notebook
- Google Colab

## 💾 Saved Model Artifacts

The notebook saves the following files:

- `delivery_type_model_final.pkl` — trained voting-ensemble model.
- `model_features.pkl` — selected feature names.

These files are generated when the relevant notebook cells are executed.

## 🚀 How to Run the Project

### 1. Clone the repository

```bash
git clone YOUR_REPOSITORY_URL
cd YOUR_PROJECT_FOLDER
```

Replace the placeholders with your actual repository URL and folder name.

### 2. Install dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost catboost joblib openpyxl jupyter
```

### 3. Open the notebook

```bash
jupyter notebook
```

Open the Delivery Type Prediction notebook.

### 4. Load the dataset

Provide the authorized Excel dataset and update the file path in the notebook to match your environment.

### 5. Run the notebook

Execute the cells in order to preprocess the data, train and evaluate the models, and save the final model artifacts.

## 🔮 Future Improvements

- Validate the model using independent external datasets.
- Investigate and eliminate potential target leakage.
- Evaluate calibration and class-specific performance.
- Assess performance across relevant patient subgroups.
- Add automated tests and reproducible dependency versions.
- Seek appropriate clinical review before considering real-world use.

## 👨‍💻 Author

**Hazem Ibrahim**

- **GitHub:** [@7azemIbrahim](https://github.com/7azemIbrahim)
- **LinkedIn:**[My Linkedin](https://www.linkedin.com/in/hazem-ibrahim-8503a628a/).
---

⭐ Thanks for visiting this project!
