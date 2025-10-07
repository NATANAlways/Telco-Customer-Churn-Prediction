# Telco Customer Churn Prediction — Machine Learning Classification (2025)

## 📘 Overview
This project aims to predict customer churn in a telecommunications company using supervised machine learning algorithms. The dataset (`WA_Fn-UseC_-Telco-Customer-Churn.csv`) contains 7,043 customer records with demographic, service usage, and billing information.

The pipeline performs **data cleaning, preprocessing, feature engineering, model selection, and ensemble learning** to accurately classify customers likely to churn.

---

## 📊 Key Steps

### 1. **Data Understanding & Cleaning**
- Removed unnecessary columns (`customerID`)
- Converted data types (`TotalCharges` → numeric)
- Handled blank values and duplicates
- Identified class imbalance (Churn: ~26.5%)

### 2. **Exploratory Data Analysis (EDA)**
- Visualized distributions of `tenure`, `MonthlyCharges`, `TotalCharges`
- Examined churn percentage with pie charts
- Used boxplots to detect outliers
- Computed correlation matrix for numeric features
- Plotted categorical feature distributions vs churn

### 3. **Feature Engineering & Encoding**
- Binary mapping for Yes/No columns  
- Ordinal encoding for contract durations  
- One-hot encoding for service/payment categories  
- Standardization using **Yeo–Johnson + StandardScaler**  
- Applied **RandomOverSampler** to balance classes

### 4. **Feature Selection**
- Used **Mutual Information** and **Random Forest Feature Importance**
- Selected top 20–25 key features contributing to churn
- Visualized cumulative feature importance (80% coverage)

### 5. **Model Building & Evaluation**
- Trained multiple models:  
  Logistic Regression, Ridge Classifier, SVM, Decision Tree, Random Forest, Extra Trees, AdaBoost, Gradient Boosting, KNN, GaussianNB, and MLP.
- Applied 5-fold **Stratified Cross Validation**.
- Tuned hyperparameters using **GridSearchCV**.
- Compared accuracy, F1-score, and ROC-AUC.

### 6. **Ensemble Learning**
- Implemented **VotingClassifier**, **StackingClassifier**, and **Blending meta-models**.
- Combined Logistic Regression, Random Forest, Gradient Boosting, and XGBoost for improved generalization.

---

## 🏆 Results
| Metric | Best Model | Value |
|---------|-------------|--------|
| Accuracy | RandomForest / Stacking | ~0.85 |
| F1-score | Ensemble (Blending) | ~0.84 |
| ROC-AUC | Ensemble (Voting/Stacking) | >0.88 |

---

## ⚙️ Tech Stack
- **Language:** Python  
- **Libraries:** Pandas, NumPy, Scikit-learn, Imbalanced-learn, XGBoost  
- **Visualization:** Matplotlib, Seaborn  
- **Environment:** Google Colab / Jupyter Notebook  

---

## 📈 Future Work
- Integrate SHAP or LIME for explainability.  
- Deploy the best model using Streamlit or Flask.  
- Automate feature selection pipeline.  

---

## 📂 Repository Structure
