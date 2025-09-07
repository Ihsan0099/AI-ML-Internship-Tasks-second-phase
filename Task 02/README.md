# 📊 Telco Customer Churn Prediction – End-to-End ML Pipeline

## 📌 Task Description
**Task 2: End-to-End ML Pipeline with Scikit-learn Pipeline API**

**Objective:**  
Build a reusable and production-ready machine learning pipeline for predicting **customer churn**.

**Dataset:**  
Telco Churn Dataset (`WA_Fn-UseC_-Telco-Customer-Churn.csv`)

**Instructions Completed:**
- ✅ Implemented preprocessing (scaling, encoding) using `Pipeline`  
- ✅ Trained **Logistic Regression** and **Random Forest** models  
- ✅ Performed hyperparameter tuning with **GridSearchCV**  
- ✅ Exported the complete pipeline with **joblib**  

---

## 📂 Project Structure

---

## ⚙️ Implementation Details

### 🔹 Data Preprocessing
- Dropped irrelevant column: `customerID`
- Converted `TotalCharges` to numeric and filled missing values
- Encoded categorical variables with **OneHotEncoder**
- Scaled numerical features with **StandardScaler**
- Combined steps in a **Scikit-learn Pipeline**

### 🔹 Model Training
- Built models with **Logistic Regression** and **Random Forest Classifier**
- Used **Pipeline + ColumnTransformer** for preprocessing integration

### 🔹 Hyperparameter Tuning
- Tuned hyperparameters using **GridSearchCV**
- Evaluated models with:
  - Accuracy Score
  - Confusion Matrix
  - Classification Report (Precision, Recall, F1-score)

### 🔹 Export
- Saved the best-performing pipeline with:
  ```python
  import joblib
  joblib.dump(best_model, "churn_pipeline.pkl")
