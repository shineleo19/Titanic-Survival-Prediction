# 🚢 Titanic Survival Prediction

This project builds a **Logistic Regression model** to predict passenger survival on the Titanic dataset using machine learning.  
It includes **data preprocessing, encoding, and model evaluation**.

---

## 📊 Dataset
The dataset used is `titanic.csv`, which contains details about Titanic passengers such as age, gender, ticket class, and survival status.

### Key Columns:
- `Survived` — Target variable (1 = Survived, 0 = Did not survive)  
- `Pclass`, `Sex`, `Age`, `Fare`, `Embarked` — Predictor features

---

## ⚙️ Features of the Model
- Handles missing values using median/mode imputation  
- Encodes categorical features using Label Encoding  
- Splits data into training and testing sets  
- Trains a Logistic Regression classifier  
- Evaluates model performance using:
  - Accuracy score  
  - Confusion matrix  
  - Classification report  

---

## 🧠 Technologies Used
- **Python 3.10+**
- **Pandas**
- **Scikit-learn**

---

## 🚀 How to Run the Project

### 1. Clone the Repository
```bash
1.git clone https://github.com/<your-username>/titanic-survival-prediction.git
2.cd titanic-survival-prediction
3.pip install -r requirements.txt
4.python main.py
