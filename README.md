# 🚕 Taxi Price Prediction using Lasso Regression

This project applies **Lasso Regression (L1 Regularization)** to predict taxi trip prices using a dataset from Kaggle.  
The goal is to build a clean and interpretable machine learning workflow that includes preprocessing, feature engineering, model training, and hyperparameter tuning.

---

## 📌 Project Overview

The main objective of this project is to predict the final **taxi trip price** based on trip-specific features such as:

- Trip distance (km)
- Trip duration (minutes)
- Base fare
- Per-km rate
- Per-minute rate
- Time of day
- Traffic conditions
- Weather

Lasso Regression is used because it:
- Performs **automatic feature selection**
- Produces a **simpler and more interpretable model**
- Prevents overfitting with L1 regularization  

---

## 🧹 Data Preprocessing

The following preprocessing steps were applied:

### ✔ Missing Value Handling
- Numeric features → filled with **median**
- Categorical features → filled with **mode**  
This ensured the dataset was clean before modeling.

### ✔ Encoding
Categorical columns were converted using **One-Hot Encoding**:
```python
pd.get_dummies(df, columns=["Time_of_Day","Day_of_Week","Traffic_Conditions","Weather"], drop_first=True)
