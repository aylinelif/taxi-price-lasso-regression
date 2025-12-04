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
## 🚀 Model Training & Evaluation

After preprocessing and scaling, the Lasso Regression model was trained using the training dataset.

**Final Test Performance:**

- **R² Score:** ~0.75  
- **RMSE:** ~18.23  

This means that the model can explain approximately **75% of the variance** in taxi fare prices.
## 📊 Feature Importance (Lasso Coefficients)

Lasso automatically reduced some coefficients to zero, performing feature selection.

Important predictors:

- Trip Distance (km)
- Trip Duration (minutes)
- Per-Km Rate
- Per-Minute Rate

Most categorical features were reduced to zero, meaning they have little effect on the prediction.

