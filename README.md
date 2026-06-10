# Car Price Prediction using Machine Learning

This repository contains a complete, end-to-end Machine Learning regression pipeline implemented in Python to predict the selling price of used cars based on various features such as historical pricing, usage metrics, age, and fuel profiles. 

Predicting used car valuations accurately is a core real-world application utilized by automotive e-commerce portals, insurance firms, and dealership networks to establish fair-market values dynamically.

## 📌 Project Objectives
* **Data Preprocessing & Cleaning:** Handle categorical variables, dropped unneeded metadata, and extract clean feature sets.
* **Feature Engineering:** Derive actionable insights (like calculating the vehicle's actual age) from raw snapshot features.
* **Pipeline Architecture:** Utilize Scikit-Learn pipelines to bundle transformers and model estimators into a clean, reproducible workflow.
* **Model Evaluation:** Measure and contrast numerical continuous errors using MAE, RMSE, and $R^2$ evaluation metrics.

## 📊 Dataset Overview
The project processes a dataset (`car data.csv`) consisting of various multi-type parameters:
* **Numerical Features:** `Present_Price` (Showroom price in Lakhs), `Driven_kms` (Total kilometers driven), `Owner` (Number of previous owners), and `Year` (Manufacturing year).
* **Categorical Features:** `Fuel_Type` (Petrol/Diesel/CNG), `Selling_type` (Dealer/Individual), and `Transmission` (Manual/Automatic).
* **Target Variable:** `Selling_Price` (The market price the car is sold at in Lakhs).

## ⚙️ Workflow Architecture
The system logic is divided into distinct, scalable modular stages:

1. **Feature Construction:** Calculated an explicit `Car_Age` feature (`Current_Year - Year`) to capture chronological depreciation value and dropped metadata columns like `Car_Name`.
2. **Column Transformations:** Used a `ColumnTransformer` to seamlessly pass through continuous variables while applying structural **One-Hot Encoding** to categorical string vectors.
3. **Model Selection:** Implemented an ensemble **Random Forest Regressor** to effectively handle non-linear feature constraints and capture high-variance distribution lines without overfitting.
4. **Data Validation:** Segmented the data matrix into an $80/20$ Train-Test split layout to guarantee an unbiased performance evaluation on unseen testing rows.

## 📈 Model Performance & Evaluation
After tuning and evaluating on the test set, the model achieved highly robust prediction metrics:

* **Mean Absolute Error (MAE):** 0.64 Lakhs *(On average, predictions deviate by just ~₹64,000)*
* **Root Mean Squared Error (RMSE):** 0.97 Lakhs
* **R² Score (Coefficient of Determination):** 0.96 *(The model explains 96% of the variance in used car selling prices)*

### Visualization
Upon executing the pipeline script, a scatter plot mapping **Actual vs. Predicted Prices** is generated automatically and saved locally as `actual_vs_predicted.png` to help visualize regression fit lines.
