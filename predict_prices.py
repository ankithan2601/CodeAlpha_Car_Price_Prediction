import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ==========================================
# 1. Load the Dataset
# ==========================================
# Read the dataset (Ensure 'car data.csv' is in the same folder)
df = pd.read_csv('car data.csv')

print("--- Initial Data Preview ---")
print(df.head())

# ==========================================
# 2. Feature Engineering & Preprocessing
# ==========================================
# Compute the age of the car to make the feature more meaningful
current_year = 2024
df['Car_Age'] = current_year - df['Year']

# Drop 'Car_Name' (too many unique strings) and the original 'Year' column
df.drop(['Car_Name', 'Year'], axis=1, inplace=True)

# Separate features (X) and target variable (y)
# 'Selling_Price' is what we want to predict
X = df.drop('Selling_Price', axis=1)
y = df['Selling_Price']

# Define numerical and categorical features
numerical_features = ['Present_Price', 'Driven_kms', 'Owner', 'Car_Age']
categorical_features = ['Fuel_Type', 'Selling_type', 'Transmission']

# Set up preprocessor: One-Hot Encode categorical strings, leave numerical as-is
preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', numerical_features),
        ('cat', OneHotEncoder(drop='first'), categorical_features)
    ])

# ==========================================
# 3. Model Pipeline Setup
# ==========================================
# Bundle preprocessing and the regression model into a single clean pipeline
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])

# ==========================================
# 4. Train-Test Split
# ==========================================
# Split data into 80% training and 20% validation testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
print("\nTraining the Random Forest Regressor model...")
model_pipeline.fit(X_train, y_train)
print("Model training complete!")

# ==========================================
# 5. Model Evaluation
# ==========================================
# Make predictions on test data
y_pred = model_pipeline.predict(X_test)

# Calculate metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\n--- Model Evaluation Results ---")
print(f"Mean Absolute Error (MAE): {mae:.2f} Lakhs")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f} Lakhs")
print(f"R² Score (Accuracy Metric): {r2:.2f}")

# ==========================================
# 6. Visualization
# ==========================================
# Plot actual vs predicted values to visually inspect accuracy
plt.scatter(y_test, y_pred, color='blue', alpha=0.6, label='Predicted vs Actual')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', lw=2, linestyle='--', label='Perfect Fit Line')
plt.xlabel('Actual Selling Price (in Lakhs)')
plt.ylabel('Predicted Selling Price (in Lakhs)')
plt.title('Actual vs Predicted Car Selling Prices')
plt.legend()
plt.tight_layout()

# Save the plot image locally in the folder
plt.savefig('actual_vs_predicted.png')
print("\nPlot saved successfully as 'actual_vs_predicted.png'!")