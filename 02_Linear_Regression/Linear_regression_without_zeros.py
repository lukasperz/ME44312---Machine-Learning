import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

base_dir = os.getcwd()
file_path = os.path.join(base_dir, "yellow_taxi_data.parquet")

data = pd.read_parquet(file_path)
data = data[['trip_distance', 'fare_amount', 'tip_amount']]

# Remove all rows where tip_amount is zero
data = data[data['tip_amount'] > 0]

# %% ----- Setup X and Y -----

features = ['trip_distance', 'fare_amount']
target = 'tip_amount'

X = data[features].values
y = data[target].values

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# %% ----- Linear Regression using SCIKIT-LEARN -----

model = LinearRegression()
model.fit(X_train, y_train)

theta0 = model.intercept_
theta1, theta2= model.coef_

print(f"Regression Equation: tip_amount = {theta0:.4f} + {theta1:.4f} * trip_distance + {theta2:.4f} * fare_amount")

# Predict on validation data
y_pred = model.predict(X_val)
y_pred = np.maximum(y_pred, 0)

# %% ----- Evaluate our model -----

mae = mean_absolute_error(y_val, y_pred)
r2 = r2_score(y_val, y_pred)

print(f"\nModel Performance:")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"R-squared (R²) Score: {r2:.4f}")

plt.figure(figsize=(10, 6))
plt.scatter(y_val, y_pred, color='blue', alpha=0.7, label='Predicted vs Actual')
plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'r--', lw=2, label='Pred. = Act.')
text_str = f"MAE: {mae:.4f}\nR² Score: {r2:.4f}"
plt.text(0.05, 0.9, text_str, transform=plt.gca().transAxes, fontsize=12, bbox=dict(facecolor='white', alpha=0.5))
plt.xlabel("Actual Tip Amount")
plt.ylabel("Predicted Tip Amount")
plt.title("Predicted vs. Actual Tip Amount")
plt.legend()
plt.show()
