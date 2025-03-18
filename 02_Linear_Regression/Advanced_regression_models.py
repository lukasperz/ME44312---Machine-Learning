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

# Remove all rows where tip_amount is zero (if wanted remove # for the next row)
# data = data[data['tip_amount'] > 0]

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


# %% ----- Improvements ----- (in each section all the variables except the X,y are redefined so that the structure of the code can be better followed)

# 1.) Polynomial features instead of linear ones

from sklearn.preprocessing import PolynomialFeatures

# Transform features into polynomial form (degree=2)                                                                       # Instead of just using the input features we extend them with a polynomial X_poly = [x1 x2 x1**2 x2**2 x1*x1] and perform again the linear regression, but now based on this extended inputs associated to each y
poly = PolynomialFeatures(degree=2, include_bias=False)                                                                    # Set degree=2, can increase to 3 if needed (no extra column of ones is added since the linear regression model already includes it)
X_poly = poly.fit_transform(X_train)                                                                                       # Transform original features
X_poly_val = poly.fit_transform(X_val)

poly_model = LinearRegression()
poly_model.fit(X_poly, y_train)                                                                                            # Fit model on transformed features

# Predict on training data
y_pred_poly = poly_model.predict(X_poly_val)
y_pred_poly = np.maximum(y_pred_poly, 0)

mae_poly = mean_absolute_error(y_val, y_pred_poly)
r2_poly = r2_score(y_val, y_pred_poly)

print(f"Polynomial Regression (Degree=2) - MAE: {mae_poly:.4f}, R² Score: {r2_poly:.4f}")

# Initially a higher degree of regression increases the accuracy of the predictions, since more the underlying pattern can be better captured. However, when the degree passes a certain order the model performs worse because then it starts to fit noise which reduces generalization


# %% Regularization applied

from sklearn.linear_model import Ridge, Lasso

# Apply regularization to the polynomial features (Ridge & Lasso Regression)
alpha_value = 1000  # Regularization strength (also with a high one the result does not change)

# Ridge Regression
ridge_model = Ridge(alpha=alpha_value)
ridge_model.fit(X_train, y_train)
y_pred_ridge = ridge_model.predict(X_val)
y_pred_ridge = np.maximum(y_pred_ridge, 0)

# Evaluate Ridge Regression
mae_ridge = mean_absolute_error(y_val, y_pred_ridge)
r2_ridge = r2_score(y_val, y_pred_ridge)

# Print results
print(f"Regularization-Ridge Regression (Degree=1) - MAE: {mae_ridge:.4f}, R² Score: {r2_ridge:.4f}")

#Same result, which means that the linear model is not overfitting and is therefore already learning from patterns and not from noise (the same holds for the polynomial model X_poly). This is understandable since our model has not a high number of features (just two) and one of the two coefficient is not very large. Therefore, it has no sense penalizing large coefficients

# %% 3.) log transform tip

# If we log transform the tip amount we can reduce skewness in the tip amount. Normally many times no tip is given and just a few times the tip is high. The outliers in the data have already been removed, but still the distribution is probably not normal. By log- transforming it we could improve the performance of the linear regression making the data better suited for it
# We convert the y in log scale and find the predictions with this new y. The predictions are then converted back into the initial scale to compare.

# Apply log transformation splitting on y_train and y_val
y_train_log = np.log1p(y_train)

# Train Linear Regression Model with log-transformed tip amounts
log_model = LinearRegression()
log_model.fit(X_train, y_train_log)  # Fit model on log-transformed y_train

# Predict log-transformed tips
y_pred_log = log_model.predict(X_val)

# Convert predictions back to original scale
y_pred = np.expm1(y_pred_log)
y_pred = np.maximum(y_pred, 0)

# Evaluate performance
mae_log = mean_absolute_error(y_val, y_pred)
r2_log = r2_score(y_val, y_pred)

print(f"Log-Transformed Linear Regression - MAE: {mae_log:.4f}, R² Score: {r2_log:.4f}")

# We see that this makes the result worse which is why y is already approximately normal and the log transformation only distort relationships instead of improving them

# %% 4.) Random forest

from sklearn.ensemble import RandomForestRegressor

# Set hyperparameters
n_estimators = 100 # Number of trees
random_state = 42  # Ensure reproducibility by defining that the trees are built randomly but once built then always in the same way

# The concept of the random forest is that many Decision Trees are trained independently on different random subsets of the training data and each tree makes then a prediction and the average of the predictions is taken

# Random Forest Regression
rf_model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
rf_model.fit(X_train, y_train)  # Train on training data

# Predict on validation data
y_pred_rf = rf_model.predict(X_val)

# Evaluate Random Forest performance
mae_rf = mean_absolute_error(y_val, y_pred_rf)
r2_rf = r2_score(y_val, y_pred_rf)

print(f"Random Forest Regression - MAE: {mae_rf:.4f}, R² Score: {r2_rf:.4f}")





# %% Gradient boosting

from sklearn.ensemble import GradientBoostingRegressor

n_estimators = 100  # Number of trees
random_state = 42   # Random data is taken, but once taken the state remains the same

# The concept of the gradient boosting is similar to the one of the random forest, but now instead of creating many trees at once and then computing the average here the same amount of trees (defined) is created, but one after another, where each new tree corrects the mistakes of the previous ones. Eg. the first tree predicts the tips but has a certain over/ underestimation pattern. The second tree then predicts the residuals to adjust this pattern

# Gradient boosting
gb_model = GradientBoostingRegressor(n_estimators=n_estimators, learning_rate=0.1, random_state=random_state)
gb_model.fit(X_train, y_train)  # Train on training data

# Predict on validation data
y_pred_gb = gb_model.predict(X_val)

# Evaluate Gradient Boosting performance
mae_gb = mean_absolute_error(y_val, y_pred_gb)
r2_gb = r2_score(y_val, y_pred_gb)

print(f"Gradient Boosting Regression - MAE: {mae_gb:.4f}, R² Score: {r2_gb:.4f}")

import matplotlib.pyplot as plt

# Create the plot
plt.figure(figsize=(10, 6))
plt.scatter(y_val, y_pred_gb, color='blue', alpha=0.7, label='Predicted vs Actual')
plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'r--', lw=2, label='Pred. = Act.')
text_str = f"MAE: {mae_gb:.4f}\nR² Score: {r2_gb:.4f}"
plt.text(0.05, 0.9, text_str, transform=plt.gca().transAxes, fontsize=12, bbox=dict(facecolor='white', alpha=0.5))
plt.xlabel("Actual Tip Amount")
plt.ylabel("Predicted Tip Amount")
plt.title("Predicted vs. Actual Tip Amount")
plt.legend()
plt.show()


# Print Feature Importance (they can only be approximated)
feature_importance = gb_model.feature_importances_

# Extract values for trip_distance and fare_amount
theta1, theta2 = float(feature_importance[0]), float(feature_importance[1])  # Convert to float values
theta0 = float(gb_model.init_.constant_.ravel()[0]) if hasattr(gb_model.init_, "constant_") else 0  # Extract first value properly as a scalar

# Print the Approximate Regression Formula
print(f"\nApproximate Regression Formula:")
print(f"Tip Amount ≈ {theta0:.4f} + ({theta1:.4f} * trip_distance) + ({theta2:.4f} * fare_amount)")









