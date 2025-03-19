import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# %% DATA SETUP for understanding and visualization

# Get the current working directory
data_dir = os.getcwd()
base_dir = os.path.join(data_dir, os.pardir, "FINAL_FOLDER/00_DATA")
base_dir = os.path.abspath(data_dir)

# Load Yellow Taxi Data
yellow_file_path = 'FINAL_FOLDER/00_DATA/yellow_taxi_data_NO_SCALE.parquet'
yellow_data = pd.read_parquet(yellow_file_path)
yellow_data = yellow_data[['trip_distance', 'fare_amount', 'tip_amount', 'passenger_count']]
yellow_filtered_data = yellow_data[yellow_data['tip_amount'] > 0]

# Load Green Taxi Data
green_file_path = 'FINAL_FOLDER/00_DATA/green_taxi_data_NO_SCALE.parquet'
green_data = pd.read_parquet(green_file_path)
green_data = green_data[['trip_distance', 'fare_amount', 'tip_amount', 'passenger_count']]
green_filtered_data = green_data[green_data['tip_amount'] > 0]

# Create visualization plot
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for i, feature in enumerate(['trip_distance', 'fare_amount', 'passenger_count']):
    # Filtering outliers
    if feature in ['trip_distance', 'fare_amount']:
        yellow_feature_data = yellow_filtered_data[
            yellow_filtered_data[feature] <= yellow_filtered_data[feature].quantile(0.99)]
        green_feature_data = green_filtered_data[
            green_filtered_data[feature] <= green_filtered_data[feature].quantile(0.99)]
    else:
        yellow_feature_data = yellow_filtered_data
        green_feature_data = green_filtered_data

    axes[i].scatter(yellow_feature_data[feature], yellow_feature_data['tip_amount'], alpha=0.3, color='yellow',
                    label="Yellow Taxi Data")
    axes[i].scatter(green_feature_data[feature], green_feature_data['tip_amount'], alpha=0.3, color='green',
                    label="Green Taxi Data")

    x_vals_yellow = np.linspace(yellow_feature_data[feature].min(), yellow_feature_data[feature].max(), 100)
    poly_coeffs_yellow = np.polyfit(yellow_feature_data[feature], yellow_feature_data['tip_amount'], deg=1)
    y_vals_yellow = np.polyval(poly_coeffs_yellow, x_vals_yellow)

    x_vals_green = np.linspace(green_feature_data[feature].min(), green_feature_data[feature].max(), 100)
    poly_coeffs_green = np.polyfit(green_feature_data[feature], green_feature_data['tip_amount'], deg=1)
    y_vals_green = np.polyval(poly_coeffs_green, x_vals_green)

    axes[i].plot(x_vals_yellow, y_vals_yellow, color="gold", linewidth=3, label="Yellow Trend Line")
    axes[i].plot(x_vals_green, y_vals_green, color="limegreen", linewidth=3, label="Green Trend Line")

    axes[i].set_xlabel(feature)
    axes[i].set_ylabel('Tip Amount')
    axes[i].ticklabel_format(style='plain', axis='x')

axes[0].set_xlim(0, max(yellow_filtered_data['trip_distance'].quantile(0.99),
                        green_filtered_data['trip_distance'].quantile(0.99)))
axes[1].set_xlim(0, max(yellow_filtered_data['fare_amount'].quantile(0.99),
                        green_filtered_data['fare_amount'].quantile(0.99)))

plt.suptitle("Relationship Between Trip Distance, Fare Amount, Passenger Count, and Tip Amount", fontsize=14)
plt.legend()
plt.show()


# %% DATA SETUP for REGRESSION (we overwrite the previous data which was just used to visualize the dependencies)

def load_data(filename):
    data_dir = os.getcwd()
    base_dir = os.path.join(data_dir, os.pardir, "00_DATA")
    base_dir = os.path.abspath(data_dir)
    file_path = os.path.join(base_dir, filename)
    data = pd.read_parquet(file_path)
    data = data[['trip_distance', 'fare_amount', 'tip_amount']]
    return data

yellow_data = load_data('FINAL_FOLDER/00_DATA/yellow_taxi_data_NO_SCALE.parquet')
green_data = load_data('FINAL_FOLDER/00_DATA/green_taxi_data_NO_SCALE.parquet')

features = ['trip_distance', 'fare_amount']
target = 'tip_amount'

X_yellow = yellow_data[features].values
y_yellow = yellow_data[target].values

X_green = green_data[features].values
y_green = green_data[target].values

X_train_yellow, X_val_yellow, y_train_yellow, y_val_yellow = train_test_split(X_yellow, y_yellow, test_size=0.2, random_state=42)
X_train_green, X_val_green, y_train_green, y_val_green = train_test_split(X_green, y_green, test_size=0.2, random_state=42)


# %% Simple Linear Regression using SCIKIT-LEARN

model_yellow = LinearRegression()
model_yellow.fit(X_train_yellow, y_train_yellow)

theta0 = model_yellow.intercept_
theta1, theta2= model_yellow.coef_

print(f"Simple Regression Equation for Green Taxis: tip_amount = {theta0:.4f} + {theta1:.4f} * trip_distance + {theta2:.4f} * fare_amount")

model_green = LinearRegression()
model_green.fit(X_train_green, y_train_green)

theta0_green = model_green.intercept_
theta1_green, theta2_green= model_green.coef_

print(f"Simple Regression Equation for Yellow Taxis: tip_amount = {theta0_green:.4f} + {theta1_green:.4f} * trip_distance + {theta2_green:.4f} * fare_amount")

# %% Evaluation of the simplest model

y_pred_yellow = model_yellow.predict(X_val_yellow)
y_pred_yellow = np.maximum(y_pred_yellow, 0)

mae_y = mean_absolute_error(y_val_yellow, y_pred_yellow)
r2_y = r2_score(y_val_yellow, y_pred_yellow)

print(f"\nYellow Model Performance:")
print(f"Mean Absolute Error (MAE): {mae_y:.4f}")
print(f"R-squared (R²) Score: {r2_y:.4f}")

y_pred_green = model_green.predict(X_val_green)
y_pred_green = np.maximum(y_pred_green, 0)

mae_g = mean_absolute_error(y_val_green, y_pred_green)
r2_g = r2_score(y_val_green, y_pred_green)

print(f"\nGreen Model Performance:")
print(f"Mean Absolute Error (MAE): {mae_g:.4f}")
print(f"R-squared (R²) Score: {r2_g:.4f}")



plt.figure(figsize=(10, 6))
plt.scatter(y_val_yellow, y_pred_yellow, color='gold', alpha=0.6, label='Yellow Taxi')
plt.scatter(y_val_green, y_pred_green, color='green', alpha=0.6, label='Green Taxi')

plt.plot([min(y_val_yellow.min(), y_val_green.min()), max(y_val_yellow.max(), y_val_green.max())],
         [min(y_val_yellow.min(), y_val_green.min()), max(y_val_yellow.max(), y_val_green.max())],
         'r--', lw=2, label='Pred. = Act.')

text_str = f"Yellow MAE: {mae_y:.4f}\nYellow R²: {r2_y:.4f}\nGreen MAE: {mae_g:.4f}\nGreen R²: {r2_g:.4f}"
plt.text(0.05, 0.85, text_str, transform=plt.gca().transAxes, fontsize=12,
         bbox=dict(facecolor='white', alpha=0.5))

plt.xlabel("Actual Tip Amount")
plt.ylabel("Predicted Tip Amount")
plt.title("Predicted vs. Actual Tip Amount for Yellow and Green Taxis (first simple regression)")
plt.legend()
plt.show()

# %% Polynomial regression

from sklearn.preprocessing import PolynomialFeatures

# Transform features into polynomial form (degree=2)                                                                       # Instead of just using the input features we extend them with a polynomial X_poly = [x1 x2 x1**2 x2**2 x1*x1] and perform again the linear regression, but now based on this extended inputs associated to each y
poly_y = PolynomialFeatures(degree=2, include_bias=False)                                                                  # Set degree=2, can increase to 3 if needed (no extra column of ones is added since the linear regression model already includes it)
X_poly_y = poly_y.fit_transform(X_train_yellow)                                                                            # Transform original features
X_poly_val_y = poly_y.fit_transform(X_val_yellow)

poly_model_y = LinearRegression()
poly_model_y.fit(X_poly_y, y_train_yellow)

y_pred_poly_y = poly_model_y.predict(X_poly_val_y)
y_pred_poly = np.maximum(y_pred_poly_y, 0)

mae_poly_y = mean_absolute_error(y_val_yellow, y_pred_poly)
r2_poly_y = r2_score(y_val_yellow, y_pred_poly)

print(f"Yellow Polynomial Regression (Degree=2) - MAE: {mae_poly_y:.4f}, R² Score: {r2_poly_y:.4f}")


# Transform features into polynomial form (degree=2)                                                                       # Instead of just using the input features we extend them with a polynomial X_poly = [x1 x2 x1**2 x2**2 x1*x1] and perform again the linear regression, but now based on this extended inputs associated to each y
poly_g = PolynomialFeatures(degree=2, include_bias=False)                                                                  # Set degree=2, can increase to 3 if needed (no extra column of ones is added since the linear regression model already includes it)
X_poly_g = poly_g.fit_transform(X_train_green)                                                                             # Transform original features
X_poly_val_g = poly_g.fit_transform(X_val_green)

poly_model_g = LinearRegression()
poly_model_g.fit(X_poly_g, y_train_green)

y_pred_poly_g = poly_model_g.predict(X_poly_val_g)
y_pred_poly = np.maximum(y_pred_poly_g, 0)

mae_poly_g = mean_absolute_error(y_val_green, y_pred_poly)
r2_poly_g = r2_score(y_val_green, y_pred_poly)

print(f"Green Polynomial Regression (Degree=2) - MAE: {mae_poly_g:.4f}, R² Score: {r2_poly_g:.4f}")

# %% Regularization

from sklearn.linear_model import Ridge

alpha_value = 1000  # Regularization strength (also with a high one the result does not change)

# Ridge Regression
ridge_model_y = Ridge(alpha=alpha_value)
ridge_model_y.fit(X_train_yellow, y_train_yellow)
y_pred_ridge_y = ridge_model_y.predict(X_val_yellow)
y_pred_ridge_y = np.maximum(y_pred_ridge_y, 0)

mae_ridge_y = mean_absolute_error(y_val_yellow, y_pred_ridge_y)
r2_ridge_y = r2_score(y_val_yellow, y_pred_ridge_y)

print(f"Yellow Regularization-Ridge Regression (Degree=1) - MAE: {mae_ridge_y:.4f}, R² Score: {r2_ridge_y:.4f}")


# Ridge Regression
ridge_model_g = Ridge(alpha=alpha_value)
ridge_model_g.fit(X_train_green, y_train_green)
y_pred_ridge_g = ridge_model_g.predict(X_val_green)
y_pred_ridge_g = np.maximum(y_pred_ridge_g, 0)

mae_ridge_g = mean_absolute_error(y_val_green, y_pred_ridge_g)
r2_ridge_g = r2_score(y_val_green, y_pred_ridge_g)

print(f"Green Regularization-Ridge Regression (Degree=1) - MAE: {mae_ridge_g:.4f}, R² Score: {r2_ridge_g:.4f}")

#Same result, which means that the linear model is not overfitting and is therefore already learning from patterns and not from noise (the same holds for the polynomial model X_poly). This is understandable since our model has not a high number of features (just two) and one of the two coefficient is not very large. Therefore, it has no sense penalizing large coefficients

# %% Log transformation

# If we log transform the tip amount we can reduce skewness in the tip amount. Normally many times no tip is given and just a few times the tip is high. The outliers in the data have already been removed, but still the distribution is probably not normal. By log- transforming it we could improve the performance of the linear regression making the data better suited for it
# We convert the y in log scale and find the predictions with this new y. The predictions are then converted back into the initial scale to compare.

y_train_log_y = np.log1p(y_train_yellow)

log_model_y = LinearRegression()
log_model_y.fit(X_train_yellow, y_train_log_y)  # Fit model on log-transformed y_train

y_pred_log_y = log_model_y.predict(X_val_yellow)

y_pred_y = np.expm1(y_pred_log_y)
y_pred_y = np.maximum(y_pred_y, 0)

mae_log_y = mean_absolute_error(y_val_yellow, y_pred_y)
r2_log_y = r2_score(y_val_yellow, y_pred_y)

print(f"Yellow Log-Transformed Linear Regression - MAE: {mae_log_y:.4f}, R² Score: {r2_log_y:.4f}")

# We see that this makes the result worse which is why y is already approximately normal and the log transformation only distort relationships instead of improving them

y_train_log_g = np.log1p(y_train_green)

log_model_g = LinearRegression()
log_model_g.fit(X_train_green, y_train_log_g)

y_pred_log_g = log_model_g.predict(X_val_green)

y_pred_g = np.expm1(y_pred_log_g)
y_pred_g = np.maximum(y_pred_g, 0)

mae_log_g = mean_absolute_error(y_val_green, y_pred_g)
r2_log_g = r2_score(y_val_green, y_pred_g)

print(f"Green Log-Transformed Linear Regression - MAE: {mae_log_g:.4f}, R² Score: {r2_log_g:.4f}")


# %% Random Forest

from sklearn.ensemble import RandomForestRegressor

# Set hyperparameters
n_estimators = 100 # Number of trees
random_state = 42  # Ensure reproducibility by defining that the trees are built randomly but once built then always in the same way

# The concept of the random forest is that many Decision Trees are trained independently on different random subsets of the training data and each tree makes then a prediction and the average of the predictions is taken

rf_model_y = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
rf_model_y.fit(X_train_yellow, y_train_yellow)  # Train on training data

y_pred_rf_y = rf_model_y.predict(X_val_yellow)

mae_rf_y = mean_absolute_error(y_val_yellow, y_pred_rf_y)
r2_rf_y = r2_score(y_val_yellow, y_pred_rf_y)

print(f"Yellow Random Forest Regression - MAE: {mae_rf_y:.4f}, R² Score: {r2_rf_y:.4f}")



rf_model_g = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
rf_model_g.fit(X_train_green, y_train_green)  # Train on training data

y_pred_rf_g = rf_model_g.predict(X_val_green)

mae_rf_g = mean_absolute_error(y_val_green, y_pred_rf_g)
r2_rf_g = r2_score(y_val_green, y_pred_rf_g)

print(f"Green Random Forest Regression - MAE: {mae_rf_g:.4f}, R² Score: {r2_rf_g:.4f}")


# %% Gradient boosting

from sklearn.ensemble import GradientBoostingRegressor

# The concept of the gradient boosting is similar to the one of the random forest, but now instead of creating many trees at once and then computing the average here the same amount of trees (defined) is created, but one after another, where each new tree corrects the mistakes of the previous ones. Eg. the first tree predicts the tips but has a certain over/ underestimation pattern. The second tree then predicts the residuals to adjust this pattern

gb_model_y = GradientBoostingRegressor(n_estimators=n_estimators, learning_rate=0.1, random_state=random_state)
gb_model_y.fit(X_train_yellow, y_train_yellow)  # Train on training data

y_pred_gb_y = gb_model_y.predict(X_val_yellow)

mae_gb_y = mean_absolute_error(y_val_yellow, y_pred_gb_y)
r2_gb_y = r2_score(y_val_yellow, y_pred_gb_y)

print(f"Green Gradient Boosting Regression - MAE: {mae_gb_y:.4f}, R² Score: {r2_gb_y:.4f}")

gb_model_g = GradientBoostingRegressor(n_estimators=n_estimators, learning_rate=0.1, random_state=random_state)
gb_model_g.fit(X_train_green, y_train_green)  # Train on training data

y_pred_gb_g = gb_model_g.predict(X_val_green)

mae_gb_g = mean_absolute_error(y_val_green, y_pred_gb_g)
r2_gb_g = r2_score(y_val_green, y_pred_gb_g)

print(f"Green Gradient Boosting Regression - MAE: {mae_gb_g:.4f}, R² Score: {r2_gb_g:.4f}")

# %% Plot and print final result

plt.figure(figsize=(10, 6))
plt.scatter(y_val_yellow, y_pred_gb_y, color='gold', alpha=0.6, label='Yellow Taxi')
plt.scatter(y_val_green, y_pred_gb_g, color='green', alpha=0.6, label='Green Taxi')

plt.plot([min(y_val_yellow.min(), y_val_green.min()), max(y_val_yellow.max(), y_val_green.max())],
         [min(y_val_yellow.min(), y_val_green.min()), max(y_val_yellow.max(), y_val_green.max())],
         'r--', lw=2, label='Pred. = Act.')

text_str = (f"Yellow MAE: {mae_gb_y:.4f}\nYellow R²: {r2_gb_y:.4f}\n"
            f"Green MAE: {mae_gb_g:.4f}\nGreen R²: {r2_gb_g:.4f}")
plt.text(0.05, 0.85, text_str, transform=plt.gca().transAxes, fontsize=12,
         bbox=dict(facecolor='white', alpha=0.5))

plt.xlabel("Actual Tip Amount")
plt.ylabel("Predicted Tip Amount")
plt.title("Gradient Boosting: Predicted vs. Actual Tip Amount for Yellow and Green Taxis")
plt.legend()
plt.show()





# Print Feature Importance (they can only be approximated)
feature_importance_y = gb_model_y.feature_importances_

# Extract values for trip_distance and fare_amount
theta1, theta2 = float(feature_importance_y[0]), float(feature_importance_y[1])  # Convert to float values
theta0 = float(gb_model_y.init_.constant_.ravel()[0]) if hasattr(gb_model_y.init_, "constant_") else 0  # Extract first value properly as a scalar

# Print the Approximate Regression Formula
print(f"\nYellow Approximate Regression Formula:")
print(f"Yellow Tip Amount ≈ {theta0:.4f} + ({theta1:.4f} * trip_distance) + ({theta2:.4f} * fare_amount)")



# Print Feature Importance (they can only be approximated)
feature_importance_g = gb_model_g.feature_importances_

# Extract values for trip_distance and fare_amount
theta1, theta2 = float(feature_importance_g[0]), float(feature_importance_g[1])  # Convert to float values
theta0 = float(gb_model_g.init_.constant_.ravel()[0]) if hasattr(gb_model_g.init_, "constant_") else 0  # Extract first value properly as a scalar

# Print the Approximate Regression Formula
print(f"\nYellow Approximate Regression Formula:")
print(f"Yellow Tip Amount ≈ {theta0:.4f} + ({theta1:.4f} * trip_distance) + ({theta2:.4f} * fare_amount)")
