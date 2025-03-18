import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score

# Define feature sets to evaluate
feature_sets = ["full", "fare_trip", "payment_passenger", "location", "no_location", "minimal"]

for FEATURE_SET in feature_sets:
    print(f"\nEvaluating feature set: {FEATURE_SET}")

    # Load trained models from the automatically created directory
    model_dir = f"NN_Models_{FEATURE_SET}"
    with open(os.path.join(model_dir, f"nn_green_{FEATURE_SET}.pkl"), "rb") as f:
        nn_green = pickle.load(f)
    with open(os.path.join(model_dir, f"nn_yellow_{FEATURE_SET}.pkl"), "rb") as f:
        nn_yellow = pickle.load(f)
    
    # Load test data from the automatically created directory
    X_test_green = np.load(os.path.join(model_dir, f"X_test_green_{FEATURE_SET}.npy"))
    y_test_green = np.load(os.path.join(model_dir, f"y_test_green_{FEATURE_SET}.npy"))
    X_test_yellow = np.load(os.path.join(model_dir, f"X_test_yellow_{FEATURE_SET}.npy"))
    y_test_yellow = np.load(os.path.join(model_dir, f"y_test_yellow_{FEATURE_SET}.npy"))

    # Make predictions
    y_pred_green = nn_green.predict(X_test_green)
    y_pred_yellow = nn_yellow.predict(X_test_yellow)

    # Evaluation Metrics
    rmse_green = np.sqrt(mean_squared_error(y_test_green, y_pred_green))
    r2_green = r2_score(y_test_green, y_pred_green)

    rmse_yellow = np.sqrt(mean_squared_error(y_test_yellow, y_pred_yellow))
    r2_yellow = r2_score(y_test_yellow, y_pred_yellow)

    print(f"Green Taxi Model - RMSE: {rmse_green:.2f}, R²: {r2_green:.2f} for feature set: {FEATURE_SET}")
    print(f"Yellow Taxi Model - RMSE: {rmse_yellow:.2f}, R²: {r2_yellow:.2f} for feature set: {FEATURE_SET}")

    # Visualization
    plt.figure(figsize=(12, 6))

    # Scatter Plot: Actual vs Predicted
    plt.subplot(1, 2, 1)
    plt.scatter(y_test_green, y_pred_green, alpha=0.5, label="Green Taxis", color="green")
    plt.scatter(y_test_yellow, y_pred_yellow, alpha=0.5, label="Yellow Taxis", color="gold")
    plt.plot([min(y_test_green), max(y_test_green)], [min(y_test_green), max(y_test_green)], 'k--', lw=2)
    plt.xlabel("Actual Fare")
    plt.ylabel("Predicted Fare")
    plt.legend()
    plt.title(f"Predicted vs Actual Taxi Fares - {FEATURE_SET}")

    # Residual Histogram
    plt.subplot(1, 2, 2)
    plt.hist(y_test_green - y_pred_green, bins=30, alpha=0.5, label="Green Taxis", color="green")
    plt.hist(y_test_yellow - y_pred_yellow, bins=30, alpha=0.5, label="Yellow Taxis", color="gold")
    plt.xlabel("Prediction Error")
    plt.ylabel("Frequency")
    plt.legend()
    plt.title(f"Residual Error Distribution - {FEATURE_SET}")

    plt.tight_layout()
    plt.show()

    # Interpretation
    print("\nInterpretation:")
    print(f"- The RMSE values indicate how far off predictions are in terms of fare prices for feature set: {FEATURE_SET}.")
    print(f"- The R² score shows how well the model explains fare variability (1 is perfect, 0 means no predictive power) for feature set: {FEATURE_SET}.")
    print(f"- The scatter plot helps visualize if predictions align with actual fares for feature set: {FEATURE_SET}.")
    print(f"- The residual histogram checks for bias in predictions; a symmetric shape is ideal for feature set: {FEATURE_SET}.")
