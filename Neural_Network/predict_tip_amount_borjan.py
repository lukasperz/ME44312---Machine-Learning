import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

def predict_taxi_tips(yellow_taxi_path, green_taxi_path):
    # Load data
    yellow_taxi_df = pd.read_parquet(yellow_taxi_path)
    green_taxi_df = pd.read_parquet(green_taxi_path)
    
    # Standardizing column names
    yellow_taxi_df.columns = yellow_taxi_df.columns.str.lower()
    green_taxi_df.columns = green_taxi_df.columns.str.lower()
    
    # Selecting common relevant columns
    common_columns = list(set(yellow_taxi_df.columns) & set(green_taxi_df.columns))
    
    # Combine datasets
    data = pd.concat([yellow_taxi_df[common_columns], green_taxi_df[common_columns]], ignore_index=True)
    
    # Selecting relevant features for prediction
    features = ['trip_distance', 'passenger_count', 'fare_amount', 'extra', 'mta_tax', 'tolls_amount', 'total_amount', 'payment_type']
    
    data = data.dropna(subset=['tip_amount'])  # Drop rows with missing tip values
    
    # Filtering necessary columns
    data = data[features + ['tip_amount']]
    
    # Handling categorical variables
    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    categorical_features = ['payment_type']
    encoded_cats = encoder.fit_transform(data[categorical_features])
    categorical_df = pd.DataFrame(encoded_cats, columns=encoder.get_feature_names_out(categorical_features))
    
    data = pd.concat([data.drop(columns=categorical_features), categorical_df], axis=1)
    
    # Splitting data
    X = data.drop(columns=['tip_amount'])
    y = data['tip_amount']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scaling numerical features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Define and train neural network model
    model = MLPRegressor(hidden_layer_sizes=(64, 32), activation='relu', solver='adam', max_iter=500, random_state=42)
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_test)
    
    # Evaluation
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    
    print(f'Mean Absolute Error: {mae}')
    print(f'Root Mean Squared Error: {rmse}')
    
    return model

# Example usage:
if __name__ == "__main__":
    model = predict_taxi_tips("C:/Users/btraj/0_TUDELFT_PythonExercises/ML/NN_Borjan/yellow_taxi_data.parquet", "C:/Users/btraj/0_TUDELFT_PythonExercises/ML/NN_Borjan/green_taxi_data.parquet")