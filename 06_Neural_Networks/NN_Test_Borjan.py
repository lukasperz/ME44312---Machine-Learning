import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, Dropout, ELU
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# Load the dataset
path = r'C:\Users\infra\PycharmProjects\ME44312---Machine-Learning\green_taxi_data.parquet'
data = pd.read_parquet(path)

# Data Preprocessing
# Select relevant columns and filter invalid values
data = data[['trip_distance', 'fare_amount', 'passenger_count', 'tip_amount']]
data = data[(data['fare_amount'] > 0) & (data['trip_distance'] > 0) & (data['passenger_count'] > 0)]

# Feature Engineering
data['fare_per_mile'] = data['fare_amount'] / data['trip_distance']
data.fillna(0, inplace=True)

# Define input and output variables
X = data[['trip_distance', 'fare_amount', 'passenger_count', 'fare_per_mile']].values
y = np.log1p(data[['tip_amount']].values)  # Apply log transformation to stabilize variance

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale input features
scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# Define the Keras Neural Network model
model = Sequential()
model.add(Dense(256, input_shape=(X_train.shape[1],)))
model.add(BatchNormalization())
model.add(ELU(alpha=0.1))
model.add(Dropout(0.3))

model.add(Dense(128))
model.add(BatchNormalization())
model.add(ELU(alpha=0.1))
model.add(Dropout(0.3))

model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='linear'))

# Compile the model
model.compile(loss='mse', optimizer='adam', metrics=['mae'])

# Define callbacks for early stopping and learning rate adjustment
callbacks = [
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6),
    ModelCheckpoint("best_model.h5", save_best_only=True)
]

# Train the model
history = model.fit(X_train, y_train, validation_data=(X_test, y_test),
                    epochs=3, batch_size=32, callbacks=callbacks, verbose=1)

# Evaluate the model
loss, mae = model.evaluate(X_test, y_test)
print(f'Test Loss: {loss}, Test MAE: {mae}')

# Make predictions
y_pred = np.expm1(model.predict(X_test))  # Convert predictions back to original scale
y_test_actual = np.expm1(y_test)

# Compute errors
print('MSE:', mean_squared_error(y_test_actual, y_pred))
print('MAE:', mean_absolute_error(y_test_actual, y_pred))

# Plot loss curve
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Loss Curve")
plt.show()

# Plot Predictions vs Actual Values
plt.scatter(y_test_actual, y_pred, alpha=0.5)
plt.xlabel("Actual Tip Amount")
plt.ylabel("Predicted Tip Amount")
plt.title("Predicted vs Actual Tip Amount")
plt.plot([min(y_test_actual), max(y_test_actual)], [min(y_test_actual), max(y_test_actual)], color='red')  # 45-degree line
plt.show()

# Prompt for accuracy
accuracy = 1 - (mae / np.mean(y_test_actual))
print(f'Accuracy: {accuracy * 100:.2f}%')