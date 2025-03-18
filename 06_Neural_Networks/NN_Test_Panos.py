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
data = data[['trip_distance', 'fare_amount', 'tip_amount']]

# Define input and output variables
X = data[['trip_distance', 'fare_amount']].values

# Normalize target variable (tip_amount) using StandardScaler
y = data[['tip_amount']].values

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
model.add(Dense(1, activation='linear'))  # Output a single continuous value

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
                    epochs=1, batch_size=32, callbacks=callbacks, verbose=1)

# Evaluate the model
loss, mae = model.evaluate(X_test, y_test)
print(f'Test Loss: {loss}, Test MAE: {mae}')

# Make predictions
y_pred = model.predict(X_test)  # Predictions in normalized scale
y_test_actual = y_test

# Compute errors
mse = mean_squared_error(y_test_actual, y_pred)
mae = mean_absolute_error(y_test_actual, y_pred)

print('MSE:', mse)
print('MAE:', mae)

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

# Compute Accuracy
accuracy = 1 - (mae / np.mean(y_test_actual))
print(f'Accuracy: {accuracy * 100:.2f}%')
