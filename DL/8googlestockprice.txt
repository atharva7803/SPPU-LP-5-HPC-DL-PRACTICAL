# Importing required libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN, Dropout

# Load training dataset
train_df = pd.read_csv('Google_Stock_Price_Train.csv')
train_df.info()  # Display information about training data

# Load testing dataset
test_df = pd.read_csv('Google_Stock_Price_Test.csv')
test_df.info()  # Display information about testing data

# Extract only the 'Open' prices for training
train = train_df.loc[:, ["Open"]].values

# Normalize the training data to range (0,1)
scaler = MinMaxScaler()
train_scaled = scaler.fit_transform(train)

# Plot the normalized training data
plt.plot(train_scaled)
plt.ylabel("Standardized Values")
plt.xlabel("Time →")
plt.title("Scaled Google Stock Opening Prices (Training Set)")
plt.show()

# Prepare sequences for RNN (60 timesteps to predict next)
x_train = []
y_train = []
time = 60  # Sequence length

# Loop to create input sequences (x_train) and corresponding output (y_train)
for i in range(time, train_scaled.shape[0]):
    x_train.append(train_scaled[i - time:i, 0])
    y_train.append(train_scaled[i, 0])

# Convert to NumPy arrays
x_train = np.array(x_train)
y_train = np.array(y_train)

# Reshape x_train for RNN input: (samples, timesteps, features)
x_train = np.reshape(x_train, newshape=(x_train.shape[0], x_train.shape[1], 1))

# ------------------------ RNN Model Building ------------------------

model = Sequential()

# First SimpleRNN layer with return_sequences=True to stack more RNNs
model.add(SimpleRNN(units=50, activation="tanh", return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(Dropout(0.2))  # Dropout for regularization

# Second RNN layer
model.add(SimpleRNN(units=50, activation="tanh", return_sequences=True))
model.add(Dropout(0.2))

# Third RNN layer
model.add(SimpleRNN(units=50, activation="tanh", return_sequences=True))
model.add(Dropout(0.2))

# Final RNN layer (no return_sequences since it's the last one)
model.add(SimpleRNN(units=50))
model.add(Dropout(0.2))

# Output layer (predict single value)
model.add(Dense(units=1))

# Compile the model with mean squared error loss
model.compile(optimizer='adam', loss='mse')

# Display model architecture
model.summary()

# Train the model on training data
model.fit(x_train, y_train, epochs=100, batch_size=30, validation_split=0.05)

# ------------------------ Prepare Test Data ------------------------

# Combine train and test data to get full time series context
data = pd.concat((train_df['Open'], test_df['Open']), axis=0)

# Create input for test data (last 60 from train + test set)
test_input = data.iloc[len(data) - len(test_df) - time:].values
test_input = test_input.reshape(-1, 1)

# Normalize test input using the same scaler
test_scaled = scaler.transform(test_input)

# Create test sequences (same format as training)
x_test = []
for i in range(time, test_scaled.shape[0]):
    x_test.append(test_scaled[i - time:i, 0])

x_test = np.array(x_test)
x_test = np.reshape(x_test, newshape=(x_test.shape[0], x_test.shape[1], 1))

# Actual test labels (ground truth)
y_test = test_df.loc[:, "Open"].values

# Predict the stock prices using trained model
y_pred = model.predict(x_test)

# Inverse transform predictions back to original price scale
y_pred = scaler.inverse_transform(y_pred)

# Evaluate model performance on test set
output = model.evaluate(x=x_test, y=y_test)

# ------------------------ Visualization ------------------------

# Plot real vs predicted stock prices
plt.plot(y_test, color='red', label='Real Google Stock Price')
plt.plot(y_pred, color='blue', label='Predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
plt.show()
