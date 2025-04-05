import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
from subprocess import check_output
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
from pandas.plotting import lag_plot
from datetime import datetime
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
warnings.filterwarnings('ignore')
data = pd.read_csv("/BHARTIAIRTEL.csv")
data.head()
data.drop(['Date'], axis=1, inplace=True)
print(data.head())
print(data.shape)
print(data.columns)
data.plot(legend=True,subplots=True, figsize = (12, 10))
plt.show()
#data['Close'].plot(legend=True, figsize = (12, 6))
#plt.show()
#data['Volume'].plot(legend=True,figsize=(12,7))
#plt.show()

data.shape
data.size
data.describe(include='all').T
data.dtypes
data.nunique()
data.reset_index(drop=True, inplace=True)
#data.fillna(data.mean(), inplace=True)
data.head()
data.nunique()

data.sort_index(axis=1,ascending=True)

cols_plot = ['Open', 'High', 'Low','Close','Volume']
axes = data[cols_plot].plot(marker='.', alpha=0.7, linestyle='None', figsize=(11, 9), subplots=True)
for ax in axes:
    ax.set_ylabel('Daily trade')

plt.plot(data['Close'], label="Close price")
plt.xlabel("Timestamp")
plt.ylabel("Closing price")
df = data
print(df)
data.isnull().sum()
cols_plot = ['Open', 'High', 'Low','Close']
axes = data[cols_plot].plot(marker='.', alpha=0.5, linestyle='None', figsize=(11, 9), subplots=True)
for ax in axes:
    ax.set_ylabel('Daily trade')
plt.plot(data['Close'], label="Close price")
plt.xlabel("Timestamp")
plt.ylabel("Closing price")
df = data
print(df)

df.describe().transpose()
## 5 fold
import numpy as np
import pandas as pd

import random

import os
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
# Function to set the seed
def set_random_seed(seed=42):
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    tf.config.experimental.enable_op_determinism()  # Ensure deterministic ops

# Set the random seed
set_random_seed(42)

# Load the dataset (ensure this is the correct dataset you want)
# Assuming the data is loaded as data with a "Close" column for the target variable
df = data[["Close"]]

# Scaling the data
scaler = MinMaxScaler()
df["Close"] = scaler.fit_transform(df)

# Function to create sequences
def create_sequences(data, time_steps=30):
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i : i + time_steps])
        y.append(data[i + time_steps])
    return np.array(X), np.array(y)

# Create sequences
X, y = create_sequences(df["Close"].values, time_steps=30)

# 5-Fold Time Series Split
tscv = TimeSeriesSplit(n_splits=5)

# Define LSTM Model
def build_lstm_model(time_step=30):
    model = Sequential([
        LSTM(units=100, return_sequences=True, input_shape=(time_step, 1)),  # First LSTM Layer
        Dropout(0.2),
        LSTM(units=50, return_sequences=False),  # Second LSTM Layer
        Dropout(0.2),
        Dense(units=1)
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
    return model

# Cross-validation loop
fold = 1
r2_scores, mse_scores, mae_scores = [], [], []

for train_index, test_index in tscv.split(X):
    print(f"\n🔹 Fold {fold}:")

    # Split data into train, validation, and test sets (60%-20%-20%)
    train_size = int(len(train_index) * 0.6)
    val_size = int(len(train_index) * 0.2)
    
    # Split data based on indices
    X_train, y_train = X[train_index[:train_size]], y[train_index[:train_size]]
    X_val, y_val = X[train_index[train_size:train_size + val_size]], y[train_index[train_size:train_size + val_size]]
    X_test, y_test = X[test_index], y[test_index]

    # Print the number of samples in train, validation, and test sets
    print(f"  Train size: {len(X_train)}")
    print(f"  Validation size: {len(X_val)}")
    print(f"  Test size: {len(X_test)}")

    # Reshape data for LSTM
    X_train = X_train.reshape(-1, 30, 1)
    X_val = X_val.reshape(-1, 30, 1)
    X_test = X_test.reshape(-1, 30, 1)

    # Build and train model
    model = build_lstm_model()
    early_stop = EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True, verbose=1)

    history = model.fit(X_train, y_train, epochs=100, validation_data=(X_val, y_val), verbose=1, batch_size=32, callbacks=[early_stop])

    # Predictions on test set
    y_pred = model.predict(X_test)

    # Inverse transform predictions and actual values
    y_pred_inv = scaler.inverse_transform(y_pred)
    y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))

    # Calculate Metrics
    r2 = r2_score(y_test_inv, y_pred_inv)
    mse = mean_squared_error(y_test_inv, y_pred_inv)
    mae = mean_absolute_error(y_test_inv, y_pred_inv)

    r2_scores.append(r2)
    mse_scores.append(mse)
    mae_scores.append(mae)

    print(f"📊 Fold {fold} - R²: {r2:.4f}, MSE: {mse:.4f}, MAE: {mae:.4f}")

    fold += 1

# Print Final Averages
print("\n📌 Final Cross-Validation Results:")
print(f"Mean R² Score: {np.mean(r2_scores):.4f}")
print(f"Mean MSE Score: {np.mean(mse_scores):.4f}")
print(f"Mean MAE Score: {np.mean(mae_scores):.4f}")
## 60-20-20
import numpy as np
import pandas as pd
import random
import tensorflow as tf
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

# Function to set the seed
def set_random_seed(seed=42):
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    tf.config.experimental.enable_op_determinism()  # Ensure deterministic ops

# Set the random seed
set_random_seed(42)

# Load the dataset (ensure this is the correct dataset you want)
df = data[["Close"]]

# Scaling the data
scaler = MinMaxScaler()
df["Close"] = scaler.fit_transform(df)

# Function to create sequences
def create_sequences(data, time_steps=30):
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i : i + time_steps])
        y.append(data[i + time_steps])
    return np.array(X), np.array(y)

# Create sequences
X, y = create_sequences(df["Close"].values, time_steps=30)

# Train-test split (60%-20%-20%)
train_size = int(len(X) * 0.6)
val_size = int(len(X) * 0.2)

X_train, y_train = X[:train_size], y[:train_size]
X_val, y_val = X[train_size:train_size + val_size], y[train_size:train_size + val_size]
X_test, y_test = X[train_size + val_size:], y[train_size + val_size:]

# Print the number of samples in train, validation, and test sets
print(f"Train size: {len(X_train)}")
print(f"Validation size: {len(X_val)}")
print(f"Test size: {len(X_test)}")

# Reshape data for LSTM
X_train = X_train.reshape(-1, 30, 1)
X_val = X_val.reshape(-1, 30, 1)
X_test = X_test.reshape(-1, 30, 1)

# Define LSTM Model
def build_lstm_model(time_step=30):
    model = Sequential([
        LSTM(units=100, return_sequences=True, input_shape=(time_step, 1)),
        Dropout(0.2),
        LSTM(units=50, return_sequences=False),
        Dropout(0.2),
        Dense(units=1)
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
    return model

# Build and train model
model = build_lstm_model()
early_stop = EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True, verbose=1)

history = model.fit(X_train, y_train, epochs=100, validation_data=(X_val, y_val), verbose=1, batch_size=32, callbacks=[early_stop])

# Predictions on test set
y_pred = model.predict(X_test)

# Inverse transform predictions and actual values
y_pred_inv = scaler.inverse_transform(y_pred)
y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))

# Calculate Metrics
r2 = r2_score(y_test_inv, y_pred_inv)
mse = mean_squared_error(y_test_inv, y_pred_inv)
mae = mean_absolute_error(y_test_inv, y_pred_inv)

# Print the results
print(f"📊 Final Results - R²: {r2:.4f}, MSE: {mse:.4f}, MAE: {mae:.4f}")
import pandas as pd
data = pd.read_csv("/kaggle/input/nifty50-stock-market-data/BHARTIARTL.csv")
print(data.head())


# Convert the date column to datetime format
data['Date'] = pd.to_datetime(data['Date'])

# Define the date range
start_date = '2017-01-03'
end_date = '2021-04-15'

# Filter the DataFrame
data = data[(data['Date'] >= start_date) & (data['Date'] <= end_date)]

# Display the filtered DataFrame
print(data)
data.drop(['Date'], axis=1, inplace=True)
data.nunique()
data.reset_index(drop=True, inplace=True)
#data.fillna(data.mean(), inplace=True)
data.head()
data.nunique()

data.sort_index(axis=1,ascending=True)
df = data
print(df)

df.describe().transpose()
## 60-20-20
import numpy as np
import pandas as pd
import random
import tensorflow as tf
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

# Function to set the seed
def set_random_seed(seed=42):
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    tf.config.experimental.enable_op_determinism()  # Ensure deterministic ops

# Set the random seed
set_random_seed(42)

# Load the dataset (ensure this is the correct dataset you want)
df = data[["Close"]]

# Scaling the data
scaler = MinMaxScaler()
df["Close"] = scaler.fit_transform(df)

# Function to create sequences
def create_sequences(data, time_steps=30):
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i : i + time_steps])
        y.append(data[i + time_steps])
    return np.array(X), np.array(y)

# Create sequences
X, y = create_sequences(df["Close"].values, time_steps=30)

# Train-test split (60%-20%-20%)
train_size = int(len(X) * 0.6)
val_size = int(len(X) * 0.2)

X_train, y_train = X[:train_size], y[:train_size]
X_val, y_val = X[train_size:train_size + val_size], y[train_size:train_size + val_size]
X_test, y_test = X[train_size + val_size:], y[train_size + val_size:]

# Print the number of samples in train, validation, and test sets
print(f"Train size: {len(X_train)}")
print(f"Validation size: {len(X_val)}")
print(f"Test size: {len(X_test)}")

# Reshape data for LSTM
X_train = X_train.reshape(-1, 30, 1)
X_val = X_val.reshape(-1, 30, 1)
X_test = X_test.reshape(-1, 30, 1)

# Define LSTM Model
def build_lstm_model(time_step=30):
    model = Sequential([
        LSTM(units=100, return_sequences=True, input_shape=(time_step, 1)),
        Dropout(0.2),
        LSTM(units=50, return_sequences=False),
        Dropout(0.2),
        Dense(units=1)
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
    return model

# Build and train model
model = build_lstm_model()
early_stop = EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True, verbose=1)

history = model.fit(X_train, y_train, epochs=100, validation_data=(X_val, y_val), verbose=1, batch_size=32, callbacks=[early_stop])

# Predictions on test set
y_pred = model.predict(X_test)

# Inverse transform predictions and actual values
y_pred_inv = scaler.inverse_transform(y_pred)
y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))

# Calculate Metrics
r2 = r2_score(y_test_inv, y_pred_inv)
mse = mean_squared_error(y_test_inv, y_pred_inv)
mae = mean_absolute_error(y_test_inv, y_pred_inv)

# Print the results
print(f"📊 Final Results - R²: {r2:.4f}, MSE: {mse:.4f}, MAE: {mae:.4f}")
from tensorflow.keras.utils import plot_model

# Print model summary
model.summary()

# Save and display the model architecture diagram
plot_model(model, to_file="model_structure.png", show_shapes=True, show_layer_names=True)

# Plot actual vs. predicted values
plt.figure(figsize=(12, 6))
plt.plot(y_test_inv, label="Actual", color="blue", linewidth=2)
plt.plot(y_pred_inv, label="Predicted", color="red", linestyle="dashed", linewidth=2)
plt.xlabel("Time")
plt.ylabel("Stock Price")
plt.title("Actual vs. Predicted Stock Prices")
plt.legend()
plt.grid()
plt.show()
import pandas as pd
data = pd.read_csv("/kaggle/input/nifty50-stock-market-data/BHARTIARTL.csv")
print(data.head())


# Convert the date column to datetime format
data['Date'] = pd.to_datetime(data['Date'])

# Define the date range
start_date = '2017-01-03'
end_date = '2021-04-15'

# Filter the DataFrame
data = data[(data['Date'] >= start_date) & (data['Date'] <= end_date)]

# Display the filtered DataFrame
print(data)



# Load the VIX dataset
vix = pd.read_csv("/kaggle/input/vix-data/vix.csv")

# Ensure both datasets have a 'date' column in datetime format
vix['Date'] = pd.to_datetime(vix['Date'])
data['Date'] = pd.to_datetime(data['Date'])

# Filter VIX to match the date range of 'data'
start_date = data['Date'].min()
end_date = data['Date'].max()
vix_filtered = vix[(vix['Date'] >= start_date) & (vix['Date'] <= end_date)]

# Merge datasets on 'date'
data = pd.merge(data, vix_filtered, on='Date', how='inner')

# Display the merged dataset
print(data.head())
data.drop(['Date'], axis=1, inplace=True)
data.nunique()
data.reset_index(drop=True, inplace=True)
#data.fillna(data.mean(), inplace=True)
data.head()
data.nunique()

data.sort_index(axis=1,ascending=True)
df = data
print(df)

df.describe().transpose()
data.head()
data.isnull().sum()
data.drop(['Vol.'],axis=1,inplace=True)
## 60-20-20
import numpy as np
import pandas as pd
import random
import tensorflow as tf
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

# Function to set the seed
def set_random_seed(seed=42):
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    tf.config.experimental.enable_op_determinism()  # Ensure deterministic ops

# Set the random seed
set_random_seed(42)

# Load the dataset (ensure this is the correct dataset you want)
df = data[["Close"]]

# Scaling the data
scaler = MinMaxScaler()
df["Close"] = scaler.fit_transform(df)

# Function to create sequences
def create_sequences(data, time_steps=30):
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i : i + time_steps])
        y.append(data[i + time_steps])
    return np.array(X), np.array(y)

# Create sequences
X, y = create_sequences(df["Close"].values, time_steps=30)

# Train-test split (60%-20%-20%)
train_size = int(len(X) * 0.6)
val_size = int(len(X) * 0.2)

X_train, y_train = X[:train_size], y[:train_size]
X_val, y_val = X[train_size:train_size + val_size], y[train_size:train_size + val_size]
X_test, y_test = X[train_size + val_size:], y[train_size + val_size:]

# Print the number of samples in train, validation, and test sets
print(f"Train size: {len(X_train)}")
print(f"Validation size: {len(X_val)}")
print(f"Test size: {len(X_test)}")

# Reshape data for LSTM
X_train = X_train.reshape(-1, 30, 1)
X_val = X_val.reshape(-1, 30, 1)
X_test = X_test.reshape(-1, 30, 1)

# Define LSTM Model
def build_lstm_model(time_step=30):
    model = Sequential([
        LSTM(units=100, return_sequences=True, input_shape=(time_step, 1)),
        Dropout(0.2),
        LSTM(units=50, return_sequences=False),
        Dropout(0.2),
        Dense(units=1)
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
    return model

# Build and train model
model = build_lstm_model()
early_stop = EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True, verbose=1)

history = model.fit(X_train, y_train, epochs=100, validation_data=(X_val, y_val), verbose=1, batch_size=32, callbacks=[early_stop])

# Predictions on test set
y_pred = model.predict(X_test)

# Inverse transform predictions and actual values
y_pred_inv = scaler.inverse_transform(y_pred)
y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))

# Calculate Metrics
r2 = r2_score(y_test_inv, y_pred_inv)
mse = mean_squared_error(y_test_inv, y_pred_inv)
mae = mean_absolute_error(y_test_inv, y_pred_inv)

# Print the results
print(f"📊 Final Results - R²: {r2:.4f}, MSE: {mse:.4f}, MAE: {mae:.4f}")
from tensorflow.keras.utils import plot_model

# Print model summary
model.summary()

# Save and display the model architecture diagram
plot_model(model, to_file="model_structure.png", show_shapes=True, show_layer_names=True)

# Plot actual vs. predicted values
plt.figure(figsize=(12, 6))
plt.plot(y_test_inv, label="Actual", color="blue", linewidth=2)
plt.plot(y_pred_inv, label="Predicted", color="red", linestyle="dashed", linewidth=2)
plt.xlabel("Time")
plt.ylabel("Stock Price")
plt.title("Actual vs. Predicted Stock Prices")
plt.legend()
plt.grid()
plt.show()
import pandas as pd
data = pd.read_csv("/kaggle/input/nifty50-stock-market-data/BHARTIARTL.csv")
print(data.head())


# Convert the date column to datetime format
data['Date'] = pd.to_datetime(data['Date'])

# Define the date range
start_date = '2017-01-03'
end_date = '2021-04-15'

# Filter the DataFrame
data = data[(data['Date'] >= start_date) & (data['Date'] <= end_date)]

# Display the filtered DataFrame
print(data)
data.drop(['Date'], axis=1, inplace=True)
data.nunique()
data.reset_index(drop=True, inplace=True)
#data.fillna(data.mean(), inplace=True)
data.head()
data.nunique()

data.sort_index(axis=1,ascending=True)
## 80-20
import numpy as np
import pandas as pd
import random
import tensorflow as tf
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

# Function to set the seed
def set_random_seed(seed=42):
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    tf.config.experimental.enable_op_determinism()  # Ensure deterministic ops

# Set the random seed
set_random_seed(42)

# Load the dataset (ensure this is the correct dataset you want)
df = data[["Close"]]

# Scaling the data
scaler = MinMaxScaler()
df["Close"] = scaler.fit_transform(df)

# Function to create sequences
def create_sequences(data, time_steps=30):
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i : i + time_steps])
        y.append(data[i + time_steps])
    return np.array(X), np.array(y)

# Create sequences
X, y = create_sequences(df["Close"].values, time_steps=30)

# Train-test split (60%-20%-20%)
train_size = int(len(X) * 0.8)
val_size = int(len(X) * 0.0)

X_train, y_train = X[:train_size], y[:train_size]
X_val, y_val = X[train_size:train_size + val_size], y[train_size:train_size + val_size]
X_test, y_test = X[train_size + val_size:], y[train_size + val_size:]

# Print the number of samples in train, validation, and test sets
print(f"Train size: {len(X_train)}")
print(f"Validation size: {len(X_val)}")
print(f"Test size: {len(X_test)}")

# Reshape data for LSTM
X_train = X_train.reshape(-1, 30, 1)
X_val = X_val.reshape(-1, 30, 1)
X_test = X_test.reshape(-1, 30, 1)

# Define LSTM Model
def build_lstm_model(time_step=30):
    model = Sequential([
        LSTM(units=100, return_sequences=True, input_shape=(time_step, 1)),
        Dropout(0.2),
        LSTM(units=50, return_sequences=False),
        Dropout(0.2),
        Dense(units=1)
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
    return model

# Build and train model
model = build_lstm_model()
early_stop = EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True, verbose=1)

history = model.fit(X_train, y_train, epochs=100, verbose=1, batch_size=32, callbacks=[early_stop])

# Predictions on test set
y_pred = model.predict(X_test)

# Inverse transform predictions and actual values
y_pred_inv = scaler.inverse_transform(y_pred)
y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))

# Calculate Metrics
r2 = r2_score(y_test_inv, y_pred_inv)
mse = mean_squared_error(y_test_inv, y_pred_inv)
mae = mean_absolute_error(y_test_inv, y_pred_inv)

# Print the results
print(f"📊 Final Results - R²: {r2:.4f}, MSE: {mse:.4f}, MAE: {mae:.4f}")
from tensorflow.keras.utils import plot_model

# Print model summary
model.summary()

# Save and display the model architecture diagram
plot_model(model, to_file="model_structure.png", show_shapes=True, show_layer_names=True)

# Plot actual vs. predicted values
plt.figure(figsize=(12, 6))
plt.plot(y_test_inv, label="Actual", color="blue", linewidth=2)
plt.plot(y_pred_inv, label="Predicted", color="red", linestyle="dashed", linewidth=2)
plt.xlabel("Time")
plt.ylabel("Stock Price")
plt.title("Actual vs. Predicted Stock Prices")
plt.legend()
plt.grid()
plt.show()
import pandas as pd
data = pd.read_csv("/kaggle/input/nifty50-stock-market-data/BHARTIARTL.csv")
print(data.head())


# Convert the date column to datetime format
data['Date'] = pd.to_datetime(data['Date'])

# Define the date range
start_date = '2017-01-03'
end_date = '2021-04-15'

# Filter the DataFrame
data = data[(data['Date'] >= start_date) & (data['Date'] <= end_date)]

# Display the filtered DataFrame
print(data)



# Load the VIX dataset
vix = pd.read_csv("/kaggle/input/vix-data/vix.csv")

# Ensure both datasets have a 'date' column in datetime format
vix['Date'] = pd.to_datetime(vix['Date'])
data['Date'] = pd.to_datetime(data['Date'])

# Filter VIX to match the date range of 'data'
start_date = data['Date'].min()
end_date = data['Date'].max()
vix_filtered = vix[(vix['Date'] >= start_date) & (vix['Date'] <= end_date)]

# Merge datasets on 'date'
data = pd.merge(data, vix_filtered, on='Date', how='inner')

# Display the merged dataset
print(data.head())
data.drop(['Date'], axis=1, inplace=True)
data.nunique()
data.reset_index(drop=True, inplace=True)
#data.fillna(data.mean(), inplace=True)
data.head()
data.nunique()

data.sort_index(axis=1,ascending=True)
df = data
print(df)

df.describe().transpose()
data.drop(['Vol.'],axis=1,inplace=True)
## 80-20
import numpy as np
import pandas as pd
import random
import tensorflow as tf
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

# Function to set the seed
def set_random_seed(seed=42):
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    tf.config.experimental.enable_op_determinism()  # Ensure deterministic ops

# Set the random seed
set_random_seed(42)

# Load the dataset (ensure this is the correct dataset you want)
df = data[["Close"]]

# Scaling the data
scaler = MinMaxScaler()
df["Close"] = scaler.fit_transform(df)

# Function to create sequences
def create_sequences(data, time_steps=30):
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i : i + time_steps])
        y.append(data[i + time_steps])
    return np.array(X), np.array(y)

# Create sequences
X, y = create_sequences(df["Close"].values, time_steps=30)

# Train-test split (60%-20%-20%)
train_size = int(len(X) * 0.8)
val_size = int(len(X) * 0.0)

X_train, y_train = X[:train_size], y[:train_size]
X_val, y_val = X[train_size:train_size + val_size], y[train_size:train_size + val_size]
X_test, y_test = X[train_size + val_size:], y[train_size + val_size:]

# Print the number of samples in train, validation, and test sets
print(f"Train size: {len(X_train)}")
print(f"Validation size: {len(X_val)}")
print(f"Test size: {len(X_test)}")

# Reshape data for LSTM
X_train = X_train.reshape(-1, 30, 1)
X_val = X_val.reshape(-1, 30, 1)
X_test = X_test.reshape(-1, 30, 1)

# Define LSTM Model
def build_lstm_model(time_step=30):
    model = Sequential([
        LSTM(units=100, return_sequences=True, input_shape=(time_step, 1)),
        Dropout(0.2),
        LSTM(units=50, return_sequences=False),
        Dropout(0.2),
        Dense(units=1)
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
    return model

# Build and train model
model = build_lstm_model()
early_stop = EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True, verbose=1)

history = model.fit(X_train, y_train, epochs=100, validation_data=(X_test, y_test), verbose=1, batch_size=32, callbacks=[early_stop])

# Predictions on test set
y_pred = model.predict(X_test)

# Inverse transform predictions and actual values
y_pred_inv = scaler.inverse_transform(y_pred)
y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))

# Calculate Metrics
r2 = r2_score(y_test_inv, y_pred_inv)
mse = mean_squared_error(y_test_inv, y_pred_inv)
mae = mean_absolute_error(y_test_inv, y_pred_inv)

# Print the results
print(f"📊 Final Results - R²: {r2:.4f}, MSE: {mse:.4f}, MAE: {mae:.4f}")
from tensorflow.keras.utils import plot_model

# Print model summary
model.summary()

# Save and display the model architecture diagram
plot_model(model, to_file="model_structure.png", show_shapes=True, show_layer_names=True)

# Plot actual vs. predicted values
plt.figure(figsize=(12, 6))
plt.plot(y_test_inv, label="Actual", color="blue", linewidth=2)
plt.plot(y_pred_inv, label="Predicted", color="red", linestyle="dashed", linewidth=2)
plt.xlabel("Time")
plt.ylabel("Stock Price")
plt.title("Actual vs. Predicted Stock Prices")
plt.legend()
plt.grid()
plt.show()
import pandas as pd
data = pd.read_csv("/kaggle/input/nifty50-stock-market-data/BHARTIARTL.csv")
print(data.head())


# Convert the date column to datetime format
data['Date'] = pd.to_datetime(data['Date'])

# Define the date range
#start_date = '2017-01-03'
#end_date = '2021-04-15'

# Filter the DataFrame
#data = data[(data['Date'] >= start_date) & (data['Date'] <= end_date)]

# Display the filtered DataFrame
print(data)



# Load the VIX dataset
vix = pd.read_csv("/kaggle/input/vix-data/vix.csv")

# Ensure both datasets have a 'date' column in datetime format
vix['Date'] = pd.to_datetime(vix['Date'])
data['Date'] = pd.to_datetime(data['Date'])

# Filter VIX to match the date range of 'data'
start_date = data['Date'].min()
end_date = data['Date'].max()
print(start_date,end_date)
vix_filtered = vix[(vix['Date'] >= start_date) & (vix['Date'] <= end_date)]
print(len(vix_filtered))
# Merge datasets on 'date'
data = pd.merge(data, vix_filtered, on='Date', how='inner')

# Display the merged dataset
print(data.head())
data.drop(['Date'], axis=1, inplace=True)
data.nunique()
data.reset_index(drop=True, inplace=True)
#data.fillna(data.mean(), inplace=True)
data.head()
data.nunique()

data.sort_index(axis=1,ascending=True)
df = data
print(df)

df.describe().transpose()

data.drop(['Vol.'],axis=1,inplace=True)
print(len(data))
3251
import numpy as np
import pandas as pd
import random
import tensorflow as tf
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout,Flatten
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

# Function to set the seed
def set_random_seed(seed=42):
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    tf.config.experimental.enable_op_determinism()  # Ensure deterministic ops

# Set the random seed
set_random_seed(42)

# Load the dataset (ensure this is the correct dataset you want)
df = data[["Close"]]

# Scaling the data
scaler = MinMaxScaler()
df["Close"] = scaler.fit_transform(df)

# Function to create sequences
def create_sequences(data, time_steps=30):
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i : i + time_steps])
        y.append(data[i + time_steps])
    return np.array(X), np.array(y)

# Create sequences
X, y = create_sequences(df["Close"].values, time_steps=30)

# Train-test split (70%-10%-20%)
train_size = int(len(X) * 0.7)
val_size = int(len(X) * 0.1)

X_train, y_train = X[:train_size], y[:train_size]
X_val, y_val = X[train_size:train_size + val_size], y[train_size:train_size + val_size]
X_test, y_test = X[train_size + val_size:], y[train_size + val_size:]

# Print the number of samples in train, validation, and test sets
print(f"Train size: {len(X_train)}")
print(f"Validation size: {len(X_val)}")
print(f"Test size: {len(X_test)}")

# Reshape data for LSTM
X_train = X_train.reshape(-1, 30, 1)
X_val = X_val.reshape(-1, 30, 1)
X_test = X_test.reshape(-1, 30, 1)

# Define LSTM-DNN Model
def build_lstm_dnn_model(time_step=30, feature_dim=1):
    model = Sequential([
        LSTM(16, activation='tanh', return_sequences=True, input_shape=(time_step, feature_dim)),
        LSTM(32, activation='tanh'),
        Dense(64, activation='relu'),
        Dense(64, activation='relu'),
        Dense(64, activation='relu'),
        Flatten(),
        Dense(1)  # Output layer should have 1 neuron
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
    return model
# Build and train model
model = build_lstm_dnn_model()
early_stop = EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True, verbose=1)

history = model.fit(X_train, y_train, epochs=100, validation_data=(X_val, y_val), verbose=1, batch_size=64, callbacks=[early_stop])

# Predictions on test set
y_pred = model.predict(X_test)

# Inverse transform predictions and actual values
y_pred_inv = scaler.inverse_transform(y_pred)
y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))

# Calculate Metrics
r2 = r2_score(y_test_inv, y_pred_inv)
mse = mean_squared_error(y_test_inv, y_pred_inv)
mae = mean_absolute_error(y_test_inv, y_pred_inv)

# Print the results
print(f"📊 Final Results - R²: {r2:.4f}, MSE: {mse:.4f}, MAE: {mae:.4f}")

from tensorflow.keras.utils import plot_model
print(model.summary())
plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)
Model: "sequential"

import numpy as np
import pandas as pd
import random
import tensorflow as tf
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout,Flatten
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

# Function to set the seed
def set_random_seed(seed=42):
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    tf.config.experimental.enable_op_determinism()  # Ensure deterministic ops

# Set the random seed
set_random_seed(42)

# Load the dataset (ensure this is the correct dataset you want)
df = data[["Close"]]

# Scaling the data
scaler = MinMaxScaler()
df["Close"] = scaler.fit_transform(df)

# Function to create sequences
def create_sequences(data, time_steps=30):
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i : i + time_steps])
        y.append(data[i + time_steps])
    return np.array(X), np.array(y)

# Create sequences
X, y = create_sequences(df["Close"].values, time_steps=30)

# Train-test split (70%-10%-20%)
train_size = int(len(X) * 0.7)
val_size = int(len(X) * 0.1)

X_train, y_train = X[:train_size], y[:train_size]
X_val, y_val = X[train_size:train_size + val_size], y[train_size:train_size + val_size]
X_test, y_test = X[train_size + val_size:], y[train_size + val_size:]

# Print the number of samples in train, validation, and test sets
print(f"Train size: {len(X_train)}")
print(f"Validation size: {len(X_val)}")
print(f"Test size: {len(X_test)}")

# Reshape data for LSTM
X_train = X_train.reshape(-1, 30, 1)
X_val = X_val.reshape(-1, 30, 1)
X_test = X_test.reshape(-1, 30, 1)

# Define LSTM-DNN Model
def build_lstm_dnn_model(time_step=30, feature_dim=1):
    model = Sequential([
        LSTM(16, activation='tanh', return_sequences=True, input_shape=(time_step, feature_dim)),
        LSTM(32, activation='tanh'),
        Dense(64, activation='relu'),
        Dense(64, activation='relu'),
        Dense(64, activation='relu'),
        Flatten(),
        Dense(1)  # Output layer should have 1 neuron
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
    return model
# Build and train model
model = build_lstm_dnn_model()
early_stop = EarlyStopping(monitor="val_loss", patience=15, restore_best_weighsts=True, verbose=1)

history = model.fit(X_train, y_train, epochs=100, validation_data=(X_val, y_val), verbose=1, batch_size=64)

# Predictions on test set
y_pred = model.predict(X_test)

# Inverse transform predictions and actual values
y_pred_inv = scaler.inverse_transform(y_pred)
y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))

# Calculate Metrics
r2 = r2_score(y_test_inv, y_pred_inv)
mse = mean_squared_error(y_test_inv, y_pred_inv)
mae = mean_absolute_error(y_test_inv, y_pred_inv)

# Print the results
print(f"📊 Final Results - R²: {r2:.4f}, MSE: {mse:.4f}, MAE: {mae:.4f}")

