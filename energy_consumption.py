import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dense, Dropout, Conv1D, MaxPooling1D, Flatten
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Data loading and preprocessing
def load_data():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00235/household_power_consumption.zip"
    df = pd.read_csv(url, sep=';', parse_dates={'datetime': ['Date', 'Time']},
                     na_values=['?'], low_memory=False)
    
    # Convert to hourly data to reduce complexity
    df.set_index('datetime', inplace=True)
    df = df.resample('H').mean()
    df.fillna(method='ffill', inplace=True)
    return df

# Feature engineering
def create_features(df):
    df['hour'] = df.index.hour
    df['dayofweek'] = df.index.dayofweek
    df['month'] = df.index.month
    df['is_weekend'] = df.index.dayofweek.isin([5, 6]).astype(int)
    return df

# Sequence creation for time series
def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data.iloc[i:(i + seq_length)].values
        y = data.iloc[i + seq_length]['Global_active_power']
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

# Custom model architecture combining CNN and LSTM
def create_hybrid_model(input_shape):
    model = tf.keras.Sequential([
        # CNN layers for feature extraction
        Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape),
        MaxPooling1D(pool_size=2),
        Conv1D(filters=128, kernel_size=3, activation='relu'),
        MaxPooling1D(pool_size=2),
        
        # LSTM layers for temporal patterns
        LSTM(100, return_sequences=True),
        Dropout(0.2),
        LSTM(50),
        Dropout(0.2),
        
        # Dense layers for final prediction
        Dense(50, activation='relu'),
        Dense(1)
    ])
    return model

# Main execution
def main():
    # Load and preprocess data
    print("Loading data...")
    df = load_data()
    df = create_features(df)
    
    # Prepare features and target
    features = ['Global_active_power', 'Global_reactive_power', 'Voltage',
                'Global_intensity', 'hour', 'dayofweek', 'month', 'is_weekend']
    
    # Scale the data
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df[features])
    scaled_df = pd.DataFrame(scaled_data, columns=features, index=df.index)
    
    # Create sequences
    seq_length = 24  # 24 hours of data to predict next hour
    X, y = create_sequences(scaled_df, seq_length)
    
    # Split data
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # Create and compile model
    model = create_hybrid_model((seq_length, len(features)))
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    
    # Train model with early stopping
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=5, restore_best_weights=True)
    
    history = model.fit(
        X_train, y_train,
        epochs=20,
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stopping],
        verbose=1
    )
    
    # Evaluate and make predictions
    test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nTest MAE: {test_mae:.4f}")
    
    # Make predictions for visualization
    predictions = model.predict(X_test)
    
    # Plot results
    plt.figure(figsize=(12, 6))
    plt.plot(y_test[:100], label='Actual')
    plt.plot(predictions[:100], label='Predicted')
    plt.legend()
    plt.title('Actual vs Predicted Power Consumption')
    plt.show()

if __name__ == "__main__":
    main()