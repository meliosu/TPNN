import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow import keras
from typing import Tuple, List, Dict, Any

def build_simple_rnn(sequence_length: int, learning_rate: float = 0.01) -> keras.Model:
    model = keras.Sequential([
        keras.layers.RNN(units=50, return_sequences=True, 
                         input_shape=(sequence_length, 1)),
        keras.layers.Dropout(0.2),
        keras.layers.RNN(units=50),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(1)
    ])
    
    model.compile(
        optimizer=keras.optimizers.SGD(learning_rate=learning_rate),
        loss='mean_squared_error'
    )
    
    return model

def build_gru(sequence_length: int, learning_rate: float = 0.01) -> keras.Model:
    model = keras.Sequential([
        keras.layers.GRU(units=50, return_sequences=True, 
                         input_shape=(sequence_length, 1)),
        keras.layers.Dropout(0.2),
        keras.layers.GRU(units=50),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(1)
    ])
    
    model.compile(
        optimizer=keras.optimizers.SGD(learning_rate=learning_rate),
        loss='mean_squared_error'
    )
    
    return model

def build_lstm(sequence_length: int, learning_rate: float = 0.01) -> keras.Model:
    model = keras.Sequential([
        keras.layers.LSTM(units=50, return_sequences=True, 
                          input_shape=(sequence_length, 1)),
        keras.layers.Dropout(0.2),
        keras.layers.LSTM(units=50),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(1)
    ])
    
    model.compile(
        optimizer=keras.optimizers.SGD(learning_rate=learning_rate),
        loss='mean_squared_error'
    )
    
    return model

def train_and_evaluate(
    model: keras.Model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    scaler: MinMaxScaler,
    epochs: int = 50,
    batch_size: int = 32,
    verbose: int = 1
) -> Dict[str, Any]:
    # Train the model
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.1,
        verbose=verbose
    )
    
    # Predict
    y_pred = model.predict(X_test)
    
    # Inverse transform the scaled data
    y_test_inv = scaler.inverse_transform(y_test)
    y_pred_inv = scaler.inverse_transform(y_pred)
    
    # Calculate metrics
    mse = mean_squared_error(y_test_inv, y_pred_inv)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test_inv, y_pred_inv)
    
    return {
        'history': history,
        'predictions': y_pred_inv,
        'actual': y_test_inv,
        'mse': mse,
        'rmse': rmse,
        'mae': mae
    }

def compare_models(results: Dict[str, Dict[str, Any]]) -> None:
    # Plot training & validation loss
    plt.figure(figsize=(15, 8))
    
    for i, (model_name, result) in enumerate(results.items()):
        plt.subplot(2, 2, i+1)
        plt.plot(result['history'].history['loss'], label='Train')
        plt.plot(result['history'].history['val_loss'], label='Validation')
        plt.title(f'{model_name} - Loss')
        plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_loss_comparison.png')
    plt.close()
    
    # Plot predictions vs actual for each model
    plt.figure(figsize=(15, 12))
    
    for i, (model_name, result) in enumerate(results.items()):
        plt.subplot(3, 1, i+1)
        plt.plot(result['actual'], label='Actual')
        plt.plot(result['predictions'], label='Predicted')
        plt.title(f'{model_name} - Predictions vs Actual')
        plt.legend()
    
    plt.tight_layout()
    plt.savefig('predictions_comparison.png')
    plt.close()
    
    # Compare metrics
    metrics = {'MSE': [], 'RMSE': [], 'MAE': []}
    model_names = list(results.keys())
    
    for model_name, result in results.items():
        metrics['MSE'].append(result['mse'])
        metrics['RMSE'].append(result['rmse'])
        metrics['MAE'].append(result['mae'])
    
    plt.figure(figsize=(12, 6))
    for i, (metric_name, values) in enumerate(metrics.items()):
        plt.subplot(1, 3, i+1)
        plt.bar(model_names, values)
        plt.title(f'Comparison by {metric_name}')
        plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig('metrics_comparison.png')
    plt.close()
    
    # Print metrics table
    print("\nModel Performance Comparison:")
    print("-" * 50)
    print(f"{'Model':<10} {'MSE':<15} {'RMSE':<15} {'MAE':<15}")
    print("-" * 50)
    
    for i, model_name in enumerate(model_names):
        print(f"{model_name:<10} {metrics['MSE'][i]:<15.4f} {metrics['RMSE'][i]:<15.4f} {metrics['MAE'][i]:<15.4f}")
