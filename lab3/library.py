import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow import keras
from typing import Tuple, List, Dict, Any

def build_simple_rnn(sequence_length: int, learning_rate: float = 0.01) -> keras.Model:
    model = keras.Sequential([
        keras.layers.Input(shape=(sequence_length, 1)),
        keras.layers.SimpleRNN(units=50, return_sequences=True),
        keras.layers.SimpleRNN(units=50),
        keras.layers.Dense(1)
    ])
    
    model.compile(
        optimizer=keras.optimizers.SGD(learning_rate=learning_rate),
        loss='mean_squared_error'
    )
    
    return model

def build_gru(sequence_length: int, learning_rate: float = 0.01) -> keras.Model:
    model = keras.Sequential([
        keras.layers.Input(shape=(sequence_length, 1)),
        keras.layers.GRU(units=50, return_sequences=True),
        keras.layers.GRU(units=50),
        keras.layers.Dense(1)
    ])
    
    model.compile(
        optimizer=keras.optimizers.SGD(learning_rate=learning_rate),
        loss='mean_squared_error'
    )
    
    return model

def build_lstm(sequence_length: int, learning_rate: float = 0.01) -> keras.Model:
    model = keras.Sequential([
        keras.layers.Input(shape=(sequence_length, 1)),
        keras.layers.LSTM(units=50, return_sequences=True),
        keras.layers.LSTM(units=50),
        keras.layers.Dense(1)
    ])
    
    model.compile(
        optimizer=keras.optimizers.SGD(learning_rate=learning_rate),
        loss='mean_squared_error'
    )
    
    return model

class EvaluationCallback(keras.callbacks.Callback):
    """Custom callback to evaluate model at regular intervals during training."""
    
    def __init__(self, validation_data, eval_interval=None, scaler=None):
        super().__init__()
        self.X_val, self.y_val = validation_data
        self.eval_interval = eval_interval  # How many batches between evaluations
        self.scaler = scaler
        
        # Storage for metrics
        self.train_losses = []
        self.val_losses = []
        self.mse_values = []
        self.rmse_values = []
        self.mae_values = []
        self.batch_numbers = []
        self.batch_count = 0
        
        # Use a smaller validation subset for faster evaluation
        val_size = min(500, len(self.X_val))
        self.X_val_sample = self.X_val[:val_size]
        self.y_val_sample = self.y_val[:val_size]
    
    def on_batch_end(self, batch, logs=None):
        self.batch_count += 1
        
        if self.eval_interval and self.batch_count % self.eval_interval == 0:
            # Record training loss
            self.train_losses.append(logs.get('loss'))
            
            # Evaluate on validation data (use sample for faster evaluation)
            val_loss = self.model.evaluate(self.X_val_sample, self.y_val_sample, verbose=0)
            self.val_losses.append(val_loss)
            
            # Get predictions and calculate metrics (on sample)
            y_pred = self.model.predict(self.X_val_sample, verbose=0)
            
            if self.scaler:
                y_val_inv = self.scaler.inverse_transform(self.y_val_sample)
                y_pred_inv = self.scaler.inverse_transform(y_pred)
            else:
                y_val_inv = self.y_val_sample
                y_pred_inv = y_pred
                
            mse = mean_squared_error(y_val_inv, y_pred_inv)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_val_inv, y_pred_inv)
            
            self.mse_values.append(mse)
            self.rmse_values.append(rmse)
            self.mae_values.append(mae)
            self.batch_numbers.append(self.batch_count)
            
            # Print progress without creating a new line (reduces console spam)
            print(f"\rBatch {self.batch_count}: Loss={logs.get('loss'):.4f}, Val Loss={val_loss:.4f}", end="")

def train_and_evaluate(
    model: keras.Model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    scaler: MinMaxScaler,
    epochs: int = 10,
    batch_size: int = 64,
    eval_frequency: int = 20
) -> Dict[str, Any]:
    # Calculate evaluation interval based on desired frequency, with a minimum to prevent too frequent evaluations
    steps_per_epoch = len(X_train) // batch_size
    eval_interval = max(10, steps_per_epoch // eval_frequency)  # At least 10 batches between evaluations
    
    # Create custom callback for evaluation
    eval_callback = EvaluationCallback(
        validation_data=(X_test, y_test),
        eval_interval=eval_interval,
        scaler=scaler
    )
    
    # Train the model
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_test, y_test),
        callbacks=[eval_callback],
        verbose=2  # Set to 2 for one line per epoch instead of progress bar
    )
    
    # Final evaluation
    print("\nPerforming final evaluation...")
    y_pred = model.predict(X_test, verbose=0)
    
    # Inverse transform to get original scale
    y_test_inv = scaler.inverse_transform(y_test)
    y_pred_inv = scaler.inverse_transform(y_pred)
    
    # Calculate metrics
    mse = mean_squared_error(y_test_inv, y_pred_inv)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test_inv, y_pred_inv)
    
    # Return results
    return {
        'model': model,
        'history': history,
        'predictions': y_pred_inv,
        'actual': y_test_inv,
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'eval_callback': eval_callback,
        # Add these keys from the eval_callback to make compatible with compare_models
        'batch_numbers': eval_callback.batch_numbers,
        'train_losses': eval_callback.train_losses,
        'val_losses': eval_callback.val_losses,
        'mse_values': eval_callback.mse_values,
        'rmse_values': eval_callback.rmse_values,
        'mae_values': eval_callback.mae_values
    }
