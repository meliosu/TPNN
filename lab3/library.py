import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow import keras
from typing import Dict, Any


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
    def __init__(self, validation_data, eval_interval=None, scaler=None):
        super().__init__()
        self.X_val, self.y_val = validation_data
        self.eval_interval = eval_interval
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
    
    def on_batch_end(self, _batch, logs=None):
        self.batch_count += 1
        
        if self.eval_interval and self.batch_count % self.eval_interval == 0:
            self.train_losses.append(logs.get('loss'))

            val_loss = self.model.evaluate(self.X_val_sample, self.y_val_sample, verbose=0)
            self.val_losses.append(val_loss)

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

            print(f"\rBatch {self.batch_count}: Loss={logs.get('loss'):.4f}, Val Loss={val_loss:.4f}", end="")


def train_and_evaluate(
    model: keras.Model,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    scaler: MinMaxScaler,
    epochs: int = 10,
    batch_size: int = 64,
    eval_frequency: int = 20
) -> Dict[str, Any]:
    steps_per_epoch = len(x_train) // batch_size
    eval_interval = max(10, steps_per_epoch // eval_frequency)

    eval_callback = EvaluationCallback(
        validation_data=(x_test, y_test),
        eval_interval=eval_interval,
        scaler=scaler
    )

    history = model.fit(
        x_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(x_test, y_test),
        callbacks=[eval_callback],
        verbose=2
    )

    print("\nEvaluating...")
    y_pred = model.predict(x_test, verbose=0)

    y_test_inv = scaler.inverse_transform(y_test)
    y_pred_inv = scaler.inverse_transform(y_pred)

    mse = mean_squared_error(y_test_inv, y_pred_inv)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test_inv, y_pred_inv)

    return {
        'model': model,
        'history': history,
        'predictions': y_pred_inv,
        'actual': y_test_inv,
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'eval_callback': eval_callback,
        'batch_numbers': eval_callback.batch_numbers,
        'train_losses': eval_callback.train_losses,
        'val_losses': eval_callback.val_losses,
        'mse_values': eval_callback.mse_values,
        'rmse_values': eval_callback.rmse_values,
        'mae_values': eval_callback.mae_values
    }
