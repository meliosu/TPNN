import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, AveragePooling2D, Dense, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import History, Callback
import numpy as np

class MetricsPerBatchCallback(Callback):
    def __init__(self, validation_data=None, batch_log_interval=10):
        super().__init__()
        self.validation_data = validation_data
        self.batch_log_interval = batch_log_interval
        self.history = {'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': [], 'batch': []}
        self.batch_counter = 0
        
    def on_batch_end(self, batch, logs=None):
        self.batch_counter += 1
        if batch % self.batch_log_interval == 0:
            self.history['loss'].append(logs.get('loss'))
            self.history['accuracy'].append(logs.get('accuracy'))
            self.history['batch'].append(self.batch_counter)
            
            if self.validation_data:
                val_logs = self.model.evaluate(
                    self.validation_data[0], self.validation_data[1], 
                    verbose=0
                )
                self.history['val_loss'].append(val_logs[0])
                self.history['val_accuracy'].append(val_logs[1])

def create_lenet5_model():
    model = Sequential([
        Conv2D(6, kernel_size=(5, 5), activation='tanh', input_shape=(28, 28, 1), padding='same'),
        AveragePooling2D(pool_size=(2, 2), strides=2),
        Conv2D(16, kernel_size=(5, 5), activation='tanh'),
        AveragePooling2D(pool_size=(2, 2), strides=2),
        Flatten(),
        Dense(120, activation='tanh'),
        Dense(84, activation='tanh'),
        Dense(10, activation='softmax')
    ])
    
    return model

def train_model(model, X_train, y_train, X_test, y_test, epochs=10, batch_size=128, learning_rate=0.001):
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    history = History()
    batch_metrics = MetricsPerBatchCallback(validation_data=(X_test, y_test), batch_log_interval=20)
    
    model.fit(
        X_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(X_test, y_test),
        callbacks=[history, batch_metrics]
    )
    
    return model, {'epoch_history': history.history, 'batch_history': batch_metrics.history}

