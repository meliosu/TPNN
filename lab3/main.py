import pandas as pd
import numpy as np
from preprocess import data, prepare_data
from library import (
    build_simple_rnn, 
    build_gru, 
    build_lstm,
    train_and_evaluate
)
from manual import (
    SimpleRNN,
    GRU,
    LSTM,
    train_numpy_model
)
from common import compare_models

if __name__ == "__main__":
    # Data parameters
    data_fraction = 0.01  # Use only 10% of the dataset
    
    df = data(data_fraction=data_fraction)
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    target_column = 'Global_active_power'
    sequence_length = 60
    epochs = 1
    batch_size = 128
    learning_rate = 0.025
    eval_frequency = 10
    
    X_train, X_test, y_train, y_test, scaler = prepare_data(
        df, 
        target_column, 
        sequence_length=sequence_length
    )
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Testing data shape: {X_test.shape}")
    
    # Set seeds for reproducibility
    np.random.seed(42)
    import tensorflow as tf
    tf.random.set_seed(42)
    
    # Train TensorFlow models
    tf_models = {
        'TF_RNN': build_simple_rnn(sequence_length, learning_rate),
        'TF_GRU': build_gru(sequence_length, learning_rate),
        'TF_LSTM': build_lstm(sequence_length, learning_rate)
    }
    
    tf_results = {}
    for name, model in tf_models.items():
        print(f"\nTraining {name} model...")
        result = train_and_evaluate(
            model, 
            X_train, y_train, 
            X_test, y_test, 
            scaler,
            epochs=epochs,
            batch_size=batch_size,
            eval_frequency=eval_frequency
        )
        tf_results[name] = result
        print(f"{name} - MSE: {result['mse']:.4f}, RMSE: {result['rmse']:.4f}, MAE: {result['mae']:.4f}")
    
    # Train NumPy models
    np_models = {
        'NP_RNN': SimpleRNN(1, 50, 1, learning_rate),
        'NP_GRU': GRU(1, 50, 1, learning_rate),
        'NP_LSTM': LSTM(1, 50, 1, learning_rate)
    }
    
    np_results = {}
    for name, model in np_models.items():
        print(f"\nTraining {name} model...")
        result = train_numpy_model(
            model, 
            X_train, y_train, 
            X_test, y_test, 
            scaler,
            epochs=epochs,
            batch_size=batch_size,
            eval_frequency=eval_frequency
        )
        np_results[name] = result
        print(f"{name} - MSE: {result['mse']:.4f}, RMSE: {result['rmse']:.4f}, MAE: {result['mae']:.4f}")
    
    # Compare TensorFlow models
    print("\nComparing TensorFlow models:")
    compare_models(tf_results)
    
    # Compare NumPy models
    print("\nComparing NumPy models:")
    compare_models(np_results)
    
    # Compare best models from each implementation
    best_models = {}
    for name, result in tf_results.items():
        if name == 'TF_RNN':  # Choose your best TF model
            best_models['TensorFlow'] = result
    
    for name, result in np_results.items():
        if name == 'NP_RNN':  # Choose your best NumPy model
            best_models['NumPy'] = result
    
    print("\nComparing best models from each implementation:")
    compare_models(best_models)
    
    print("\nTraining and evaluation completed.")
