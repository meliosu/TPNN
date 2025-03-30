import pandas as pd
import numpy as np
from preprocess import data, prepare_data
from library import (
    build_simple_rnn, 
    build_gru, 
    build_lstm,
    train_and_evaluate,
    compare_models
)

if __name__ == "__main__":
    # Load the preprocessed data
    df = data()
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    # Configuration
    target_column = 'Global_active_power'  # Column to predict
    sequence_length = 60  # Use 60 minutes (1 hour) to predict next value
    epochs = 1  # Train for only 1 epoch
    batch_size = 128  # Increased batch size for faster training
    learning_rate = 0.01
    eval_frequency = 10  # Reduced number of evaluations during training
    
    # Prepare data
    X_train, X_test, y_train, y_test, scaler = prepare_data(
        df, 
        target_column, 
        sequence_length=sequence_length
    )
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Testing data shape: {X_test.shape}")
    
    # Initialize models with the same random seed for reproducibility
    np.random.seed(42)
    import tensorflow as tf
    tf.random.set_seed(42)
    
    models = {
        'RNN': build_simple_rnn(sequence_length, learning_rate),
        'GRU': build_gru(sequence_length, learning_rate),
        'LSTM': build_lstm(sequence_length, learning_rate)
    }
    
    # Train and evaluate each model
    results = {}
    
    for name, model in models.items():
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
        results[name] = result
        
        print(f"{name} - MSE: {result['mse']:.4f}, RMSE: {result['rmse']:.4f}, MAE: {result['mae']:.4f}")
    
    # Compare model performance
    compare_models(results)
    
    print("\nTraining and evaluation completed.")
