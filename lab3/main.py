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
from plotting import ensure_plot_dir, plot_training_metrics, plot_final_comparison

if __name__ == "__main__":
    plot_dir = ensure_plot_dir("plots")

    data_fraction = 0.2
    
    df = data(data_fraction=data_fraction)
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    target_column = 'Global_active_power'
    sequence_length = 60
    epochs = 1
    batch_size = 128
    learning_rate = 0.025
    eval_frequency = 25
    
    X_train, X_test, y_train, y_test, scaler = prepare_data(
        df, 
        target_column, 
        sequence_length=sequence_length
    )
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Testing data shape: {X_test.shape}")
    
    # Train library implementation models
    tf_models = {
        'LIB_RNN': build_simple_rnn(sequence_length, learning_rate),
        'LIB_GRU': build_gru(sequence_length, learning_rate),
        'LIB_LSTM': build_lstm(sequence_length, learning_rate)
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
    
    # Train manual implementation models
    np_models = {
        'MAN_RNN': SimpleRNN(1, 50, 1, learning_rate),
        'MAN_GRU': GRU(1, 50, 1, learning_rate),
        'MAN_LSTM': LSTM(1, 50, 1, learning_rate)
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

    print("\nGenerating Library models plot...")
    plot_training_metrics(tf_results, 'Library', plot_dir)

    print("\nGenerating Manual models plot...")
    plot_training_metrics(np_results, 'Manual', plot_dir)

    print("\nGenerating final comparison plot...")
    plot_final_comparison(tf_results, np_results, plot_dir)
