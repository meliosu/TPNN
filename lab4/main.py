from preprocess import load_data
from library import create_lenet5_model as create_tf_lenet5_model, train_model
from manual import create_lenet5_model as create_manual_lenet5_model
from plots import visualize_results

def main():
    # Configuration for hyperparameters
    # TensorFlow implementation hyperparameters
    tf_epochs = 6
    tf_batch_size = 128
    tf_learning_rate = 0.001
    
    # Manual implementation hyperparameters
    manual_epochs = 6
    manual_batch_size = 128
    manual_learning_rate = 0.001
    
    X_train, y_train, X_test, y_test = load_data()

    print("Training with TensorFlow implementation:")
    lenet5_model = create_tf_lenet5_model()
    trained_model, tf_history = train_model(lenet5_model, X_train, y_train, X_test, y_test, 
                                        epochs=tf_epochs, batch_size=tf_batch_size,
                                        learning_rate=tf_learning_rate)

    test_loss, test_acc = trained_model.evaluate(X_test, y_test)
    print(f"TensorFlow Test accuracy: {test_acc:.4f}")
    
    print("\nTraining with manual NumPy implementation:")
    manual_model = create_manual_lenet5_model()
    manual_history = manual_model.fit(X_train, y_train,
                              epochs=manual_epochs, 
                              batch_size=manual_batch_size, 
                              learning_rate=manual_learning_rate,
                              validation_data=(X_test, y_test))
    
    test_loss, test_acc = manual_model.evaluate(X_test, y_test)
    print(f"Manual Test accuracy: {test_acc:.4f}")
    
    # Visualize results
    visualize_results(trained_model, manual_model, tf_history, manual_history, 
                     X_test, y_test, save_path='./plots')

if __name__ == "__main__":
    main()
