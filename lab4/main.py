from preprocess import load_data
from library import create_lenet5_model as create_tf_lenet5_model, train_model
from manual import create_lenet5_model as create_manual_lenet5_model

def main():
    X_train, y_train, X_test, y_test = load_data()
    
    # TensorFlow implementation
    print("Training with TensorFlow implementation:")
    lenet5_model = create_tf_lenet5_model()
    trained_model, history = train_model(lenet5_model, X_train, y_train, X_test, y_test, epochs=2)

    test_loss, test_acc = trained_model.evaluate(X_test, y_test)
    print(f"TensorFlow Test accuracy: {test_acc:.4f}")
    
    # Manual implementation
    print("\nTraining with manual NumPy implementation:")
    manual_model = create_manual_lenet5_model()
    # Use smaller batch size and fewer epochs for manual implementation
    history = manual_model.fit(X_train, y_train, epochs=1, batch_size=32, learning_rate=0.001,
                              validation_data=(X_test, y_test))
    
    test_loss, test_acc = manual_model.evaluate(X_test, y_test)
    print(f"Manual Test accuracy: {test_acc:.4f}")

if __name__ == "__main__":
    main()
