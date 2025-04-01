import numpy as np
import pandas as pd

def load_data():
    train_data = pd.read_csv('../mnist/train.csv')
    test_data = pd.read_csv('../mnist/test.csv')
    
    X_train = train_data.iloc[:, 1:].values
    y_train = train_data.iloc[:, 0].values
    X_test = test_data.iloc[:, 1:].values
    y_test = test_data.iloc[:, 0].values
    
    X_train = X_train.reshape(X_train.shape[0], 28, 28, 1).astype('float32') / 255.0
    X_test = X_test.reshape(X_test.shape[0], 28, 28, 1).astype('float32') / 255.0
    
    y_train = np.eye(10)[y_train]
    y_test = np.eye(10)[y_test]
    
    return X_train, y_train, X_test, y_test
