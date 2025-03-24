import h5py
import numpy as np

from manual import MLP, ReLU, Sigmoid, MSE, MAE, MBE
import students

load = True


def load_weights_from_h5(filename):
    weights = []
    biases = []
    
    with h5py.File(filename, 'r') as f:
        layers_group = f['layers']

        for i in range(4):
            layer_name = f'dense_{i}' if i > 0 else 'dense'
            layer_group = layers_group[layer_name]['vars']

            layer_weights = np.array(layer_group['0'])
            layer_biases = np.array(layer_group['1']).reshape(1, -1)
            
            weights.append(layer_weights)
            biases.append(layer_biases)
            
    return weights, biases


(X_train, Y_train), (X_test, Y_test) = students.data()
features = X_train.shape[1]

Y_train = Y_train.reshape(-1, 1)
Y_test = Y_test.reshape(-1, 1)

layer_sizes = [features, 64, 48, 24, 1]
activations = [ReLU(), ReLU(), ReLU(), Sigmoid()]
loss = MSE()

if load:
    print("Loading weights from MLP.weights.h5...")
    weights, biases = load_weights_from_h5('MLP.weights.h5')
else:
    weights, biases = None, None

model = MLP(
    layer_sizes=layer_sizes,
    activations=activations,
    loss=loss,
    weights=weights,
    biases=biases
)

if not load:
    model.train(
        x=X_train, 
        y=Y_train, 
        epochs=100, 
        batch_size=10, 
        learning_rate=0.025
    )

metrics = [MAE(), MBE()]
results = model.evaluate(X_test, Y_test, metrics=metrics)

print(f"Loss: {results['loss']}")
print(f"Mean Absolute Error: {results['Mean Absolute Error'] * 20}")
print(f"Mean Bias Error: {results['Mean Bias Error'] * 20}")
