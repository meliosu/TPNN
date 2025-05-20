import numpy as np


class Activation:
    def forward(self, x):
        raise NotImplementedError
    
    def backward(self, x):
        raise NotImplementedError


class ReLU(Activation):
    def forward(self, x):
        return np.maximum(0, x)
    
    def backward(self, x):
        return np.where(x > 0, 1, 0)


class Sigmoid(Activation):
    def forward(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def backward(self, x):
        s = self.forward(x)
        return s * (1 - s)


class Loss:
    def calculate(self, y_true, y_pred):
        raise NotImplementedError
    
    def backward(self, y_true, y_pred):
        raise NotImplementedError


class MSE(Loss):
    def calculate(self, y_true, y_pred):
        return np.mean(np.power(y_true - y_pred, 2))
    
    def backward(self, y_true, y_pred):
        return -2 * (y_true - y_pred) / y_true.shape[0]


class Metric:
    def __init__(self, name):
        self.name = name
    
    def calculate(self, y_true, y_pred):
        raise NotImplementedError


class MAE(Metric):
    def __init__(self):
        super().__init__("Mean Absolute Error")
    
    def calculate(self, y_true, y_pred):
        return np.mean(np.abs(y_true - y_pred))


class MBE(Metric):
    def __init__(self):
        super().__init__("Mean Bias Error")
    
    def calculate(self, y_true, y_pred):
        return np.mean(y_pred - y_true)


class Layer:
    def __init__(self, input_size, output_size, activation, weights=None, biases=None, seed=None):
        self.input_size = input_size
        self.output_size = output_size
        self.activation = activation
        
        if weights is None:
            if seed is not None:
                np.random.seed(seed)

            self.weights = np.random.randn(input_size, output_size) * np.sqrt(1 / input_size)
        else:
            self.weights = weights
        
        if biases is None:
            if seed is not None:
                np.random.seed(seed)

            self.biases = np.zeros((1, output_size))
        else:
            self.biases = biases
    
    def forward(self, inputs):
        self.inputs = inputs
        self.z = np.dot(inputs, self.weights) + self.biases
        self.output = self.activation.forward(self.z)
        return self.output
    
    def backward(self, grad):
        activation_grad = grad * self.activation.backward(self.z)
        self.weights_grad = np.dot(self.inputs.T, activation_grad)
        self.biases_grad = np.sum(activation_grad, axis=0, keepdims=True)
        return np.dot(activation_grad, self.weights.T)


class MLP:
    def __init__(self, layer_sizes, activations=None, loss=None, weights=None, biases=None, seed=None):
        self.layers = []
        self.loss = loss if loss is not None else MSE()
        
        # If a single activation is provided, use it for all layers
        if activations is not None and not isinstance(activations, list):
            activations = [activations] * (len(layer_sizes) - 1)
        elif activations is None:
            # Default to ReLU for hidden layers and Sigmoid for the output layer
            activations = [ReLU()] * (len(layer_sizes) - 2) + [Sigmoid()]
        
        for i in range(len(layer_sizes) - 1):
            input_size = layer_sizes[i]
            output_size = layer_sizes[i+1]
            activation = activations[i]
            
            layer_weights = None if weights is None else weights[i]
            layer_biases = None if biases is None else biases[i]
            
            # If seed is provided, create a unique seed for each layer
            layer_seed = None
            if seed is not None:
                layer_seed = seed + i
            
            layer = Layer(input_size, output_size, activation, layer_weights, layer_biases, layer_seed)
            self.layers.append(layer)
    
    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)

        return x
    
    def backward(self, y_true, y_pred):
        grad = self.loss.backward(y_true, y_pred)
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
    
    def train(self, x, y, epochs=100, batch_size=10, learning_rate=0.01):
        n_samples = x.shape[0]
        
        for epoch in range(epochs):
            epoch_loss = 0
            
            # Shuffle data
            indices = np.random.permutation(n_samples)
            x_shuffled = x[indices]
            y_shuffled = y[indices]
            
            # Process in batches
            for i in range(0, n_samples, batch_size):
                x_batch = x_shuffled[i:i+batch_size]
                y_batch = y_shuffled[i:i+batch_size]
                
                # Forward pass
                y_pred = self.forward(x_batch)
                batch_loss = self.loss.calculate(y_batch, y_pred)
                epoch_loss += batch_loss * len(x_batch)
                
                # Backward pass and update
                self.backward(y_batch, y_pred)
                
                for layer in self.layers:
                    layer.weights -= learning_rate * layer.weights_grad
                    layer.biases -= learning_rate * layer.biases_grad
            
            epoch_loss /= n_samples
            print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss}")
    
    def evaluate(self, x, y, metrics=None):
        y_pred = self.forward(x)
        results = {"loss": self.loss.calculate(y, y_pred)}
        
        if metrics:
            for metric in metrics:
                results[metric.name] = metric.calculate(y, y_pred)
        
        return results
