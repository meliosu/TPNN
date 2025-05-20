import numpy as np
from scipy import signal
from scipy import ndimage

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

def tanh(x):
    return np.tanh(x)

def tanh_derivative(x):
    return 1 - np.tanh(x)**2

def categorical_crossentropy(y_true, y_pred):
    m = y_true.shape[0]
    loss = -np.sum(y_true * np.log(np.clip(y_pred, 1e-15, 1.0))) / m
    return loss

class Layer:
    def __init__(self):
        self.params = {}
        self.grads = {}
    
    def forward(self, X):
        raise NotImplementedError
    
    def backward(self, dY):
        raise NotImplementedError

class Conv2D(Layer):
    def __init__(self, filters, kernel_size, activation, input_shape=None, padding='valid'):
        super().__init__()
        self.filters = filters
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.activation_name = activation
        self.activation = tanh if activation == 'tanh' else None
        self.activation_derivative = tanh_derivative if activation == 'tanh' else None
        self.padding = padding
        self.input_shape = input_shape
        
        if input_shape:
            self.initialize_params(input_shape)
        
        self.X = None
        self.Z = None
    
    def initialize_params(self, input_shape):
        if len(input_shape) == 3:
            h, w, channels = input_shape
        else:
            _, h, w, channels = input_shape
            
        scale = np.sqrt(2.0 / (self.kernel_size[0] * self.kernel_size[1] * channels))
        self.params['W'] = np.random.normal(scale=scale, size=(self.filters, self.kernel_size[0], self.kernel_size[1], channels))
        self.params['b'] = np.zeros((self.filters,))
        
    def forward(self, X):
        # Add check to initialize weights if they don't exist
        if 'W' not in self.params:
            self.initialize_params(X.shape)
            
        self.X = X
        batch_size, h, w, channels = X.shape
        
        pad_h = pad_w = 0
        if self.padding == 'same':
            pad_h = (self.kernel_size[0] - 1) // 2
            pad_w = (self.kernel_size[1] - 1) // 2
        
        output_h = h - self.kernel_size[0] + 1 + 2 * pad_h
        output_w = w - self.kernel_size[1] + 1 + 2 * pad_w
        
        Z = np.zeros((batch_size, output_h, output_w, self.filters))
        
        if pad_h > 0 or pad_w > 0:
            X_padded = np.pad(X, ((0, 0), (pad_h, pad_h), (pad_w, pad_w), (0, 0)), mode='constant')
        else:
            X_padded = X
        
        # Process each batch item and each filter
        for i in range(batch_size):
            for j in range(self.filters):
                # Process each channel and sum the results
                for c in range(channels):
                    kernel_channel = self.params['W'][j, :, :, c]
                    Z[i, :, :, j] += signal.correlate2d(X_padded[i, :, :, c], kernel_channel, mode='valid')
                
                # Add bias
                Z[i, :, :, j] += self.params['b'][j]
        
        self.Z = Z
        A = self.activation(Z) if self.activation else Z
        return A
    
    def backward(self, dA):
        batch_size, h, w, channels = self.X.shape
        batch_size, dh, dw, filters = dA.shape
        
        if self.activation:
            dZ = dA * self.activation_derivative(self.Z)
        else:
            dZ = dA
        
        pad_h = pad_w = 0
        if self.padding == 'same':
            pad_h = (self.kernel_size[0] - 1) // 2
            pad_w = (self.kernel_size[1] - 1) // 2
        
        self.grads['W'] = np.zeros_like(self.params['W'])
        self.grads['b'] = np.sum(dZ, axis=(0, 1, 2))
        
        dX = np.zeros_like(self.X)
        
        if pad_h > 0 or pad_w > 0:
            X_padded = np.pad(self.X, ((0, 0), (pad_h, pad_h), (pad_w, pad_w), (0, 0)), mode='constant')
            dX_padded = np.zeros_like(X_padded)
        else:
            X_padded = self.X
            dX_padded = np.zeros_like(X_padded)
        
        # Calculate gradients using ndimage.correlate
        for j in range(filters):
            kernel_shape = self.params['W'][j].shape
            for i in range(batch_size):
                # Calculate gradients for weights
                for c in range(channels):
                    # Compute correlation between input and output gradient
                    self.grads['W'][j, :, :, c] += signal.correlate2d(
                        X_padded[i, :, :, c], dZ[i, :, :, j], mode='valid')
                
                # Calculate gradient for inputs using convolution
                rotated_kernel = np.rot90(self.params['W'][j], 2, (0, 1))
                # Handle multiple channels for the input gradient
                for c in range(channels):
                    dX_padded[i, :, :, c] += signal.convolve2d(
                        dZ[i, :, :, j], rotated_kernel[:, :, c], mode='full')
        
        if pad_h > 0 or pad_w > 0:
            dX = dX_padded[:, pad_h:-pad_h, pad_w:-pad_w, :]
        else:
            dX = dX_padded
            
        return dX

class AveragePooling2D(Layer):
    def __init__(self, pool_size, strides):
        super().__init__()
        self.pool_size = pool_size if isinstance(pool_size, tuple) else (pool_size, pool_size)
        self.strides = strides if isinstance(strides, tuple) else (strides, strides)
        self.X = None
    
    def forward(self, X):
        self.X = X
        batch_size, h, w, channels = X.shape
        
        out_h = (h - self.pool_size[0]) // self.strides[0] + 1
        out_w = (w - self.pool_size[1]) // self.strides[1] + 1
        
        output = np.zeros((batch_size, out_h, out_w, channels))
        
        for i in range(0, out_h):
            for j in range(0, out_w):
                h_start = i * self.strides[0]
                h_end = h_start + self.pool_size[0]
                w_start = j * self.strides[1]
                w_end = w_start + self.pool_size[1]
                
                output[:, i, j, :] = np.mean(X[:, h_start:h_end, w_start:w_end, :], axis=(1, 2))
        
        return output
    
    def backward(self, dA):
        batch_size, h, w, channels = self.X.shape
        _, dh, dw, _ = dA.shape
        
        dX = np.zeros_like(self.X)
        
        for i in range(dh):
            for j in range(dw):
                h_start = i * self.strides[0]
                h_end = h_start + self.pool_size[0]
                w_start = j * self.strides[1]
                w_end = w_start + self.pool_size[1]
                
                dX[:, h_start:h_end, w_start:w_end, :] += dA[:, i:i+1, j:j+1, :] / (self.pool_size[0] * self.pool_size[1])
        
        return dX

class Flatten(Layer):
    def __init__(self):
        super().__init__()
        self.input_shape = None
    
    def forward(self, X):
        self.input_shape = X.shape
        return X.reshape(X.shape[0], -1)
    
    def backward(self, dA):
        return dA.reshape(self.input_shape)

class Dense(Layer):
    def __init__(self, units, activation):
        super().__init__()
        self.units = units
        self.activation_name = activation
        
        if activation == 'tanh':
            self.activation = tanh
            self.activation_derivative = tanh_derivative
        elif activation == 'softmax':
            self.activation = softmax
            self.activation_derivative = None
        else:
            self.activation = None
            self.activation_derivative = None
            
        self.X = None
        self.Z = None
    
    def initialize_params(self, input_dim):
        scale = np.sqrt(2.0 / input_dim)
        self.params['W'] = np.random.normal(scale=scale, size=(input_dim, self.units))
        self.params['b'] = np.zeros((1, self.units))
    
    def forward(self, X):
        if 'W' not in self.params:
            self.initialize_params(X.shape[1])
            
        self.X = X
        self.Z = np.dot(X, self.params['W']) + self.params['b']
        
        if self.activation:
            return self.activation(self.Z)
        return self.Z
    
    def backward(self, dA):
        if self.activation_name == 'softmax':
            batch_size = dA.shape[0]
            dZ = dA
        elif self.activation:
            dZ = dA * self.activation_derivative(self.Z)
        else:
            dZ = dA
        
        self.grads['W'] = np.dot(self.X.T, dZ)
        self.grads['b'] = np.sum(dZ, axis=0, keepdims=True)
        
        dX = np.dot(dZ, self.params['W'].T)
        return dX

class Sequential:
    def __init__(self, layers=None):
        self.layers = layers if layers else []
    
    def add(self, layer):
        self.layers.append(layer)
    
    def forward(self, X):
        A = X
        for layer in self.layers:
            A = layer.forward(A)
        return A
    
    def backward(self, dA):
        for layer in reversed(self.layers):
            dA = layer.backward(dA)
        return dA
    
    def update_params(self, learning_rate):
        for layer in self.layers:
            if hasattr(layer, 'params') and hasattr(layer, 'grads'):
                for param_name in layer.params:
                    if param_name in layer.grads:
                        layer.params[param_name] -= learning_rate * layer.grads[param_name]
    
    def fit(self, X, y, epochs=10, batch_size=32, learning_rate=0.01, validation_data=None):
        history = {
            'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': [],
            'batch_loss': [], 'batch_accuracy': [], 'batch_val_loss': [], 
            'batch_val_accuracy': [], 'batch': []
        }
        samples = X.shape[0]
        batch_counter = 0
        
        for epoch in range(epochs):
            indices = np.random.permutation(samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            epoch_loss = 0
            epoch_correct = 0
            
            for i in range(0, samples, batch_size):
                batch_counter += 1
                end = min(i + batch_size, samples)
                X_batch = X_shuffled[i:end]
                y_batch = y_shuffled[i:end]
                
                y_pred = self.forward(X_batch)
                
                batch_loss = categorical_crossentropy(y_batch, y_pred)
                batch_correct = np.sum(np.argmax(y_pred, axis=1) == np.argmax(y_batch, axis=1))
                batch_accuracy = batch_correct / (end - i)
                
                epoch_loss += batch_loss * (end - i)
                epoch_correct += batch_correct
                
                dA = y_pred - y_batch
                self.backward(dA)
                self.update_params(learning_rate)

                if (batch_counter - 1) % 30 == 0:
                    history['batch_loss'].append(batch_loss)
                    history['batch_accuracy'].append(batch_accuracy)
                    history['batch'].append(batch_counter)
                    
                    if validation_data:
                        X_val, y_val = validation_data
                        y_val_pred = self.forward(X_val)
                        val_loss = categorical_crossentropy(y_val, y_val_pred)
                        val_accuracy = np.mean(np.argmax(y_val_pred, axis=1) == np.argmax(y_val, axis=1))
                        history['batch_val_loss'].append(val_loss)
                        history['batch_val_accuracy'].append(val_accuracy)
            
            # Record epoch metrics
            epoch_loss = epoch_loss / samples
            epoch_accuracy = epoch_correct / samples
            
            history['loss'].append(epoch_loss)
            history['accuracy'].append(epoch_accuracy)
            
            if validation_data:
                X_val, y_val = validation_data
                y_val_pred = self.forward(X_val)
                val_loss = categorical_crossentropy(y_val, y_val_pred)
                val_accuracy = np.mean(np.argmax(y_val_pred, axis=1) == np.argmax(y_val, axis=1))
                history['val_loss'].append(val_loss)
                history['val_accuracy'].append(val_accuracy)
                print(f"Epoch {epoch+1}/{epochs} - loss: {epoch_loss:.4f} - accuracy: {epoch_accuracy:.4f} - val_loss: {val_loss:.4f} - val_accuracy: {val_accuracy:.4f}")
            else:
                print(f"Epoch {epoch+1}/{epochs} - loss: {epoch_loss:.4f} - accuracy: {epoch_accuracy:.4f}")
        
        return history
    
    def evaluate(self, X, y):
        y_pred = self.forward(X)
        loss = categorical_crossentropy(y, y_pred)
        accuracy = np.mean(np.argmax(y_pred, axis=1) == np.argmax(y, axis=1))
        return loss, accuracy

def create_lenet5_model():
    model = Sequential()
    model.add(Conv2D(6, kernel_size=5, activation='tanh', input_shape=(28, 28, 1), padding='same'))
    model.add(AveragePooling2D(pool_size=2, strides=2))
    model.add(Conv2D(16, kernel_size=5, activation='tanh', padding='valid'))
    model.add(AveragePooling2D(pool_size=2, strides=2))
    model.add(Flatten())
    model.add(Dense(120, activation='tanh'))
    model.add(Dense(84, activation='tanh'))
    model.add(Dense(10, activation='softmax'))
    return model

