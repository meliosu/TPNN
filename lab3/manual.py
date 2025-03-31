import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Helper functions with improved numerical stability
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -15, 15)))

def tanh(x):
    return np.tanh(np.clip(x, -15, 15))

def vector_mse(y_pred, y_true):
    return np.mean(np.square(y_pred - y_true))

def clip_gradients(gradients, threshold=5.0):
    """Clip gradients to prevent exploding gradients"""
    for grad in gradients:
        np.clip(grad, -threshold, threshold, out=grad)
    return gradients

class Optimizer:
    """Simple SGD optimizer with momentum and learning rate decay"""
    def __init__(self, learning_rate=0.01, momentum=0.9, decay=0.0):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.decay = decay
        self.iterations = 0
        self.velocity = {}
        
    def initialize(self, params):
        """Initialize velocity for each parameter"""
        for i, param in enumerate(params):
            self.velocity[i] = np.zeros_like(param)
            
    def update(self, params, grads):
        """Update parameters using momentum"""
        if not self.velocity:
            self.initialize(params)
            
        self.iterations += 1
        lr = self.learning_rate * (1.0 / (1.0 + self.decay * self.iterations))
        
        for i, (param, grad) in enumerate(zip(params, grads)):
            self.velocity[i] = self.momentum * self.velocity[i] - lr * grad
            param += self.velocity[i]

class SimpleRNN:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate

        # Initialize weights with Xavier/Glorot initialization
        scale = np.sqrt(2.0 / (input_size + hidden_size))
        self.Wxh = np.random.randn(hidden_size, input_size) * scale
        self.Whh = np.random.randn(hidden_size, hidden_size) * scale
        # Orthogonal initialization for recurrent weights often works better
        u, _, v = np.linalg.svd(np.random.randn(hidden_size, hidden_size))
        self.Whh = u @ v * scale
        self.Why = np.random.randn(output_size, hidden_size) * scale
        self.bh = np.zeros((hidden_size, 1))
        self.by = np.zeros((output_size, 1))
        
        # Initialize optimizer
        self.optimizer = Optimizer(learning_rate=learning_rate, momentum=0.9, decay=0.0001)

    def forward(self, inputs_batch):
        """Vectorized forward pass handling a batch of sequences"""
        batch_size = len(inputs_batch)
        seq_length = inputs_batch[0].shape[0]
        outputs = np.zeros((batch_size, self.output_size))

        # Store hidden states for each sample in the batch
        all_hidden_states = []

        for b in range(batch_size):
            inputs = inputs_batch[b]
            h = np.zeros((self.hidden_size, 1))
            hidden_states = [h]

            # Process sequence using matrix operations
            for t in range(seq_length):
                x = inputs[t].reshape(-1, 1)
                h = tanh(np.dot(self.Wxh, x) + np.dot(self.Whh, h) + self.bh)
                hidden_states.append(h)

            outputs[b] = (np.dot(self.Why, hidden_states[-1]) + self.by).flatten()
            all_hidden_states.append(hidden_states)

        return outputs, all_hidden_states

    def backward(self, inputs_batch, all_hidden_states, outputs, targets):
        """Vectorized backward pass with gradient clipping"""
        batch_size = len(inputs_batch)

        # Initialize weight gradients for batch
        dWxh = np.zeros_like(self.Wxh)
        dWhh = np.zeros_like(self.Whh)
        dWhy = np.zeros_like(self.Why)
        dbh = np.zeros_like(self.bh)
        dby = np.zeros_like(self.by)

        total_loss = 0

        for b in range(batch_size):
            inputs = inputs_batch[b]
            hidden_states = all_hidden_states[b]
            y_pred = outputs[b].reshape(-1, 1)
            y_true = targets[b].reshape(-1, 1)

            # Calculate loss and output gradients
            dy = y_pred - y_true
            total_loss += np.mean(np.square(dy))

            # Gradients for output layer
            dWhy += np.dot(dy, hidden_states[-1].T)
            dby += dy

            # Backpropagate through time
            dh_next = np.dot(self.Why.T, dy)

            for t in reversed(range(len(inputs))):
                h = hidden_states[t+1]
                h_prev = hidden_states[t]
                x = inputs[t].reshape(-1, 1)

                # Backprop through tanh: tanh'(x) = 1 - tanh²(x)
                dtanh = (1 - h * h) * dh_next
                dbh += dtanh
                dWxh += np.dot(dtanh, x.T)
                dWhh += np.dot(dtanh, h_prev.T)

                # Pass gradients to the next timestep
                dh_next = np.dot(self.Whh.T, dtanh)

        # Average gradients across the batch
        grads = [dWxh/batch_size, dWhh/batch_size, dWhy/batch_size, 
                 dbh/batch_size, dby/batch_size]
        
        # Clip gradients to prevent exploding gradients
        grads = clip_gradients(grads, threshold=5.0)
        
        # Update weights using optimizer
        params = [self.Wxh, self.Whh, self.Why, self.bh, self.by]
        self.optimizer.update(params, grads)

        return total_loss / batch_size

    def predict(self, inputs):
        """Predict for a batch of sequences"""
        outputs, _ = self.forward(inputs)
        return outputs

class GRU:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate

        # Initialize weights with Xavier/Glorot initialization
        scale = np.sqrt(2.0 / (input_size + hidden_size))
        
        # Gate weights
        self.Wz = np.random.randn(hidden_size, input_size + hidden_size) * scale
        self.Wr = np.random.randn(hidden_size, input_size + hidden_size) * scale
        self.Wh = np.random.randn(hidden_size, input_size + hidden_size) * scale
        self.Why = np.random.randn(output_size, hidden_size) * scale

        # Initialize biases
        self.bz = np.zeros((hidden_size, 1))
        self.br = np.zeros((hidden_size, 1))
        self.bh = np.zeros((hidden_size, 1))
        self.by = np.zeros((output_size, 1))
        
        # Initialize optimizer
        self.optimizer = Optimizer(learning_rate=learning_rate, momentum=0.9, decay=0.0001)

    def forward(self, inputs_batch):
        """Vectorized forward pass for GRU"""
        batch_size = len(inputs_batch)
        outputs = np.zeros((batch_size, self.output_size))
        all_caches = []

        for b in range(batch_size):
            inputs = inputs_batch[b]
            h = np.zeros((self.hidden_size, 1))

            h_states = [h]
            z_list, r_list, h_tilde_list = [], [], []

            for x in inputs:
                x = x.reshape(-1, 1)
                xh = np.vstack((x, h))

                # Update gate
                z = sigmoid(np.dot(self.Wz, xh) + self.bz)
                # Reset gate
                r = sigmoid(np.dot(self.Wr, xh) + self.br)

                # Candidate hidden state
                h_reset = r * h
                xh_reset = np.vstack((x, h_reset))
                h_tilde = tanh(np.dot(self.Wh, xh_reset) + self.bh)

                # New hidden state with interpolation via update gate
                h = (1 - z) * h + z * h_tilde

                h_states.append(h)
                z_list.append(z)
                r_list.append(r)
                h_tilde_list.append(h_tilde)

            outputs[b] = (np.dot(self.Why, h_states[-1]) + self.by).flatten()
            all_caches.append((h_states, z_list, r_list, h_tilde_list))

        return outputs, all_caches

    def backward(self, inputs_batch, all_caches, outputs, targets):
        """Vectorized backward pass for GRU with gradient clipping"""
        batch_size = len(inputs_batch)

        # Initialize gradients
        dWz = np.zeros_like(self.Wz)
        dWr = np.zeros_like(self.Wr)
        dWh = np.zeros_like(self.Wh)
        dWhy = np.zeros_like(self.Why)

        dbz = np.zeros_like(self.bz)
        dbr = np.zeros_like(self.br)
        dbh = np.zeros_like(self.bh)
        dby = np.zeros_like(self.by)

        total_loss = 0

        for b in range(batch_size):
            inputs = inputs_batch[b]
            h_states, z_list, r_list, h_tilde_list = all_caches[b]
            y_pred = outputs[b].reshape(-1, 1)
            y_true = targets[b].reshape(-1, 1)

            # MSE loss
            dy = y_pred - y_true
            total_loss += np.mean(np.square(dy))

            # Output layer gradients
            dWhy += np.dot(dy, h_states[-1].T)
            dby += dy

            # Backpropagate through time
            dh_next = np.dot(self.Why.T, dy)

            for t in reversed(range(len(inputs))):
                h = h_states[t+1]
                h_prev = h_states[t]
                z = z_list[t]
                r = r_list[t]
                h_tilde = h_tilde_list[t]
                x = inputs[t].reshape(-1, 1)

                # Gradient through hidden state update
                dh = dh_next
                dh_tilde = dh * z
                dz = dh * (h_tilde - h_prev)

                # Gradient through candidate hidden state
                xh_reset = np.vstack((x, r * h_prev))
                dhtilde_raw = (1 - h_tilde**2) * dh_tilde
                dWh += np.dot(dhtilde_raw, xh_reset.T)
                dbh += dhtilde_raw

                # Gradients for reset gate components
                dxh_reset = np.dot(self.Wh.T, dhtilde_raw)
                dx_reset = dxh_reset[:self.input_size]
                dh_reset = dxh_reset[self.input_size:]

                dr = dh_reset * h_prev

                # Gradients for update and reset gates
                xh = np.vstack((x, h_prev))
                dz_raw = z * (1 - z) * dz
                dWz += np.dot(dz_raw, xh.T)
                dbz += dz_raw

                dr_raw = r * (1 - r) * dr
                dWr += np.dot(dr_raw, xh.T)
                dbr += dr_raw

                # Propagate gradients back to input and previous hidden state
                dxh = np.dot(self.Wz.T, dz_raw) + np.dot(self.Wr.T, dr_raw)
                dx = dxh[:self.input_size] + dx_reset
                dh_prev = dxh[self.input_size:] + dh * (1 - z) + dr * r

                dh_next = dh_prev

        # Average gradients across batch
        grads = [dWz/batch_size, dWr/batch_size, dWh/batch_size, dWhy/batch_size,
                 dbz/batch_size, dbr/batch_size, dbh/batch_size, dby/batch_size]
        
        # Clip gradients
        grads = clip_gradients(grads, threshold=5.0)
        
        # Update weights using optimizer
        params = [self.Wz, self.Wr, self.Wh, self.Why, 
                  self.bz, self.br, self.bh, self.by]
        self.optimizer.update(params, grads)

        return total_loss / batch_size

    def predict(self, inputs):
        outputs, _ = self.forward(inputs)
        return outputs

class LSTM:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate

        # Xavier/Glorot initialization
        scale = np.sqrt(2.0 / (input_size + hidden_size))
        
        # Gate weights
        self.Wf = np.random.randn(hidden_size, input_size + hidden_size) * scale
        self.Wi = np.random.randn(hidden_size, input_size + hidden_size) * scale
        self.Wc = np.random.randn(hidden_size, input_size + hidden_size) * scale
        self.Wo = np.random.randn(hidden_size, input_size + hidden_size) * scale
        self.Why = np.random.randn(output_size, hidden_size) * scale

        # Initialize bias values (following TensorFlow defaults)
        # Forget gate bias initialized to 1.0 (important LSTM optimization)
        self.bf = np.ones((hidden_size, 1))
        self.bi = np.zeros((hidden_size, 1))
        self.bc = np.zeros((hidden_size, 1))
        self.bo = np.zeros((hidden_size, 1))
        self.by = np.zeros((output_size, 1))
        
        # Initialize optimizer
        self.optimizer = Optimizer(learning_rate=learning_rate, momentum=0.9, decay=0.0001)

    def forward(self, inputs_batch):
        """Vectorized forward pass for LSTM"""
        batch_size = len(inputs_batch)
        outputs = np.zeros((batch_size, self.output_size))
        all_caches = []

        for b in range(batch_size):
            inputs = inputs_batch[b]
            h = np.zeros((self.hidden_size, 1))
            c = np.zeros((self.hidden_size, 1))

            h_states, c_states = [h], [c]
            f_list, i_list, o_list, c_tilde_list = [], [], [], []

            for x in inputs:
                x = x.reshape(-1, 1)
                xh = np.vstack((x, h))

                # Compute gates
                f = sigmoid(np.dot(self.Wf, xh) + self.bf)  # forget gate
                i = sigmoid(np.dot(self.Wi, xh) + self.bi)  # input gate
                o = sigmoid(np.dot(self.Wo, xh) + self.bo)  # output gate
                c_tilde = tanh(np.dot(self.Wc, xh) + self.bc)  # candidate cell state

                # Update cell state
                c = f * c + i * c_tilde
                
                # Compute hidden state (output)
                h = o * tanh(c)

                h_states.append(h)
                c_states.append(c)
                f_list.append(f)
                i_list.append(i)
                o_list.append(o)
                c_tilde_list.append(c_tilde)

            outputs[b] = (np.dot(self.Why, h_states[-1]) + self.by).flatten()
            all_caches.append((h_states, c_states, f_list, i_list, o_list, c_tilde_list))

        return outputs, all_caches

    def backward(self, inputs_batch, all_caches, outputs, targets):
        """Vectorized backward pass for LSTM with gradient clipping"""
        batch_size = len(inputs_batch)

        # Initialize gradients
        dWf = np.zeros_like(self.Wf)
        dWi = np.zeros_like(self.Wi)
        dWc = np.zeros_like(self.Wc)
        dWo = np.zeros_like(self.Wo)
        dWhy = np.zeros_like(self.Why)
        
        dbf = np.zeros_like(self.bf)
        dbi = np.zeros_like(self.bi)
        dbc = np.zeros_like(self.bc)
        dbo = np.zeros_like(self.bo)
        dby = np.zeros_like(self.by)

        total_loss = 0

        for b in range(batch_size):
            inputs = inputs_batch[b]
            h_states, c_states, f_list, i_list, o_list, c_tilde_list = all_caches[b]
            y_pred = outputs[b].reshape(-1, 1)
            y_true = targets[b].reshape(-1, 1)

            # MSE loss
            dy = y_pred - y_true
            total_loss += np.mean(np.square(dy))

            # Output layer gradients
            dWhy += np.dot(dy, h_states[-1].T)
            dby += dy

            # Backpropagate through time
            dh_next = np.dot(self.Why.T, dy)
            dc_next = np.zeros_like(c_states[0])

            for t in reversed(range(len(inputs))):
                h = h_states[t+1]
                h_prev = h_states[t]
                c = c_states[t+1]
                c_prev = c_states[t]
                f = f_list[t]
                i = i_list[t]
                o = o_list[t]
                c_tilde = c_tilde_list[t]
                x = inputs[t].reshape(-1, 1)

                # Gradient from next timestep
                dh = dh_next
                
                # Gradient through output gate
                do = dh * tanh(c)
                dc = dc_next + dh * o * (1 - tanh(c)**2)
                
                # Gradient through cell state update
                df = dc * c_prev
                di = dc * c_tilde
                dc_tilde = dc * i
                
                # Gate gradients through sigmoid and tanh
                do_raw = o * (1 - o) * do
                df_raw = f * (1 - f) * df
                di_raw = i * (1 - i) * di
                dc_tilde_raw = (1 - c_tilde**2) * dc_tilde

                # Accumulate gradients for weights
                xh = np.vstack((x, h_prev))
                dWf += np.dot(df_raw, xh.T)
                dWi += np.dot(di_raw, xh.T)
                dWo += np.dot(do_raw, xh.T)
                dWc += np.dot(dc_tilde_raw, xh.T)

                # Bias gradients
                dbf += df_raw
                dbi += di_raw
                dbo += do_raw
                dbc += dc_tilde_raw

                # Gradient to previous timestep
                dxh = (np.dot(self.Wf.T, df_raw) +
                      np.dot(self.Wi.T, di_raw) +
                      np.dot(self.Wo.T, do_raw) +
                      np.dot(self.Wc.T, dc_tilde_raw))

                dx = dxh[:self.input_size]
                dh_prev = dxh[self.input_size:]

                # Pass cell state gradient to previous timestep
                dc_prev = dc * f

                dh_next = dh_prev
                dc_next = dc_prev

        # Average gradients across batch
        grads = [dWf/batch_size, dWi/batch_size, dWc/batch_size, dWo/batch_size, dWhy/batch_size,
                 dbf/batch_size, dbi/batch_size, dbc/batch_size, dbo/batch_size, dby/batch_size]
        
        # Clip gradients
        grads = clip_gradients(grads, threshold=5.0)
        
        # Update weights using optimizer
        params = [self.Wf, self.Wi, self.Wc, self.Wo, self.Why,
                  self.bf, self.bi, self.bc, self.bo, self.by]
        self.optimizer.update(params, grads)

        return total_loss / batch_size

    def predict(self, inputs):
        outputs, _ = self.forward(inputs)
        return outputs

def train_numpy_model(model, X_train, y_train, X_test, y_test, scaler, epochs=10, batch_size=64, eval_frequency=20):
    n_samples = len(X_train)
    steps_per_epoch = n_samples // batch_size
    eval_interval = max(10, steps_per_epoch // eval_frequency)

    train_losses = []
    val_losses = []
    mse_values = []
    rmse_values = []
    mae_values = []
    batch_numbers = []

    batch_count = 0
    
    # Process validation data in smaller batches for efficient evaluation
    val_batch_size = 128
    
    for epoch in range(epochs):
        # Shuffle training data
        indices = np.random.permutation(n_samples)
        X_train_shuffled = X_train[indices]
        y_train_shuffled = y_train[indices]

        for start in range(0, n_samples, batch_size):
            batch_count += 1
            end = min(start + batch_size, n_samples)
            
            X_batch = X_train_shuffled[start:end]
            y_batch = y_train_shuffled[start:end]

            # Forward and backward pass
            y_pred, cache = model.forward(X_batch)
            batch_loss = model.backward(X_batch, cache, y_pred, y_batch)

            if batch_count % eval_interval == 0:
                # Record training loss
                train_losses.append(batch_loss)

                # Evaluate on validation set in batches
                val_size = min(500, len(X_test))
                val_indices = np.random.choice(len(X_test), val_size, replace=False)
                X_val_sample = X_test[val_indices]
                y_val_sample = y_test[val_indices]

                # Get predictions
                val_preds = model.predict(X_val_sample)
                val_loss = np.mean(np.square(val_preds - y_val_sample))
                val_losses.append(val_loss)

                # Calculate metrics in original scale
                y_val_inv = scaler.inverse_transform(y_val_sample)
                y_pred_inv = scaler.inverse_transform(val_preds.reshape(-1, 1)).flatten()

                mse = mean_squared_error(y_val_inv, y_pred_inv)
                rmse = np.sqrt(mse)
                mae = mean_absolute_error(y_val_inv, y_pred_inv)

                mse_values.append(mse)
                rmse_values.append(rmse)
                mae_values.append(mae)
                batch_numbers.append(batch_count)

                print(f"\rEpoch {epoch+1}/{epochs}, Batch {batch_count}: Loss={batch_loss:.4f}, Val Loss={val_loss:.4f}", end="")

    print("\nPerforming final evaluation...")
    
    # Evaluate on full test set in batches to prevent memory issues
    all_preds = []
    for i in range(0, len(X_test), val_batch_size):
        batch_end = min(i + val_batch_size, len(X_test))
        batch_preds = model.predict(X_test[i:batch_end])
        all_preds.append(batch_preds)
    
    y_pred = np.vstack(all_preds)
    
    # Transform back to original scale
    y_test_inv = scaler.inverse_transform(y_test)
    y_pred_inv = scaler.inverse_transform(y_pred)

    # Final metrics
    mse = mean_squared_error(y_test_inv, y_pred_inv)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test_inv, y_pred_inv)

    return {
        'model': model,
        'predictions': y_pred_inv.flatten(),  # Flatten for consistency with TF output
        'actual': y_test_inv.flatten(),
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'mse_values': mse_values,
        'rmse_values': rmse_values,
        'batch_numbers': batch_numbers
    }
