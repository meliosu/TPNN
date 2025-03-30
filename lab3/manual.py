import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

class SimpleRNN:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        
        self.Wxh = np.random.randn(hidden_size, input_size) * 0.01
        self.Whh = np.random.randn(hidden_size, hidden_size) * 0.01
        self.Why = np.random.randn(output_size, hidden_size) * 0.01
        self.bh = np.zeros((hidden_size, 1))
        self.by = np.zeros((output_size, 1))
    
    def forward(self, inputs):
        h = np.zeros((self.hidden_size, 1))
        hidden_states = [h]
        
        for x in inputs:
            x = x.reshape(-1, 1)
            h = np.tanh(np.dot(self.Wxh, x) + np.dot(self.Whh, h) + self.bh)
            hidden_states.append(h)
        
        y = np.dot(self.Why, hidden_states[-1]) + self.by
        return y, hidden_states
    
    def backward(self, inputs, hidden_states, y_pred, y_true):
        dWxh = np.zeros_like(self.Wxh)
        dWhh = np.zeros_like(self.Whh)
        dWhy = np.zeros_like(self.Why)
        dbh = np.zeros_like(self.bh)
        dby = np.zeros_like(self.by)
        
        dy = y_pred - y_true
        dWhy = np.dot(dy, hidden_states[-1].T)
        dby = dy
        
        dh_next = np.dot(self.Why.T, dy)
        
        for t in reversed(range(len(inputs))):
            h = hidden_states[t+1]
            h_prev = hidden_states[t]
            x = inputs[t].reshape(-1, 1)
            
            dtanh = (1 - h * h) * dh_next
            dbh += dtanh
            dWxh += np.dot(dtanh, x.T)
            dWhh += np.dot(dtanh, h_prev.T)
            
            dh_next = np.dot(self.Whh.T, dtanh)
        
        for param, dparam in zip([self.Wxh, self.Whh, self.Why, self.bh, self.by], 
                                 [dWxh, dWhh, dWhy, dbh, dby]):
            param -= self.learning_rate * dparam
            
        mse = np.mean(np.square(dy))
        return mse

    def predict(self, inputs):
        results = []
        for sequence in inputs:
            output, _ = self.forward(sequence)
            results.append(output.flatten())
        return np.array(results)

class GRU:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        
        self.Wz = np.random.randn(hidden_size, input_size + hidden_size) * 0.01
        self.Wr = np.random.randn(hidden_size, input_size + hidden_size) * 0.01
        self.Wh = np.random.randn(hidden_size, input_size + hidden_size) * 0.01
        self.Why = np.random.randn(output_size, hidden_size) * 0.01
        
        self.bz = np.zeros((hidden_size, 1))
        self.br = np.zeros((hidden_size, 1))
        self.bh = np.zeros((hidden_size, 1))
        self.by = np.zeros((output_size, 1))
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -15, 15)))
    
    def forward(self, inputs):
        h = np.zeros((self.hidden_size, 1))
        hidden_states = [h]
        z_list, r_list, h_tilde_list = [], [], []
        
        for x in inputs:
            x = x.reshape(-1, 1)
            xh = np.vstack((x, h))
            
            z = self.sigmoid(np.dot(self.Wz, xh) + self.bz)
            r = self.sigmoid(np.dot(self.Wr, xh) + self.br)
            
            h_reset = r * h
            xh_reset = np.vstack((x, h_reset))
            h_tilde = np.tanh(np.dot(self.Wh, xh_reset) + self.bh)
            
            h = (1 - z) * h + z * h_tilde
            
            hidden_states.append(h)
            z_list.append(z)
            r_list.append(r)
            h_tilde_list.append(h_tilde)
        
        y = np.dot(self.Why, hidden_states[-1]) + self.by
        cache = (hidden_states, z_list, r_list, h_tilde_list)
        return y, cache
    
    def backward(self, inputs, cache, y_pred, y_true):
        hidden_states, z_list, r_list, h_tilde_list = cache
        
        dWz = np.zeros_like(self.Wz)
        dWr = np.zeros_like(self.Wr)
        dWh = np.zeros_like(self.Wh)
        dWhy = np.zeros_like(self.Why)
        
        dbz = np.zeros_like(self.bz)
        dbr = np.zeros_like(self.br)
        dbh = np.zeros_like(self.bh)
        dby = np.zeros_like(self.by)
        
        dy = y_pred - y_true
        dWhy = np.dot(dy, hidden_states[-1].T)
        dby = dy
        
        dh_next = np.dot(self.Why.T, dy)
        
        for t in reversed(range(len(inputs))):
            h = hidden_states[t+1]
            h_prev = hidden_states[t]
            z = z_list[t]
            r = r_list[t]
            h_tilde = h_tilde_list[t]
            x = inputs[t].reshape(-1, 1)
            
            dh = dh_next
            dh_tilde = dh * z
            dz = dh * (h_tilde - h_prev)
            
            xh_reset = np.vstack((x, r * h_prev))
            dhtilde_raw = (1 - h_tilde**2) * dh_tilde
            dWh += np.dot(dhtilde_raw, xh_reset.T)
            dbh += dhtilde_raw
            
            dxh_reset = np.dot(self.Wh.T, dhtilde_raw)
            dx_reset = dxh_reset[:self.input_size]
            dh_reset = dxh_reset[self.input_size:]
            
            dr = dh_reset * h_prev
            
            xh = np.vstack((x, h_prev))
            dz_raw = z * (1 - z) * dz
            dWz += np.dot(dz_raw, xh.T)
            dbz += dz_raw
            
            dr_raw = r * (1 - r) * dr
            dWr += np.dot(dr_raw, xh.T)
            dbr += dr_raw
            
            dxh = np.dot(self.Wz.T, dz_raw) + np.dot(self.Wr.T, dr_raw)
            dx = dxh[:self.input_size] + dx_reset
            dh_prev = dxh[self.input_size:] + dh * (1 - z) + dr * r
            
            dh_next = dh_prev
        
        for param, dparam in zip([self.Wz, self.Wr, self.Wh, self.Why, self.bz, self.br, self.bh, self.by],
                                [dWz, dWr, dWh, dWhy, dbz, dbr, dbh, dby]):
            param -= self.learning_rate * dparam
        
        mse = np.mean(np.square(dy))
        return mse

    def predict(self, inputs):
        results = []
        for sequence in inputs:
            output, _ = self.forward(sequence)
            results.append(output.flatten())
        return np.array(results)

class LSTM:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        
        self.Wf = np.random.randn(hidden_size, input_size + hidden_size) * 0.01
        self.Wi = np.random.randn(hidden_size, input_size + hidden_size) * 0.01
        self.Wc = np.random.randn(hidden_size, input_size + hidden_size) * 0.01
        self.Wo = np.random.randn(hidden_size, input_size + hidden_size) * 0.01
        self.Why = np.random.randn(output_size, hidden_size) * 0.01
        
        self.bf = np.zeros((hidden_size, 1))
        self.bi = np.zeros((hidden_size, 1))
        self.bc = np.zeros((hidden_size, 1))
        self.bo = np.zeros((hidden_size, 1))
        self.by = np.zeros((output_size, 1))
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -15, 15)))
    
    def forward(self, inputs):
        h = np.zeros((self.hidden_size, 1))
        c = np.zeros((self.hidden_size, 1))
        
        h_states, c_states = [h], [c]
        f_list, i_list, o_list, c_tilde_list = [], [], [], []
        
        for x in inputs:
            x = x.reshape(-1, 1)
            xh = np.vstack((x, h))
            
            f = self.sigmoid(np.dot(self.Wf, xh) + self.bf)
            i = self.sigmoid(np.dot(self.Wi, xh) + self.bi)
            o = self.sigmoid(np.dot(self.Wo, xh) + self.bo)
            c_tilde = np.tanh(np.dot(self.Wc, xh) + self.bc)
            
            c = f * c + i * c_tilde
            h = o * np.tanh(c)
            
            h_states.append(h)
            c_states.append(c)
            f_list.append(f)
            i_list.append(i)
            o_list.append(o)
            c_tilde_list.append(c_tilde)
        
        y = np.dot(self.Why, h_states[-1]) + self.by
        cache = (h_states, c_states, f_list, i_list, o_list, c_tilde_list)
        return y, cache
    
    def backward(self, inputs, cache, y_pred, y_true):
        h_states, c_states, f_list, i_list, o_list, c_tilde_list = cache
        
        dWf, dWi, dWc, dWo, dWhy = np.zeros_like(self.Wf), np.zeros_like(self.Wi), np.zeros_like(self.Wc), np.zeros_like(self.Wo), np.zeros_like(self.Why)
        dbf, dbi, dbc, dbo, dby = np.zeros_like(self.bf), np.zeros_like(self.bi), np.zeros_like(self.bc), np.zeros_like(self.bo), np.zeros_like(self.by)
        
        dy = y_pred - y_true
        dWhy = np.dot(dy, h_states[-1].T)
        dby = dy
        
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
            
            dh = dh_next
            dc = dc_next + dh * o * (1 - np.tanh(c)**2)
            
            do = dh * np.tanh(c)
            df = dc * c_prev
            di = dc * c_tilde
            dc_tilde = dc * i
            
            do_raw = o * (1 - o) * do
            df_raw = f * (1 - f) * df
            di_raw = i * (1 - i) * di
            dc_tilde_raw = (1 - c_tilde**2) * dc_tilde
            
            xh = np.vstack((x, h_prev))
            dWf += np.dot(df_raw, xh.T)
            dWi += np.dot(di_raw, xh.T)
            dWo += np.dot(do_raw, xh.T)
            dWc += np.dot(dc_tilde_raw, xh.T)
            
            dbf += df_raw
            dbi += di_raw
            dbo += do_raw
            dbc += dc_tilde_raw
            
            dxh = (np.dot(self.Wf.T, df_raw) + 
                  np.dot(self.Wi.T, di_raw) + 
                  np.dot(self.Wo.T, do_raw) + 
                  np.dot(self.Wc.T, dc_tilde_raw))
            
            dx = dxh[:self.input_size]
            dh_prev = dxh[self.input_size:]
            
            dc_prev = dc * f
            
            dh_next = dh_prev
            dc_next = dc_prev
        
        for param, dparam in zip([self.Wf, self.Wi, self.Wc, self.Wo, self.Why, self.bf, self.bi, self.bc, self.bo, self.by],
                                [dWf, dWi, dWc, dWo, dWhy, dbf, dbi, dbc, dbo, dby]):
            param -= self.learning_rate * dparam
        
        mse = np.mean(np.square(dy))
        return mse

    def predict(self, inputs):
        results = []
        for sequence in inputs:
            output, _ = self.forward(sequence)
            results.append(output.flatten())
        return np.array(results)

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
    for epoch in range(epochs):
        indices = np.random.permutation(n_samples)
        for start in range(0, n_samples, batch_size):
            batch_count += 1
            end = min(start + batch_size, n_samples)
            batch_indices = indices[start:end]
            
            batch_loss = 0
            for idx in batch_indices:
                X = X_train[idx]
                y_true = y_train[idx].reshape(-1, 1)
                y_pred, cache = model.forward(X)
                loss = model.backward(X, cache, y_pred, y_true)
                batch_loss += loss
            
            batch_loss /= len(batch_indices)
            
            if batch_count % eval_interval == 0:
                train_losses.append(batch_loss)
                
                val_size = min(500, len(X_test))
                val_indices = np.random.choice(len(X_test), val_size, replace=False)
                X_val_sample = X_test[val_indices]
                y_val_sample = y_test[val_indices]
                
                val_preds = model.predict(X_val_sample)
                val_loss = np.mean(np.square(val_preds - y_val_sample))
                val_losses.append(val_loss)
                
                y_val_inv = scaler.inverse_transform(y_val_sample)
                y_pred_inv = scaler.inverse_transform(val_preds.reshape(-1, 1)).flatten()
                
                mse = mean_squared_error(y_val_inv, y_pred_inv)
                rmse = np.sqrt(mse)
                mae = mean_absolute_error(y_val_inv, y_pred_inv)
                
                mse_values.append(mse)
                rmse_values.append(rmse)
                mae_values.append(mae)
                batch_numbers.append(batch_count)
                
                print(f"\rBatch {batch_count}: Loss={batch_loss:.4f}, Val Loss={val_loss:.4f}", end="")
    
    print("\nPerforming final evaluation...")
    y_pred = model.predict(X_test)
    
    y_test_inv = scaler.inverse_transform(y_test)
    y_pred_inv = scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()
    
    mse = mean_squared_error(y_test_inv, y_pred_inv)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test_inv, y_pred_inv)
    
    return {
        'model': model,
        'predictions': y_pred_inv,
        'actual': y_test_inv,
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'mse_values': mse_values,
        'rmse_values': rmse_values,
        'batch_numbers': batch_numbers
    }
