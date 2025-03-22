from manual import MLP, ReLU, Sigmoid, MSE, MAE, MBE
import students

(X_train, Y_train), (X_test, Y_test) = students.data()
features = X_train.shape[1]

Y_train = Y_train.reshape(-1, 1)
Y_test = Y_test.reshape(-1, 1)

layer_sizes = [features, 64, 48, 24, 1]
activations = [ReLU(), ReLU(), ReLU(), Sigmoid()]
loss = MSE()

model = MLP(layer_sizes=layer_sizes, activations=activations, loss=loss)

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
