import keras
import students

activation = 'relu'
save = True


def mean_bias_error(y_true, y_pred):
    return keras.ops.mean(y_pred - y_true)


(X_train, Y_train), (X_test, Y_test) = students.data()
features = X_train.shape[1]

model = keras.Sequential(
    [
        keras.layers.Input(shape=(features,)),
        keras.layers.Dense(64, activation=activation),
        keras.layers.Dense(48, activation=activation),
        keras.layers.Dense(24, activation=activation),
        keras.layers.Dense(1, activation='sigmoid'),
    ]
)

model.summary()

model.compile(
    optimizer=keras.optimizers.SGD(learning_rate=0.025),
    loss=keras.losses.MeanSquaredError,
    metrics=[
        keras.metrics.MeanAbsoluteError(name='MAE'),
        mean_bias_error,
    ],
)

model.fit(
    X_train,
    Y_train,
    batch_size=10,
    epochs=100,
    validation_data=(X_test, Y_test),
    validation_freq=10,
)

if save:
    model.save_weights('MLP.weights.h5')

score = model.evaluate(X_test, Y_test)

print(f'loss: {score[0]}, Mean Absolute Error: {score[1] * 20}, Mean Bias Error: {score[2] * 20}')
