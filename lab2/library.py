import keras
import students

activation = 'relu'

(X_train, Y_train), (X_test, Y_test) = students.data()

model = keras.Sequential(
    [
        keras.layers.Input(shape=(37,)),
        keras.layers.Dense(256, activation=activation),
        keras.layers.Dense(512, activation=activation),
        keras.layers.Dense(1024, activation=activation),
        keras.layers.Dense(1024, activation=activation),
        keras.layers.Dense(512, activation=activation),
        keras.layers.Dense(128, activation=activation),
        keras.layers.Dense(1, activation=activation),
    ]
)

model.summary()

model.compile(
    optimizer='sgd',
    loss='mse',
    metrics=['mae'],
)

model.fit(
    X_train,
    Y_train,
    batch_size=10,
    epochs=100,
    validation_data=(X_test, Y_test),
)

score = model.evaluate(X_test, Y_test)

print(f'loss: {score[0]}, Mean Absolute Error: {score[1] * 20}')
