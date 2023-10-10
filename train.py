import tensorflow as tf
from tensorflow import keras

#define model
model = keras.Sequential([
    keras.layers.Dense(64, activation='relu', input_shape=(num_features,)),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

#Load data
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))
