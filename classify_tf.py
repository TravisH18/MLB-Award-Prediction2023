import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

print('GPU name: ', tf.config.experimental.list_physical_devices('GPU'))


df1 = pd.read_csv('2018_batting.csv')
#df1.drop(columns=['Rk', 'Name', 'Age', 'Lg', 'Tm', 'Pos Summary','Name-additional'], axis=1)
df2 = pd.read_csv('2019_batting.csv')
df3 = pd.read_csv('2020_batting.csv')
df4 = pd.read_csv('2021_batting.csv')
df5 = pd.read_csv('2022_batting.csv')
#print(df1.head)
combined_df = pd.concat([df1, df2, df3, df4, df5], ignore_index=True)
combined_df = combined_df.dropna()
#print(combined_df.head)

X = combined_df.drop(['MVP', 'Name-additional'], axis=1)
Y = combined_df['MVP']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
batch_size = 64
# Define the RNN model
model = keras.Sequential([
    keras.layers.LSTM(64, return_sequences=True, input_shape=(X_train.shape[1], 1)),
    keras.layers.LSTM(64),
    keras.layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Display model summary
model.summary()

# Define Linear Model
# model = keras.Sequential([
#     tf.keras.layers.Input(shape=(X_train.shape[1],1)),
#     tf.keras.layers.Dense(128, activation='relu'),
#     tf.keras.layers.Dense(64, activation='relu'),
#     tf.keras.layers.Dense(1, activation='sigmoid')
# ])

# model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# model.summary()

num_epochs = 10  # Adjust as needed

model.fit(X_train, Y_train, epochs=num_epochs, batch_size=batch_size)

test_loss, test_accuracy = model.evaluate(X_test, Y_test, batch_size=batch_size)
model.save("tensorflowRNN_model")
print(f'Accuracy: {test_accuracy}')

# df2023 = pd.read_csv('2023_batting.csv')
# df2023 = df2023.dropna()

# df2023 = df2023.drop(['MVP', 'Name-additional'], axis=1)
# model.predict(df2023)
