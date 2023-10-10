import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
import pandas as pd


def get_model():
    model = keras.Sequential([
        keras.layers.Dense(64, activation='relu', input_shape=(BATCH_SIZE, num_features)),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model
#Load data
# 'MVP' is your target variable and you want to predict it
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

X = combined_df.drop(['MVP', 'Name-additional'], axis=1)  # Features
#X = X.drop('Name-additional', axis=1)
#print(X.head)
#could use X.values
X = tf.convert_to_tensor(X.values)
#y = combined_df['MVP']  # Target variable
y = combined_df.pop('MVP')
key = combined_df['Name-additional']
BATCH_SIZE = 32
EPOCHS = 50
num_features = X.shape[1]

#define normalizer
normalizer = tf.keras.layers.Normalization(axis=-1)
normalizer.adapt(X)

#define model
model = get_model()
# Split the data into a training set (80%) and a testing set (20%)
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model.fit(X, y, epochs=EPOCHS, batch_size=BATCH_SIZE)
