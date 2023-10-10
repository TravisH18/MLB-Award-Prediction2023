import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split


#define model
model = keras.Sequential([
    keras.layers.Dense(64, activation='relu', input_shape=(num_features,)),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

#Load data
# Assuming 'MVP' is your target variable and you want to predict it
df1 = pd.read_csv('2018_batting.csv')
df2 = pd.read_csv('2019_batting.csv')
df3 = pd.read_csv('2020_batting.csv')
df4 = pd.read_csv('2021_batting.csv')
df5 = pd.read_csv('2022_batting.csv')
combined_df = pd.concat([df1, df2, df3, df4, df5], ignore_index=True)
X = combined_df.drop('MVP', axis=1)  # Features
y = combined_df['MVP']  # Target variable

# Split the data into a training set (80%) and a testing set (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))
