import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

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
#y = combined_df['MVP']  # Target variable
Y = combined_df['MVP']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

classifier = RandomForestClassifier(n_estimators=100, random_state=42)
classifier.fit(X_train, Y_train)

Y_pred = classifier.predict(X_test)

accuracy = accuracy_score(Y_test, Y_pred)
print(f'Accuracy: {accuracy}')

report = classification_report(Y_test, Y_pred)
print('Classification Report:\n', report)

conf_matrix = confusion_matrix(Y_test, Y_pred)
print('Confusion Matrix:\n', conf_matrix)
