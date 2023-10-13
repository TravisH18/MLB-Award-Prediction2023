import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class BinaryClassifier(nn.Module):
    def __init__(self, input_size):
        super(BinaryClassifier, self).__init()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        x = self.sigmoid(x)
        return x


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
X = X.values
#y = combined_df['MVP']  # Target variable
Y = combined_df['MVP'].values
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)


train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float), torch.tensor(Y_train, dtype=torch.float))
test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float), torch.tensor(Y_test, dtype=torch.float))

batch_size = 64  # Adjust as needed
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

input_size = X_train.shape[1]  # Adjust according to the number of features
model = BinaryClassifier(input_size)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 10  # Adjust as needed

for epoch in range(num_epochs):
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels.view(-1, 1))
        loss.backward()
        optimizer.step()

model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for inputs, labels in test_loader:
        outputs = model(inputs)
        predicted = (outputs > 0.5).float()
        total += labels.size(0)
        correct += (predicted == labels.view(-1, 1)).sum().item()

accuracy = correct / total
print(f'Accuracy: {accuracy}')

