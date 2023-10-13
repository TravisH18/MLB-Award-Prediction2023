import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class BinaryClassifier(nn.Module):
    def __init__(self, input_size):
        super(BinaryClassifier, self).__init__()
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

class RNNClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, batch_size):
        super(RNNClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers)
        self.fc = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        print(f"Forward input {x.shape}")
        h0 = torch.zeros(self.num_layers, self.hidden_size).to("cuda") 
        c0 = torch.zeros(self.num_layers, self.hidden_size).to("cuda")
        out, _ = self.rnn(x, (h0, c0))
        out = self.fc(out[:, -1, :])  # Use the output of the last time step
        out = self.sigmoid(out)  # Apply sigmoid for binary classification
        return out

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    df1 = pd.read_csv('2018_batting.csv')
    df2 = pd.read_csv('2019_batting.csv')
    df3 = pd.read_csv('2020_batting.csv')
    df4 = pd.read_csv('2021_batting.csv')
    df5 = pd.read_csv('2022_batting.csv')
    combined_df = pd.concat([df1, df2, df3, df4, df5], ignore_index=True)
    combined_df = combined_df.dropna()

    X = combined_df.drop(['MVP', 'Name-additional'], axis=1)
    Y = combined_df['MVP']
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
    train_dataset = TensorDataset(torch.tensor(X_train.values, dtype=torch.float), torch.tensor(Y_train.values, dtype=torch.float))
    test_dataset = TensorDataset(torch.tensor(X_test.values, dtype=torch.float), torch.tensor(Y_test.values, dtype=torch.float))

    batch_size = 64  # Adjust as needed
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # Define hyperparameters
    input_size = X_train.shape[1]  # Number of features
    hidden_size = 64  # Number of hidden units
    num_layers = 2  # Number of LSTM layers
    output_size = 1  # Binary classification output

    # Create the RNN model
    model = RNNClassifier(input_size, hidden_size, num_layers, output_size, batch_size).to(device)
    print(model)
    # input_size = X_train.shape[1]  # Adjust according to the number of features
    # model = BinaryClassifier(input_size).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 10  # Adjust as needed

    for epoch in range(num_epochs):
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            print(f"Inputs {inputs.shape}")
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
    print(f'Pytorch Model Accuracy: {accuracy}')

    # df2023 = pd.read_csv('2023_batting.csv')
    # df2023 = df2023.dropna()

    # df2023 = df2023.drop(['MVP', 'Name-additional'], axis=1)
    # input_sequence = torch.Tensor(df2023.values) #Convert 2023df.values to tensor here
    # output = model(input_sequence)
    # prediction = output[-1].item()
    # print(prediction)

if __name__ == '__main__':
    main()