import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch import nn
from torch import optim
import pandas as pd
import itertools as it


class NeuralNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(NeuralNetwork, self).__init__()
        self.layer_1 = nn.Linear(input_dim, hidden_dim)
        nn.init.kaiming_uniform_(self.layer_1.weight, nonlinearity="relu")
        self.layer_2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer_3 = nn.Linear(hidden_dim, output_dim)
       
    def forward(self, x):
        x = torch.nn.functional.relu(self.layer_1(x))
        x = torch.nn.functional.relu(self.layer_2(x))
        x = torch.nn.functional.sigmoid(self.layer_3(x))
        return x

class NeuralNetwork5(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(NeuralNetwork5, self).__init__()
        self.layer_1 = nn.Linear(input_dim, hidden_dim)
        nn.init.kaiming_uniform_(self.layer_1.weight, nonlinearity="relu")
        self.layer_2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer_3 = nn.Linear(hidden_dim, hidden_dim)
        self.layer_4 = nn.Linear(hidden_dim, hidden_dim)
        self.layer_5 = nn.Linear(hidden_dim, output_dim)
       
    def forward(self, x):
        x = torch.nn.functional.relu(self.layer_1(x))
        x = torch.nn.functional.relu(self.layer_2(x))
        x = torch.nn.functional.relu(self.layer_3(x))
        x = torch.nn.functional.relu(self.layer_4(x))
        x = torch.nn.functional.sigmoid(self.layer_5(x))

        return x

class NeuralNetwork9(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(NeuralNetwork9, self).__init__()
        self.layer_1 = nn.Linear(input_dim, hidden_dim)
        nn.init.kaiming_uniform_(self.layer_1.weight, nonlinearity="relu")
        self.layer_2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer_3 = nn.Linear(hidden_dim, hidden_dim)
        self.layer_4 = nn.Linear(hidden_dim, hidden_dim)
        self.layer_5 = nn.Linear(hidden_dim, hidden_dim)
        self.layer_6 = nn.Linear(hidden_dim, hidden_dim)
        self.layer_7 = nn.Linear(hidden_dim, hidden_dim)
        self.layer_8 = nn.Linear(hidden_dim, hidden_dim)
        self.layer_9 = nn.Linear(hidden_dim, output_dim)
       
    def forward(self, x):
        x = torch.nn.functional.relu(self.layer_1(x))
        x = torch.nn.functional.relu(self.layer_2(x))
        x = torch.nn.functional.relu(self.layer_3(x))
        x = torch.nn.functional.relu(self.layer_4(x))
        x = torch.nn.functional.relu(self.layer_5(x))
        x = torch.nn.functional.relu(self.layer_6(x))
        x = torch.nn.functional.relu(self.layer_7(x))
        x = torch.nn.functional.relu(self.layer_8(x))
        x = torch.nn.functional.sigmoid(self.layer_9(x))
        return x

class Data(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X.to_numpy().astype(np.float32))
        self.y = torch.from_numpy(y.to_numpy().astype(np.float32))
        self.len = self.X.shape[0]
       
    def __getitem__(self, index):
        return self.X[index], self.y[index]
   
    def __len__(self):
        return self.len
   
batch_size = 20

data_train = pd.read_csv("train.csv", delimiter=',', header=None)
X = data_train.iloc[: , :-1]
y = data_train.iloc[: , -1]
print(X.shape)
print(y.shape)
train_data = Data(X, y)
train_dataloader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)

data_test = pd.read_csv("test.csv", delimiter=',', header=None)
X_test = data_test.iloc[:, :-1]
y_test = data_test.iloc[:, -1]
test_data = Data(X_test, y_test)
test_dataloader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=True)

input_dim = 4
hidden_dims = [5, 10, 25, 50, 100]

output_dim = 1
for hidden_dim in hidden_dims:
    model1 = NeuralNetwork(input_dim, hidden_dim, output_dim)
    model2 = NeuralNetwork5(input_dim, hidden_dim, output_dim)
    model3 = NeuralNetwork9(input_dim, hidden_dim, output_dim)
    models = [model1, model2, model3]
    layers = [3,5,9]
    for i, model in enumerate(models):
        loss_fn = nn.BCELoss()
        optimizer = torch.optim.Adam(model.parameters())
        num_epochs = 100
        loss_values = []
        errors = 0

        for epoch in range(num_epochs):
            for X, y in train_dataloader:
                optimizer.zero_grad()
                pred = model(X)
                loss = loss_fn(pred, y.unsqueeze(-1))
                loss.backward()
                optimizer.step()

        with torch.no_grad():
            for X, y in test_dataloader:
                outputs = model(X)
                predicted = np.where(outputs < 0.5, 0, 1)
                predicted = list(it.chain(*predicted))
                errors += abs(predicted - y.numpy()).sum()

        print(f"for hidden node count {hidden_dim} and model with depth {layers[i]}, error rate was ={errors/len(test_data)}")