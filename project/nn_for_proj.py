import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch import nn
from torch import optim
import pandas as pd
import itertools as it
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt



class NeuralNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(NeuralNetwork, self).__init__()
        self.layer_1 = nn.Linear(input_dim, hidden_dim)
        self.layer_2 = nn.Linear(hidden_dim, output_dim)

        nn.init.kaiming_uniform_(self.layer_1.weight, nonlinearity="relu")
        #nn.init.kaiming_uniform_(self.layer_2.weight, nonlinearity="relu")


    def forward(self, x):
        x = torch.nn.functional.relu(self.layer_1(x))
        x = torch.nn.functional.sigmoid(self.layer_2(x))
        return x

class NeuralNetwork3(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(NeuralNetwork3, self).__init__()
        self.layer_1 = nn.Linear(input_dim, hidden_dim)
        self.layer_2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer_3 = nn.Linear(hidden_dim, output_dim)

        nn.init.kaiming_uniform_(self.layer_1.weight, nonlinearity="relu")
        nn.init.kaiming_uniform_(self.layer_2.weight, nonlinearity="relu")
       
    def forward(self, x):
        x = torch.nn.functional.relu(self.layer_1(x))
        x = torch.nn.functional.relu(self.layer_2(x))
        x = torch.sigmoid(self.layer_3(x))
        return x


class NeuralNetwork5(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(NeuralNetwork5, self).__init__()
        self.layer_1 = nn.Linear(input_dim, hidden_dim)
        self.layer_2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer_3 = nn.Linear(hidden_dim, hidden_dim)
        self.layer_4 = nn.Linear(hidden_dim, hidden_dim)
        self.layer_5 = nn.Linear(hidden_dim, hidden_dim)
        self.layer_6 = nn.Linear(hidden_dim, hidden_dim)
        self.layer_7 = nn.Linear(hidden_dim, hidden_dim)
        self.layer_8 = nn.Linear(hidden_dim, hidden_dim)
        self.layer_9 = nn.Linear(hidden_dim, hidden_dim)
        self.layer_10 = nn.Linear(hidden_dim, output_dim)
        nn.init.kaiming_uniform_(self.layer_1.weight, nonlinearity="relu")
        nn.init.kaiming_uniform_(self.layer_2.weight, nonlinearity="relu")
        nn.init.kaiming_uniform_(self.layer_3.weight, nonlinearity="relu")
        nn.init.kaiming_uniform_(self.layer_4.weight, nonlinearity="relu")
        nn.init.kaiming_uniform_(self.layer_5.weight, nonlinearity="relu")
        nn.init.kaiming_uniform_(self.layer_6.weight, nonlinearity="relu")
        nn.init.kaiming_uniform_(self.layer_7.weight, nonlinearity="relu")
        nn.init.kaiming_uniform_(self.layer_8.weight, nonlinearity="relu")
        nn.init.kaiming_uniform_(self.layer_9.weight, nonlinearity="relu")
        # nn.init.kaiming_uniform_(self.layer_10.weight, nonlinearity="relu")
       
    def forward(self, x):
        x = torch.nn.functional.relu(self.layer_1(x))
        x = torch.nn.functional.relu(self.layer_2(x))
        x = torch.nn.functional.relu(self.layer_3(x))
        x = torch.nn.functional.relu(self.layer_4(x))
        x = torch.nn.functional.relu(self.layer_5(x))
        x = torch.nn.functional.relu(self.layer_6(x))
        x = torch.nn.functional.relu(self.layer_7(x))
        x = torch.nn.functional.relu(self.layer_8(x))
        x = torch.nn.functional.relu(self.layer_9(x))
        x = torch.sigmoid(self.layer_10(x))
        return x

class NeuralNetwork9(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(NeuralNetwork9, self).__init__()
        self.layer_1 = nn.Linear(input_dim, hidden_dim)
        self.layer_2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer_3 = nn.Linear(hidden_dim, hidden_dim)
        self.layer_4 = nn.Linear(hidden_dim, hidden_dim)
        self.layer_5 = nn.Linear(hidden_dim, hidden_dim)
        self.layer_6 = nn.Linear(hidden_dim, hidden_dim)
        self.layer_7 = nn.Linear(hidden_dim, hidden_dim)
        self.layer_8 = nn.Linear(hidden_dim, hidden_dim)
        self.layer_9 = nn.Linear(hidden_dim, hidden_dim)
        self.layer_10 = nn.Linear(hidden_dim, hidden_dim)
        self.layer_11 = nn.Linear(hidden_dim, hidden_dim)
        self.layer_12 = nn.Linear(hidden_dim, hidden_dim)
        self.layer_13 = nn.Linear(hidden_dim, hidden_dim)
        self.layer_14 = nn.Linear(hidden_dim, hidden_dim)
        self.layer_15 = nn.Linear(hidden_dim, hidden_dim)
        self.layer_16 = nn.Linear(hidden_dim, hidden_dim)
        self.layer_17 = nn.Linear(hidden_dim, hidden_dim)
        self.layer_18 = nn.Linear(hidden_dim, hidden_dim)
        self.layer_19 = nn.Linear(hidden_dim, output_dim)

        nn.init.kaiming_uniform_(self.layer_1.weight, nonlinearity="relu")
        nn.init.kaiming_uniform_(self.layer_2.weight, nonlinearity="relu")
        nn.init.kaiming_uniform_(self.layer_3.weight, nonlinearity="relu")
        nn.init.kaiming_uniform_(self.layer_4.weight, nonlinearity="relu")
        nn.init.kaiming_uniform_(self.layer_5.weight, nonlinearity="relu")
        nn.init.kaiming_uniform_(self.layer_6.weight, nonlinearity="relu")
        nn.init.kaiming_uniform_(self.layer_7.weight, nonlinearity="relu")
        nn.init.kaiming_uniform_(self.layer_8.weight, nonlinearity="relu")
        nn.init.kaiming_uniform_(self.layer_9.weight, nonlinearity="relu")
        nn.init.kaiming_uniform_(self.layer_10.weight, nonlinearity="relu")
        nn.init.kaiming_uniform_(self.layer_11.weight, nonlinearity="relu")
        nn.init.kaiming_uniform_(self.layer_12.weight, nonlinearity="relu")
        nn.init.kaiming_uniform_(self.layer_13.weight, nonlinearity="relu")
        nn.init.kaiming_uniform_(self.layer_14.weight, nonlinearity="relu")
        nn.init.kaiming_uniform_(self.layer_15.weight, nonlinearity="relu")
        nn.init.kaiming_uniform_(self.layer_16.weight, nonlinearity="relu")
        nn.init.kaiming_uniform_(self.layer_17.weight, nonlinearity="relu")
        nn.init.kaiming_uniform_(self.layer_18.weight, nonlinearity="relu")
        nn.init.kaiming_uniform_(self.layer_19.weight, nonlinearity="relu")
       
    def forward(self, x):
        x = torch.nn.functional.relu(self.layer_1(x))
        x = torch.nn.functional.relu(self.layer_2(x))
        x = torch.nn.functional.relu(self.layer_3(x))
        x = torch.nn.functional.relu(self.layer_4(x))
        x = torch.nn.functional.relu(self.layer_5(x))
        x = torch.nn.functional.relu(self.layer_6(x))
        x = torch.nn.functional.relu(self.layer_7(x))
        x = torch.nn.functional.relu(self.layer_8(x))
        x = torch.nn.functional.relu(self.layer_9(x))
        x = torch.nn.functional.relu(self.layer_10(x))
        x = torch.nn.functional.relu(self.layer_11(x))
        x = torch.nn.functional.relu(self.layer_12(x))
        x = torch.nn.functional.relu(self.layer_13(x))
        x = torch.nn.functional.relu(self.layer_14(x))
        x = torch.nn.functional.relu(self.layer_15(x))
        x = torch.nn.functional.relu(self.layer_16(x))
        x = torch.nn.functional.relu(self.layer_17(x))
        x = torch.nn.functional.relu(self.layer_18(x))
        x = torch.nn.functional.sigmoid(self.layer_19(x))
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
   

def update_df_with_median(S):
    for col in ["age", "fnlwgt", "capital.gain", "capital.loss", "hours.per.week", "education.num"]:
        # median = S[col].median()
        # S[col] = pd.to_numeric(S[col])
        # S[col] = np.where(S[col]<=median , 0, 1)
        S[col] = preprocessing.normalize(np.array(S[col]).reshape(-1,1))

    return S



batch_size = 100

data = pd.read_csv("train_final.csv", delimiter=',', header=None) # names=["age", "job", "marital", "education", "default", "balance", "housing", "loan", "contact", "day", "month", "duration", "campaign", "pdays", "previous", "poutcome", "label"])
data_test = pd.read_csv("test_final.csv", delimiter=',', header=None) # names=["age", "job", "marital", "education", "default", "balance", "housing", "loan", "contact", "day", "month", "duration", "campaign", "pdays", "previous", "poutcome", "label"])
headers = data.iloc[0]
data.columns = headers
data = data[1:]
headers_test = data_test.iloc[0]
data_test = data_test[1:]
data_test.columns = headers_test
data_test= data_test.drop('ID', axis=1)

data = update_df_with_median(data)
data_test = update_df_with_median(data_test)

# for i in data.columns:
#     plt.figure()
#     plt.hist(data[i])
#     plt.savefig(f"{i}.png")

# data = scale(data)
# data_test = scale(data_test)
# find  NaN entries in your df
nanEntries = data[data['label'] == '0'].index.tolist()
# choose 10% randomly
dropIndices = np.random.choice(nanEntries, size = int(data.shape[0]*0.5))
# drop them
data = data.drop(dropIndices)

true_output = data["label"]
pd.options.display.max_columns = 100

data["sex"] = np.where(data["sex"] == 'Male', 0, 1)
data_test["sex"] = np.where(data_test["sex"] == 'Male', 0, 1)


data = data.drop("label", axis=1)


#one_hot_encoded_data = pd.get_dummies(data, columns = headers[:-1])
#one_hot_encoded_data_test = pd.get_dummies(data_test, columns = headers_test[1:])
categorical_columns = data.columns.difference(["age", "fnlwgt", "capital.gain", "capital.loss", "hours.per.week", "education.num", "sex"])
one_hot_encoded_data = pd.get_dummies(data, columns=categorical_columns)
one_hot_encoded_data_test = pd.get_dummies(data_test, columns=categorical_columns)
one_hot_encoded_data_reduced = one_hot_encoded_data.drop(columns=one_hot_encoded_data.columns.difference(one_hot_encoded_data_test.columns))
one_hot_encoded_data_test_reduced = one_hot_encoded_data_test.drop(columns=one_hot_encoded_data_test.columns.difference(one_hot_encoded_data.columns))

X_test = one_hot_encoded_data_test_reduced
y_test = one_hot_encoded_data_test_reduced.iloc[:, -1]


X_train, X_traintest, y_train, y_traintest = train_test_split(one_hot_encoded_data_reduced, true_output, test_size=0.2, random_state=42)

crosstrain_data = Data(X_traintest, y_traintest)
crosstrain_dataloader = DataLoader(dataset=crosstrain_data, batch_size=batch_size, shuffle=True)


test_data = Data(X_test, y_test)
test_dataloader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)

train_data = Data(X_train, y_train)
train_dataloader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)

input_dim = 106
hidden_dims = [80, 100, 200]

output_dim = 1
for hidden_dim in hidden_dims:
    model1 = NeuralNetwork(input_dim, hidden_dim, output_dim)
    model2 = NeuralNetwork5(input_dim, hidden_dim, output_dim)
    model3 = NeuralNetwork9(input_dim, hidden_dim, output_dim)
    model4 = NeuralNetwork3(input_dim, hidden_dim, output_dim)

    models =[model3, model2]
    layers = [3, 5]
    for i, model in enumerate(models):
        loss_fn = nn.BCELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=.0001)
        num_epochs = 250
        loss_values = []
        errors = 0
        final_predictions_train = list()
        final_predictions = list()
        loss_values = list()
        last_crosstrain = 10500
        counter = 0


        for epoch in range(num_epochs):
            crosstrain_error = 0

            for X, y in train_dataloader:
                optimizer.zero_grad()
                pred = model(X)
                loss = loss_fn(pred, y.unsqueeze(-1))
                loss_values.append(loss.item())
                loss.backward()
                optimizer.step()

            print("epoch, loss", epoch, loss.item())
            with torch.no_grad():
                for X, y in crosstrain_dataloader:
                    outputs = model(X).numpy()
                    predicted = list(it.chain(*outputs))
                    crosstrain_error += abs(predicted - y.numpy()).sum()
            
            print("crosstrain loss", crosstrain_error)

            if crosstrain_error > last_crosstrain:
                counter +=1
                #if counter > 10:
                 #   print("epoch", epoch)
                  #  break
            else:
                last_crosstrain = crosstrain_error       

        with torch.no_grad():
            for X, y in test_dataloader:
                outputs = model(X).numpy()
                predicted = list(it.chain(*outputs))
                final_predictions += predicted

        print("final loss value", loss_values[-1])
        print("error, hidden dim", errors, hidden_dim)

        ids = np.arange(1, len(data_test) + 1)
        df2 = pd.DataFrame({"Id" : ids, "Prediction" : np.where(np.asarray(final_predictions) < .5, 0 ,1)})
        df = pd.DataFrame({"Id" : ids, "Prediction" : final_predictions})
        df.to_csv(f"nn-{i}-{hidden_dim}-no_shuffle.csv", index=False)   
        df2.to_csv(f"nn-{i}-{hidden_dim}-final.csv", index=False)   

