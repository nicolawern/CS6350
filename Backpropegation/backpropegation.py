import numpy as np
import sys
import pandas as pd


def sig(weights, input):
    x = sum([weights[i] * input[i] for i in range(len(weights))])
    return 1/(1 + np.exp(-x))


def feedforward(network, row): #todo something is wrong here
    input = row
    num_layers = len(network)
    for i in range(0, num_layers-1): #skip output layer
        new_input = [1] #bias term
        for neuron in network[i]:
            if len(neuron["weights"]) != 0:
                neuron["output"] = sig(neuron["weights"], input)
                new_input.append(neuron["output"])
        input = new_input
    
    weights = network[num_layers-1][0]["weights"] #output weights
    prediction = sum([weights[i] * input[i] for i in range(len(weights))])
    network[num_layers-1][0]["output"] = prediction
    return prediction

def build_network_0(num_inputs, width): #is num_inputs count of fields? ithinkso
    network = list()
    layer1 =[{"weights": [0] * num_inputs} for i in range(width)]
    layer1[0]["weights"] = []
    layer1[0]["output"] = 1 #todo is this bias
    network.append(layer1)
    layer2 = [{"weights": [0] * width} for i in range(width)]
    layer2[0]["weights"] = []
    layer2[0]["output"] = 1 #todo is this bias
    network.append(layer2)
    output = [{"weights": [0] * width}] #assumes 1 output
    network.append(output)
    return network


def build_network(num_inputs, width): #is num_inputs count of fields? ithinkso
    network = list()
    layer1 =[{"weights": np.random.normal(size=num_inputs)} for i in range(width)]
    layer1[0]["weights"] = []
    layer1[0]["output"] = 1 #todo is this bias
    network.append(layer1)
    layer2 = [{"weights": np.random.normal(size=width)} for i in range(width)]
    layer2[0]["weights"] = []
    layer2[0]["output"] = 1 #todo is this bias
    network.append(layer2)
    output = [{"weights": np.random.normal(size=width)}] #assumes 1 output
    network.append(output)
    return network

#todo where to add bias
def print_partials_backpropegation(network, y):
    for i in reversed(range(len(network))):
        layer = network[i]
        errors = [] #errors is partial of node
        if i != len(network)-1:
            for j in range(1, len(layer)): 
                error = 0.0
                for neuron in network[i + 1]:
                    if len(neuron["weights"]) != 0:
                        error += (neuron['weights'][j] * neuron['derivative'])
                errors.append(error)
        else:
            for j in range(0, len(layer)):
                neuron = layer[j]
                errors.append(neuron['output'] - y)
        for k in range(len(layer)): #todo something wrong with bias here. either need to add it with empty weights? it just needs neuron_derivative since error is 1
            neuron = layer[k] #a layer has the two not bias nodes, the derivative is the weight which is error (node) * deriv
            if i == len(network)-1:
                for j in network[i-1]:
                    print("partial", errors[k] * j["output"])
                neuron['derivative'] = errors[k] #derivative is partial of node 
            elif len(neuron['weights']) != 0:
                if errors[0] != np.array(1):
                    errors.insert(0, np.array(1))
                #print partial
                if(i > 0):
                    for j in network[i-1]:
                        print("partial", errors[k] * neuron_derivative(neuron['output']) * j["output"])
                else:
                    for j in range(3): #todo hwo do we know how many inputs
                        print("partial", errors[k] * neuron_derivative(neuron['output'])) #todo here should be * input param
                neuron['derivative'] = errors[k] * neuron_derivative(neuron['output']) #i think *neuron(output) is wrong here


def backpropegation(network, y):
    for i in reversed(range(len(network))):
        layer = network[i]
        errors = [] #errors is partial of node
        if i != len(network)-1:
            for j in range(1, len(layer)): 
                error = 0.0
                for neuron in network[i + 1]:
                    if neuron["weights"] != []:
                        error += (neuron['weights'][j] * neuron['derivative'])
                errors.append(error)
        else:
            for j in range(0, len(layer)):
                neuron = layer[j]
                errors.append(neuron['output'] - y)
        for k in range(len(layer)): #todo something wrong with bias here. either need to add it with empty weights? it just needs neuron_derivative since error is 1
            neuron = layer[k] #a layer has the two not bias nodes, the derivative is the weight which is error (node) * deriv
            if i == len(network)-1:
                neuron['derivative'] = errors[k] #derivative is partial of node 
            elif neuron['weights'] != []:
                if errors[0] != np.array(1):
                    errors.insert(0, np.array(1))
                #print partial
                neuron['derivative'] = errors[k] * neuron_derivative(neuron['output']) #i think *neuron(output) is wrong here


def neuron_derivative(output):
    return output * (1.0 - output)

def train_network(network, train):
    for epoch in range(50):
        total_error = 0 
        train = train.sample(frac=1, replace=False).reset_index(drop=True) #shuffle
        for index, row in train.iterrows():
            y = row.label
            row = row.drop("label")
            lr = lr_update(index)
            prediction = feedforward(network, row)
            total_error += abs(np.rint(prediction) - y)
            backpropegation(network, y)
            update_weights(network, row, lr)
        print(f"epoch {epoch}, lr{lr}, error={total_error}")

def lr_update(t):
    return .01/(1 + .01 * t / 3)

def update_weights(network, row, lr): #todo something with bias here
    for i in range(len(network)):
        inputs = row
        if i != 0:
            inputs = [neuron['output'] for neuron in network[i - 1]]
        for neuron in network[i]:
            if len(neuron["weights"]) != 0:
                for j in range(len(inputs)):
                    neuron['weights'][j] -= lr * neuron['derivative'] * inputs[j]
                neuron['weights'][-1] -= lr * neuron['derivative']

def run(run_partial=True, run_ff=True):
    #todo test bias in feed forward
    col_names=["variance","skewness","curtosis", "entropy", "label"]
    data = pd.read_csv("train.csv", delimiter=',', header=None, names=col_names)
    data_test = pd.read_csv("test.csv", delimiter=',', header=None, names=col_names)

    if run_ff:
        width = [5,10,25,50,100]
        for i in width:
            network = build_network(4, i)
            train_network(network, data)
            total_error = 0
            for index, row in data_test.iterrows():
                y = row["label"]
                row = row.drop("label")
                prediction = feedforward(network, row)
                total_error += abs(np.rint(prediction) - y)
            print(f"width = {i} and test error is {total_error}")

        if run_partial:
            network = build_network(4, 4)
            row = data_test.iloc[0]
            y = row["label"]
            row = row.drop("label")
            prediction = feedforward(network, row)
            print_partials_backpropegation(network, y)


#run(print_partials=sys.argv[1], run_ff=sys.argv[2])
run(True, True)

    # network2 = [[{'weights': [], 'output': 1}, {'weights': [-1, -2, -3]}, {'weights': [1, 2, 3]}],
    # [{'weights': [], 'output': 1}, {'weights': [-1, -2, -3]}, {'weights': [1, 2, 3]}], [{'weights' : [-1, 2, -1.5]}]]
    #print_partials_backpropegation(network, [1])
    # prediction = sum([output_weights[i] * output[i] for i in range(len(output_weights))])
    # print(prediction)