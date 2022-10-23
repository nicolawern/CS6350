from cProfile import run
from re import T
from typing import final
import pandas as pd
import math
import numpy as np
import random
import datetime
import matplotlib.pyplot as plt
import sys


def lms_batch(S):
    lr = [.001, .005]
    features = S.drop("label", axis=1) #todo drop other labels

    for i in range(0, len(lr)):
        cost_fn = []
        print("learning rate ", lr[i])
        weights = [0] * (len(S.columns) -1)

        for t in range(0, 15000):
            grad = pd.Series(calc_gradient(weights, S))
            new_weights = weights - lr[i] * grad
            cost_fn.append(.5 * np.sum((S["label"] - np.inner(weights, features))**2))
            norm = np.linalg.norm(new_weights - weights)
            if norm < .0000006: 
                print("error below threshold, weights and lr ", np.asarray(new_weights), lr[i])
                plot_from_array(cost_fn, "Cost fn", "Cost Function For Learning Rate " + str(lr[i]))
                return weights

            weights = new_weights

    return [0] * (len(S.columns) -1)


def calc_gradient(weights, S):
    features = S.drop("label", axis=1)
    grad = []
    for i in range(0, len(features.columns)):
        grad.append(-1 * np.sum((S["label"] - np.inner(weights, features)) * features.iloc[:, i]))

    return grad


def lms_stochastic(S):
    lr = [.03, .01]
    features = S.drop("label", axis=1)
    for i in range(0, len(lr)):
        print("learning rate ", lr[i])
        weights = [0] * len(features.columns)
        cost_fn = []
        for t in range(0, 150000):
            row = S.iloc[(t % len(S)), :]
            row_features = row.drop("label")
            grad = (row["label"] - np.inner(weights, row_features)) * row_features
            new_weights = weights + lr[i] * grad
            weights = new_weights
            cost_fn.append(.5 * np.sum((S["label"] - np.inner(weights, features))**2))

            if len(cost_fn) > 2 and np.abs(cost_fn[-1] - cost_fn[-2]) < .000000006:
                print("convergence weights", np.asarray(weights))
                plot_from_array(cost_fn, "Cost fn", "Cost Function For Learning Rate " + str(lr[i]))
                return weights
    
    return [0] * len(features.columns)
    


def plot_from_array(line1, label1, title):
    plt.plot(np.asarray(line1), label=label1)
    plt.title(title)
    plt.xlabel("Iterations")
    plt.ylabel("Cost")
    plt.legend()
    plt.show()

def run_lms_batch(train="concrete-train.csv", test="concrete-test.csv", col_names=["Cement","Slag","Fly ash", "Water", "SP", "Coarse Aggr", "Fine Aggr", "label"]):

    data = pd.read_csv(train, delimiter=',', header=None, names=col_names)
    data_test = pd.read_csv(test, delimiter=',', header=None, names=col_names)


    weights = lms_batch(data)
    cost = .5 * np.sum((data_test["label"] - np.inner(weights, data_test.drop("label", axis=1)))**2)

    print("cost of test data is", cost)

def run_lms_gradient(train="concrete-train.csv", test="concrete-test.csv", col_names=["Cement","Slag","Fly ash", "Water", "SP", "Coarse Aggr", "Fine Aggr", "label"]):

    data = pd.read_csv(train, delimiter=',', header=None, names=col_names)
    data_test = pd.read_csv(test, delimiter=',', header=None, names=col_names)
    weights = lms_stochastic(data)
    cost = .5 * np.sum((data_test["label"] - np.inner(weights, data_test.drop("label", axis=1)))**2)
    print("cost of test data is", cost)

def run(batch, stochastic):
    print(batch, stochastic)
    if batch == "True":
        print("running batch gradient descent")
        run_lms_batch()
    
    if stochastic == "True":
        print("running stochastic gradient descent")
        run_lms_gradient()

run(batch=sys.argv[1], stochastic=sys.argv[2])