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

def averaged_perceptron(S, data_test):
    lrs = [.1]
    features = S.drop("label", axis=1)
    for j in range(0, len(lrs)):
        weights = [0] * len(features.columns)
        average = [0] * len(features.columns)
        for t in range(0, 10):
            S = S.sample(frac=1, replace=False).reset_index(drop=True)
            for i in S.index:
                row = S.iloc[i, :]
                row_features = row.drop("label")
                prediction = np.sign(np.inner(weights, row_features))
                if prediction != row["label"]:
                    weights = weights + lrs[j] * np.inner(row["label"], row_features)
                average += weights
        
        data_test_no_label = data_test.drop("label", axis=1)
        final_predictions = np.sign(np.inner(weights, data_test_no_label))
        average_error = np.sum(((final_predictions - data_test["label"])/2).replace(-1, 1))/len(data_test)
        print("average perceptron error", average_error)
        print("final weights", weights)


def voted_perceptron(S, data_test):
    lrs = [.1]
    successes = 1
    successes_array = []
    features = S.drop("label", axis=1)
    for j in range(0, len(lrs)):
        weights = [0] * len(features.columns)
        weights_and_successes = []
        for t in range(0, 10):
            S = S.sample(frac=1, replace=False).reset_index(drop=True)
            for i in S.index:
                row = S.iloc[i, :]
                row_features = row.drop("label")
                prediction = np.sign(np.inner(weights, row_features))
                if prediction != row["label"]:
                    successes_array.append(successes)
                    weights_and_successes.append([weights])
                    weights = weights + lrs[j] * np.inner(row["label"], row_features)
                    successes = 1
                else:
                    successes += 1

        sign_for_all_weights_and_data = np.sign(np.inner(weights_and_successes, data_test.drop("label", axis=1))[:,0])
        df = pd.DataFrame(sign_for_all_weights_and_data)
        df = df.multiply(successes_array, axis='rows')
        final_predictions = np.sign(np.sum(df))
        average_error = np.sum(((final_predictions - data_test["label"])/2).replace(-1, 1))/len(data_test)
        print("voted perceptron average error", average_error)
        print("final weights", weights)
 

def perceptron(S, data_test):
    lrs = [.1]
    features = S.drop("label", axis=1)
    for j in range(0, len(lrs)):
        print("learning rate ", lrs[j])
        weights = [0] * len(features.columns) #todo add bias
        for t in range(0, 10):
            S = S.sample(frac=1, replace=False).reset_index(drop=True)
            for i in S.index:
                row = S.iloc[i, :]
                row_features = row.drop("label")
                prediction = np.sign(np.inner(weights, row_features))
                if prediction != row["label"]:
                    weights = weights + lrs[j] * np.inner(row["label"], row_features)
    
        
        data_test_no_label = data_test.drop("label", axis=1)
        final_predictions = np.sign(np.inner(weights, data_test_no_label))
        average_error = np.sum(((final_predictions - data_test["label"])/2).replace(-1, 1))/len(data_test)
        print("standard perceptron error", average_error)
        print("final weights", weights)


def update_df_with_numerical_label(S):
    S["label"] = np.where(S["label"]==0 , -1, 1)
    return S


def plot_from_array(line1, label1, title):
    plt.plot(np.asarray(line1), label=label1)
    plt.title(title)
    plt.xlabel("Iterations")
    plt.ylabel("Cost")
    plt.legend()
    plt.show()


def run_perceptron(train="train.csv", test="test.csv", col_names=["variance","skewness","curtosis", "entropy", "label"]):
    data = pd.read_csv(train, delimiter=',', header=None, names=col_names)
    data = update_df_with_numerical_label(data)
    data.insert(0, "bias", 1)

    data_test = pd.read_csv(test, delimiter=',', header=None, names=col_names)
    data_test = update_df_with_numerical_label(data_test)
    data_test.insert(0, "bias", 1)

    perceptron(data, data_test)
    voted_perceptron(data, data_test)
    averaged_perceptron(data, data_test)
        

run_perceptron()