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
from sklearn import ensemble

def svm_learn(data, data_test):
    clf = ensemble.RandomForestRegressor(n_estimators=6000)
    X = data.drop("label", axis=1)
    y = data["label"]
    clf.fit(X, y)
    predictions = clf.predict(data_test)

    ids = np.arange(1, len(predictions) + 1)
    df = pd.DataFrame({"Id" : ids, "Prediction" : predictions})
    df.to_csv("rf-6k_buckets.csv", index=False) 

    

def plot_from_array(line1, label1, title):
    plt.plot(np.asarray(line1), label=label1)
    plt.title(title)
    plt.xlabel("Iterations")
    plt.ylabel("Cost")
    plt.legend()
    plt.show()

def change_strings_to_int(df):
    for features in ["workclass", "education", "marital.status", "occupation", "relationship", "race", "sex", "native.country"]:
            # vals = df[features].unique()
            # for i, val in enumerate(vals):
            #     df[features] = df[features].replace(val, i/10)
            one_hot = pd.get_dummies(df[features])
            
            #this didn't work because different unknowns
            if "?" in one_hot.columns:
                one_hot = one_hot.rename(columns={"?": features + "?"})
            else:
                one_hot[features + "?"] = 0
            # Drop column B as it is now encoded
            df = df.drop(features,axis = 1)
            # Join the encoded df
            df = df.join(one_hot)


    for features in df.columns:
        df[features] = df[features].astype(float)

    return df

def replace_unknown(S):
    for col in S.columns:
        most_common_val = S[col].value_counts()[:1].index.tolist()[0]
        S[col] = S[col].replace({'?': most_common_val})
    return S

def remove_unknown(df):
    for col in df.columns:
        df = df[df[col] != '?']
    return df

def update_df_with_median(S):
    for col in ["fnlwgt", "capital.gain", "capital.loss", "hours.per.week"]:
        median = S[col].astype(float).median()
        S[col] = pd.to_numeric(S[col])
        S[col] = np.where(S[col]<=median , 0, 1)

    return S

def addBucket(data):
    data["age"] = np.where(data["age"] < 30, 1, data["age"])
    data["age"] = np.where(np.logical_and(data["age"] < 45, data["age"] > 30), 2, data["age"])
    data["age"] = np.where(np.logical_and(data["age"] < 60, data["age"] > 45), 3, data["age"])
    data["age"] = np.where(data["age"] >= 60, 4, data["age"]) 

    data["education.num"] = np.where(data["education.num"] < 8, 1, data["education.num"])
    data["education.num"] = np.where(np.logical_and(data["education.num"] < 16, data["education.num"] > 8), 2, data["education.num"])
    data["education.num"] = np.where(data["education.num"] >= 16, 3, data["education.num"]) 

    data["hours.per.week"] = np.where(data["hours.per.week"] < 20, 1, data["hours.per.week"])
    data["hours.per.week"] = np.where(np.logical_and(data["hours.per.week"] <= 40, data["hours.per.week"] > 20), 2, data["hours.per.week"])
    data["hours.per.week"] = np.where(data["hours.per.week"] > 40, 3, data["hours.per.week"]) 
    return data


def run(train="train_final.csv", test="test_final.csv"):
    data = pd.read_csv(train, delimiter=',')
    data_test = pd.read_csv(test, delimiter=',')

    #data = remove_unknown(data)

    #data = replace_unknown(data)
    #data_test = replace_unknown(data_test)

    # data = pd.read_csv(train, delimiter=',', header=None, names=col_names)
    # data_test = pd.read_csv(test, delimiter=',', header=None, names=col_names)
    data = addBucket(data)
    data_test = addBucket(data_test)


    data = change_strings_to_int(data)
    data_test = change_strings_to_int(data_test)

    data = update_df_with_median(data)
    data_test = update_df_with_median(data_test)
    data["Holand-Netherlands"] = 0
    #data_test = data_test.drop("Holand-Netherlands", axis=1)

    nanEntries = data[data['label'] == 0].index.tolist()
    # choose 10% randomly
    #dropIndices = np.random.choice(nanEntries, size = int(data.shape[0]*0.3))
    # drop them
    #data = data.drop(dropIndices)


    # matrix = np.asarray(data.drop('label', axis=1))
    # y_a = np.asarray(data["label"])
    # mT = matrix.transpose()
    # output = mT.dot(matrix)
    # trans = np.linalg.inv(output)
    # weights = trans.dot(mT.dot(y_a))

    #weights = lms_stochastic(data)
    data_test = data_test.drop("ID", axis=1)

    # rf = pd.read_csv("rf2-10000.csv", delimiter=',', header=None) # names=["age", "job", "marital", "education", "default", "balance", "housing", "loan", "contact", "day", "month", "duration", "campaign", "pdays", "previous", "poutcome", "label"])
    # rf = rf.iloc[1: , :]
    # rf[1] = np.where(rf[1].astype(float)<=.1 , 0, rf[1])
    # rf[1] = np.where(rf[1].astype(float)>=.9 , 1, rf[1])
    # rf.to_csv("rf-janky.csv", index=False) 

    svm_learn(data, data_test)

    # predictions = np.inner(weights, data_test)
    # ids = np.arange(1, len(predictions) + 1)
    # df = pd.DataFrame({"Id" : ids, "Prediction" : predictions})
    # df.to_csv("batch_submission.csv", index=False) 



run()