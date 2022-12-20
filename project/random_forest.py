from cProfile import run
from re import T
from typing import final
import pandas as pd
import math
import numpy as np
import random
import datetime
import sys
import matplotlib.pyplot as plt


def plot_from_array(line1, line2, label1, label2, title):
    plt.plot(np.asarray(line1), label=label1)
    plt.plot(np.asarray(line2), label=label2)
    plt.title(title)
    plt.xlabel("Trees")
    plt.ylabel("Error")
    plt.legend()
    plt.show()

def calc_entropy(S):
    labels = S.label.unique()
    total = S.shape[0]
    entropy = 0
    for l in labels:
        count = len(S.loc[S.label == l])
        entropy -= (count/total * math.log2(count/total))

    return entropy


def calc_information_gain_for_attribute(S, attribute):
    information_gain = 0
    for a in S[attribute].unique():
        entries = S.loc[S[attribute] == a]
        information_gain += calc_entropy(entries) * len(entries)/len(S)
    return information_gain


def ID3_entropy(S, attribute, max_depth, current_depth, leaves):
    if len(S.label.unique()) == 1:
        return {S[attribute].values[0] : S.label.values[0]}

    elif current_depth == max_depth:
        label = S['label'].value_counts()[:1].index.tolist()[0]
        return {S[attribute].values[0] : label}

    else:
         total_entropy = calc_entropy(S)
         information_gain = []
         if attribute != "":
            S = S.drop(columns=[attribute])
         for col in S.columns: #all col's except label
            if col not in ["label", "labelno_count", "labelyes_count"]:
                information_gain.append(total_entropy - calc_information_gain_for_attribute(S, col))

         max_gain = max(information_gain)
         attribute_for_split_index = information_gain.index(max_gain)
         attribute_for_split = S.columns[attribute_for_split_index]
         vals_of_attribute = S[attribute_for_split].unique()
         current_depth += 1
         tree = {}
         for i in vals_of_attribute:
            subset_s = S.loc[S[attribute_for_split] == i]
            result = ID3_entropy(subset_s, attribute_for_split, max_depth, current_depth, leaves)
            if attribute_for_split in tree.keys() and i in result.keys():
                tree[attribute_for_split][i] = result[i]
            elif attribute_for_split in tree.keys():
                tree[attribute_for_split][i] = result

            else: 
                tree[attribute_for_split] = result

            if i not in tree[attribute_for_split].keys():
                tree[attribute_for_split] = {i : tree[attribute_for_split]}

         if len(tree[attribute_for_split].keys()) < len(leaves[attribute_for_split]):
            missing_leaves = [x for x in leaves[attribute_for_split] if x not in list(tree[attribute_for_split])] 
            for missing in missing_leaves:
                label = S['label'].value_counts()[:1].index.tolist()[0]
                tree[attribute_for_split][missing] = label

         return tree
         

def decision_tree_random_forest(S, attribute, max_depth, current_depth, leaves, count_features_to_keep):
    if len(S.label.unique()) == 1:
        return {S[attribute].values[0] : S.label.values[0]}

    elif current_depth == max_depth:
        label = S['label'].value_counts()[:1].index.tolist()[0]
        return {S[attribute].values[0] : label}

    else:
         total_entropy = calc_entropy(S)
         information_gain = []
         if attribute != "":
            S = S.drop(columns=[attribute])
         if len(S.columns) > count_features_to_keep + 4: #label + modes + at least one feature to drop
            features = list(S.columns)
            features.remove("label")
            features.remove("labelno_count")
            features.remove("labelyes_count")
            features_to_drop = random.sample(features, len(features) - count_features_to_keep)
            sample = S.drop(columns=features_to_drop)
         else:
            sample = S
         for col in sample.columns: #all col's except label
            if col not in ["label", "labelno_count", "labelyes_count"]:
                information_gain.append(total_entropy - calc_information_gain_for_attribute(sample, col))

         max_gain = max(information_gain)
         attribute_for_split_index = information_gain.index(max_gain)
         attribute_for_split = sample.columns[attribute_for_split_index]
         vals_of_attribute = S[attribute_for_split].unique()
         current_depth += 1
         tree = {}
         for i in vals_of_attribute:
            subset_s = S.loc[S[attribute_for_split] == i]
            result = ID3_entropy(subset_s, attribute_for_split, max_depth, current_depth, leaves)
            if attribute_for_split in tree.keys() and i in result.keys():
                tree[attribute_for_split][i] = result[i]
            elif attribute_for_split in tree.keys():
                tree[attribute_for_split][i] = result

            else: 
                tree[attribute_for_split] = result

            if i not in tree[attribute_for_split].keys():
                tree[attribute_for_split] = {i : tree[attribute_for_split]}

         if len(tree[attribute_for_split].keys()) < len(leaves[attribute_for_split]):
            missing_leaves = [x for x in leaves[attribute_for_split] if x not in list(tree[attribute_for_split])] 
            for missing in missing_leaves:
                label = S['label'].value_counts()[:1].index.tolist()[0]
                tree[attribute_for_split][missing] = label

         return tree

def calc_error(tree, df):
    len_data = df.shape[0]
    error = 0

    final_predictions = []
    #for each row in df, get value of first key
    for i in range(0, len_data):
        row = df.iloc[i, :]
        original_tree = tree
        attribute_for_split = list(tree.keys())[0]
        val = row[attribute_for_split]
        prediction = tree[attribute_for_split][val]
        while type(prediction) == list or type(prediction) == dict:
            tree = tree[attribute_for_split][val]
            attribute_for_split = list(tree.keys())[0]
            val = row[attribute_for_split]
            prediction = tree[attribute_for_split][val]

        tree = original_tree
                
        count_yes = row["labelyes_count"]
        count_no = row["labelno_count"]

        if prediction > 0:
            count_yes += 1
            df["labelyes_count"].iloc[i] = count_yes
        else:
            count_no +=1
            df["labelno_count"].iloc[i] = count_no

        if count_yes > count_no:
            final_prediction = 1
            avg = (count_yes - count_no)/(count_yes + count_no)
        else:
            final_prediction = -1
            avg = (count_no - count_yes)/(count_yes + count_no)

        final_predictions.append(avg)

        if "label" in row and row["label"] != final_prediction:
            error += 1

    if "label" not in df.columns:
        df["predictions"] = final_predictions

    return error/len_data, df



def update_df_with_median(S):
    for col in ["age", "fnlwgt", "capital.gain", "capital.loss", "hours.per.week", "education.num"]:
        median = S[col].median()
        S[col] = np.where(S[col]<=median , "below", "above")
    return S


def replace_unknown(S):
    for col in S.columns:
        most_common_val = S[col].value_counts()[:1].index.tolist()[0]
        S[col] = S[col].replace({'?': most_common_val})
    return S

def update_df_with_numerical_label(S):
    S["label"] = np.where(S["label"]==0 , -1, 1)
    return S

leaves_without_unknown = {"age": ["above", "below"], 
"workclass": ["Private","Self-emp-not-inc","Self-emp-inc","Federal-gov","Local-gov","State-gov",
                                       "Without-pay","self-employed","Never-worked", "?"], 
"fnlwgt": ["above", "below"], 
"education.num": ["above", "below"], 
"education": ["Bachelors", "Some-college", "11th", "HS-grad", "Prof-school", 
    "Assoc-acdm", "Assoc-voc", "9th", "7th-8th", "12th", "Masters", "1st-4th", "10th", 
    "Doctorate", "5th-6th", "Preschool", "?"],
"marital.status": ["Married-civ-spouse", "Divorced", "Never-married", "Separated", "Widowed",
     "Married-spouse-absent", "Married-AF-spouse", "?"], 
"occupation": ["Tech-support", "Craft-repair", "Other-service", "Sales", "Exec-managerial", "Prof-specialty", 
    "Handlers-cleaners", "Machine-op-inspct", "Adm-clerical", "Farming-fishing", "Transport-moving", 
    "Priv-house-serv", "Protective-serv", "Armed-Forces", "?"], 
"relationship": ["Wife", "Own-child", "Husband", "Not-in-family", "Other-relative", "Unmarried", "?"], 
"race": ["White", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other", "Black", "?"], 
"sex": ["Female", "Male", "?"], 
"capital.gain": ["above", "below"], 
"capital.loss": ["above", "below"],
"hours.per.week": ["above", "below"],
"native.country" : ["United-States", "Cambodia", "England", "Puerto-Rico", "Canada", "Germany", "Outlying-US(Guam-USVI-etc)", 
    "India", "Japan", "Greece", "South", "China", "Cuba", "Iran", "Honduras", "Philippines", "Italy", "Poland", "Jamaica", 
    "Vietnam", "Mexico", "Portugal", "Ireland", "France", "Dominican-Republic", "Laos", "Ecuador", "Taiwan", "Haiti", "Columbia",
     "Hungary", "Guatemala", "Nicaragua", "Scotland", "Thailand", "Yugoslavia", "El-Salvador", "Trinadad&Tobago", "Peru", "Hong", 
     "Holand-Netherlands", "?"] }


def load_data(train_data_file="train-final.csv", test_data_file="test-final.csv"):
    data = pd.read_csv(train_data_file, delimiter=',', header=None, names=["age", "job", "marital", "education", "default", "balance", "housing", "loan", "contact", "day", "month", "duration", "campaign", "pdays", "previous", "poutcome", "label"])
    data_test = pd.read_csv(test_data_file, delimiter=',', header=None, names=["age", "job", "marital", "education", "default", "balance", "housing", "loan", "contact", "day", "month", "duration", "campaign", "pdays", "previous", "poutcome", "label"])
    data = update_df_with_median(data)
    data_test = update_df_with_median(data_test)
    data = update_df_with_numerical_label(data)
    data_test = update_df_with_numerical_label(data_test)
    data["labelno_count"] = np.zeros(data.shape[0])
    data["labelyes_count"] = np.zeros(data.shape[0])
    data_test["labelno_count"] = np.zeros(data_test.shape[0])
    data_test["labelyes_count"] = np.zeros(data_test.shape[0])
    return data, data_test



def random_forest():
    data = pd.read_csv("train_final.csv", delimiter=',')
    data_test = pd.read_csv("test_final.csv", delimiter=',')
    data = update_df_with_median(data)
    data_test = update_df_with_median(data_test)
    data = update_df_with_numerical_label(data)
    data["labelno_count"] = np.zeros(data.shape[0])
    data["labelyes_count"] = np.zeros(data.shape[0])
    data_test["labelno_count"] = np.zeros(data_test.shape[0])
    data_test["labelyes_count"] = np.zeros(data_test.shape[0])

    feature_count = [4,6]

    for index, j in enumerate(feature_count):
        total_error_ig = []
        total_error_ig_test = []
        for i in range(0, 50) : #how many trees to create
            sample = data.sample(frac=.6, replace=True).reset_index(drop=True)
            tree = decision_tree_random_forest(sample, "", len(sample.columns)-3, 0, leaves_without_unknown,j)
            error, d = calc_error(tree, data)
            total_error_ig.append(error)
            test_error, data_test = calc_error(tree, data_test)
            total_error_ig_test.append(test_error)
            

        data_test["labelno_count"] = np.zeros(data_test.shape[0])
        data_test["labelyes_count"] = np.zeros(data_test.shape[0])  

        labels = data_test["predictions"]
        ids = np.arange(1, len(labels) + 1)
        df = pd.DataFrame({"Id" : ids, "Prediction" : labels})
        df.to_csv("rf_submission.csv", index=False)   

     
random_forest()
