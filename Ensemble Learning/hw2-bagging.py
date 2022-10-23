from cProfile import run
from re import T
from typing import final
import pandas as pd
import math
import numpy as np
import random
import datetime
import sys


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
        else:
            final_prediction = -1
        
        if row["label"] != final_prediction:
            error += 1

    return error/len_data

def calc_predictions_for_bias_and_variance(tree, df):
    len_data = df.shape[0]
    error = 0

    final_predictions = np.empty(len_data)

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
        else:
            final_prediction = -1
        
        if row["label"] != final_prediction:
            error += 1

        final_predictions[i] = final_prediction

    return final_predictions

def calc_bias_variance_and_error(predictions, labels):
    predictions_df = pd.DataFrame(predictions).T
    print(predictions_df.shape)
    len_data = predictions_df.shape[0]
    variance = np.empty(len_data)
    bias = np.empty(len_data)

    for i in range(0, len_data):
        bias[i] = (np.mean(predictions_df.iloc[i, :]) - labels[i]) ** 2
        variance[i] = np.var(predictions_df.iloc[i, :])


    avg_bias = np.mean(bias)
    avg_var = np.mean(variance)
    sq_error = avg_var + avg_bias    
    print("bias, variance, mean sq error", avg_bias, avg_var, sq_error)


def update_df_with_median(S):
    for col in ["age", "balance", "day", "duration", "campaign", "pdays", "previous"]:
        median = S[col].median()
        S[col] = np.where(S[col]<=median , "below", "above")
    return S


def update_df_with_numerical_label(S):
    S["label"] = np.where(S["label"]=="no" , -1, 1)
    return S

leaves_with_unknown_as_val = {"age": ["above", "below"], 
"job": ["admin.","unknown","unemployed","management","housemaid","entrepreneur","student",
                                       "blue-collar","self-employed","retired","technician","services", "poutcome"], 
"marital": ["married","divorced","single"], 
"education": ["unknown","secondary","primary","tertiary"], 
"default": ["yes", "no"], 
"balance": ["above", "below"], 
"housing": ["yes", "no"],
"loan": ["yes", "no"], 
"contact": ["unknown","telephone","cellular"], 
"day": ["above", "below"], 
"month": ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], 
"duration": ["above", "below"], 
"campaign": ["above", "below"], 
"pdays": ["above", "below"], 
"previous": ["above", "below"],
"poutcome": ["unknown","other","failure","success"] }


def load_data(train_data_file="train.csv", test_data_file="test.csv"):
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


def build_500_bagged_trees():
    data, data_test = load_data()
    total_error = np.empty(500)
    total_error_test = np.empty(500)
    
    for i in range(0, 500) :
        sample = data.sample(frac=.6, replace=True).reset_index(drop=True)
        entropy_tree = ID3_entropy(sample, "", len(sample.columns)-3, 0, leaves_with_unknown_as_val)
        total_error[i] = calc_error(entropy_tree, data)
        total_error_test[i] = calc_error(entropy_tree, data_test)
        if i%50 == 0:
            print("evaluating tree ", i, datetime.datetime.now())

    print("Information Gain Error", total_error)
    print("Information Gain Error Test", total_error_test)


def bagged_tree_with_bias():
    data, data_test = load_data()
    #data = pd.read_csv("play_tennis.csv")
    
    predictions_1 = np.zeros((100, len(data_test)))
    predictions_500 = np.zeros((100, len(data_test)))

    for j in range(0, 100):
        sample1 = data.sample(frac=.2, replace=False).reset_index(drop=True)
        print("building forest", j)
        for i in range(0, 500) : #todo should frac=1? slow to create trees
            sample2 = sample1.sample(frac=.25, replace=True).reset_index(drop=True)

            tree = ID3_entropy(sample2, "", len(sample2.columns)-3, 0, leaves_with_unknown_as_val)
            predictions = calc_predictions_for_bias_and_variance(tree, data_test)
            
            if i == 0:
                predictions_1[j] = predictions   

            if i == 499:
                predictions_500[j] = predictions

        data_test["labelno_count"] = np.zeros(data_test.shape[0])
        data_test["labelyes_count"] = np.zeros(data_test.shape[0])
    
    print("first tree")
    calc_bias_variance_and_error(predictions_1, data_test["label"])
    print("bagged trees")    
    calc_bias_variance_and_error(predictions_500, data_test["label"])    


def random_forest():
    data, data_test = load_data()

    feature_count = [2,4,6]

    predictions_1 = np.zeros((3, len(data_test)))
    predictions_500 = np.zeros((3, len(data_test)))

    for index, j in enumerate(feature_count):
        total_error_ig = []
        total_error_ig_test = []
        for i in range(0, 500) : #how many trees to create
            sample = data.sample(frac=.6, replace=True).reset_index(drop=True)
            tree = decision_tree_random_forest(sample, "", len(sample.columns)-3, 0, leaves_with_unknown_as_val,j)
            error, predictions = calc_predictions_for_bias_and_variance(tree, data)
            total_error_ig.append(error)
            test_error, test_predictions = calc_predictions_for_bias_and_variance(tree, data_test)
            total_error_ig_test.append(test_error)
            if i%50 == 0:
                print("evaluating tree ", i, datetime.datetime.now())
            
            if i == 0:
                predictions_1[index] = test_predictions   

            if i == 499:
                predictions_500[index] = test_predictions


        print("first tree")
        calc_bias_variance_and_error(predictions_1, data_test["label"])
        print("bagged trees")    
        calc_bias_variance_and_error(predictions_500, data_test["label"])  
        print("Random Forest Error for feature count", j, total_error_ig)
        print("Random Forest Error Test for feature count", j, total_error_ig_test)
        
        data_test["labelno_count"] = np.zeros(data_test.shape[0])
        data_test["labelyes_count"] = np.zeros(data_test.shape[0])      

def random_forests():
    data, data_test = load_data()

    forests_to_make = 100
    trees_per_forest = 500
    predictions_1 = np.zeros((forests_to_make, len(data_test)))
    predictions_500 = np.zeros((forests_to_make, len(data_test)))

    for index in range(0, forests_to_make):
        total_error_ig = []
        total_error_ig_test = []
        data_test["labelno_count"] = np.zeros(data_test.shape[0])
        data_test["labelyes_count"] = np.zeros(data_test.shape[0]) 
        print("evaluating forest ", index, datetime.datetime.now())

        for i in range(0, trees_per_forest) : #how many trees to create
            sample = data.sample(frac=.2, replace=True).reset_index(drop=True)
            tree = decision_tree_random_forest(sample, "", len(sample.columns)-3, 0, leaves_with_unknown_as_val,4)
            error, predictions = calc_predictions_for_bias_and_variance(tree, data)
            total_error_ig.append(error)
            test_error, test_predictions = calc_predictions_for_bias_and_variance(tree, data_test)
            total_error_ig_test.append(test_error)
            
            if i == 0:
                predictions_1[index] = test_predictions   

            if i == trees_per_forest-1:
                predictions_500[index] = test_predictions


    print("first tree")
    calc_bias_variance_and_error(predictions_1, data_test["label"])
    print("bagged trees")    
    calc_bias_variance_and_error(predictions_500, data_test["label"])  
    print("len data, data_test, error, error_test", len(data), len(data_test), len(total_error_ig), len(total_error_ig_test))  
    
     


def run(run_random_forest, run_random_forests, run_build_500_bagged_trees, run_bagged_tree_with_bias):

    if run_random_forest == "True":
        print("running random forest")
        random_forest()

    if run_random_forests == "True":
        print("running generate 500 forests")
        random_forests()

    if run_build_500_bagged_trees == "True":
        print("running build 500 trees")
        build_500_bagged_trees()
    
    if run_bagged_tree_with_bias == "True":
        print("running run_bagged_tree_with_bias creates 500 trees * 100 times and reports bias/variance/error of tree 0 and 500 for each of 100 runs")
        bagged_tree_with_bias()


run(run_random_forest=sys.argv[1], run_random_forests=sys.argv[2], run_build_500_bagged_trees=sys.argv[3], run_bagged_tree_with_bias=sys.argv[4],)