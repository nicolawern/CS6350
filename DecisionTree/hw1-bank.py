from re import T
import pandas as pd
import math
import numpy as np


def calc_entropy(S):
    labels = S.label.unique()
    total = S.shape[0]
    entropy = 0
    for l in labels:
        count = len(S.loc[S.label == l])
        entropy -= (count/total * math.log2(count/total))

    return entropy

def calc_information_gain_for_attribute_gini(S, attribute):
    information_gain = 0
    for a in S[attribute].unique():
        entries = S.loc[S[attribute] == a]
        information_gain += calc_gini_index(entries) * len(entries)/len(S)

    return information_gain


def ID3_gini(S, attribute, max_depth, current_depth, leaves):
    if len(S.label.unique()) == 1:
                 return {S[attribute].values[0] : S.label.values[0]}

    elif current_depth == max_depth:
        #todo figure out is it most common label? probably
        label = S['label'].value_counts()[:1].index.tolist()[0]
        return {S[attribute].values[0] : label}

    else:
         #calculate best label
         total_entropy = calc_gini_index(S)
         information_gain = []
         if attribute != "":
            S = S.drop(columns=[attribute])
         for col in S.columns:
            if col != "label":
                information_gain.append(total_entropy - calc_information_gain_for_attribute_gini(S, col))

         max_gain = max(information_gain)
         attribute_for_split_index = information_gain.index(max_gain)
         attribute_for_split = S.columns[attribute_for_split_index]

         vals_of_attribute = S[attribute_for_split].unique()
         current_depth += 1
         tree = {}
         for i in vals_of_attribute:
            result = ID3_gini(S.loc[S[attribute_for_split] == i], attribute_for_split, max_depth, current_depth, leaves)
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


def calc_gini_index(S):
        labels = S.label.unique()
        total = float(S.shape[0])
        gini = 1.0
        for l in labels:
            count = float(len(S.loc[S.label == l]))
            gini -= (count/total)**2

        return gini


def calc_max_error(S):
    labels = S.label.unique()
    total = S.shape[0]
    entries = []
    for l in labels:
        entries.append(len(S.loc[S.label == l]))

    if min(entries) == total:
        return 0

    return min(entries)/total


def calc_information_gain_for_attribute_ME(S, attribute):
    information_gain = 0
    for a in S[attribute].unique():
        entries = S.loc[S[attribute] == a]
        information_gain += calc_max_error(entries) * len(entries)/len(S)
    return information_gain


def ID3_ME(S, attribute, max_depth, current_depth, leaves):
    if len(S.label.unique()) == 1:
        return {S[attribute].values[0] : S.label.values[0]}

    elif current_depth == max_depth:
        #todo figure out is it most common label? probably
        label = S['label'].value_counts()[:1].index.tolist()[0]
        return {S[attribute].values[0] : label}

    else:
         #calculate best label
         total_entropy = calc_max_error(S)
         information_gain = []
         if attribute != "":
            S = S.drop(columns=[attribute])
         for col in S.columns: #all col's except label
            if col != "label":
                information_gain.append(total_entropy - calc_information_gain_for_attribute_ME(S, col))

         max_gain = max(information_gain)
         attribute_for_split_index = information_gain.index(max_gain)
         attribute_for_split = S.columns[attribute_for_split_index]
         vals_of_attribute = S[attribute_for_split].unique()
         current_depth += 1
         tree = {}
         for val in vals_of_attribute:
            subset_s = S.loc[S[attribute_for_split] == val]
            result = ID3_ME(subset_s, attribute_for_split, max_depth, current_depth, leaves)
            if attribute_for_split in tree.keys() and val in result.keys():
                tree[attribute_for_split][val] = result[val]
            elif attribute_for_split in tree.keys():
                tree[attribute_for_split][val] = result

            else: 
                tree[attribute_for_split] = result

            if val not in tree[attribute_for_split].keys():
                tree[attribute_for_split] = {val : tree[attribute_for_split]}

         if len(vals_of_attribute) < len(leaves[attribute_for_split]):
            missing_leaves = [x for x in leaves[attribute_for_split] if x not in list(tree[attribute_for_split])] 
            for missing in missing_leaves:
                label = S['label'].value_counts()[:1].index.tolist()[0]
                tree[attribute_for_split][missing] = label

         return tree



def calc_information_gain_for_attribute(S, attribute):
    information_gain = 0
    for a in S[attribute].unique():
        entries = S.loc[S[attribute] == a]
        information_gain += calc_entropy(entries) * len(entries)/len(S)
    return information_gain


        #todo add leaf node for most common value even if not seen
def ID3_entropy(S, attribute, max_depth, current_depth, leaves):
    if len(S.label.unique()) == 1:
        return {S[attribute].values[0] : S.label.values[0]}

    elif current_depth == max_depth:
        label = S['label'].value_counts()[:1].index.tolist()[0]
        return {S[attribute].values[0] : label}

    else:
         #calculate best label
         total_entropy = calc_entropy(S)
         information_gain = []
         if attribute != "":
            S = S.drop(columns=[attribute])
         for col in S.columns: #all col's except label
            if col != "label":
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


def calc_error(tree, df):
    error = 0
    original_tree = tree
    
    #for each row in df, get value of first key
    for i in range(0, df.shape[0]):
        attribute_for_split = list(tree.keys())[0]
        val = df[attribute_for_split][i]
        label = tree[attribute_for_split][val]
        while type(label) == list or type(label) == dict:
            tree = tree[attribute_for_split][val]
            attribute_for_split = list(tree.keys())[0]
            val = df[attribute_for_split][i]
            label = tree[attribute_for_split][val]

        tree = original_tree
        if df.iloc[i, -1] != label:
            error+= 1

    return error/df.shape[0]

def update_df_with_median(S):
    for col in ["age", "balance", "day", "duration", "campaign", "pdays", "previous"]:
        median = S[col].median()
        S[col] = np.where(S[col]<=median , "below", "above")
    return S


def replace_unknown(S):
    for col in S.columns:
        most_common_val = S[col].value_counts()[:1].index.tolist()[0]
        S[col] = S[col].replace({'unknown': most_common_val})
    return S


#data = pd.read_csv("play_tennis.csv")

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

leaves_without_unknown = {"age": ["above", "below"], 
"job": ["admin.","unemployed","management","housemaid","entrepreneur","student",
                                       "blue-collar","self-employed","retired","technician","services", "poutcome"], 
"marital": ["married","divorced","single"], 
"education": ["secondary","primary","tertiary"], 
"default": ["yes", "no"], 
"balance": ["above", "below"], 
"housing": ["yes", "no"],
"loan": ["yes", "no"], 
"contact": ["telephone","cellular"], 
"day": ["above", "below"], 
"month": ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], 
"duration": ["above", "below"], 
"campaign": ["above", "below"], 
"pdays": ["above", "below"], 
"previous": ["above", "below"],
"poutcome": ["other","failure","success"] }


def build_tree_and_test(train_data_file="train-bank.csv", test_data_file="test-bank.csv", missing_values=True):
    
    data = pd.read_csv(train_data_file, delimiter=',', header=None, names=["age", "job", "marital", "education", "default", "balance", "housing", "loan", "contact", "day", "month", "duration", "campaign", "pdays", "previous", "poutcome", "label"])
    data_test = pd.read_csv(test_data_file, delimiter=',', header=None, names=["age", "job", "marital", "education", "default", "balance", "housing", "loan", "contact", "day", "month", "duration", "campaign", "pdays", "previous", "poutcome", "label"])

    total_error_me = []
    total_error_me_test = []

    total_error_gi = []
    total_error_gi_test = []

    total_error_ig = []
    total_error_ig_test = []

    if missing_values :
        data = replace_unknown(data)
        data_test = replace_unknown(data_test)


    data = update_df_with_median(data)
    data_test = update_df_with_median(data_test)

    for i in range(1,17) :

        if missing_values:
            entropy_tree = ID3_entropy(data, "", i, 0, leaves_without_unknown)
            me_tree = ID3_ME(data, "", i, 0, leaves_without_unknown)
            gini_tree = ID3_gini(data, "", i, 0, leaves_without_unknown)

        else: 
            entropy_tree = ID3_entropy(data, "", i, 0, leaves_with_unknown_as_val)
            me_tree = ID3_ME(data, "", i, 0, leaves_with_unknown_as_val)
            gini_tree = ID3_gini(data, "", i, 0, leaves_with_unknown_as_val)


        total_error_ig.append(calc_error(entropy_tree, data))
        total_error_me.append(calc_error(me_tree, data))
        total_error_gi.append(calc_error(gini_tree, data))

        total_error_me_test.append(calc_error(me_tree, data_test))
        total_error_ig_test.append(calc_error(entropy_tree, data_test))
        total_error_gi_test.append(calc_error(gini_tree, data_test))

    print("Max Error error", total_error_me)
    print("avg ME", sum(total_error_me) / len(total_error_me))
    print("Max Error test error", total_error_me_test)
    print("avg ME Test", sum(total_error_me_test) / len(total_error_me_test))

    print("Gini Index Error", total_error_gi)
    print("Gini Index Error Test", total_error_gi_test)
    print("avg GI", sum(total_error_gi) / len(total_error_gi))
    print("avg GI Test", sum(total_error_gi_test) / len(total_error_gi_test))

    print("Information Gain Error", total_error_ig)
    print("Information Gain Error Test", total_error_ig_test)
    print("avg IG", sum(total_error_ig) / len(total_error_ig))
    print("avg IG Test", sum(total_error_ig_test) / len(total_error_ig_test))

print("Unknown replaced with most common value")
build_tree_and_test()

print("Unknown treated as value")
build_tree_and_test(missing_values=False)