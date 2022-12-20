from distutils.dep_util import newer_pairwise
from re import T
import pandas as pd
import math
import numpy as np
import matplotlib.pyplot as plt


def calc_entropy_adaboost(S):
    labels = S.label.unique()
    total = sum(S["weight"])
    entropy = 0
    for l in labels:
        weight = sum(S[S.label == l]["weight"])
        entropy -= ((weight/total) * math.log2(weight/total))

    return entropy


def calc_information_gain_for_attribute_adaboost(S, attribute):
    information_gain = 0
    total_weight = sum(S["weight"])

    for a in S[attribute].unique():
        entries = S.loc[S[attribute] == a]
        weight_of_attribute = sum(S[S[attribute] == a]["weight"])
        information_gain += calc_entropy_adaboost(entries) * weight_of_attribute/total_weight
    return information_gain


def decision_tree_adaboost(S, attribute, max_depth, current_depth, leaves):
    if len(S.label.unique()) == 1:
        return {S[attribute].values[0] : S.label.values[0]}

    elif current_depth == max_depth:
        label = S.groupby(['label'])['weight'].sum().idxmax()
        return {S[attribute].values[0] : label}

    else:
         #calculate best label
         total_entropy = calc_entropy_adaboost(S)
         information_gain = []
         if attribute != "":
            S = S.drop(columns=[attribute])
         for col in S.columns: #all col's except label
            if col not in ["label", "weight", "prediction"]:
                information_gain.append(total_entropy - calc_information_gain_for_attribute_adaboost(S, col))

         max_gain = max(information_gain)
         attribute_for_split_index = information_gain.index(max_gain)
         attribute_for_split = S.columns[attribute_for_split_index]
         vals_of_attribute = S[attribute_for_split].unique()
         current_depth += 1
         tree = {}
         for i in vals_of_attribute:
            subset_s = S.loc[S[attribute_for_split] == i]
            result = decision_tree_adaboost(subset_s, attribute_for_split, max_depth, current_depth, leaves)
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
                label = S.groupby(['label'])['weight'].sum().idxmax()
                tree[attribute_for_split][missing] = label

         return tree


def calc_error(tree, df, training_data=True):
    error = 0
    original_tree = tree
    predictions = []
    count_wrong=0
    #for each row in df, get value of first key
    for i in range(0, df.shape[0]):
        attribute_for_split = list(tree.keys())[0]
        val = df[attribute_for_split][i]
        prediction = tree[attribute_for_split][val]
        while type(prediction) == list or type(prediction) == dict:
            tree = tree[attribute_for_split][val]
            attribute_for_split = list(tree.keys())[0]
            val = df[attribute_for_split][i]
            prediction = tree[attribute_for_split][val]

        tree = original_tree
        #if label != prediction
        if training_data and df["label"].iloc[i] != prediction:
            error+= df["weight"].iloc[i]
            count_wrong += 1

        predictions.append(prediction)

    df["prediction"] = predictions
    #print(count_wrong)
    return error


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


def run_adaboost(train_data_file="train_final.csv", to_label="test_final.csv"):
    
    data = pd.read_csv(train_data_file, delimiter=',')
    to_label = pd.read_csv(to_label, delimiter=',')

    data = update_df_with_median(data)
    data = update_df_with_numerical_label(data)
    #data = replace_unknown(data)

    to_label = update_df_with_median(to_label)
    #to_label = replace_unknown(to_label)

    adaboost_sum = np.zeros(len(data))
    adaboost_sum_test = np.zeros(len(to_label))
    predictions = np.empty(len(to_label))
    adaboost_errors = []

    weights = [1/len(data)] * len(data)
    for t in range(0, 100):
        #run the adaboost
        data["weight"] = weights
        entropy_tree = decision_tree_adaboost(data, "", 1, 0, leaves_without_unknown)
        error = calc_error(entropy_tree, data) #this calcs error for each tree
        error_test = calc_error(entropy_tree, to_label, training_data=False) #this calcs error for each tree

        if error == 0:
            print("error is 0")
            break

        alpha = .5 * np.log((1-error)/error)
        new_weights = weights * np.exp(-alpha * (data["label"].astype(int) * data["prediction"].astype(int)))
        norm = sum(new_weights)

        
        #calc adaboost errors
        adaboost_sum =  alpha * data["prediction"].astype(int) + adaboost_sum
        adaboost_error = (np.sign(adaboost_sum) - data["label"])/2
        adaboost_error_sum_weights = np.sum(adaboost_error.replace(-1, 1))/len(data)
        adaboost_errors.append(adaboost_error_sum_weights)

        adaboost_sum_test =  alpha * to_label["prediction"].astype(int) + adaboost_sum_test
        predictions = (np.sign(adaboost_sum_test))
        weights = new_weights/norm

    labels = np.where(predictions ==-1, 0, 1)
    ids = np.arange(1, len(labels) + 1)
    df = pd.DataFrame({"Id" : ids, "Prediction" : labels})
    df.to_csv("adaboost_submission.csv", index=False) 

    print("adaboost error", adaboost_errors)

run_adaboost()

def plot_from_array(line1, line2, label1, label2, title):
    plt.plot(np.asarray(line1), label=label1)
    plt.plot(np.asarray(line2), label=label2)
    plt.title(title)
    plt.xlabel("Trees")
    plt.ylabel("Error")
    plt.legend()
    plt.show()
