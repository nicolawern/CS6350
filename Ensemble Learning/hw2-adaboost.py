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
        label = S.groupby(['label'])['weight'].sum().idxmax() #todo this is right
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
                label = S['label'].value_counts()[:1].index.tolist()[0]
                tree[attribute_for_split][missing] = label

         return tree


def calc_error(tree, df):
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
        if df["label"].iloc[i] != prediction:
            error+= df["weight"].iloc[i]
            count_wrong += 1

        predictions.append(prediction)

    df["prediction"] = predictions
    #print(count_wrong)
    return error

def update_df_with_median(S):
    for col in ["age", "balance", "day", "duration", "campaign", "pdays", "previous"]:
        median = S[col].median()
        S[col] = np.where(S[col]<=median , "below", "above")
    return S


def update_df_with_numerical_label(S):
    S["label"] = np.where(S["label"]=="no" , -1, 1)
    return S


def replace_unknown(S):
    for col in S.columns:
        most_common_val = S[col].value_counts()[:1].index.tolist()[0]
        S[col] = S[col].replace({'unknown': most_common_val})
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


# leaves_with_unknown_as_val = {"outlook" : ["Sunny", "Overcast", "Rain"],
# "temp" : ["Hot", "Mild", "Cool"],
# "humidity" : ["High", "Normal"],
# "wind" : ["Weak", "Strong"] }

def run_adaboost(train_data_file="train.csv", test_data_file="test.csv"):
    
    data = pd.read_csv(train_data_file, delimiter=',', header=None, names=["age", "job", "marital", "education", "default", "balance", "housing", "loan", "contact", "day", "month", "duration", "campaign", "pdays", "previous", "poutcome", "label"])
    data_test = pd.read_csv(test_data_file, delimiter=',', header=None, names=["age", "job", "marital", "education", "default", "balance", "housing", "loan", "contact", "day", "month", "duration", "campaign", "pdays", "previous", "poutcome", "label"])

    data = update_df_with_median(data)
    data = update_df_with_numerical_label(data)

    data_test = update_df_with_median(data_test)
    data_test = update_df_with_numerical_label(data_test)

    adaboost_sum = np.zeros(len(data))
    adaboost_sum_test = np.zeros(len(data_test))


    errors_of_stump = []
    errors_of_stump_test = []
    adaboost_errors = []
    adaboost_errors_test = []

    weights = [1/len(data)] * len(data)
    for t in range(0, 500):
        #run the adaboost
        data["weight"] = weights
        data_test["weight"] = weights
        entropy_tree = decision_tree_adaboost(data, "", 1, 0, leaves_with_unknown_as_val)
        error = calc_error(entropy_tree, data) #this calcs error for each tree
        errors_of_stump.append(error)
        error_test = calc_error(entropy_tree, data_test)
        errors_of_stump_test.append(error_test)

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

        adaboost_sum_test =  alpha * data_test["prediction"].astype(int) + adaboost_sum_test
        adaboost_error_test = (np.sign(adaboost_sum_test) - data_test["label"])/2
        adaboost_error_test_sum = np.sum(adaboost_error_test.replace(-1, 1))/len(data_test)
        adaboost_errors_test.append(adaboost_error_test_sum)
        
        weights = new_weights/norm


    print("adaboost error", adaboost_errors)
    print("adaboost test error", adaboost_errors_test)

#todo for some reason my test error is wack. I'm using not test alpha, weights and tree
    plot_from_array(line1=adaboost_errors, line2=adaboost_errors_test, label1="Error", label2="Test Error", title="Adaboost Error by Count Tree")
    plot_from_array(line1=errors_of_stump, line2=errors_of_stump_test, label1="Error", label2="Test Error", title="Stump Error by Count Tree")


def plot_from_array(line1, line2, label1, label2, title):
    plt.plot(np.asarray(line1), label=label1)
    plt.plot(np.asarray(line2), label=label2)
    plt.title(title)
    plt.xlabel("Trees")
    plt.ylabel("Error")
    plt.legend()
    plt.show()

def run(adaboost):
    if adaboost == "True":
        run_adaboost()

run(adaboost=sys.argv[1])