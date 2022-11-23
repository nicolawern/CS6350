from cProfile import run
from re import T
from typing import final
import pandas as pd
import math
import numpy as np
from scipy import optimize
import datetime
import matplotlib.pyplot as plt
import sys
from numpy import linalg

def lr1_update(gamma, t, a):
    return gamma/(1 + gamma * t / a)

def lr2_update(gamma, t):
    return gamma/(1+t)

def SVM_primal(S, data_test):

    gamma_array = [.1, .5, 1, 5, 100]
    for gamma in gamma_array:
        a_array = [.1, .5, 1, 5, 50, 100]
        C_array = [100/873, 500/873, 700/873]
        #C_array = [1/3]  #remove
        N = len(S)
        with open('primal_out2.txt', 'w') as writer:

            S["bias_term"] = 1
            for a in a_array:
                #lr_for_question = [.01, .005, .0025]#remove
                for lr_scheduler in range(0,1): #todo fix this
                    weights = np.zeros(len(S.columns)-1)
                    weights2 = np.zeros(len(S.columns)-1)
                    for C in C_array:
                        for i in range(0,6):
                            average_error = []
                            for t in range(0, 100):
                                S = S.sample(frac=1, replace=False).reset_index(drop=True) #shuffle
                                for index, row in S.iterrows():
                                    features = row.drop("label")

                                    if lr_scheduler == 0:
                                        lr = lr1_update(gamma, index, a)
                                    else:
                                        lr = lr2_update(gamma, index)

                                    #derivative of SVM objective
                                    if max(0, 1 - row.label * np.inner(weights, features)) == 0:
                                        weights[:-1] = (1 - lr) * weights[:-1] 

                                    else:
                                        weights_0_bias = weights
                                        weights_0_bias[-1] = 0
                                        weights = weights - lr * weights_0_bias + C * N * lr * row.label * features.values


                            data_test_no_label = data_test.drop("label", axis=1)
                            data_test_no_label["bias_term"] = 1
                            final_predictions = np.sign(np.inner(weights, data_test_no_label))
                            average_error.append(np.sum(((final_predictions - data_test["label"])/2).replace(-1, 1))/len(data_test)) #todo upate for new label

                        writer.write(f"average error={np.mean(average_error)}, C={C}, a={a}, gamma scheduler {lr_scheduler}, gamma={gamma}, weights={weights}")

                        print("final weights are", weights) #todo are there any other model parameters?
                        print(f"average error={np.mean(average_error)}, C={C}, a={a}, gamma scheduler {lr_scheduler}, gamma={gamma} ")



def optimize_fun(alphas, H) :
    return .5 * np.dot(np.dot(alphas.T, H), alphas) - np.sum(alphas)

def alphas_y_constraint(y, alphas):
    return np.dot(y, alphas)

def gaussian(x, y, gamma):
    return np.exp(-gamma*linalg.norm(x - y) ** 2 )

def SVM_dual(S, data_test):

    C_array = [100/873, 500/873, 700/873]
    y = S.label * 1.
    X = S.drop("label", axis=1) * 1.

    y_times_X = X.mul(y, axis=0)
    H = np.dot(y_times_X, y_times_X.T)

    constraints = optimize.LinearConstraint(A=y, lb=0, ub=0)
    with open('primal_out2.txt', 'w') as writer:

        for C in C_array:
            bound = optimize.Bounds(0, C)

            output = optimize.minimize(fun=optimize_fun, x0=np.zeros(len(S)), method="SLSQP", constraints=constraints, bounds=bound, args=(H))
            alphas = output.x
            weights = ((y * alphas).T @ X)
            SV = (alphas > 1e-4).flatten()
            b = y[SV] - np.dot(X[SV], weights)
            
            #Display results
            print('Alphas = ',alphas[alphas > 1e-4])
            print('w = ', weights)
            print("all b = ", b)
            print('b = ', np.sum(b)) #todo find true b
            print("C =", C)
            
            data_test_no_label = data_test.drop("label", axis=1)
            data_test_no_label["bias-term"] = 1
            weights = weights.to_numpy()

            weights = np.append(weights, b.values[0])

            num_SV = len(alphas[alphas > 1e-4])

            final_predictions = np.sign(np.inner(weights, data_test_no_label))

            average_error = np.sum(((final_predictions - data_test["label"])/2).replace(-1, 1))/len(data_test)

            writer.write(f"average error={average_error}, C={C}, weights={weights}, SV={num_SV}")

            print(f"average error={average_error}, C={C}, weights={weights}, SV={num_SV}")

            
def SVM_dual_Gaussian(S, data_test):

    C_array = [100/873, 500/873, 700/873]
    m, n = S.shape
    y = S.label * 1.
    X = S.drop("label", axis=1) * 1.

    gamma_array = [.1, .5, 1, 5, 100]
    
    for gamma in gamma_array:
    
        X_gaus=np.array([[gaussian(X.iloc[x1], X.iloc[x2], gamma) for x1 in range(m)] for x2 in range(m)])
        H = np.outer(y, y) * X_gaus
        constraints = optimize.LinearConstraint(A=y, lb=0, ub=0)
        with open('dual_out_G.txt', 'w') as writer:

            for C in C_array:
                bound = optimize.Bounds(0, C)
                #constraints = {"type": "eq", "fun":alphas_y_constraint }
                output = optimize.minimize(fun=optimize_fun, x0=np.zeros(len(S)), method="SLSQP", constraints=constraints, bounds=bound, args=(H))
                alphas = output.x
                SV = (alphas > 1e-4).flatten()                
                
                data_test_no_label = data_test.drop("label", axis=1)
               

                final_predictions_test = np.array([[gaussian(data_test_no_label.iloc[x1], sv_i, gamma) 
                    for i, sv_i in X[SV].iterrows()] for x1 in range(len(data_test_no_label))])
                predictions_test = np.sign(np.inner(final_predictions_test, alphas[SV] * y[SV]))
                #np.sign(np.inner(weights, data_test_no_label))
                average_test_error = np.sum(((predictions_test - data_test["label"])/2).replace(-1, 1))/len(data_test)
                print("average test errror", average_test_error)

                final_predictions =np.array([[gaussian(X.iloc[x1], sv_i, gamma) 
                    for i, sv_i in X[SV].iterrows()] for x1 in range(len(X))])
                predictions_train = np.sign(np.inner(final_predictions, alphas[SV] * y[SV]))
                average_train_error = np.sum(((predictions_train - S["label"])/2).replace(-1, 1))/len(X)
                print("average train error", average_train_error)

                if C == (500/873):
                    writer.write(f'\n SV for gamma {gamma}\n {X[SV]}')

                writer.write(f"average train error={average_train_error}, average test error={average_test_error}, C={C}, gamma={gamma}, SV={len(alphas[alphas > 1e-4])}")

                print(f"average train error={average_train_error}, average test error={average_test_error}, C={C}, gamma={gamma}, SV={len(alphas[alphas > 1e-4])}")



def run(SVM_1, SVM_2, SVM_3):

    col_names=["variance","skewness","curtosis", "entropy", "label"]
    
    data = pd.read_csv("train.csv", delimiter=',', header=None, names=col_names)
    data["label"] = np.where(data["label"]==0 , -1, 1)

    data_test = pd.read_csv("test.csv", delimiter=',', header=None, names=col_names)
    data_test["label"] = np.where(data_test["label"]==0 , -1, 1)

    if SVM_1:
        print("running SVM primal")
        SVM_primal(data, data_test)
    
    if SVM_2:
        print("running SVM dual")
        SVM_dual(data, data_test)

    if SVM_3:
        print("running SVM with Gaussian Kernal")
        SVM_dual_Gaussian(data, data_test)

run(SVM_1=sys.argv[1], SVM_2=sys.argv[2], SVM_3=sys.argv[2])
