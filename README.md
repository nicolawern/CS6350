# CS6350

This is a machine learning library developed by Nicola Wernecke for
CS5350/6350 in University of Utah.

HW1
To run the Decision Tree code please download all files and run run.sh. If you want to replace the train/test for either cars or bank please make sure they are named the same as the train/test files in the repository.
The output is an array of the incorrect labels/total labels from tree depth 1-6 for cars and 1-17 for bank. The average error is also printed for both test and train and all calculations. 

HW2 - Ensemble
To run the Ensemble Learning code please download all files in the folder and call run.sh. If you want to replace the train/test please make sure they are named the same as the train/test files in the repository. Please set options True or False for each component. Option 1 is run_adaboost, Option 2 is run random forest creation, Option 3 for run random forests with bias and variance (VERY LONG RUNNING), Option 4 for building bagged trees, Option 5 for run bagged tree with bias and variance (VERY LONG RUNNING). Options 1, 2, 4 output a graph of errors from 0-500 iterations of the algorithm while 3 and 5 output bias, variance and GSE for forests of 100 trees on either their first or 500th iteration. 
Eg: sh run.sh True True True True True

HW2 - Linear Regression
To run the Linear Regression code please download all files in the folder and run run.sh with option 1=True to run batch gradient descent and 2=True to run stochastic gradient descent. Set to False for either to not run them. They will each output a plot of the cost fn, the learning rate that lead to convergence, the final weight vector, and the calculated test error based on weights at train convergence. If you want to replace the train/test for please make sure they are named the same as the train/test files in the repository.
Eg: sh run.sh True True

HW3 - Perceptron
run.sh will run all 3 versions of Perceptron with T=10 and print the average error and final weights. To use different test/train files please name the replacements the same as the test/train in the repo.

HW4 - SVM
Call run.sh with three paramaters - each being True or False. 1st input is for primal, second for dual, and third for dual Gaussian.
