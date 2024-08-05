import time

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
import util_funcs as u
import training_functions as t
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression

# init properties
pd.options.display.float_format = '{:.5f}'.format
np.set_printoptions(suppress=True, precision=3)
np.random.seed(0)
pd.set_option("display.max_rows", None, "display.max_columns", None)


# import datasets
print("Importing Datasets")
test = pd.read_csv('fraudTest.csv')
train = pd.read_csv('fraudTrain.csv')

# sample 250k from train
print("Preprocess Data")
train, _ = u.sample_rows(train, n=250000)

# preprocess
x_train, y_train, train_dicts, left_over, all_counts = u.data_preprocess(train)
x_test, y_test, test_dics, _ = u.data_preprocess(test, train=False, train_counts=all_counts)
full_x_train, full_x_test = u.align_feature_matrices(x_train, x_test)

# correct feature selection using the sampling result
sample = False
if sample:
    t.custom_grid_search(x_train,y_train)

summary, cols = u.combine_csv_files('dt_hp.csv', 'logistic_hp.csv', 'xgboost_hp.csv', 'random_forest_hp.csv', 15, 15)

x_train = x_train[cols]
x_test = x_test[cols]

# grid search
param_grid_xgb = {
    'learning_rate': [0.1, 0.05, 0.03, 0.01],
    'max_depth': [5, 6, 7, 8, 9],
    'n_estimators': [20, 30, 40, 50],
    'n_jobs': [8],
}

param_grid_random_forest = {
    'max_depth': [6, 7, 8, 9],
    'n_estimators': [20, 30, 40, 50, 60, 70],
    'class_weight': [{0: 1, 1: 2}, {0: 1, 1: 3}],
    'bootstrap': [True, False],
    'n_jobs': [8]
}

param_grid_dt = {
    'max_depth': [8, 9, 10],
    'class_weight': [{0: 1, 1: 2}],
    'ccp_alpha': [0.0, 0.02, 0.005, 0.1],
    'criterion': ['gini', 'entropy', 'log_loss']
}

param_grid_mlp = {
    'solver' : ['adam'],
    'activation' : ['relu'],
    'hidden_layer_sizes': [[64,32]],
    'early_stopping': [True],
    'validation_fraction' : [0.2]
}

param_grid_logistic = {
    'penalty': ['l1', 'l2','none'],
    'class_weight': [{0: 1, 1: 2}, {0: 1, 1: 3}, {0: 1, 1: 4}, {0: 1, 1: 5}],
    'max_iter' : [500,800],
    'n_jobs': [6]
}

print("Starting DecisionTreeClassifier Grid Search")
u.param_search(DecisionTreeClassifier(), param_grid_dt, [], 3, x_train, y_train, x_test, y_test)
print("Done. Starting XGBClassifier Grid Search")
u.param_search(XGBClassifier(), param_grid_xgb, [], 3, x_train, y_train, x_test, y_test)
print("Done. Starting RandomForestClassifier Grid Search")
u.param_search(RandomForestClassifier(), param_grid_random_forest, [], 3, x_train, y_train, x_test, y_test)
print("Done. Starting MLPClassifier Grid Searches")
u.param_search(MLPClassifier(), param_grid_mlp, [], 3, x_train, y_train, x_test, y_test)
print("Done. Starting LogisticRegression Grid Search")
u.param_search(LogisticRegression(), param_grid_logistic, [], 3, x_train, y_train, x_test, y_test)
print("Done")
