import xgboost as xgb
import os
import sys
import numpy as np
from sklearn import tree
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score
from sklearn.metrics import make_scorer
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from multiprocessing import cpu_count
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

import util_funcs as u
import time
import random
import datetime
import csv


# functions for running models
def xgboost_func(x_train, y_train, x_test, y_test, param):

    # Train the model
    train_matrix = xgb.DMatrix(data=x_train, label=y_train)
    test_matrix = xgb.DMatrix(data=x_test, label=y_test)

    null_device = open(os.devnull, 'w')
    old_stdout = sys.stdout
    sys.stdout = null_device

    bst = xgb.train(param, train_matrix, param['n_estimators'], evals=[(train_matrix, 'train')])

    sys.stdout = old_stdout
    null_device.close()

    # Make predictions
    y_pred_test = bst.predict(test_matrix)
    y_pred_test = np.round(y_pred_test)
    y_pred_train = bst.predict(train_matrix)
    y_pred_train = np.round(y_pred_train)

    return u.print_stats(y_train, y_pred_train, y_test, y_pred_test, "XGBoost")


def dt_func(x_train, y_train, x_test, y_test, params):
    # decision tree
    clf = tree.DecisionTreeClassifier(max_depth=params['depth'],
                                      criterion='entropy', class_weight=params['weights'],
                                      min_weight_fraction_leaf=params['min_weight_fraction_leaf'])
    clf = clf.fit(x_train, y_train)

    y_pred_test = clf.predict(x_test)
    y_pred_train = clf.predict(x_train)

    u.print_stats(y_train, y_pred_train, y_test, y_pred_test, "Decision Tree")


def random_forest_func(x_train, y_train, x_test, y_test, params):
    # random forest
    clf = RandomForestClassifier(max_depth=params['depth'], random_state=0, criterion='entropy',
                                 class_weight=params['weights'], max_features=params['max_features'])
    clf = clf.fit(x_train, y_train)

    y_pred_test = clf.predict(x_test)
    y_pred_train = clf.predict(x_train)

    u.print_stats(y_train, y_pred_train, y_test, y_pred_test, "Random Forest")


def custom_grid_search(x_train, y_train):
    #full grid search
    num_features = [8, 12, 16]
    cv = 3
    param_grid_xgb = {
        'learning_rate': [0.1, 0.15, 0.2],
        'max_depth': [6, 7, 8],
        'n_estimators': [40, 50, 60],
    }

    param_grid_random_forest = {
        'max_depth': [6, 7, 8],
        'n_estimators': [40, 50, 60],
        'class_weight': [{0: 1, 1: 2}, {0: 1, 1: 3}, {0: 1, 1: 4}, {0: 1, 1: 5}],
        'bootstrap': [True, False]
    }

    param_grid_logistic = {
        'C': [0.1, 0.5, 1],
        'class_weight': [{0: 1, 1: 2}, {0: 1, 1: 3}, {0: 1, 1: 4}, {0: 1, 1: 5}],
        'penalty': ['l2', 'none'],
        'max_iter': [1000, 2000, 3000]
    }

    param_grid_dt = {
        'max_depth': [8, 9, 10, 11, 12, 13, 14],
        'class_weight': [{0: 1, 1: 2}, {0: 1, 1: 3}, {0: 1, 1: 4}, {0: 1, 1: 4}]
    }

    start_time = time.time()
    end_time = start_time + 60 * 60 * 4
    timed_hp_search_and_feature_selection(x_train, y_train, cv=cv, model=XGBClassifier(), end_time=end_time,
                                            grid=param_grid_xgb, num_features=num_features, filename='xgboost_hp')

    start_time = time.time()
    end_time = start_time + 60 * 60 * 4
    timed_hp_search_and_feature_selection(x_train, y_train, cv=cv, model=RandomForestClassifier(), end_time=end_time,
                                            grid=param_grid_random_forest, num_features=num_features,
                                            filename='random_forest_hp')

    start_time = time.time()
    end_time = start_time + 60 * 60 * 4
    timed_hp_search_and_feature_selection(x_train, y_train, cv=cv, model=LogisticRegression(), end_time=end_time,
                                            grid=param_grid_logistic, num_features=num_features,
                                            filename='logistic_hp')

    start_time = time.time()
    end_time = start_time + 60 * 60 * 4
    timed_hp_search_and_feature_selection(x_train, y_train, cv=cv, model=DecisionTreeClassifier(), end_time=end_time,
                                            grid=param_grid_dt, num_features=num_features,
                                            filename='dt_hp')


def timed_hp_search_and_feature_selection(x, y, cv, model, end_time, grid, num_features, filename):
    # function for feature selection based on model sampling and fixed running time
    print(f"Starting Grid Search for {model.__class__.__name__} until {datetime.datetime.fromtimestamp(end_time)}")
    best_score = 0
    best_features = None
    best_params = None
    all_stats = []

    while time.time() < end_time:
        # choose the number of features
        n = random.choice(num_features)

        # add them to the model
        x_train = x.sample(n=n, axis=1)

        # grid search
        scoring = {"AUC": "roc_auc", "Accuracy": make_scorer(accuracy_score), "f1": make_scorer(f1_score)}
        grid_search = GridSearchCV(model, grid, cv=cv, scoring=scoring, refit='f1', n_jobs=cpu_count() - 1)
        grid_search.fit(x_train, y)

        best_combination = str(grid_search.best_params_)
        score = round(grid_search.best_score_, 4)
        str_score = str(score)
        best_features = str(x_train.columns.tolist())
        features = [1 if x in x_train.columns.tolist() else 0 for x in x]
        res = grid_search.cv_results_
        index = list(res['mean_test_AUC']).index(max(res['mean_test_AUC']))
        stats_no_features = [best_combination, "f1:" + str(round(res['mean_test_f1'][index], 3)), "AUC:" +
                 str(round(res['mean_test_AUC'][index], 3)),
                 "Acc:" + str(round(res['mean_test_Accuracy'][index], 3)), str_score]

        stats = features+stats_no_features
        all_stats.append(stats)

        if time.time() > end_time:
            break

    title = x.columns.tolist() + ["Hyper Parameters", "F1","AUC", "Accuracy", "Best Score"]

    all_stats.sort(key=lambda x: x[-1], reverse=True)
    with open(filename+'.csv', 'a', newline='') as csv_file:
        # Create a writer object
        writer = csv.writer(csv_file)
        writer.writerow(title)
        for row in all_stats:
            writer.writerow(row)
