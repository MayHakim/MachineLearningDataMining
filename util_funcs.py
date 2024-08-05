import pandas as pd
import tensorflow as tf
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix, make_scorer
import numpy as np
from sklearn.model_selection import GridSearchCV


# utility functions: conversions, alignment, etc

def flatten_list(input_list):
    flat_list = []
    for sublist in input_list:
        flat_list.extend(sublist)
    return flat_list


def align_feature_matrices(x, y):
    # Get the set of column names in x and y
    x_cols = set(x.columns)
    y_cols = set(y.columns)

    # Find the set of columns that are only in x
    cols_only_in_x = x_cols - y_cols

    # Add those columns to y, filled with zeros
    for col in cols_only_in_x:
        y[col] = 0

    # Find the set of columns that are only in y
    cols_only_in_y = y_cols - x_cols

    # Add those columns to x, filled with zeros
    for col in cols_only_in_y:
        x[col] = 0

    # Sort the columns in both x and y to match
    x = x.sort_index(axis=1)
    y = y.sort_index(axis=1)

    return x, y


def sample_rows(df, n=10000000):
    df_to_sample = df[df['is_fraud'] == 0]
    # Sample n rows from the dataframe
    sampled_df = df_to_sample.sample(n=n)

    # Get the index values of the sampled rows
    sampled_index = sampled_df.index

    # Create a new dataframe with the rows not in the sampled index
    remaining_df = df_to_sample[~df_to_sample.index.isin(sampled_index)]
    sampled_df = pd.concat([sampled_df, df[df['is_fraud'] == 1]])

    # Return the sampled and remaining dataframes
    return sampled_df, remaining_df


def category_to_num(dataframe, col):
    # get the unique categories in the column
    categories = dataframe[col].unique()

    # create a dictionary with the categories as keys and numeric values as values
    category_dict = {categories[i]: i for i in range(len(categories))}

    # convert the column of categories to a column of numeric values using the dictionary
    dataframe[col] = dataframe[col].map(category_dict)

    return dataframe[col], category_dict


def to_tensor(x):
    new_tensor = tf.convert_to_tensor(x, dtype='float32')
    return new_tensor


def print_stats(y_train, y_pred_train, y_test, y_pred_test, name):
    print('_' * 100)
    print("Now using: ", name)
    train_acc = np.sum(y_pred_train == y_train) / len(y_train)
    test_acc = np.sum(y_pred_train == y_train) / len(y_train)

    train_f1 = f1_score(y_train, y_pred_train)
    test_f1 = f1_score(y_test, y_pred_test)

    train_auc = roc_auc_score(y_train, y_pred_train)
    test_auc = roc_auc_score(y_test, y_pred_test)

    print(f"Train accuracy: {train_acc:.4f}, f1: {train_f1:.4f}, AUC score: {train_auc:.4f}")
    print("train confusion matrix:")
    cm = confusion_matrix(y_train, y_pred_train)
    print(cm)

    print(f"Test accuracy: {test_acc:.4f}, f1: {test_f1:.4f}, AUC score: {test_auc:.4f}")
    print("test confusion matrix:")
    cm = confusion_matrix(y_test, y_pred_test)
    print(cm)

    return train_f1, test_f1, train_auc, test_auc


def categorize(x, y, z, letter):
    categories = {}
    for i, value in enumerate(y):
        found = False
        for j, threshold in enumerate(z):
            if value <= threshold:
                categories[x[i]] = f"{letter}{j + 1}"
                found = True
                break
        if not found:
            categories[x[i]] = f"{letter}{len(z) + 1}"
    return categories


def categorical_probs(data, cats, ratios, train=False, mapping={}):
    groups = data.groupby(cats)

    # Get the counts for each class in each category
    counts = groups[ratios].value_counts().unstack()

    # Calculate the ratio of class 0 to class 1 in each category
    for index, num in enumerate(counts[0]):
        if np.isnan(counts[0][index]):
            counts[0][index] = 1
            counts[1][index] = 0
        if np.isnan(counts[1][index]):
            counts[1][index] = 0

    ratio = counts[1] / counts[0]
    counts['ratio'] = ratio

    return counts


def make_big_categories(df, old_cat_name, letter, train, counts=None, th=[0.02]):
    if train:
        counts = categorical_probs(df, old_cat_name, 'is_fraud')

    mapping = categorize(counts.index, counts['ratio'], th, letter)
    new_cat = df[old_cat_name].map(mapping)

    return new_cat, mapping


def data_preprocess(sample, train=True, train_counts=[]):

    # main function to prepare the data for models

    left_over = []
    all_counts = []
    new_categories = []

    # geneder to binary
    sample['gender'], dict_gender = category_to_num(sample, 'gender')
    _dicts = {'gender': dict_gender}

    # birthyear extraction
    birth_year = sample['dob'].str.slice(0, 4)
    sample['byear'] = birth_year

    # month and hour extraction and categorization
    sample['month'] = sample['trans_date_trans_time'].str.slice(5, 7)
    sample['hour'] = sample['trans_date_trans_time'].str.slice(11, 13)

    months = {'01': 'Jan', '02': 'Feb', '04': 'March', '04': 'April', '05 ': 'May', '06': 'June',
              '07': 'July', '08': 'Aug', '09': 'Sept', '10': 'Oct', '11': 'Nov', '12': 'Dec', }
    sample['month'] = sample['month'].map(months)

    hours = {'01': "B", '02': "B", '03': "B", '04': "A", '05': "A", '06': "A", '07': "A",
             '08': "A", '09': "A", '10': "A", '11': "A", '12': "A", '13': "A", '14': "A",
             '15': "A", '16': "A", '17': "A", '18': "A", '19': "A", '20': "A", '21': "A",
             '21': "C", '22': "C", '00': "B"}
    sample['hour'] = sample['hour'].map(hours)


    # catagroy creation based on probabilites for varibales with > 100 categories
    if train:
        merchants, merchant_index = make_big_categories(sample, 'merchant', 0, train, th=[0.03])
        merchant_index = [merchant_index, 'merchant_index']
        all_counts.append(merchant_index)
        new_categories.append(merchants)

        jobs, job_index = make_big_categories(sample, 'job', 1, train, th=[0.03])
        job_index = [job_index, 'job_index']
        all_counts.append(job_index)
        new_categories.append(jobs)

        cities, city_index = make_big_categories(sample, 'city', 2, train, th=[0.03])
        city_index = [city_index, 'city_index']
        all_counts.append(city_index)
        new_categories.append(cities)

        streets, street_index = make_big_categories(sample, 'street', 3, train, th=[0.03])
        street_index = [street_index, 'street_index']
        all_counts.append(street_index)
        new_categories.append(streets)

        states, states_index = make_big_categories(sample, 'state', 4, train, th=[0.03])
        states_index = [states_index, 'state_index']
        all_counts.append(states_index)
        new_categories.append(states)

        month, month_index = make_big_categories(sample, 'month', 5, train, th=[0.03])
        month_index = [month_index, 'month_index']
        all_counts.append(month_index)
        new_categories.append(month)

        byear, byear_index = make_big_categories(sample, 'byear', 6, train, th=[0.03])
        byear_index = [byear_index, 'byear_index']
        all_counts.append(byear_index)
        new_categories.append(byear)
    else:
        # same categories for test set
        for index, item in enumerate(train_counts):
            old_name = item[1].replace("_index", "")
            new_categories.append(sample[old_name].map(item[0]))

    sample['byear'] = sample['byear'].astype(int)


    # categories into dummies
    states = pd.get_dummies(sample['state'])
    months = pd.get_dummies(sample['month'])
    hours = pd.get_dummies(sample['hour'])
    category = pd.get_dummies(sample['category'])
    merchant_cats = pd.get_dummies(new_categories[0])
    job_cats = pd.get_dummies(new_categories[1])
    city_cats = pd.get_dummies(new_categories[2])
    street_cats = pd.get_dummies(new_categories[3])
    states_cats = pd.get_dummies(new_categories[4])
    month_cats = pd.get_dummies(new_categories[5])
    birth_year_cats = pd.get_dummies(new_categories[6])

    y = sample['is_fraud']

    # sets preperation
    sample = sample.drop(
        columns=['trans_date_trans_time', 'cc_num', 'first', 'last', 'trans_num', 'dob', 'is_fraud', 'unix_time',
                 'street', 'job', 'state', 'city', 'merchant', 'category', 'zip', 'Unnamed: 0', 'city_pop', 'month',
                 'hour', ], axis=1)

    # sample = sample.drop(columns=['lat', 'long'])
    sample = sample.drop(columns=['merch_long', 'merch_lat'])

    x = pd.concat([sample, hours, category, months, states['AK'], states['RI'], street_cats,
                   city_cats, merchant_cats, job_cats], axis=1)

    if train:
        return x, y, _dicts, left_over, all_counts

    return x, y, _dicts, left_over



def accuracy_print(model, x_train, y_train, x_test=None, y_test=None, train=False):
    if train:
        preds = model.predict(x_train)
        print("train confusion matrix")
        print(confusion_matrix(y_train, preds))
        print("train accuracy is ", model.score(x_train, y_train))
        print("train f1 is ", f1_score(y_train, preds))
        print('train roc auc is ', roc_auc_score(y_train, preds))
    else:
        preds = model.predict(x_test)
        print("test confusion matrix")
        print(confusion_matrix(y_test, preds))
        print("test accuracy is ", model.score(x_test, y_test))
        print("test f1 is ", f1_score(y_test, preds))
        print('test roc auc is ', roc_auc_score(y_test, preds))


def combine_csv_files(file1, file2, file3, file4, n=None, k=None):
    df1 = pd.read_csv(file1, nrows=n)
    df2 = pd.read_csv(file2, nrows=n)
    df3 = pd.read_csv(file3, nrows=n)
    df4 = pd.read_csv(file4, nrows=n)
    # df = pd.concat([df1, df2, df3, df4])

    df = pd.concat([df1, df3, df4]).reset_index(drop=True)

    # Create an empty list to store the row sums
    row_sums = []

    # Iterate through the rows of the DataFrame
    for index, row in df.iterrows():
        # Compute the sum of the row
        row_sum = sum([x for x in row if x in [0, 1]])
        # Append the sum to the list
        row_sums.append(row_sum)

    # Add a new column with the row sums
    df['row_sum'] = row_sums

    # Create an empty list to store the sums
    sums = []

    # Iterate through the columns of the DataFrame
    for col in df:
        # Check if the column contains only values of 0 and 1
        if all(x in [0, 1] for x in df[col]):
            # If the column contains only values of 0 and 1, compute the sum and append it to the list
            sums.append(sum(df[col]))
        else:
            # If the column contains other values, append a NaN value to the list
            sums.append(float('NaN'))

    # Create a new row with the sums
    sum_row = pd.DataFrame([sums], columns=df.columns)

    # Concatenate the new row to the bottom of the DataFrame
    df = pd.concat([df, sum_row])
    df = df.reindex(sorted(df.columns, key=lambda x: df[x].iloc[-1], reverse=True), axis=1)

    return df, [col for col in df if df[col].iloc[-1] >= k]


def param_search(model, random_grid, list, cv, x_train, y_train, x_test, y_test):
    grid_search = GridSearchCV(estimator=model, param_grid=random_grid, cv=cv, scoring=make_scorer(f1_score))
    grid_search.fit(x_train, y_train)
    print("Best hyperparameters:", grid_search.best_params_)
    best = grid_search.best_estimator_
    print("Test set accuracy:", grid_search.score(x_test, y_test))
    list.append(best)
    print("the best model is ", best)
    print("accuracy after adjustment is ")
    accuracy_print(best, x_train, y_train, x_test, y_test, train=True)
    accuracy_print(best, x_train, y_train, x_test, y_test)
