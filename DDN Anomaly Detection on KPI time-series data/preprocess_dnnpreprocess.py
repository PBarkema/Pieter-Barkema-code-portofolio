import pandas as pd
from sklearn.preprocessing import MinMaxScaler, Binarizer
import numpy as np
from os import path
from preprocess_filler_deseasonalize import deseasonalize
from pandas import *
realdir = path.dirname(path.realpath(__file__))

file = read_csv("test.csv")
x=preprocess(file)
def preprocess(test, sliding_windows=10):#train, 
    return [preprocess_test(test, sliding_windows)]#preprocess_train(train, sliding_windows), 


def preprocess_train(train, sliding_windows=10):
    dnn_preprocess_train_path = realdir + '/dnn_preprocess_train.csv'
    if path.exists(dnn_preprocess_train_path):
        train = pd.read_csv(dnn_preprocess_train_path)
    else:
        insert_index = 2
        train.insert(insert_index, 'mean', 0)
        train.insert(insert_index, 'std', 0)
        train.insert(insert_index, 'median', 0)
        # train.insert(insert_index, 'min', 0)
        # train.insert(insert_index, 'max', 0)
        # train.insert(insert_index, 'skew', 0)
        # train.insert(insert_index, 'kurt', 0)
        train.insert(insert_index, 'diff', 0)
        train.insert(insert_index, 'mean_diff', 0)
        train.insert(insert_index, 'std_diff', 0)
        train.insert(insert_index, 'median_diff', 0)
        # train.insert(insert_index, 'diff_2', 0)
        kpi_names = train['KPI ID'].values
        kpi_names = np.unique(kpi_names)
        for kpi_name in kpi_names:
            kpi_train = (train[train["KPI ID"] == kpi_name])
            train.loc[train['KPI ID'] == kpi_name, 'value'] = MinMaxScaler(feature_range=(0, 1)).fit_transform(np.array(kpi_train['value']).reshape(-1, 1))
            kpi_train = (train[train["KPI ID"] == kpi_name])
            kpi_train_rolling = kpi_train['value'].rolling(sliding_windows, min_periods=1)
            train.loc[train['KPI ID'] == kpi_name, 'value'] = kpi_train['value'].fillna(0)
            kpi_train['mean'] = train.loc[train['KPI ID'] == kpi_name, 'mean'] = kpi_train_rolling.mean().fillna(0)
            kpi_train['std'] = train.loc[train['KPI ID'] == kpi_name, 'std'] = kpi_train_rolling.std().fillna(0)
            kpi_train['median'] = train.loc[train['KPI ID'] == kpi_name, 'median'] = kpi_train_rolling.median().fillna(0)
            # train.loc[train['KPI ID'] == kpi_name, 'min'] = kpi_train_rolling.min()
            # train.loc[train['KPI ID'] == kpi_name, 'max'] = kpi_train_rolling.max()
            # train.loc[train['KPI ID'] == kpi_name, 'skew'] = kpi_train_rolling.skew()
            # train.loc[train['KPI ID'] == kpi_name, 'kurt'] = kpi_train_rolling.kurt()
            train.loc[train['KPI ID'] == kpi_name, 'diff'] = kpi_train['value'].diff().fillna(0)
            train.loc[train['KPI ID'] == kpi_name, 'mean_diff'] = kpi_train['mean'].diff().fillna(0)
            train.loc[train['KPI ID'] == kpi_name, 'std_diff'] = kpi_train['std'].diff().fillna(0)
            train.loc[train['KPI ID'] == kpi_name, 'median_diff'] = kpi_train['median'].diff().fillna(0)
            # train.loc[train['KPI ID'] == kpi_name, 'diff_2'] = kpi_train['value'].diff(periods=2).fillna(kpi_train['value'])
        train.fillna(0)
        train.to_csv(dnn_preprocess_train_path, index=False)
    return train


def preprocess_test(test, sliding_windows=10):
    dnn_preprocess_test_path = realdir + '/dnn_preprocess_test.csv'
    if path.exists(dnn_preprocess_test_path):
        test = pd.read_csv(dnn_preprocess_test_path)
    else:
        insert_index = 2
        test.insert(insert_index, 'mean', 0)
        test.insert(insert_index, 'std', 0)
        test.insert(insert_index, 'median', 0)
        # test.insert(insert_index, 'min', 0)
        # test.insert(insert_index, 'max', 0)
        # test.insert(insert_index, 'skew', 0)
        # test.insert(insert_index, 'kurt', 0)
        test.insert(insert_index, 'diff', 0)
        # test.insert(insert_index, 'diff_2', 0)
        test.insert(insert_index, 'mean_diff', 0)
        test.insert(insert_index, 'std_diff', 0)
        test.insert(insert_index, 'median_diff', 0)
        kpi_names = test['KPI ID'].values
        kpi_names = np.unique(kpi_names)
        for kpi_name in kpi_names:
            
            kpi_test = (test[test["KPI ID"] == kpi_name])
            test.loc[test['KPI ID'] == kpi_name, 'value'] = MinMaxScaler(feature_range=(0, 1)).fit_transform(np.array(kpi_test['value']).reshape(-1, 1))
            kpi_test = (test[test["KPI ID"] == kpi_name])
            kpi_test_rolling = kpi_test['value'].rolling(sliding_windows, min_periods=1)
            kpi_test['mean'] = test.loc[test['KPI ID'] == kpi_name, 'mean'] = kpi_test_rolling.mean().fillna(0)
            kpi_test['std'] = test.loc[test['KPI ID'] == kpi_name, 'std'] = kpi_test_rolling.std().fillna(0)
            kpi_test['median'] = test.loc[test['KPI ID'] == kpi_name, 'median'] = kpi_test_rolling.median().fillna(0)
            # test.loc[test['KPI ID'] == kpi_name, 'min'] = kpi_test_rolling.min()
            # test.loc[test['KPI ID'] == kpi_name, 'max'] = kpi_test_rolling.max()
            # test.loc[test['KPI ID'] == kpi_name, 'skew'] = kpi_test_rolling.skew()
            # test.loc[test['KPI ID'] == kpi_name, 'kurt'] = kpi_test_rolling.kurt()
            test.loc[test['KPI ID'] == kpi_name, 'diff'] = kpi_test['value'].diff().fillna(0)
            test.loc[test['KPI ID'] == kpi_name, 'mean_diff'] = kpi_test['mean'].diff().fillna(0)
            test.loc[test['KPI ID'] == kpi_name, 'std_diff'] = kpi_test['std'].diff().fillna(0)
            test.loc[test['KPI ID'] == kpi_name, 'median_diff'] = kpi_test['median'].diff().fillna(0)
#        #prepare data
#        seasonarray = []
#        x=0
#        for t in test:
#            if x>0:
#                seasonarray.append([t['timestamp'],t['value']]) 
#            x+=1
#        # get deseasonalized data
#        fulldict = deseasonalize(seasonarray)
#        x=0
#        for i in test:
#            if x>0:
#                try: i['deseason_day'] = fulldict[i['timestamp']]
#                except: pass
#            x+=1
            # test.loc[test['KPI ID'] == kpi_name, 'diff_2'] = kpi_test['value'].diff(periods=2).fillna(kpi_test['value'])
        test.fillna(0)
        test.to_csv(dnn_preprocess_test_path, index=False)
    return test
