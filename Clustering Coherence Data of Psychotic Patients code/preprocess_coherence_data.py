# -*- coding: utf-8 -*-
"""
Created on Wed May 15 12:51:59 2019

@author: Pieter Barkema

Preprocessing coherence data

This program contains three functions.
One should extract specified demographic/diagnostic features and return a featuremap
One should do the same for coherence features
One should take these coherence features and append the right demographic and diagnostic features
"""


import pandas as pd
import csv
import os,time
import numpy as np
import pickle
from numpy import genfromtxt
from sklearn import preprocessing


# Loading data hard coded path
os.chdir(r"C:\Users\piete\OneDrive\Documenten\BSc\scriptie\code")
# Raw coherence measures
with open("4april", 'rb') as f:
    u = pickle._Unpickler(f)
    u.encoding = 'latin1'
    p = u.load()
    data = p

demo_data = []
with open("demographic_data_29apr.csv") as csvfile:
    reader = csv.reader(csvfile) # change contents to floats
    for row in reader: # each row is a list
        demo_data.append(row)
medicine = []
with open("lijst_medicijnen.csv")  as csvfile:
    reader = csv.reader(csvfile,delimiter=";") # change contents to floats
    next(reader, None)  # skip header
    for row in reader: # each row is a list
        medicine.append(row)

 #%%
def demo_Features(demo_data, PANSS = False, remove_NA = False):
    """ This function extracts relevant demographic measures.
        Parameters:
        demo_data: demographic data as provided
        PANSS: True/False include PANSS score or not
        wdw_size: window size to extract
    """
    # remove labels
    from operator import itemgetter
    demo_data.sort(key=itemgetter(0))
    demo_data_NA = [demo_data[0]]
    # remove NA's and replace with 0's
    
    for dp in demo_data[1:]:
        if remove_NA:
            # if no demographic data is NA
            if "NA" not in dp:
                demo_data_NA.append(dp)
            # if PANSS data is NA skip data point
            elif "NA" in dp[:7]:
                pass
            # if actual demographic NA is found
            else: 
                demo_data_NA.append(dp[:4] + [0 if x=="NA" else float(x) for x in dp[4:]])
        # if remove_NA = False replace with 0's
        else: demo_data_NA.append(dp[:4] + [0 if x=="NA" else float(x) for x in dp[4:]])
    # remove coherence measures and group labels but include PANSS or not
    if PANSS:
        filter_dem = [i[4:11] for i in demo_data_NA]
    else: 
        filter_dem = [i[4:7] for i in demo_data_NA]
    # extract labels
    if PANSS: labels = demo_data_NA[0][3:11]
    else: labels = [i for i in demo_data_NA[0][3:11] if "PANSS" not in i]
    # Encode gender in binary
    gender = [i[3] for i in demo_data_NA]
    gender_bin = []
    for i in gender:
        if i == "Vrouw":
            gender_bin.append([0])
        elif i == "Man":
            gender_bin.append([1])
        else: gender_bin.append([i])
    
    # extract P0 numbers
    dem_P0 = [i[1] for i in demo_data_NA]
    # combine features into one datapoint with subject nr as key
    zip_dem_fts = zip(dem_P0,gender_bin,filter_dem)
    dem_featuremap = {}
    for dp in zip_dem_fts:     
        complete_dp = dp[1] + dp[2] 
        dem_featuremap[dp[0]] = complete_dp
        
    # remove labels
    dem_featuremap.pop("subject")
    return dem_featuremap, labels
# testrun
#demo_features, labels2 = demo_Features(demo_data)
#%%
def coherence_Features(raw_data,wdw_size = 10):
    """ This function extracts relevant coherence data measures.
        Parameters:
            raw_data: data
            wdw_size: window size to extract
    """
    featuremap = {}
    for subject in raw_data:
        subject_list = []
        for key,value in subject.items():
            subject_list.append((key,value))
        subject_list.sort()
        # select relevant features per data point
        subject_nr = "Not found"
        rel_features = []
        labels = []
        for i in subject_list:
            if i[0] == "subject":
                subject_nr = str(i[1])
            elif i[0] == "group":
                rel_features.append(i[1])
                
            # Feature extraction
                        # train.loc[train['KPI ID'] == kpi_name, 'min'] = kpi_train_rolling.min()
            # train.loc[train['KPI ID'] == kpi_name, 'max'] = kpi_train_rolling.max()

            else:
                # simple only
                if i[0].split("_")[2] == str(wdw_size) and i[0].split("_")[1] != "summary":# and i[0].split("_")[0] != "raw":
                    labels.append(i[0])
                    rel_features.append(i[1])
        
        featuremap[subject_nr] = rel_features
    set(labels)
    return featuremap,labels
# testrun
#featuremap,labels = coherence_Features(data)
#%%
    
def full_Featuremap(demo_data, coherence_data , PANSS = True, remove_NA = False,wdw_size = 10):
    """ The function which adds demographic features to coherence features
        Parameters:
            demo_data: demographic data as provided
            coherence_data: coherence data as provided
            PANSS: True/False if PANSS scores are added or not
            remove_NA: True/False if NA's are removed or not
    """
    demo_fts, labels2 = demo_Features(demo_data, PANSS, remove_NA)
    coherence_fts, labels = coherence_Features(coherence_data,wdw_size)
    # Combine demographic data with coherence measures
    X = [] 
    Y = []
    nr_keys = []
    for dp in demo_fts:
        # Binary encode Y 
        if coherence_fts[dp].pop(0) == "Psychose":
            Y.append(1)
        else: Y.append(0)
        X.append(demo_fts[dp] + coherence_fts[dp])   
        nr_keys.append(dp)
    # Scale data
    #print(X)
    X_scaled = X#preprocessing.scale(X)
    return X_scaled,Y, nr_keys, labels2 + labels
# testrun
#X,Y, keys, labels = full_Featuremap(demo_data, data)
