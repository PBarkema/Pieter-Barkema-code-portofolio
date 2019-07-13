# -*- coding: utf-8 -*-
"""
Created on Sun May 19 10:59:35 2019

@author: Pieter Barkema



Functions:
    Feature extraction
    Feature validation
    Feature plotter
    Feature selector
    Test significance patients vs controls (for all features)

Functionality for:
    Preprocessing data
    Test membership similarity
    K-means clustering with random restart
        and silhouette score
        plus member analysis
        and PCA
    Hierarchical clustering
        with dendrogram
        and member analysis
        centroid calculation
    K-means elbow method model selection
    PCA visualization
    
"""

""" Import libraries and prepare data """

# Path
import os,time
os.chdir(r"C:\Users\piete\OneDrive\Documenten\scriptie\code")

from sklearn import preprocessing

import pandas as pd
import csv
import numpy as np
import pickle
from numpy import genfromtxt
import preprocess_coherence as preprocess
from preprocess_coherence import full_Featuremap as features
from sklearn.cluster import KMeans
import importlib
from preprocess_coherence import coherence_Features
from operator import itemgetter
from scipy.stats import skew
from scipy.stats import kurtosis
from numpy import mean
import pylab as pl
from sklearn.decomposition import PCA
from numpy import std
from scipy.stats import mannwhitneyu
# Reload after preprocess changes
importlib.reload(preprocess)
# Loading data


# Raw coherence measures
with open("4april", 'rb') as f:
    u = pickle._Unpickler(f)
    u.encoding = 'latin1'
    p = u.load()
    data = p

# Get demographics data
demo_data = []
with open("demographic_data_29apr.csv") as csvfile:
    reader = csv.reader(csvfile) 
    for row in reader: 
        demo_data.append(row)

# Get medicine data
medicine = []
with open("lijst_medicijnen.csv")  as csvfile:
    reader = csv.reader(csvfile,delimiter=";") 
    next(reader, None)  # skip header
    for row in reader: 
        medicine.append(row)

# Get X, Y, keys and labels from the original data sets, with or without PANSS/NA
wdw = 10
X,Y,keys, labels = features(demo_data, data, PANSS = True, remove_NA = False, wdw_size = wdw)
        #%%
""" Get medicine and divide in categories 'high' and not 'high' D2R (low, none) """
#highd2r = ["amisulpride","aripiprazol","fluanxol","haloperidol","risperdal"]
#lowd2r = ["clozapine","olanzapine","paliperidon", "paliperidon depot","quetiapine"]
#p0_medicine = {}
#count_medicine = {}
#for i in medicine:
#    p0 = i[0]
#    med = i[1].split(" ")[0]
#    doses = i[2:]
#    
#    if med in highd2r:
#        p0_medicine[p0] = ["high"] + doses
#    elif med in lowd2r:
#        p0_medicine[p0] = ["low"] + doses
#    else:
#        p0_medicine[p0] = ["none"] + doses

#%%


def feature_extraction(X,wdw, mode = "simple"):
    """ 
        Extracts designated features from raw coherence measures
        Make sure to include the new label when adding features
        
        Parameters
        ----------
        X : data as provided by coherence_Features
        wdw : window size range 2 - 20
        mode :  ["simple"] ( "summary" functionality removed due to unnecessity)
                changeable in preprocess_coherence.py in coherence_Features
    """
    # Gather raw data arrays per person
    raw_data = []
    for key in X:
        arrays = []
        # For every raw feature per person
        for ft in X[key]:
            if type(ft) is np.ndarray:  # All raw data is type ndarray
                arrays = arrays + [ft]
        raw_data.append([key] + arrays)
    # Sorted by P0 number
    raw_data.sort(key=itemgetter(0))
      
    # Gather features for one mode
    new_fts = []
    for person in raw_data:
        if mode == "simple":
            raw_array = person[1]
        if mode == "summary":
            raw_array = person[2]
        # Engineer new features
        temp = [] 
        temp.append(skew(raw_array))
        temp.append(kurtosis(raw_array))
        temp.append(np.median(raw_array))
        temp.append(np.std(raw_array))
        diff = []
        index =0
        while index< len(raw_array)-1:
            diff.append(abs(raw_array[index] - raw_array[index+1]))
            index+=1
        temp.append(mean(diff))
        new_fts.append(temp)
    # Create new labels
    labels.append("skew" +  "_" + mode +  "_" + str(wdw))
    labels.append("kurt" + "_" + mode + "_" +  str(wdw))
    labels.append("median" + "_" + mode + "_" +  str(wdw))
    labels.append("std" + "_" + mode +  "_" + str(wdw))
    labels.append("diff" + "_" + mode +  "_" + str(wdw))
    return new_fts


## Snippet for adding new features to old features + test run
#raw = coherence_Features(data,wdw_size=wdw)[0]
#raw_fts= feature_extraction(raw, wdw,mode="simple")
#x=0
#while x< len(X):
#    X[x] = X[x] + raw_fts[x]
#    x+=1


#%%

def feature_validation(X,Y,labels):
    """ 
        Print significance levels of features (normality assumed)
                
        Parameters
        ----------
        X : X as provided by coherence_Features
        Y : Y as provided by coherence_Features
        labels: labels as provided by coherence_Features
    """
    
    i=0
    patients = []
    controls = []
    # Split into patients and controls
    while i < len(X):
        if Y[i] == 1:
            patients.append(X[i])
        if Y[i] == 0:
            controls.append(X[i])
        i+=1
    coherence_fts = labels
    for ft in coherence_fts[8:]:
        # for features, not raw data
        if "raw" not in ft:
            print("\n",ft,significance(patients,controls,coherence_fts.index(ft)))

def significance(patients, controls, label):
    """
        Return statistical results of the features 
        compared between patients and controls.
                
        Parameters
        ----------
        patients : patient subject data as provided by feature_validation
        controls : control subject data as provided by feature_validation
        label: current label as provided by feature_validation
    """
    
    from scipy.stats import mannwhitneyu, shapiro, ranksums, wilcoxon
    curr_patients = [x[label] for x in patients]
    curr_controls = [y[label] for y in controls]
    # Snippet for normality test before mannwhitney u
    #if shapiro(curr_patients)[1] >0.05 and shapiro(curr_controls)[1] > 0.05:
    return "normal with significant difference of p-val: ", str(mannwhitneyu(curr_patients, curr_controls, use_continuity=True)[1])
    
    # Snippet for non-normal non-parametric test
    # return wilcoxon(curr_patients, curr_controls, zero_method='pratt', correction=True)[1]
    # Snippet for rank sums test
    # return "not both normally distributed with Wilcoxon p-val: ", str(ranksums(curr_patients, curr_controls)[1])
    # Snippet for normality test
    # return "patients normality" + str(shapiro(curr_patients)) + " controls normality: " + str(shapiro(curr_controls))

# test run
#feature_validation(X,Y,labels)

#%%
 
def feature_plotter(X,Y,labels):
    """
        Plot every feature cumulatively
        between patients and controls
                        
        Parameters
        ----------
        X : X as provided by coherence_Features
        Y : Y as provided by coherence_Features
        labels: labels as provided by coherence_Features
    """ 
    
    i=0
    patients = []
    controls = []
    # Split into patients and controls
    while i < len(X):
        if Y[i] == 1:
            patients.append(X[i])
        if Y[i] == 0:
            controls.append(X[i])
        i+=1
    i=0
    while i< len(labels):
        if "raw" not in labels[i]:
            # Select feature rows
            ft_patients = [j[i] for j in patients]
            ft_controls = [j[i] for j in controls]
            ft_patients.sort()
            ft_controls.sort()
            cum_patients =ft_patients 
            cum_controls =ft_controls
            index = 0
            # Accumulate scores
            for t in ft_patients:
                cum_patients.append(sum(ft_patients[:index+1]))
                cum_controls.append(sum(ft_controls[:index+1]))
                index+=1
            pl.figure('plot: ' + str(labels[i]))
            pl.plot(cum_patients, c="r") 
            pl.plot(cum_controls, c="b")
            pl.xlabel('index')
            pl.ylabel('feature: ' + labels[i])
            pl.title('patients vs controls: ' + str(labels[i]))
            pl.show()
        i+=1
        
# Test run
#feature_plotter(X,Y, labels)
#%%
        
"""
    The snippet for selecting the best window size
    per feature. 
"""        

def feature_selector(X,Y,labels):
    """
        Select window size per feature
        based on which window size maximizes the difference
        between patients and controls
                                
        Parameters
        ----------
        X : X as provided by coherence_Features
        Y : Y as provided by coherence_Features
        labels: labels as provided by coherence_Features
    """
    i=0
    patients = []
    controls = []
    # Split into patients and controls
    while i < len(X):
        if Y[i] == 1:
            patients.append(X[i])
        if Y[i] == 0:
            controls.append(X[i])
        i+=1
    i=0 
    wdw_allft_diff = {}
    wdw_ft = {}
    while i< len(labels):
        if "raw" not in labels[i]:
            # Select feature rows
            ft_patients = [j[i] for j in patients]
            ft_controls = [j[i] for j in controls]
            # Create full feature map per window size divided into features and patients/controls
            wdw_ft[labels[i].split("_")[0]] = [ft_patients,ft_controls]
            # Compare total scores for patients and controls in one feature and wdw size
            tot_patients = sum(ft_patients)
            tot_controls = sum(ft_controls)
            
            wdw_allft_diff[labels[i]] = tot_patients-tot_controls
        i+=1
    return  wdw_allft_diff, wdw_ft

"""
    Go through the whole process of 
        * extracting all features
        * engineering all new features
        * optimize best window size per feature
"""

# bottom and top window size (goes up from 2 to 20)
i = 5
top_wdw = 15
# Maximize absolute difference per feature    
best_fts = {}
total_fts = {}
all_features = {}

while i <= top_wdw:
    # Extract relevant demographic, PANSS and coherence features
    X,Y,keys, labels = features(demo_data, data, PANSS = True, remove_NA = False, wdw_size = i)
    # Prepare data for feature extraction
    raw = coherence_Features(data,wdw_size=i)[0]
    # Use simple mode to extract features from data
    raw_fts = feature_extraction(raw,i, mode="simple")
    # Add new features to existing set
    x=0
    while x< len(X):
        X[x] = X[x] + raw_fts[x]
        x+=1
    # Calculate total distance per feature for one window size
    curr_fts,wdw_dict = feature_selector(X,Y,labels)
    all_features[i] = wdw_dict
    # Extract coherence features
    for l in labels[8:]:
        curr_wdw = l.split("_")[2]
        curr_ft = l.split("_")[0]
        if curr_ft != "raw":
            total_fts[l] = curr_fts[l]
            # Create dictionary entry for window size per feature
            if curr_ft not in best_fts:
                best_fts[curr_ft] = (curr_wdw, curr_fts[l])
            # Update new best for maximum difference for window size per feature
            if curr_ft in best_fts:
                if abs(best_fts[curr_ft][1]) >= abs(curr_fts[l]):
                    best_fts[curr_ft] = best_fts[curr_ft]
                else: best_fts[curr_ft] = (curr_wdw,curr_fts[l])
    i+=1
# Best window sizes: 13: diff, max; 12:std; 11: median, min;  10: mean; 8: skew; 6: kurt;  5: var;   
# Collect all top features

mean = all_features[10]["mean"]
median = all_features[11]["median"]
diff = all_features[13]["diff"]
var = all_features[5]["var"]
std = all_features[12]["std"]
mini = all_features[11]["min"]
maxi = all_features[13]["max"]
skew = all_features[8]["skew"]
kurt = all_features[6]["kurt"]

# Create list of all top features and labels
cluster_features = [mean,median,diff,var,std,mini,maxi,skew,kurt]
cluster_features_nm = ["10 mean","11 median","13 diff","5 var", "12 std", "11 min","13 max","8 skew","6 kurt"]
#%% 
"""
    Reconstruct people using P0 numbers and selected top features
    This snippet is necessary to combine the selected top features
    with the P0 numbers for identification.
"""
# Fill with correct P0 numbers
member_dict = {}
patients_full = []
controls_full = []
person = 0
while person < len(keys):
    if Y[person] == 1:
        patients_full.append(keys[person])
    if Y[person] == 0:
        controls_full.append(keys[person])
    person+=1

cluster_data = {}
index = 0

# Combine P0 numbers with features separating patients and controls
while index < len(cluster_features):
    person_num = 0
    pt = cluster_features[index][0]
    cont = cluster_features[index][1]
    while person_num < len(pt):
        P0 = patients_full[person_num]
        feature = pt[person_num]
        if P0 not in cluster_data:
            cluster_data[P0] = [feature]
        else: cluster_data[P0].append(feature)
        person_num+=1
    index+=1

    #%%
    
# Plot features with significance test adjusted for 'cluster_features' data
h=0
while h< len(cluster_features):
    ft_patients = cluster_features[h][0]
    ft_controls = cluster_features[h][1]
    cum_patients =[]
    cum_controls =[]
    index =0
    # accumulate scores
    for t in ft_patients:
        cum_patients.append(ft_patients)
        cum_controls.append(ft_controls)
        index+=1
    
    pl.figure('plot: ' + str(cluster_features_nm[h]))
    pl.plot(cum_patients, c="r") #
    pl.plot(cum_controls, c="b")
    pl.xlabel('index')
    pl.ylabel('feature: ' + cluster_features_nm[h])
    pl.title('patients vs controls: ' + str(cluster_features_nm[h]))
    pl.show()
    print(cluster_features_nm[h], " has significant difference of p-val: ", str(mannwhitneyu(cluster_features[h][0], cluster_features[h][1], use_continuity=True)[1]))
    h+=1
#%%
# From source: https://stackoverflow.com/questions/16592222/matplotlib-group-boxplots
""" 
    Create boxplots for every feature: patients vs controls
"""
import matplotlib.pyplot as plt
import numpy as np
slicert = 0
endr = 2
data_a = [i[0] for i in cluster_features[slicert:endr]] #patients
data_b = [i[1] for i in cluster_features[slicert:endr]] #controls

ticks = cluster_features_nm[slicert:endr]

def set_box_color(bp, color):
    plt.setp(bp['boxes'], color=color)
    plt.setp(bp['whiskers'], color=color)
    plt.setp(bp['caps'], color=color)
    plt.setp(bp['medians'], color=color)

plt.figure()

bpl = plt.boxplot(data_a, positions=np.array(range(len(data_a)))*2.0-0.4, sym='', widths=0.6)
bpr = plt.boxplot(data_b, positions=np.array(range(len(data_b)))*2.0+0.4, sym='', widths=0.6)
set_box_color(bpl, '#D7191C')
set_box_color(bpr, '#2C7BB6')

# draw temporary red and blue lines and use them to create a legend
plt.plot([], c='#D7191C', label='Patients')
plt.plot([], c='#2C7BB6', label='Controls')
plt.legend()

plt.xticks(range(0, len(ticks) * 2, 2), ticks)
plt.xlim(-2, len(ticks)*2)
plt.ylim(0.55, 0.65)
plt.tight_layout()
plt.savefig('boxcompare.png')

#%%

"""
    Compare every available clustering for k-means in terms of membership
    by getting the most similar cluster pair for two clusterings.
    This is done through symmetric set difference. The difference in set
    is the amount of data points that is not common. The most similar is 1.00
    and the lowest depends on k of k-means.
"""

memberships = {}
for dicto in member_dict:
    for otherdicto in member_dict:
        index = 0 
        difference=0
        diffdiff=0
        clust1 = set()
        clust2 = set()
        clust3 = set()
        o1 = set()
        o2 = set()
        o3 = set()
        for i in member_dict[dicto][0]:
    
            # if not same cluster number assigned: count difference
            original_c = member_dict[dicto][1][index]
            compare_c = member_dict[otherdicto][1][member_dict[otherdicto][0].index(i)]
            if original_c == 0:
                clust1.add(i)
            elif original_c == 1:
                clust2.add(i)
            elif original_c == 2:
                clust3.add(i)
            if compare_c == 0:
                o1.add(i)
            elif compare_c == 1:
                o2.add(i)
            elif compare_c == 2:
                o3.add(i)
                
            if original_c != compare_c:
                difference +=1
            if original_c == compare_c:
                diffdiff +=1
            index+=1
        # Get the smallest symmetric set difference of all clusters
        diff1 = min(len(clust1^o1),len(clust1^o2))#,len(clust1^o3))
        diff2 = min(len(clust2^o1),len(clust2^o2))#,len(clust2^o3))
        #diff3 = max(len(clust3^o1),len(clust3^o2)#,len(clust3^o3))
        best = min(diff1,diff2)#,diff3)
        memberships[otherdicto + " vs " + dicto] = 1-best/50#max(difference/index,diffdiff/index) 
        print(dicto, " ", otherdicto)
        print(difference/index)
        print(diffdiff/index)
   #%%
"""
    This large snippet of code:
        * creates a K-means model with random restart with specified k and specified features.
        * plots the silhouette scores with the average.
        * plots the clustered data in the dimensions of the first two features.
        * uses PCA analysis on the used features (with option for visualization).
        * saves memberships per cluster.
        * performs statistical tests on demographic, medicine and diagnostic data.
        * shows bar plots with demographic and diagnostic data.

"""

from statistics import mean, stdev
from math import sqrt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from scipy.stats import mannwhitneyu
# Specify k for k-means
range_n_clusters = [2]

cluster_list = [[i] + cluster_data[i] for i in cluster_data]

# Randomly shuffle for fair play
np.random.shuffle(cluster_list)
# Remember member_order after shuffle
member_order = [i[0] for i in cluster_list]
clusterable = [i[1:] for i in cluster_list]# if p0_medicine[i][0] == "high"] # Medicine group specification
cluster_features_nm = ["10 mean","11 median","13 diff","5 var", "12 std", "11 min","13 max","8 skew","6 kurt"]
# Specify indexes of the cluster features to use from cluster_features_nm
specif = [8]
X_scaled_nomin = [[x[i] for i in specif] for x in clusterable]

# Get amount of features
f_nr = len(X_scaled_nomin[0])

# Preprocess by standardizing values
X_scaled_nominmed = preprocessing.scale(X_scaled_nomin)
# Specify data set to use
cluster_X = X_scaled_nominmed

# Begin the clustering and visualization process
for n_clusters in range_n_clusters:
    # Create a subplot with 1 row and 2 columns
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(18, 7)

    # The 1st subplot is the silhouette plot
    # The silhouette coefficient can range from -1, 1 but in this example all
    # lie within [-0.1, 1]
    ax1.set_xlim([-0.1, 1])
    # The (n_clusters+1)*10 is for inserting blank space between silhouette
    # plots of individual clusters, to demarcate them clearly.
    ax1.set_ylim([0, 50 + (n_clusters + 1) * 10])

    # Initialize the clusterer with n_clusters value and a random generator
    # seed of NA for reproducibility.
    # random restart 500 times
    silhouette_avg=0
    rr= 500
    clusterer = KMeans(n_clusters=n_clusters, n_init = rr)#, random_state=8)#, random_state=1)
    cluster_labels = clusterer.fit_predict(cluster_X)
    
        # The silhouette_score gives the average value for all the samples.
        # This gives a perspective into the density and separation of the formed
        # clusters
    silhouette_avg = silhouette_score(cluster_X, cluster_labels)
    print("For n_clusters =", n_clusters,
          "The average silhouette_score with ", rr, " random restarts is :", silhouette_avg) 
    # Compute the silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(cluster_X, cluster_labels)
    y_lower = 10
    for i in range(n_clusters):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values = \
            sample_silhouette_values[cluster_labels == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.nipy_spectral(float(i) / n_clusters)
        ax1.fill_betweenx(np.arange(y_lower, y_upper),
                          0, ith_cluster_silhouette_values,
                          facecolor=color, edgecolor=color, alpha=0.7)

        # Label the silhouette plots with their cluster numbers at the middle
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

    ax1.set_title("The silhouette plot for the various clusters.")
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")

    # The vertical line for average silhouette score of all the values
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

    ax1.set_yticks([])  # Clear the yaxis labels / ticks
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
    
    # Execute PCA analysis on used features
#    pca = PCA(n_components=f_nr).fit(X_scaled_nominmed)
#    pca_c= pca.transform(X_scaled_nominmed)
#    pca_d = [x[0] for x in pca_c]
#    pca_e = [x[1] for x in pca_c]
#    components =pca.components_
#    # Snippet for visualizing in top two PCA components
#    ax2.scatter(pca_d, pca_e, marker='.', s=30, lw=0, alpha=0.7,
#                c=colors, edgecolor='k')
    
    # 2nd Plot showing the actual clusters formed
#    colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
#    ax2.scatter(cluster_X[:, 0], cluster_X[:, 1], marker='.', s=30, lw=0, alpha=0.7,
#                c=colors, edgecolor='k')
#
#    # Labeling the clusters
#    centers = clusterer.cluster_centers_
#    # Draw white circles at cluster centers
#    ax2.scatter(centers[:, 0], centers[:, 1], marker='o',
#                c="white", alpha=1, s=200, edgecolor='k')
#
#    for i, c in enumerate(centers):
#        ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1,
#                    s=50, edgecolor='k')
#
#    ax2.set_title("The visualization of the clustered data.")
#    ax2.set_xlabel("Standardized mean feature space")
#    ax2.set_ylabel("Standardized variance feature space")
#
#    plt.suptitle(("Silhouette analysis for KMeans clustering on sample data "
#                  "with n_clusters = %d" % n_clusters),
#                 fontsize=14, fontweight='bold')

plt.show()

# Create dictionary with members per cluster label
name = ""
for i in specif:
    name = name + cluster_features_nm[i]
member_dict[name] = [member_order, cluster_labels]

# Print importance of feature per PCA component
feature_names = range(1,f_nr+1)
feature_var =[]
for component in pca.components_[:3]: 
    print (" + ".join("%.2f x %s" % (value, name) for value,name in zip(component, feature_names)))
    
rank =0
for result in pca.explained_variance_ratio_[:5]:
    print(rank,": ",result)
    rank+=1

# Add demographic features and clusters to P0 subjects
full_cluster_kmeans = []
index = 0
for cluster_point in member_order:
    p0 = cluster_point
    cluster_nr = cluster_labels[index]
    index+=1
    demos = []
    for person in demo_data:
        if person[1] == p0:
            demos = person[1:11]
#    needs to be enabled for medicine analysis
#    meds_bin = p0_medicine[p0][0]
    full_cluster_kmeans.append([cluster_nr,demos]) #"demos + [meds_bin]" instead of demos for medicine analysis
  
import numpy as np
import matplotlib.pyplot as plt
from numpy import nanmean

# Source code from: https://pythonspot.com/matplotlib-bar-chart/

n_groups = 7
# Extract demographic feature averages per cluster

cluster_means_1 = [ np.array([float(dm) if 'NA' not in dm else np.nan for dm in p[1][3:10]]) for p in full_cluster_kmeans if p[0] == 0]
cluster_means_2 = [ np.array([float(dm) if 'NA' not in dm else np.nan for dm in p[1][3:10]]) for p in full_cluster_kmeans if p[0] == 1]
# Optional third and fourth cluster
#cluster_means_3 = [ np.array([float(dm) if 'NA' not in dm else np.nan for dm in p[1][3:]]) for p in full_cluster_kmeans if p[0] == 2]
#cluster_means_4 = [ np.array([float(dm) if 'NA' not in dm else np.nan for dm in p[1][3:]]) for p in full_cluster_kmeans if p[0] == 3]

## Statistical tests for gender difference
#cluster_gen1 = [ p[1][2] for p in full_cluster_kmeans if p[0] == 0]
#cluster_gen2 = [ p[1][2] for p in full_cluster_kmeans  if p[0] == 1]
#gen1m = cluster_gen1.count("Man")
#gen1v = cluster_gen1.count("Vrouw")
#gen2m = cluster_gen2.count("Man")
#gen2v = cluster_gen2.count("Vrouw")
#print("gender diff: ", chisquare([gen1m,gen1v],[gen2m,gen2v]))

## Statistical tests for medicine difference
#cluster_meds_1 = [ p[1][10] for p in full_cluster_kmeans if p[0] == 0]
#cluster_meds_2 = [ p[1][10] for p in full_cluster_kmeans if p[0] == 1]
#meds1=[]
#meds2=[]
#for j in cluster_meds_1: meds1.append(j) 
#for j in cluster_meds_2: meds2.append(j) 
#med_count1 = [meds1.count("high"),meds1.count("low")+meds1.count("none")]
#med_count2 = [meds2.count("high"),meds2.count("low")+meds2.count("none")]
#chi2 =  chisquare(med_count1,med_count2)[0]
#print(chisquare(med_count1,med_count2))
#print(mannwhitneyu(feat1[6],feat2[6]))
#print("effect size", math.sqrt(chi2/len(clusterable)))

feat3 = []
feat4 = []
feat1 = []
feat2 = []
for i in range (7):
    feat1.append([j[i] for j in cluster_means_1])
    feat2.append([j[i] for j in cluster_means_2])
    #feat3.append([j[i] for j in cluster_means_3])
 #   feat4.append([j[i] for j in cluster_means_4])
demo_labels = demo_data[0][4:11]
for i in range (7):        
    print(demo_labels[i], ": ", mannwhitneyu(feat1[i],feat2[i]))

c0 = feat1[6]
c1 = feat2[6]
cohens_d = (mean(c0) - mean(c1)) / (sqrt((stdev(c0) ** 2 + stdev(c1) ** 2) / 2))
print("Cohen's d effect size: ", cohens_d)
print("Absolute difference of mean: ", mean(c0) - mean(c1))

# Translate demographic features to bar plots
feat1 = np.array([nanmean(x) for x in feat1]).T
feat2 = np.array([nanmean(x) for x in feat2]).T
#feat3 = np.array([nanmean(x) for x in feat3]).T
#feat4 = np.array([nanmean(x) for x in feat4]).T

fig, ax = plt.subplots()
index = np.arange(n_groups)
bar_width = 0.35
opacity = 0.5

rects1 = plt.bar(index*2, feat1, bar_width,
alpha=opacity,
color='b',
label='Cluster 1')

rects2 = plt.bar(index*2 + bar_width, feat2, bar_width,
alpha=opacity,
color='g',
label='Cluster 2')

#rects3 = plt.bar(index*2 + 2*bar_width, feat3, bar_width,
#alpha=opacity,
#color='r',
#label='Cluster 3')

#rects4 = plt.bar(index*2 + 3*bar_width, feat4, bar_width,
#alpha=opacity,
#color='y',
#label='4')

plt.xlabel('Member properties')
plt.ylabel('Average score')
#plt.title('Scores by person')
plt.xticks(index*2 -0.2+ bar_width, ('Age', 'YOE', 'YOEP', 'PANSS \n total', 'PANSS \n negative', 'PANSS \n positive', 'PANSS \n general'))
plt.legend()

plt.tight_layout()
plt.show()

#%%
"""
    Create a dendrogram with linkage from the data and plot it
"""


# Dendrogram code from: https://stackabuse.com/hierarchical-clustering-with-python-and-scikit-learn/
from scipy.cluster.hierarchy import dendrogram, cophenet, linkage 
from matplotlib import pyplot as plt
from scipy.spatial.distance import pdist

cluster_list = [[i] + cluster_data[i] for i in cluster_data]
#np.random.shuffle(cluster_list)
clusterable = [i[1:] for i in cluster_list ]#if p0_medicine[i[0]][0] == "high"
ind = 0
specif = [3]
X_scaled_nomin = [[x[i] for i in specif] for x in clusterable] #sliced data set
X_scaled_nominmed = preprocessing.scale(X_scaled_nomin)

#linked_comp=linkage(X_scaled_nominmed, 'complete')

linked = linkage(X_scaled_nominmed, 'average')

hier_X = pdist(X_scaled_nominmed)
hier_Z = cophenet(linked)

labelList = [c[0] for c in cluster_list]#range(0,50)#cluster_features_nm
plt.figure(figsize=(10, 7))
dendo = dendrogram(linked,  
            orientation='top',
            labels=labelList,
            distance_sort='descending',
            show_leaf_counts=True)
dendo_order = dendo["ivl"]

plt.show()
corr_coef = np.corrcoef(hier_X,hier_Z)[0,1]
print(corr_coef)
#%%    
"""
    Cut off the dendrogram and create clusters.
    Analyse the created clusters and compare the members with 
    significant tests and bar plots.
"""
from statistics import mean, stdev
from math import sqrt
# Set boundaries for cut-off of dendogram by order of dendo_order
boundary1 = dendo_order.index("P022")
boundary2 = dendo_order.index("P015") 
#boundary3 = dendo_order.index("P088") 
#boundary4 = len(dendo_order)-1
hierarchy_1 = dendo_order[:boundary1]
hierarchy_2 = dendo_order[boundary1:]
#hierarchy_3 = dendo_order[boundary2:]#boundary4] 
  
for person in demo_data:
    demos = person[1:11]
#    meds_bin = p0_medicine[p0][0]
    if person[1] in hierarchy_1:  
        hierarch_c1.append(demos)
    elif person[1] in hierarchy_2:
        hierarch_c2.append(demos)
    elif person[1] in hierarchy_3:
        hierarch_c3.append(demos)

# Select relevant demographics (from index 3) per cluster
cluster_means_1 = [ np.array([float(dm) if 'NA' not in dm else np.nan for dm in p[3:]]) for p in hierarch_c1 if p[0] in hierarchy_1]
cluster_means_2 = [ np.array([float(dm) if 'NA' not in dm else np.nan for dm in p[3:]]) for p in hierarch_c2 if p[0] in hierarchy_2]




## Statistical tests for gender
#cluster_gen1 = [ p[2] for p in hierarch_c1 if p[0] in hierarchy_1]
#cluster_gen2 = [ p[2] for p in hierarch_c2 if p[0] in hierarchy_2]
#gen1m = cluster_gen1.count("Man")
#gen1v = cluster_gen1.count("Vrouw")
#gen2m = cluster_gen2.count("Man")
#gen2v = cluster_gen2.count("Vrouw")
#print("gender diff: ", chisquare([gen1m,gen1v],[gen2m,gen2v]))

## Statistical tests for medicine
#cluster_meds_1 = [ p0_medicine[p[0]][0] for p in hierarch_c1 if p[0] in hierarchy_1]
#cluster_meds_2 = [ p0_medicine[p[0]][0] for p in hierarch_c2 if p[0] in hierarchy_2]
#meds1=[]
#meds2=[]
#for j in cluster_meds_1: meds1.append(j) 
#for j in cluster_meds_2: meds2.append(j) 
#med_count1 = [meds1.count("high"),meds1.count("low")+meds1.count("none")]
#med_count2 = [meds2.count("high"),meds2.count("low")+meds2.count("none")]
#chi2 =  chisquare(med_count1,med_count2)[0]
#print(chisquare(med_count1,med_count2))

# Optional third or fourth cluster
#cluster_means_3 = [ np.array([float(dm) if 'NA' not in dm else np.nan for dm in p[3:]]) for p in hierarch_c3 if p[0] in hierarchy_3]
#cluster_means_4 = [ np.array([float(dm) if 'NA' not in dm else np.nan for dm in p[1][3:]]) for p in full_cluster_kmeans if p[0] == 3]

ofeat1 = []
ofeat2 = []
ofeat3 = []
ofeat4 = []
# Change data points into arrays of demographics
for i in range (7):
    ofeat1.append([j[i] for j in cluster_means_1])
    ofeat2.append([j[i] for j in cluster_means_2])
    ofeat3.append([j[i] for j in cluster_means_3])
    ofeat4.append([j[i] for j in cluster_means_4])
    
#print("effect size", math.sqrt(chi2/len(clusterable)))
print("mean", mean(ofeat1[6]) - mean(ofeat2[6]))


# test conditions
c0 = ofeat1[6]
c1 = ofeat2[6]

cohens_d = (mean(c0) - mean(c1)) / (sqrt((stdev(c0) ** 2 + stdev(c1) ** 2) / 2))
print("Cohens d effect size: ", cohens_d)

# Turn demographic data into bar plots: every cluster has a bar
feat1 = np.array([nanmean(x) for x in feat1]).T
feat2 = np.array([nanmean(x) for x in feat2]).T
feat3 = np.array([nanmean(x) for x in feat3]).T
feat4 = np.array([nanmean(x) for x in feat4]).T
# create plot
fig, ax = plt.subplots()
index = np.arange(n_groups)
bar_width = 0.35
opacity = 0.5

rects1 = plt.bar(index*2, feat1, bar_width,
alpha=opacity,
color='b',
label='1')

rects2 = plt.bar(index*2 + bar_width, feat2, bar_width,
alpha=opacity,
color='g',
label='2')

#rects3 = plt.bar(index*2 + 2*bar_width, feat3, bar_width,
#alpha=opacity,
#color='r',
#label='3')

#rects4 = plt.bar(index*2 + 3*bar_width, feat4, bar_width,
#alpha=opacity,
#color='y',
#label='4')

plt.xlabel('Demographics')
plt.ylabel('Scores')
plt.title('Scores by person')
plt.xticks(index*2 + bar_width, ('Age', 'YOE', 'YOEP', 'PANSS total', 'PANSS neg', 'PANSS pos', 'PANSS gen'))
plt.legend()

plt.tight_layout()
plt.show()

# Run significance test per demographic label
from scipy.stats import f_oneway
from scipy.stats import chisquare
start = 4
demo_labels = demo_data[0][start:11]
cluster_0 = hierarch_c1
cluster_1 = hierarch_c2
cluster_2 = hierarch_c3
indexer = start-1
for lab in demo_labels:
    if indexer > 50:
        c_huh1= [float(dm[indexer])   for dm in cluster_0 ]
        c_huh2 = [float(dm[indexer])  for dm in cluster_1 ]
        c_huh3 = [float(dm[indexer])  for dm in cluster_2 ]
    else:
        c_huh1= [float(dm[indexer])   if 'NA' not in dm[indexer] else np.nan for dm in cluster_0 ]
        c_huh2= [float(dm[indexer])   if 'NA' not in dm[indexer] else np.nan for dm in cluster_1 ]
        c_huh3= [float(dm[indexer])   if 'NA' not in dm[indexer] else np.nan for dm in cluster_2 ]
    y= c_huh1
    z = c_huh2
    print(lab, " has significant difference of p-val: ", str(mannwhitneyu(c_huh1,c_huh2)))
    indexer+=1

#%%  
"""
    Snippets for KMeans model selection with elbow curve
    and snippets for PCA analysis and visualization
"""

## k-Model selection for kmeans
#data = X_scalednominmed
#Nc = range(1, 20)
#kmeans = [KMeans(n_clusters=i) for i in Nc]
#score = [kmeans[i].fit(X_pro).score(X_pro) for i in range(len(kmeans))]
#pl.plot(Nc,score)
#pl.xlabel('Number of Clusters')
#pl.ylabel('Score')
#pl.title('Elbow Curve')
#pl.show()

# Select PC's and transform data to 2D
pca = PCA(n_components=2).fit(X_scaled_nominmed)
pca_c= pca.transform(X_scaled_nominmed)
pca_d = [x[0] for x in pca_c]
pca_e = [x[1] for x in pca_c]
components =pca.components_

# Use Model selection's K-means
kmeans= KMeans(n_clusters=3) 
kmeansoutput=kmeans.fit(X_scaled_nominmed)

# Print importance of feature per component
feature_names = cluster_features_nm
feature_var =[]
for component in pca.components_[:3]: 
    print (" + ".join("%.2f x %s" % (value, name) for value,name in zip(component, feature_names)))
    
rank =0
for result in pca.explained_variance_ratio_[:5]:
    print(rank,": ",result)
    rank+=1

# Get eigenvalues
#from numpy import array
#X_pro = array(X_pro)
#centered_matrix = X_pro - X_pro.mean(axis=1)[:, np.newaxis]
#cov = np.dot(centered_matrix, centered_matrix.T)
#eigvals, eigvecs = np.linalg.eig(cov)

# Plot data in terms of PC's
pl.figure('PCA 1 to 2')
pl.scatter(pca_d, pca_e, c=kmeansoutput.labels_)#
pl.xlabel('pc 1')
pl.ylabel('pc 2')
pl.title('Cluster K-Means')
pl.show()
#%%

""" Calculate centroid coordinates for hierarchical clustering """
cluster_list = [[i] + cluster_data[i] for i in cluster_data]
#np.random.shuffle(cluster_list)
clusterable = [i[1:] for i in cluster_list ]
clusterable = preprocessing.scale(clusterable)
scaled_dic = {}

ind = 0
for i in cluster_list:
    num = i[0]
    scaled_fts = clusterable[ind]
    scaled_dic[num] = scaled_fts
    ind+=1
    
cluster_1_fts = []
cluster_2_fts = []
for i in hierarchy_1:
    cluster_1_fts.append(scaled_dic[i])
for i in hierarchy_2:
    cluster_2_fts.append(scaled_dic[i])

mean1 = mean([i[0] for i in cluster_1_fts])
diff1 = mean([i[1] for i in cluster_1_fts])
var1 = mean([i[2] for i in cluster_1_fts])
min1 = mean([i[3] for i in cluster_1_fts])
mean2 = mean([i[0] for i in cluster_2_fts])
diff2 = mean([i[1] for i in cluster_2_fts])
var2 = mean([i[2] for i in cluster_2_fts])
min2 = mean([i[3] for i in cluster_2_fts])

coords_1 = [mean1,diff1,var1,min1]
coords_2 = [mean2,diff2,var2,min2]