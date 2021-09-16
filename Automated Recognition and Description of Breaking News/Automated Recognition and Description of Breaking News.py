"""
    Author: Pieter Barkema
    Title: Breaking news identification and description
    Functionality:
        Cluster the Echobox data set into events
        Preprocess the likely to be anomalous articles
        Extract useful information and report on it \
        # Model selection for DBSCAN clustering
"""

import pandas as pd
import csv
import numpy as np
import spacy
import os
from os import path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn import metrics
from collections import Counter 
from datetime import datetime
import random
realdir = path.dirname(path.realpath(__file__))

# Load the data ANSI encoded
data = pd.read_csv('20190710 - DS Challenge - Breaking News Articles Detection - DSChallengeArticleId.csv',encoding = 'ANSI')

# Threshold of 8 unique sources
source_thres = 8
# Threshold of 6 hours before and after
time_thres = 32400
# Model parameters for conservativity and minimum amount of samples for cluster creation
conservativity_eps = 0.5
min_samples = 8

#  Returns the labeled data
labeled_data = predict_news_labels(data, source_thres, time_thres, conservativity_eps, min_samples)  

def predict_news_labels(data, source_thres, time_thres, conservativity_eps, min_samples):
    """ 
        Predict EventId based on the Echobox challenge data
        Parameters:
            data = pandas DataFrame of the Echobox challenge data
            conservativity_eps = DBSCAN eps parameter
            min_samples = DBSCAN min_samples parameter
    """
    result_path = realdir + r'\DSchallenge_echobox_labeled'
    news_path = realdir +  r'\DSchallenge_echobox_breaking_news'
    
    # Preprocessing
    insert_index = 5
    data.insert(insert_index, 'Relative time', 0)
    data.insert(insert_index, 'PredictedId', 0)
    data['Relative time'] = data['ArticlePublishedTime'] - min(data['ArticlePublishedTime'])
    
    # tfidf feature extraction
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_title = vectorizer.fit_transform(data['ArticleTitle'])
    
    # Clustering
    #model = DBSCAN(eps=0.5, min_samples=12, metric='cosine').fit(tfidf_title)
    model = DBSCAN(eps=conservativity_eps, min_samples=min_samples, metric='cosine').fit(tfidf_title)
    EventIds = model.fit_predict(tfidf_title)
    data['PredictedId'] = EventIds
    
    # Postprocessing
    data = postprocessing(data, source_thres, time_thres)
    
    # Extract data from clusters to create breaking news information
    extracted_data = data_extract(data)
    
    # Extract keywords per event
    keywords = []
    for c in range(max(data['EventId'])+1):
        # Concatenate all article descriptions and transformed into tf idf vectors
        descriptions = [" ".join(list(data[data['EventId'] == c]['ArticleDescription']))]  
        descr_vector = vectorizer.transform(descriptions)
        # Extract the indexes of the most important features
        tfidf_sorting = np.argsort(descr_vector.toarray()).flatten()[::-1]
        features = vectorizer.get_feature_names()
        
        # Extract the top n tf-idf features, to become the keywords
        n = 6
        top_n = []
        for i in tfidf_sorting[:n]:
            top_n.append(features[i])
        keywords.append(" ".join(top_n))
    
    # Name all columns 
    extracted_data["Keywords"] = keywords
    extracted_data.columns=["EventId","Day", "Time/Timestamp", "Location", "Title","Article count","Keywords"]
    # Drop the duplicates from breaking news based on same keywords
    extracted_data.drop_duplicates(subset = "Keywords", keep = 'first', inplace = True)
    # Drop unnecessary columns
    data = data.drop(['Relative time','PredictedId'], axis = 1)
    # Removing empty clusters
    extracted_data = extracted_data.loc[extracted_data['Article count']!=0]
    # Create two csv files containing the breaking news information and a csv with resulted predictions
    extracted_data.to_csv(news_path + ".csv", index=False, encoding = 'ANSI')
    data.to_csv(result_path + ".csv", index=False, encoding = 'ANSI')
    return data

def data_extract(data):
    """ 
        Extract relevant data from the clusters of articles to represent a breaking news article.
        Parameters:
            data = pandas DataFrame of the Echobox challenge data
    """
    nlp = spacy.load("en_core_web_sm")
    event_info = []
    
    # Extract information per cluster
    for c in range(max(data['EventId'])+1):
        
        # Breaking news format [EventId, date, time, location, title, keywords]
        # If data not found, provide empty data.
        date = ""
        location = ""
        title = ""
        first_locs = []
        first_dates =[]
        first_times = []
        
        # Preferably use the description, but title if description == nan
        data['ArticleDescription'] = data['ArticleDescription'].fillna(data['ArticleTitle'])
        
        # Return the first occurence of time, date and geographical entities
        for text_data in data[data['EventId'] == c]['ArticleDescription']:   
            entities = nlp(text_data).ents
            for word in entities:
                if word.label_ == 'TIME':
                    first_times.append(word.text)
                    break
            for word in entities:
                if word.label_ == 'DATE':
                    first_dates.append(word.text)
                    break
            for word in entities:
                if word.label_ == 'GPE':
                    first_locs.append(word.text)
                    break
        # Count articles per cluster
        article_count = len(data[data['EventId'] == c]['ArticleId'])
                
        # For date, location and time the most common first occurence was used
        if len(Counter(first_dates)) != 0: date = Counter(first_dates).most_common(1)[0][0]
        if len(Counter(first_locs)) != 0: location = Counter(first_locs).most_common(1)[0][0]
        
        if len(Counter(first_times)) != 0: 
            t = Counter(first_times).most_common(1)[0][0]
        # Alternatively, use median timestamp of the cluster's publish times
        else: 
            t = np.median(data[data['EventId'] == c]['ArticlePublishedTime'])
            if t >0:
                t = datetime.fromtimestamp(t).strftime("%d-%b")
        
        # Naively use a random title to represent the breaking news article
        titles = list(data[data['EventId'] == c]['ArticleTitle'])
        if len(titles) !=0:
            title = titles[random.randint(0,len(titles)-1)]
        
        event_info.append([c,date,t,location, title, article_count])
    event_info = pd.DataFrame(event_info)

    return event_info

def postprocessing(labeled_data, source_thres, time_thres):
      """
        Use additional information to postprocess the labelling.
        Parameters:
            labeled_data = The labeled version of the Echobox challenge data set.
            source_thres = Set minimum amount of sources as breaking news threshold.
            time_thres = set minimum time deviation as breaking news threshold.
      """
      # Label articles further away than time_thres from the median as noise
      n_clusters = max(labeled_data['PredictedId'])
      for c in range(n_clusters):
          cluster_rows = labeled_data.loc[labeled_data['PredictedId']==c]
          timestamps = cluster_rows['Relative time']
          time_median = np.median(timestamps)
          IDs = cluster_rows[abs(cluster_rows['Relative time']-time_median) > time_thres]['ArticleId']
          for ID in IDs: 
              labeled_data.loc[labeled_data['ArticleId'] == ID,'PredictedId'] = -1
      
      # Label clusters with less than source_thres sources as noise
      for c in range(n_clusters):
          cluster_rows = labeled_data.loc[labeled_data['PredictedId']==c]
          sources = set([r.split("://")[1].split("/")[0] for r in cluster_rows['ArticleURL']])  
          if len(sources)<source_thres:
              labeled_data['PredictedId'] = labeled_data['PredictedId'].replace(c,-1)
      labeled_data['EventId'] = labeled_data['PredictedId']
      return labeled_data
       
      
###############################################
      # Only used for model selection of DBSCAN
###############################################
      
#vectorizer = TfidfVectorizer(stop_words='english')
#tfidf_title = vectorizer.fit_transform(data['ArticleTitle'])
## Model selection
#cluster_data = tfidf_title
#from sklearn.cluster import DBSCAN
#from sklearn import metrics
#import random
#title_results_dict = {}
#descr_results_dict = {}
#db_results = {}
#sel_eps = [0.3,0.4, 0.5, 0.6, 0.7,0.8,0.9]
#sel_ms = [5,7,9,11,15,20]#[3,5,7,9,11,15]
#
## Create a DBSCAN model for each combination of eps and min_samples
#for ep in sel_eps:
#    for ms in sel_ms:
#        print("\n", ep)
#        db = DBSCAN(eps=ep, min_samples=ms, metric='cosine').fit(cluster_data)
#        clusters = db.fit_predict(cluster_data)
#        labels = db.labels_
#        cluster_n = max(clusters)+1
#        
#        # Create a dictionary with key =  cluster number and value = list of article titles in cluster
#        cluster_members = {}
#        for label, article in zip(clusters,list(data['ArticleTitle'])):
#            if label not in cluster_members:
#                cluster_members[label] = [article]
#            else:
#                cluster_members[label].append(article)
#        
#        # Average silhouette score
#        
#        try: sill_score = metrics.silhouette_score(cluster_data, labels)
#        except ValueError: sill_score = 0
#        # Amount of noise articles
#        noise_n = len(cluster_members[-1])
#        
#        # Create a dictionary for each eps and ms with the corresponding results
#        db_results[str(ep) + "_" + str(ms)] = [sill_score, noise_n, cluster_n]
