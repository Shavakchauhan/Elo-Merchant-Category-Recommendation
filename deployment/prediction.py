import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import gc
import lightgbm as lgb
import json
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import gensim
from gensim.models import Word2Vec
from tqdm import tqdm
import datetime
import pickle
import random

train = pd.read_csv('feature_engineered_train.csv')
test = pd.read_csv('feature_engineered_test.csv')
train.drop(['Unnamed: 0'],axis=1,inplace=True)
test.drop(['Unnamed: 0'],axis=1,inplace=True)

# Here we are already predicting the score for all the cards variable so at deploy time it does not need to calculated
def prediction_of_loyalty_score_for_single_value_1(X,train=train,test=test):
  # train,test = load_data() we cannot use load data because it takes to much time to execute approximately half hour
  features_1 = [c for c in train.columns.values if c not in ['target','outliers']]
  features = [c for c in train.columns.values if c not in ['card_id','target','outliers']]
  train = pd.concat([train[features_1],test[features_1]],axis=0)
  train = train.loc[train['card_id'].isin(X)]
  with open('best_model_lightgbm.sav', 'rb') as pickle_file:
    model = pickle.load(pickle_file)
  prediction_lgb = model.predict(train[features] , num_iteration=model.best_iteration,predict_disable_shape_check=True) 
  predcition_target = pd.DataFrame()
  predcition_target['card_id'] = train['card_id'] 
  predcition_target['predicted_target'] = prediction_lgb
  predcition_target.set_index('card_id',inplace=True)
  return predcition_target





card_ids = list(set(train['card_id']).union(set(test['card_id'])))
predictions_df = prediction_of_loyalty_score_for_single_value_1(card_ids)
predictions_df.to_csv('predicted_targets.csv')