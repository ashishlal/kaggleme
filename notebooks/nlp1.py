import pandas as pd
import numpy as np
from sklearn.feature_extraction import text
from nltk.stem.porter import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
import dask.dataframe as dd
from datetime import datetime
from csv import DictReader
from math import exp, log, sqrt
from random import random,shuffle
import pickle
import sys
#from ngram import getUnigram
import string
import nltk
from nltk.util import ngrams # function for making ngrams
import re
from sklearn import preprocessing as pe
from sklearn.model_selection import StratifiedKFold
from scipy import sparse as ssp
import lightgbm as lgbm
import time
from scipy.sparse import csr_matrix, hstack
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import Ridge
from textblob import TextBlob

# vectorized error calc
def rmsle(y, y0):
    assert len(y) == len(y0)
    return np.sqrt(np.mean(np.power(np.log1p(y)-np.log1p(y0), 2)))

#looping error calc
def rmsle_loop(y, y_pred):
    assert len(y) == len(y_pred)
    terms_to_sum = [(math.log(y_pred[i] + 1) - math.log(y[i] + 1)) ** 2.0 for i,pred in enumerate(y_pred)]
    return (sum(terms_to_sum) * (1.0/len(y))) ** 0.5

def evalerror(preds, dtrain):
    labels = dtrain.get_label()
    return 'rmsle', rmsle(labels, preds), False

def rmse0(y, y_pred):
    return sqrt(mean_squared_error(y, y_pred))

seed = 1024
np.random.seed(seed)

start_time = time.time()

data = '../data/'
cache = '../cache/'

train = pd.read_csv(data+"train.tsv", sep='\t')
test = pd.read_csv(data+"test.tsv", sep='\t')

abbr = {}
abbr['BNWT'] = ['bnwt', 'brand new with tags']
abbr['NWT'] = ['nwt', 'new with tags']
abbr['BNWOT'] = ['bnwot', 'brand new with out tags', 'brand new without tags']
abbr['NWOT'] = ['nwot', 'new with out tags', 'new without tags']
abbr['BNIP'] = ['bnip', 'brand new in packet', 'brand new in packet']
abbr['NIP'] = ['nip', 'new in packet', 'new in packet']
abbr['BNIB'] = ['bnib', 'brand new in box']
abbr['NIB'] = ['nib', 'new in box']
abbr['MIB'] = ['mib', 'mint in box']
abbr['MWOB'] = ['mwob', 'mint with out box', 'mint without box']
abbr['MIP'] = ['mip', 'mint in packet']
abbr['MWOP'] = ['mwop', 'mint with out packet', 'mint without packet']


price = train['price'].values
train_id = train['train_id'].values
test_id = test['test_id'].values

train_label = train['price']

del train['price']
del train['train_id']
del test['test_id']

df_all = pd.concat((train, test), axis=0, ignore_index=True)

df_all['tag'] = df_all['item_description'].map(lambda a: 'BNWT' if any(x in a.lower() for x in abbr['BNWT'])
                                               else 'NWT' if any(x in a.lower() for x in abbr['NWT'])
                                               else 'BNWOT' if any(x in a.lower() for x in abbr['BNWOT'])
                                               else 'NWOT' if any(x in a.lower() for x in abbr['NWOT'])
                                               else 'BNIP' if any(x in a.lower() for x in abbr['BNIP'])
                                               else 'NIP' if any(x in a.lower() for x in abbr['NIP'])
                                               else 'BNIB' if any(x in a.lower() for x in abbr['BNIB'])
                                               else 'NIB' if any(x in a.lower() for x in abbr['NIB'])
                                               else 'MIB' if any(x in a.lower() for x in abbr['MIB'])
                                               else 'MWOB' if any(x in a.lower() for x in abbr['MWOB'])
                                               else 'MIP' if any(x in a.lower() for x in abbr['MIP'])
                                               else 'MWOP' if any(x in a.lower() for x in abbr['MWOP'])
                                               else 'None')

d = df_all['tag'].value_counts()

df_all['tag_count'] = df_all['tag'].map(lambda x: d[x])


df_all['brand_name'].fillna('UnkB', inplace=True)
df_all['category_name'].fillna('UnkC', inplace=True)

for c in ['brand_name', 'category_name', 'item_condition_id']:
    d = df_all[c].value_counts()
    df_all[c+'_count'] = df_all[c].map(lambda x: d[x])

df_all['bci'] = df_all['brand_name'].astype('str') + '_' + df_all['category_name'].astype('str') + '_' + \
                df_all['item_condition_id'].astype('str')
    
df_all['bc'] = df_all['brand_name'].astype('str') + '_' + df_all['category_name'].astype('str')
    
df_all['bi'] = df_all['brand_name'].astype('str') + '_' +   df_all['item_condition_id'].astype('str')
    
df_all['ci'] = df_all['category_name'].astype('str') + '_' + df_all['item_condition_id'].astype('str')

for c in ['bci', 'bc', 'bi', 'ci']:
    d = df_all[c].value_counts()
    df_all[c+'_count'] = df_all[c].map(lambda x: d[x])

for c in ['brand_name', 'category_name', 'item_condition_id', 'ci', 'bi']:
    lbl = pe.LabelEncoder()
    df_all[c+'_lbl_enc'] = lbl.fit_transform(df_all[c].values)

for c in ['bci', 'bc']:
    df_all = df_all.drop(c, axis=1)

df_all['sentiment_score'] = df_all['item_description'].map(lambda x: TextBlob(x).sentiment.polarity)

df_all['sentiment'] = df_all['sentiment_score'].map(lambda x: 'Pos' if x > 0 else 'Neu' if  x == 0 else 'Neg')

d = df_all['category_name'].value_counts()

lbl = pe.LabelEncoder()
df_all['sentiment'] = lbl.fit_transform(df_all['sentiment'].values)

for c in ['name', 'item_description']:
    df_all[c+'_len'] = df_all[c].map(lambda x: len(str(x)))
    df_all[c+'_words'] = df_all[c].map(lambda x: len(str(x).split(' '))) 

lbl = pe.LabelEncoder()
df_all['tag'] = lbl.fit_transform(df_all['tag'].values)

cv = CountVectorizer(min_df=10)
X_name = cv.fit_transform(df_all['name'])
print('[{}] Finished count vectorize `name`'.format(time.time() - start_time))

cv = CountVectorizer()
X_category = cv.fit_transform(df_all['category_name'])
print('[{}] Finished count vectorize `category_name`'.format(time.time() - start_time))

tv = TfidfVectorizer(max_features=2**16,
                     ngram_range=(1, 3),
                     stop_words='english')
X_description = tv.fit_transform(df_all['item_description'])
print('[{}] Finished TFIDF vectorize `item_description`'.format(time.time() - start_time))

X_num = df_all
for c in ['name', 'category_name', 'brand_name', 'item_description', 'bi','ci', 'item_condition_id']:
    X_num = X_num.drop(c, axis=1)

sparse_merge = hstack((X_num, X_description, X_category, X_name)).tocsr()
print('[{}] Finished to create sparse merge'.format(time.time() - start_time))

X = sparse_merge[:len(df_train)]
X_test = sparse_merge[len(df_train):]

params2 = {
        'learning_rate': 0.75,
        'application': 'regression',
        'max_depth': 4,
        'num_leaves': 100,
        'verbosity': -1,
        'metric': 'RMSE',
    }

y = np.log1p(price)
train_X, valid_X, train_y, valid_y = train_test_split(X, y, test_size = 0.1, random_state = 144) 
d_train = lgbm.Dataset(train_X, label=train_y, max_bin=1024)
d_valid = lgbm.Dataset(valid_X, label=valid_y, max_bin=1024)
watchlist = [d_train, d_valid]

model = lgbm.train(params2, train_set=d_train, num_boost_round=4000, valid_sets=watchlist, \
early_stopping_rounds=50, verbose_eval=100) 
preds_lgbm_1024 = model.predict(X_test)

model = Ridge(solver="sag", fit_intercept=True, random_state=205)
model.fit(X, y)
print('[{}] Finished to train ridge'.format(time.time() - start_time))
preds_ridge = model.predict(X=X_test)
print('[{}] Finished to predict ridge'.format(time.time() - start_time))


preds = (0.6 * preds_lgbm_1024) + (0.4 * preds_ridge)
submission['price'] = np.expm1(preds)
submission.to_csv("submission_lgbm_ridge_nlp2.csv", index=False)



