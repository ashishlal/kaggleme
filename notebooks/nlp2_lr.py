# Based on Bojan's -> https://www.kaggle.com/tunguz/more-effective-ridge-lgbm-script-lb-44944
#
import gc
import time
import numpy as np
import pandas as pd

from joblib import Parallel, delayed

from scipy.sparse import csr_matrix, hstack

from sklearn.linear_model import Ridge
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import LabelBinarizer, MinMaxScaler, Normalizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import SGDRegressor
from textblob import TextBlob
import lightgbm as lgb
import os, psutil
from multiprocessing import Pool
from sklearn.metrics.pairwise import cosine_similarity, pairwise_distances
from scipy.spatial.distance import pdist, squareform
from collections import Counter
import re
import lzma
import Levenshtein
from numba import jit

NUM_BRANDS = 4000
NUM_CATEGORIES = 1000
NAME_MIN_DF = 10
MAX_FEATURES_ITEM_DESCRIPTION = 2 ** 14
NUM_PARTITIONS = 12 #number of partitions to split dataframe
NUM_CORES = 8 #number of cores on your machine

###################################################################################
import random, copy, struct
from hashlib import sha1

# The size of a hash value in number of bytes
hashvalue_byte_size = len(bytes(np.int64(42).data))

# http://en.wikipedia.org/wiki/Mersenne_prime
_mersenne_prime = (1 << 61) - 1
_max_hash = (1 << 32) - 1
_hash_range = (1 << 32)

class MinHash(object):
    '''MinHash is a probabilistic data structure for computing 
    `Jaccard similarity`_ between sets.
 
    Args:
        num_perm (int, optional): Number of random permutation functions.
            It will be ignored if `hashvalues` is not None.
        seed (int, optional): The random seed controls the set of random 
            permutation functions generated for this MinHash.
        hashobj (optional): The hash function used by this MinHash. 
            It must implements
            the `digest()` method similar to hashlib_ hash functions, such
            as `hashlib.sha1`.
        hashvalues (`numpy.array` or `list`, optional): The hash values is 
            the internal state of the MinHash. It can be specified for faster 
            initialization using the existing state from another MinHash. 
        permutations (optional): The permutation function parameters. This argument
            can be specified for faster initialization using the existing
            state from another MinHash.
    
    Note:
        To save memory usage, consider using :class:`datasketch.LeanMinHash`.
        
    Note:
        Since version 1.1.1, MinHash will only support serialization using 
        `pickle`_. ``serialize`` and ``deserialize`` methods are removed, 
        and are supported in :class:`datasketch.LeanMinHash` instead. 
        MinHash serialized before version 1.1.1 cannot be deserialized properly 
        in newer versions (`need to migrate? <https://github.com/ekzhu/datasketch/issues/18>`_). 
    Note:
        Since version 1.1.3, MinHash uses Numpy's random number generator 
        instead of Python's built-in random package. This change makes the 
        hash values consistent across different Python versions.
        The side-effect is that now MinHash created before version 1.1.3 won't
        work (i.e., ``jaccard``, ``merge`` and ``union``)
        with those created after. 
    .. _`Jaccard similarity`: https://en.wikipedia.org/wiki/Jaccard_index
    .. _hashlib: https://docs.python.org/3.5/library/hashlib.html
    .. _`pickle`: https://docs.python.org/3/library/pickle.html
    '''

    def __init__(self, num_perm=128, seed=1, hashobj=sha1,
            hashvalues=None, permutations=None):
        if hashvalues is not None:
            num_perm = len(hashvalues)
        if num_perm > _hash_range:
            # Because 1) we don't want the size to be too large, and
            # 2) we are using 4 bytes to store the size value
            raise ValueError("Cannot have more than %d number of\
                    permutation functions" % _hash_range)
        self.seed = seed
        self.hashobj = hashobj
        # Initialize hash values
        if hashvalues is not None:
            self.hashvalues = self._parse_hashvalues(hashvalues)
        else:
            self.hashvalues = self._init_hashvalues(num_perm)
        # Initalize permutation function parameters
        if permutations is not None:
            self.permutations = permutations
        else:
            generator = np.random.RandomState(self.seed)
            # Create parameters for a random bijective permutation function
            # that maps a 32-bit hash value to another 32-bit hash value.
            # http://en.wikipedia.org/wiki/Universal_hashing
            self.permutations = np.array([(generator.randint(1, _mersenne_prime, dtype=np.uint64),
                                           generator.randint(0, _mersenne_prime, dtype=np.uint64))
                                          for _ in range(num_perm)], dtype=np.uint64).T
        if len(self) != len(self.permutations[0]):
            raise ValueError("Numbers of hash values and permutations mismatch")

    def _init_hashvalues(self, num_perm):
        return np.ones(num_perm, dtype=np.uint64)*_max_hash

    def _parse_hashvalues(self, hashvalues):
        return np.array(hashvalues, dtype=np.uint64)
    @jit
    def update(self, b):
        '''Update this MinHash with a new value.
        
        Args:
            b (bytes): The value of type `bytes`.
            
        Example:
            To update with a new string value:
            
            .. code-block:: python
                minhash.update("new value".encode('utf-8'))
        '''
        hv = struct.unpack('<I', self.hashobj(b).digest()[:4])[0]
        a, b = self.permutations
        phv = np.bitwise_and((a * hv + b) % _mersenne_prime, np.uint64(_max_hash))
        self.hashvalues = np.minimum(phv, self.hashvalues)
    @jit
    def jaccard(self, other):
        '''Estimate the `Jaccard similarity`_ (resemblance) between the sets
        represented by this MinHash and the other.
        
        Args:
            other (datasketch.MinHash): The other MinHash.
            
        Returns:
            float: The Jaccard similarity, which is between 0.0 and 1.0.
        '''
        if other.seed != self.seed:
            raise ValueError("Cannot compute Jaccard given MinHash with\
                    different seeds")
        if len(self) != len(other):
            raise ValueError("Cannot compute Jaccard given MinHash with\
                    different numbers of permutation functions")
        return np.float(np.count_nonzero(self.hashvalues==other.hashvalues)) /\
                np.float(len(self))
    @jit
    def count(self):
        '''Estimate the cardinality count based on the technique described in
        `this paper <http://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=365694>`_.
        
        Returns:
            int: The estimated cardinality of the set represented by this MinHash.
        '''
        k = len(self)
        return np.float(k) / np.sum(self.hashvalues / np.float(_max_hash)) - 1.0
    @jit
    def merge(self, other):
        '''Merge the other MinHash with this one, making this one the union
        of both.
        
        Args:
            other (datasketch.MinHash): The other MinHash.
        '''
        if other.seed != self.seed:
            raise ValueError("Cannot merge MinHash with\
                    different seeds")
        if len(self) != len(other):
            raise ValueError("Cannot merge MinHash with\
                    different numbers of permutation functions")
        self.hashvalues = np.minimum(other.hashvalues, self.hashvalues)
    @jit
    def digest(self):
        '''Export the hash values, which is the internal state of the
        MinHash.
        
        Returns:
            numpy.array: The hash values which is a Numpy array.
        '''
        return copy.copy(self.hashvalues)
    @jit
    def is_empty(self):
        '''
        Returns: 
            bool: If the current MinHash is empty - at the state of just
                initialized.
        '''
        if np.any(self.hashvalues != _max_hash):
            return False
        return True
    @jit
    def clear(self):
        '''
        Clear the current state of the MinHash.
        All hash values are reset.
        '''
        self.hashvalues = self._init_hashvalues(len(self))

    @jit
    def copy(self):
        '''
        Returns:
            datasketch.MinHash: A copy of this MinHash by exporting its
                state.
        '''
        return MinHash(seed=self.seed, hashvalues=self.digest(),
                permutations=self.permutations)

    def __len__(self):
        '''
        Returns:
            int: The number of hash values.
        '''
        return len(self.hashvalues)

    def __eq__(self, other):
        '''
        Returns:
            bool: If their seeds and hash values are both equal then two
                are equivalent.
        '''
        return self.seed == other.seed and \
                np.array_equal(self.hashvalues, other.hashvalues)
                
    @classmethod
    @jit
    def union(cls, *mhs):
        '''Create a MinHash which is the union of the MinHash objects passed as arguments.
        Args:
            *mhs: The MinHash objects to be united. The argument list length is variable,
                but must be at least 2.
        
        Returns:
            datasketch.MinHash: A new union MinHash.
        '''
        if len(mhs) < 2:
            raise ValueError("Cannot union less than 2 MinHash")
        num_perm = len(mhs[0])
        seed = mhs[0].seed
        if any((seed != m.seed or num_perm != len(m)) for m in mhs):
            raise ValueError("The unioning MinHash must have the\
                    same seed and number of permutation functions")
        hashvalues = np.minimum.reduce([m.hashvalues for m in mhs])
        permutations = mhs[0].permutations
        return cls(num_perm=num_perm, seed=seed, hashvalues=hashvalues,
                permutations=permutations)
###################################################################################

def rmsle(y, y0):
    assert len(y) == len(y0)
    return np.sqrt(np.mean(np.power(np.log1p(y)-np.log1p(y0), 2)))

def split_cat(text):
    try: return text.split("/")
    except: return ("No Label", "No Label", "No Label")

def handle_missing_inplace(dataset):
    dataset['category_name'].fillna(value='missing', inplace=True)
    dataset['brand_name'].fillna(value='missing', inplace=True)
    dataset['item_description'].fillna(value='missing', inplace=True)

def cutting(dataset):
    pop_brand = dataset['brand_name'].value_counts().loc[lambda x: x.index != 'missing'].index[:NUM_BRANDS]
    dataset.loc[~dataset['brand_name'].isin(pop_brand), 'brand_name'] = 'missing'
    pop_category = dataset['category_name'].value_counts().loc[lambda x: x.index != 'missing'].index[:NUM_BRANDS]
    dataset.loc[~dataset['category_name'].isin(pop_category), 'category_name'] = 'missing'

def to_categorical(dataset):
    dataset['category_name'] = dataset['category_name'].astype('category')
    dataset['brand_name'] = dataset['brand_name'].astype('category')
    dataset['item_condition_id'] = dataset['item_condition_id'].astype('category')

def print_memory_usage():
    print('cpu: {}'.format(psutil.cpu_percent()))
    print('consuming {:.2f}GB RAM'.format(
    	   psutil.Process(os.getpid()).memory_info().rss / 1073741824),
    	  flush=True)


def _sigmoid(score):
    p = 1. / (1. + np.exp(-score))
    return p


def _logit(p):
    return np.log(p/(1.-p))


def _softmax(score):
    score = np.asarray(score, dtype=float)
    score = np.exp(score - np.max(score))
    score /= np.sum(score, axis=1)[:,np.newaxis]
    return score


def _cast_proba_predict(proba):
    N = proba.shape[1]
    w = np.arange(1,N+1)
    pred = proba * w[np.newaxis,:]
    pred = np.sum(pred, axis=1)
    return pred


def _one_hot_label(label, n_classes):
    num = label.shape[0]
    tmp = np.zeros((num, n_classes), dtype=int)
    tmp[np.arange(num),label.astype(int)] = 1
    return tmp


def _majority_voting(x, weight=None):
    ## apply weight
    if weight is not None:
    	assert len(weight) == len(x)
    	x = np.repeat(x, weight)
    c = Counter(x)
    value, count = c.most_common()[0]
    return value


def _voter(x, weight=None):
    idx = np.isfinite(x)
    if sum(idx) == 0:
    	value = config.MISSING_VALUE_NUMERIC
    else:
    	if weight is not None:
    		value = _majority_voting(x[idx], weight[idx])
    	else:
    		value = _majority_voting(x[idx])
    return value


def _array_majority_voting(X, weight=None):
    y = np.apply_along_axis(_voter, axis=1, arr=X, weight=weight)
    return y


def _mean(x):
    idx = np.isfinite(x)
    if sum(idx) == 0:
    	value = float(config.MISSING_VALUE_NUMERIC) # cast it to float to accommodate the np.mean
    else:
    	value = np.mean(x[idx]) # this is float!
    return value


def _array_mean(X):
    y = np.apply_along_axis(_mean, axis=1, arr=X)
    return y


def _corr(x, y_train):
    if _dim(x) == 1:
    	corr = pearsonr(x.flatten(), y_train)[0]
    	if str(corr) == "nan":
    		corr = 0.
    else:
    	corr = 1.
    return corr


def _dim(x):
    d = 1 if len(x.shape) == 1 else x.shape[1]
    return d

@jit
def _entropy(proba):
    entropy = -np.sum(proba*np.log(proba))
    return entropy

@jit
def _try_divide(x, y, val=0.0):
    """try to divide two numbers"""
    if y != 0.0:
    	val = float(x) / y
    return val

@jit
def _jaccard_coef(A, B):
    if not isinstance(A, set):
        A = set(A)
    if not isinstance(B, set):
        B = set(B)
    return _try_divide(float(len(A.intersection(B))), len(A.union(B)))

@jit
def _dice_dist(A, B):
    if not isinstance(A, set):
        A = set(A)
    if not isinstance(B, set):
        B = set(B)
    return _try_divide(2.*float(len(A.intersection(B))), (len(A) + len(B)))

@jit    
def entropy(obs, token_pattern=' '):
    obs_tokens = obs.split(token_pattern)
    counter = Counter(obs_tokens)
    count = np.asarray(list(counter.values()))
    proba = count/np.sum(count)
    # del obs_tokens
    return _entropy(proba)
        
def digit_count(obs):
    return len(re.findall(r"\d", obs))

def digit_ratio(obs, token_pattern = ' '):
    obs_tokens = obs.split(token_pattern)
    return _try_divide(len(re.findall(r"\d", obs)), len(obs_tokens))

def emoji_count(obs):
    return len(re.findall(r'[^\w\s,]', obs))

def emoji_ratio(obs, token_pattern = ' '):
    obs_tokens = obs.split(token_pattern)
    return _try_divide(len(re.findall(r'[^\w\s,]', obs)), len(obs_tokens))

@jit
def _unigrams(words):
    """
    	Input: a list of words, e.g., ["I", "am", "Denny"]
    	Output: a list of unigram
    """
    assert type(words) == list
    return words

@jit
def _bigrams(words, join_string, skip=0):
    """
       Input: a list of words, e.g., ["I", "am", "Denny"]
       Output: a list of bigram, e.g., ["I_am", "am_Denny"]
       I use _ as join_string for this example.
    """
    assert type(words) == list
    L = len(words)
    if L > 1:
    	lst = []
    	for i in range(L-1):
    		for k in range(1,skip+2):
    			if i+k < L:
    				lst.append( join_string.join([words[i], words[i+k]]) )
    else:
    	# set it as unigram
    	lst = _unigrams(words)
    return lst


def _trigrams(words, join_string, skip=0):
    """
       Input: a list of words, e.g., ["I", "am", "Denny"]
       Output: a list of trigram, e.g., ["I_am_Denny"]
       I use _ as join_string for this example.
    """
    assert type(words) == list
    L = len(words)
    if L > 2:
    	lst = []
    	for i in range(L-2):
    		for k1 in range(1,skip+2):
    			for k2 in range(1,skip+2):
    				if i+k1 < L and i+k1+k2 < L:
    					lst.append( join_string.join([words[i], words[i+k1], words[i+k1+k2]]) )
    else:
    	# set it as bigram
    	lst = _bigrams(words, join_string, skip)
    return lst

def UniqueCount_Ngram(obs, count, token_pattern=' '):
    obs_tokens = obs.lower().split(token_pattern)
    obs_ngrams = _ngrams(obs_tokens, count)
    l = len(set(obs_ngrams))
    del obs_tokens
    del obs_ngrams
    return l

def UniqueRatio_Ngram(obs, count, token_pattern=' '):
    obs_tokens = obs.lower().split(token_pattern)
    obs_ngrams = _ngrams(obs_tokens, count)
    r = _try_divide(len(set(obs_ngrams)), len(obs_ngrams))
    del obs_tokens
    del obs_ngrams
    return r

def _ngrams(words, ngram, join_string=" "):
    """wrapper for ngram"""
    if ngram == 1:
    	return _unigrams(words)
    elif ngram == 2:
    	return _bigrams(words, join_string)
    elif ngram == 3:
    	return _trigrams(words, join_string)
    elif ngram == 4:
    	return _fourgrams(words, join_string)
    elif ngram == 12:
    	unigram = _unigrams(words)
    	bigram = [x for x in _bigrams(words, join_string) if len(x.split(join_string)) == 2]
    	return unigram + bigram
    elif ngram == 123:
    	unigram = _unigrams(words)
    	bigram = [x for x in _bigrams(words, join_string) if len(x.split(join_string)) == 2]
    	trigram = [x for x in _trigrams(words, join_string) if len(x.split(join_string)) == 3]
    	return unigram + bigram + trigram
    	
def parallelize_dataframe(df, func):
    df_split = np.array_split(df, NUM_PARTITIONS)
    pool = Pool(NUM_CORES)
    df = pd.concat(pool.map(func, df_split))
    pool.close()
    pool.join()
    return df

def get_sentiment_score(df):
    df['sentiment_score'] = df['item_description'].map(lambda x: TextBlob(x).sentiment.polarity)
    return df

# def main():
start_time = time.time()

train = pd.read_table('../input/train.tsv', engine='c')
test = pd.read_table('../input/test.tsv', engine='c')
print('[{}] Finished to load data'.format(time.time() - start_time))
print('Train shape: ', train.shape)
print('Test shape: ', test.shape)

nrow_test = test.shape[0]

test_id = test['test_id'].values
submission: pd.DataFrame = test[['test_id']]

if nrow_test < 700000:
    test = pd.concat([test,test,test,test,test])
    print('Test shape ', test.shape)


nrow_train = train.shape[0]
y = np.log1p(train["price"])
del train['price']
merge: pd.DataFrame = pd.concat([train, test])

train_cols = set(train.columns)
del train
del test
gc.collect()


handle_missing_inplace(merge)
print('[{}] Handle missing completed.'.format(time.time() - start_time))

def get_doclen_name(df):
    df['name_doclen'] = df['name'].map(lambda x: len(str(x).lower().split(' ')))
    return df

def get_doclen_itemdesc(df):
    df['item_description_doclen'] = df['item_description'].map(lambda x: len(str(x).lower().split(' ')))
    return df

def get_doclen_brand_name(df):
    df['brand_name_doclen'] = df['brand_name'].map(lambda x: len(str(x).lower().split(' ')))
    return df

def get_entropy_name(df):
    df['name_entropy'] = df['name'].map(lambda x: entropy(str(x).lower(), ' '))
    return df

def get_entropy_itemdesc(df):
    df['item_description_entropy'] = \
    	df['item_description'].map(lambda x: entropy(str(x).lower(), ' '))
    return df

def get_entropy_brand_name(df):
    df['brand_name_entropy'] = \
    	df['brand_name'].map(lambda x: entropy(str(x).lower(), ' '))
    return df

def get_digit_count_name(df):
    df['name_dc'] = df['name'].map(lambda x: digit_count(str(x).lower()))
    return df

def get_digit_count_itemdesc(df):
    df['item_description_dc'] = \
    	df['item_description'].map(lambda x: digit_count(str(x).lower()))
    return df

def get_digit_count_brand_name(df):
    df['brand_name_dc'] = \
    	df['brand_name'].map(lambda x: digit_count(str(x).lower()))
    return df

def get_digit_ratio_name(df):
    df['name_dr'] = df['name'].map(lambda x: digit_ratio(str(x).lower()))
    return df

def get_digit_ratio_itemdesc(df):
    df['item_description_dr'] = \
    	df['item_description'].map(lambda x: digit_ratio(str(x).lower()))
    return df

def get_digit_ratio_brand_name(df):
    df['brand_name_dr'] = \
    	df['brand_name'].map(lambda x: digit_ratio(str(x).lower()))
    return df

def get_emoji_count_name(df):
    df['name_ec'] = df['name'].map(lambda x: emoji_count(str(x).lower()))
    return df

def get_emoji_count_itemdesc(df):
    df['item_description_ec'] = \
    	df['item_description'].map(lambda x: emoji_count(str(x).lower()))
    return df

def get_emoji_count_brand_name(df):
    df['brand_name_ec'] = \
    	df['brand_name'].map(lambda x: emoji_count(str(x).lower()))
    return df
        
def get_emoji_ratio_name(df):
    df['name_er'] = df['name'].map(lambda x: emoji_ratio(str(x).lower()))
    return df

def get_emoji_ratio_itemdesc(df):
    df['item_description_er'] = \
    	df['item_description'].map(lambda x: emoji_ratio(str(x).lower()))
    return df

def get_emoji_ratio_brand_name(df):
    df['brand_name_er'] = \
    	df['brand_name'].map(lambda x: emoji_ratio(str(x).lower()))
    return df

cols1 = set(merge.columns)
cols = []
obs_fields = ['name', 'brand_name', 'item_description']
merge = parallelize_dataframe(merge, get_doclen_name)
merge = parallelize_dataframe(merge, get_doclen_itemdesc)
merge = parallelize_dataframe(merge, get_doclen_brand_name)

merge = parallelize_dataframe(merge, get_entropy_name)
merge = parallelize_dataframe(merge, get_entropy_itemdesc)
merge = parallelize_dataframe(merge, get_entropy_brand_name)

merge = parallelize_dataframe(merge, get_digit_count_name)
merge = parallelize_dataframe(merge, get_digit_count_itemdesc)
merge = parallelize_dataframe(merge, get_digit_count_brand_name)

merge = parallelize_dataframe(merge, get_digit_ratio_name)
merge = parallelize_dataframe(merge, get_digit_ratio_itemdesc)
merge = parallelize_dataframe(merge, get_digit_ratio_brand_name)

# merge = parallelize_dataframe(merge, get_emoji_count_name)
# merge = parallelize_dataframe(merge, get_emoji_count_itemdesc)
# merge = parallelize_dataframe(merge, get_emoji_count_brand_name)

# merge = parallelize_dataframe(merge, get_emoji_ratio_name)
# merge = parallelize_dataframe(merge, get_emoji_ratio_itemdesc)
# merge = parallelize_dataframe(merge, get_emoji_ratio_brand_name)

print('[{}] Finished basic creation for name, bn, item_desc'.format(time.time() - start_time))

for f in obs_fields:
    counter = Counter(merge[f].values)
    merge[f+'_docfreq'] = merge[f].map(lambda x: counter[x])
    
    cols.append(f+'_doclen')
    cols.append(f+'_docfreq')
    cols.append(f+'_docEntropy')
    cols.append(f+'_digitCount')
    cols.append(f+'_digitRatio')
    # cols.append(f+'_emojiCount')
    # cols.append(f+'_emojiRatio')

f = 'category_name'
def get_category_name_doclen(df):
    df[f+'_doclen'] = df[f].map(lambda x: len(str(x).lower().split('/')))
    return df

merge = parallelize_dataframe(merge, get_category_name_doclen)

counter = Counter(merge[f].values)
merge[f+'_docfreq'] = merge[f].map(lambda x: counter[x])

token_pattern = '/'

def get_category_name_entropy(df):
	df[f+'_docEntropy'] = df[f].map(lambda x: entropy(str(x).lower(),token_pattern))
	return df
merge = parallelize_dataframe(merge, get_category_name_entropy)

def get_category_name_dc(df):
	df[f+'_dc'] = df[f].map(lambda x: digit_count(str(x).lower()))
	return df
merge = parallelize_dataframe(merge, get_category_name_dc)

def get_category_name_dr(df):
	df[f+'_dr'] = df[f].map(lambda x: digit_ratio(str(x).lower(), token_pattern))
	return df
merge = parallelize_dataframe(merge, get_category_name_dr)

def get_category_name_ec(df):
	df[f+'_emojiCount'] = df[f].map(lambda x: emoji_count(str(x).lower()))
	return df
# merge = parallelize_dataframe(merge, get_category_name_ec)

def get_category_name_er(df):
	df[f+'_emojiRatio'] = df[f].map(lambda x: emoji_ratio(str(x).lower()))
	return df
# merge = parallelize_dataframe(merge, get_category_name_er)

cols.append(f+'_doclen')
cols.append(f+'_docfreq')
cols.append(f+'_docEntropy')
cols.append(f+'_digitCount')
cols.append(f+'_digitRatio')
# cols.append(f+'_emojiCount')
# cols.append(f+'_emojiRatio')

print('[{}] Finished basic creation for category_name'.format(time.time() - start_time))

obs_fields = ["name", "item_description"]

# def get_onegram_uc_name(df):
# 	df['name_1_uc'] = df['name'].map(lambda x: UniqueCount_Ngram(str(x), 1))
# 	return df
# merge = parallelize_dataframe(merge, get_onegram_uc_name)

# def get_onegram_uc_item_desc(df):
# 	df['item_desc_1_uc'] = \
# 		df['item_description'].map(lambda x: UniqueCount_Ngram(str(x), 1))
# 	return df
# merge = parallelize_dataframe(merge, get_onegram_uc_item_desc)

# def get_onegram_ur_name(df):
# 	df['name_1_ur'] = df['name'].map(lambda x: UniqueRatio_Ngram(str(x), 1))
# 	return df
# merge = parallelize_dataframe(merge, get_onegram_ur_name)

# def get_onegram_ur_item_desc(df):
# 	df['item_desc_1_ur'] = \
# 		df['item_description'].map(lambda x: UniqueRatio_Ngram(str(x), 1))
# 	return df
# merge = parallelize_dataframe(merge, get_onegram_ur_item_desc)

def get_bigram_uc_name(df):
	df['name_2_uc'] = df['name'].map(lambda x: UniqueCount_Ngram(str(x), 2))
	return df
merge = parallelize_dataframe(merge, get_bigram_uc_name)

def get_bigram_uc_item_desc(df):
	df['item_desc_2_uc'] = \
		df['item_description'].map(lambda x: UniqueCount_Ngram(str(x), 2))
	return df
merge = parallelize_dataframe(merge, get_bigram_uc_item_desc)

def get_bigram_ur_name(df):
	df['name_2_ur'] = df['name'].map(lambda x: UniqueRatio_Ngram(str(x), 2))
	return df
merge = parallelize_dataframe(merge, get_bigram_ur_name)

def get_bigram_ur_item_desc(df):
	df['item_desc_2_ur'] = \
		df['item_description'].map(lambda x: UniqueRatio_Ngram(str(x), 2))
	return df
merge = parallelize_dataframe(merge, get_bigram_ur_item_desc)

# def get_trigram_uc_name(df):
# 	df['name_3_uc'] = df['name'].map(lambda x: UniqueCount_Ngram(str(x), 3))
# 	return df
# merge = parallelize_dataframe(merge, get_trigram_uc_name)

# def get_trigram_uc_item_desc(df):
# 	df['item_desc_3_uc'] = \
# 		df['item_description'].map(lambda x: UniqueCount_Ngram(str(x), 3))
# 	return df
# merge = parallelize_dataframe(merge, get_trigram_uc_item_desc)

# def get_trigram_ur_name(df):
# 	df['name_3_ur'] = df['name'].map(lambda x: UniqueRatio_Ngram(str(x), 3))
# 	return df
# merge = parallelize_dataframe(merge, get_trigram_ur_name)

# def get_trigram_ur_item_desc(df):
# 	df['item_desc_3_ur'] = \
# 		df['item_description'].map(lambda x: UniqueRatio_Ngram(str(x), 3))
# 	return df
# merge = parallelize_dataframe(merge, get_trigram_ur_item_desc)

# print('[{}] Finished ngram count for name, item_desc'.format(time.time() - start_time))


# ngrams = [1,2,3]
# token_pattern =' '
# for f in obs_fields:
# 	for n in ngrams:
# 		cols.append(f+'_{}_uc'.format(n))
# 		cols.append(f+'_{}_ur'.format(n))

# f = 'category_name'
# merge[f+'_{}_uc'.format(n)] = merge[f].map(lambda x: UniqueCount_Ngram(str(x), n, '/'))
# merge[f+'_{}_ur'.format(n)] = merge[f].map(lambda x: UniqueRatio_Ngram(str(x), n, '/'))
# cols.append(f+'_{}_uc'.format(n))
# cols.append(f+'_{}_ur'.format(n))
		
# remove constatnt cols
merge =  merge.loc[:, (merge != merge.iloc[0]).any()]
print(len(cols))
del cols
cols = list(set(merge.columns) - cols1)
print(len(cols))

X_b = merge[cols]

print('[{}] Finished X_basic1'.format(time.time() - start_time))

scaler = MinMaxScaler()
X_b = scaler.fit_transform(X_b)
X_basic = csr_matrix(X_b)
print('basic: ', X_basic.data.nbytes)
print('[{}] Finished X_basic2'.format(time.time() - start_time))
del X_b
for c in cols:
    merge = merge.drop(c, axis=1)
print_memory_usage()

# jaccard and dice
merge['n_id'] = merge['name'].astype('str') + '___' + merge['item_description'].astype('str')

from sklearn.metrics.pairwise import pairwise_distances

def jaccard_skl(text):
    obs, target = text.split('___')
    obs_tokens = obs.split(' ')
    target_tokens = target.split(' ')
    j1 = 1 - pairwise_distances(obs_tokens, target_tokens, metric = "hamming")
    del obs, target, target_tokens, obs_tokens
    return j1

@jit    
def jaccard_1(text):
    obs, target = text.split('___')
    obs_tokens = obs.split(' ')
    target_tokens = target.split(' ')
    obs_ngrams = _ngrams(obs_tokens, 1)
    target_ngrams = _ngrams(target_tokens, 1)
    j1 = _jaccard_coef(obs_ngrams, target_ngrams)
    # del obs, target, target_tokens, obs_tokens, obs_ngrams, target_ngrams
    return j1
    
def jaccard_2(text):
    obs, target = text.split('___')
    obs_tokens = obs.split(' ')
    target_tokens = target.split(' ')
    obs_ngrams = _ngrams(obs_tokens, 2)
    target_ngrams = _ngrams(target_tokens, 2)
    j2 = _jaccard_coef(obs_ngrams, target_ngrams)
    del obs, target, target_tokens, obs_tokens, obs_ngrams, target_ngrams
    return j2

def jaccard_3(text):
    obs, target = text.split('___')
    obs_tokens = obs.split(' ')
    target_tokens = target.split(' ')
    obs_ngrams = _ngrams(obs_tokens, 3)
    target_ngrams = _ngrams(target_tokens, 3)
    j3 = _jaccard_coef(obs_ngrams, target_ngrams)
    del obs, target, target_tokens, obs_tokens, obs_ngrams, target_ngrams
    return j3

@jit
def jaccard_minhash(text):
    obs, target = text.split('___')
    obs_tokens = obs.split(' ')
    target_tokens = target.split(' ')
    m1, m2 = MinHash(), MinHash()
    for d in obs_tokens:
        m1.update(d.encode('utf8'))
    for d in target_tokens:
        m2.update(d.encode('utf8'))
    j = m1.jaccard(m2)
    return j

# def get_j2(df):
    # merge['j2'] = merge['n_id'].map(lambda x: jaccard_2(x))
    # return df
# merge = parallelize_dataframe(merge, get_j2)

def get_j1(df):
    merge['j1'] = merge['n_id'].map(lambda x: jaccard_1(x))
    return df
merge = parallelize_dataframe(merge, get_j1)

X_j = merge[['j1']]
del merge['n_id']
np.min(X_j)
np.max(X_j)
print('[{}] Finished X_j'.format(time.time() - start_time))
print_memory_usage()

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

merge['tag'] = merge['item_description'].map(lambda a: 'BNWT' if any(x in a.lower() for x in abbr['BNWT'])
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
print('[{}] Finished tag'.format(time.time() - start_time))
del abbr
print_memory_usage()

merge['bci'] = merge['brand_name'].astype('str') + ' ' + merge['category_name'].astype('str') + ' ' + \
			merge['item_condition_id'].astype('str')

merge['bc'] = merge['brand_name'].astype('str') + ' ' + merge['category_name'].astype('str')

merge['bcis'] = merge['brand_name'].astype('str') + ' ' \
				+ merge['category_name'].astype('str') + ' ' + \
				merge['item_condition_id'].astype('str') + ' ' + \
				merge['shipping'].astype('str')

merge['bcs'] = merge['brand_name'].astype('str') + ' ' + \
				merge['category_name'].astype('str') + ' ' + \
				merge['shipping'].astype('str')

# merge['bi'] = merge['brand_name'].astype('str') + '_' +   merge['item_condition_id'].astype('str')
	
# merge['ci'] = merge['category_name'].astype('str') + '_' + merge['item_condition_id'].astype('str')

print('[{}] Finished creating bci bc bi ci bcs bcis'.format(time.time() - start_time))
print_memory_usage()


# merge.drop(['bci', 'bc'], axis=1, inplace=True)

# merge = parallelize_dataframe(merge, get_sentiment_score)
# merge['sentiment_score'] = merge['item_description'].map(lambda x: TextBlob(x).sentiment.polarity)

# print('[{}] Finished sentiment score'.format(time.time() - start_time))
# a = merge['sentiment_score'].values
# print(np.min(a))
# print(np.max(a))

# print_memory_usage()
# merge['sentiment'] = merge['sentiment_score'].map(lambda x: 'VPos' if x > 0.5 
													# else 'Pos' if (x <= 0.5) and (x > 0)
													# else 'Neu' if  x == 0 
													# else 'Neg' if (x < 0) and (x >= -0.5)
													# else 'VNeg')

# print('[{}] Finished sentiment'.format(time.time() - start_time))

cutting(merge)
print('[{}] Finished to cut'.format(time.time() - start_time))

to_categorical(merge)
print('[{}] Finished to convert categorical'.format(time.time() - start_time))

# tv = TfidfVectorizer(max_features=2 ** 14,
#                      min_df=NAME_MIN_DF,
# 					 ngram_range=(1, 3),
# 					 stop_words='english')
# X_name1 = tv.fit_transform(merge['name'])
# print('[{}] Finished TFIDF vectorize `name`'.format(time.time() - start_time))
# print(X_name1.shape)
# print(np.min(X_name1))
# print(np.max(X_name1))
# # del merge['item_description']
# print_memory_usage()

cv = CountVectorizer(min_df=NAME_MIN_DF, stop_words='english')
X_name = cv.fit_transform(merge['name'])
norm = Normalizer()
X_name = norm.fit_transform(X_name)
print('[{}] Finished count vectorize `name`'.format(time.time() - start_time))
print(X_name.shape)
print(np.min(X_name))
print(np.max(X_name))
del merge['name']
print_memory_usage()

cv = CountVectorizer()
X_category = cv.fit_transform(merge['category_name'])
norm = Normalizer()
X_category = norm.fit_transform(X_category)
print('[{}] Finished count vectorize `category_name`'.format(time.time() - start_time))
print(X_category.shape)
print(np.min(X_category))
print(np.max(X_category))
del merge['category_name']
gc.collect()
print_memory_usage()

# cv = CountVectorizer()
# X_bci_cv = cv.fit_transform(merge['bci'])
# norm = Normalizer()
# X_bci_cv = norm.fit_transform(X_bci_cv)
# print('[{}] Finished count vectorize `X_bci_cv`'.format(time.time() - start_time))
# print(X_bci_cv.shape)
# print(np.min(X_bci_cv))
# print(np.max(X_bci_cv))
# del merge['bci']
# gc.collect()
# print_memory_usage()


tv = TfidfVectorizer(max_features=MAX_FEATURES_ITEM_DESCRIPTION,
					 ngram_range=(1, 3),
					 stop_words='english')
X_description = tv.fit_transform(merge['item_description'])
print('[{}] Finished TFIDF vectorize `item_description`'.format(time.time() - start_time))
print(X_description.shape)
print(np.min(X_description))
print(np.max(X_description))
del merge['item_description']
print_memory_usage()

# X_cos = cosine_similarity(X_description, dense_output=False)
# X_cos = squareform(pdist(np.asarray(X_description.toarray()), 'cosine'))
# print(X_cos.shape)
# print('[{}] Finished cosine similarity'.format(time.time() - start_time))
# print_memory_usage()

lb = LabelBinarizer(sparse_output=True)
X_brand = lb.fit_transform(merge['brand_name'])
print('[{}] Finished label binarize `brand_name`'.format(time.time() - start_time))
print(X_brand.shape)
del merge['brand_name']
print_memory_usage()

lb = LabelBinarizer(sparse_output=True)
X_bci = lb.fit_transform(merge['bci'])
print('[{}] Finished label binarize `bci`'.format(time.time() - start_time))
print(X_bci.shape)
del merge['bci']
print_memory_usage()

# lb = LabelBinarizer(sparse_output=True)
# X_bc = lb.fit_transform(merge['bc'])
# print('[{}] Finished label binarize `bc`'.format(time.time() - start_time))
# print(X_bc.shape)
# del merge['bc']
# print_memory_usage()

lb = LabelBinarizer(sparse_output=True)
X_bcis = lb.fit_transform(merge['bcis'])
print('[{}] Finished label binarize `bcis`'.format(time.time() - start_time))
print(X_bcis.shape)
del merge['bcis']
gc.collect()
print_memory_usage()

lb = LabelBinarizer(sparse_output=True)
X_bcs = lb.fit_transform(merge['bcs'])
print('[{}] Finished label binarize `bcs`'.format(time.time() - start_time))
print(X_bcs.shape)
del merge['bcs']
gc.collect()
print_memory_usage()

X_dummies = csr_matrix(pd.get_dummies(merge[['item_condition_id', 'shipping', 
											'tag']], sparse=True).values)
print('[{}] Finished to get dummies on `item_condition_id` and `shipping`'.format(time.time() - start_time))
print(X_dummies.shape)
print_memory_usage()

del merge
gc.collect()
print_memory_usage()

print('basic: ', X_basic.data.nbytes)
print('bcis: ', X_bcis.data.nbytes)
print('bci: ', X_bci.data.nbytes)
print('dummies: ', X_dummies.data.nbytes)
print('description: ', X_description.data.nbytes)
print('brand: ', X_brand.data.nbytes)
print('category: ', X_category.data.nbytes)
print('name: ', X_name.data.nbytes)
print('name1: ', X_name1.data.nbytes)

sparse_merge = hstack((X_j, X_basic, X_bci, X_bcis, X_dummies, X_description, X_brand, X_category, X_name)).tocsr()
print('[{}] Finished to create sparse merge'.format(time.time() - start_time))

del X_j, X_basic, X_bcis, X_bci, X_bcs, X_dummies, X_description, X_brand, X_category, X_name, X_name1
gc.collect()
print_memory_usage()

X = sparse_merge[:nrow_train]
X_test = sparse_merge[nrow_train:]

print(X.shape)
print_memory_usage()

del sparse_merge
gc.collect()
print_memory_usage()

np.random.seed(0)
train_X, valid_X, train_y, valid_y = train_test_split(X, y, test_size = 0.01, random_state = 0) 
print_memory_usage()
# d_train = lgb.Dataset(X, label=y, max_bin=8192)
d_train = lgb.Dataset(train_X, label=train_y, max_bin=8192)
d_valid = lgb.Dataset(valid_X, label=valid_y, max_bin=8192)
watchlist = [d_train, d_valid]
print_memory_usage()

params = {
	'learning_rate': 0.75,
	'application': 'regression',
	'max_depth': 3,
	'num_leaves': 100,
	'verbosity': -1,
	'metric': 'RMSE',
	'num_threads': 4
}


model = lgb.train(params, train_set=d_train, valid_sets=watchlist,
					num_boost_round=5000,early_stopping_rounds=100,verbose_eval=500) 
print('[{}] Finished to train lgbm'.format(time.time() - start_time))
preds = model.predict(X_test)
print('[{}] Finished to train predict lgbm'.format(time.time() - start_time))
del model, d_train, d_valid
print_memory_usage()

# submission=pd.DataFrame()
# submission['test_id'] = test_id
# submission['price'] = np.expm1(preds)
# submission.to_csv("submission_lgbm_nlp2.csv", index=False)
preds *= 0.6
# print('[{}] Finished submission lgbm'.format(time.time() - start_time))
if nrow_test < 700000:
	preds = preds[:nrow_test]

model = Ridge(solver="saga", fit_intercept=True, random_state=205)
model.fit(X, y)
print('[{}] Finished to train ridge'.format(time.time() - start_time))
preds1 = model.predict(X=X_test)
print('[{}] Finished to predict ridge'.format(time.time() - start_time))
# submission['price'] = np.expm1(preds1)
# submission.to_csv("submission_ridge_nlp2.csv", index=False)
print_memory_usage()
if nrow_test < 700000:
	preds1 = preds1[:nrow_test]
	
preds += 0.4*preds1
submission['price'] = np.expm1(preds)
submission.to_csv("submission_lgbm_ridge_nlp2.csv", index=False)
# if __name__ == '__main__':
# main()
