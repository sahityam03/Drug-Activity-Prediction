
# coding: utf-8

# In[1]:

import numpy as np
import pandas as pd
import scipy.sparse as sp
import re
from numpy.linalg import norm
from collections import Counter, defaultdict
from scipy.sparse import csr_matrix
from pyparsing import anyOpenTag, anyCloseTag
from xml.sax.saxutils import unescape as unescape


# In[3]:

# read in the dataset
with open("data/train.dat", "r") as tr:
    docs2 = tr.readlines()
with open("data/test.dat", "r") as te:
    docs3 = te.readlines()
print len(docs2)

cls = []
docs2_new = []
for x in docs2:
    cls.append(x[0:1])
    docs2_new.append(x[2:])
print len(cls)
print (cls[0])


# In[4]:

docs = docs2_new + docs3
docs2_new1 = []
for i in range(0, len(docs)):
    docs2_new1.append(docs[i].split())
print("done")


# In[5]:

def build_matrix(docs):
    r""" Build sparse matrix from a list of documents, 
    each of which is a list of word/terms in the document.  
    """
    nrows = len(docs)
    idx = {}
    tid = 0
    nnz = 0
    for d in docs:
        nnz += len(set(d))
        for w in d:
            if w not in idx:
                idx[w] = tid
                tid += 1
    ncols = len(idx)
        
    # set up memory
    ind = np.zeros(nnz, dtype=np.int)
    val = np.zeros(nnz, dtype=np.double)
    ptr = np.zeros(nrows+1, dtype=np.int)
    i = 0  # document ID / row counter
    n = 0  # non-zero counter
    # transfer values
    for d in docs:
        cnt = Counter(d)
        keys = list(k for k,_ in cnt.most_common())
        l = len(keys)
        for j,k in enumerate(keys):
            ind[j+n] = idx[k]
            val[j+n] = cnt[k]
        ptr[i+1] = ptr[i] + l
        n += l
        i += 1
            
    mat = csr_matrix((val, ind, ptr), shape=(nrows, ncols), dtype=np.double)
    mat.sort_indices()
    
    return mat


# In[6]:

csrmat1 = build_matrix(docs2_new1)
print("done")


# In[7]:

def csr_info(mat, name="", non_empy=False):
    r""" Print out info about this CSR matrix. If non_empy, 
    report number of non-empty rows and cols as well
    """
    if non_empy:
        print("%s [nrows %d (%d non-empty), ncols %d (%d non-empty), nnz %d]" % (
                name, mat.shape[0], 
                sum(1 if mat.indptr[i+1] > mat.indptr[i] else 0 
                for i in range(mat.shape[0])), 
                mat.shape[1], len(np.unique(mat.indices)), 
                len(mat.data)))
    else:
        print( "%s [nrows %d, ncols %d, nnz %d]" % (name, 
                mat.shape[0], mat.shape[1], len(mat.data)) )


# In[8]:

csr_info(csrmat1)
train1 = csrmat1[:800, :]
test1 = csrmat1[800:, :]
csr_info(train1)
print train1.shape
print test1.shape
csr_info(test1)
#print(train1)


# In[9]:

#truncated svd
from sklearn.decomposition import TruncatedSVD
svd2 = TruncatedSVD(n_components=800, n_iter=15, random_state=35)
svdfit2 = svd2.fit(train1)
train_reduced = svdfit2.transform(train1)
test_reduced = svdfit2.transform(test1)
print("done")
print train_reduced.shape
print test_reduced.shape



# In[82]:

#neural networks
#from sklearn.neural_network import MLPClassifier
#clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
#clf.fit(train_reduced, cls)


# In[10]:

from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(class_weight='balanced', max_depth=21, 
             n_estimators=25, n_jobs=1,
             random_state=42, )
rfc.fit(train_reduced, cls)


# In[15]:

#print rfc.predict(test_reduced)


# In[12]:

test_cls = []
test_cls = rfc.predict(test_reduced)


# In[13]:

test = open("data/format.dat", "w")
for i in range(0, len(test_cls)):
    test.write(test_cls[i])
    test.write('\n')
test.close()


# In[14]:

print("done")


# In[ ]:

#dense matrix creation
#col, row = 100001, 1150;
#densemat = [[0 for x in range(col)] for y in range(row)] 

#for x in range(0, 1150):
#    for y in range(0, len(docs2_new1[x])):
#        temp = int(docs2_new1[x][y])
        #print temp
#        densemat[x][temp] = 1


# In[ ]:

#pca implementation
#from sklearn.decomposition import PCA
#pca = PCA(copy=True, iterated_power='auto', n_components=800, random_state=None,
#  svd_solver='auto', tol=0.0, whiten=False)
#pcafit = pca.fit(densemat)
#dense_reduced = pcafit.transform(densemat)
#print dense_reduced.shape


# In[ ]:

#random forest implementation
#from sklearn.ensemble import RandomForestClassifier
#rfc = RandomForestClassifier(bootstrap=True, class_weight='balanced', criterion='gini',
#            max_depth=1, max_features='auto', max_leaf_nodes=200,
#           min_impurity_split=0,
#           min_samples_leaf=1, min_samples_split=2,
#           min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=1,
#           oob_score=False, random_state=0, verbose=0, warm_start=False)
#rfc.fit(dense_reduced, cls)

