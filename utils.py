
# coding: utf-8

# In[2]:


import os
import numpy as np
import time
import scipy.stats as stats
from scipy.special import factorial
import scipy.stats as stats
import csv
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.stats.multitest as sm
from fisher import pvalue_npy
import fisher
from sklearn import linear_model
from sklearn import metrics
from sklearn.model_selection import train_test_split
import get_reverse_index

# In[4]:


def rev_index(a,b,c,d,rev_threshold):
    num_disease = np.sum(y_train)
    num_control = len(y_train) - num_disease
    rev = rev_threshold
    at = ((a.astype(float)/num_control) >= rev)
    ct = ((c.astype(float)/num_disease) <= (1-rev))

    index0, index1 = np.where(at * ct)
    for i in range(0,len(index0)):
        if index0[i] > index1[i]:
            index0[i],index1[i] = index1[i],index0[i]
    print("reversal percentage = %f" % rev)
    print("number of pairs = %d" % len(index0))
    index = np.stack((index0, index1))
    return index,len(index0)
def fisher_index(pvalue_Bonf2,threshold):
    # find indices of pvalue < fdr_threshold
    length = len(pvalue_Bonf2[0,:])
    # add 1 in triangle matrix to remove duplicated index
    pvalue_matrix = pvalue_Bonf2 + np.triu(np.ones([length,length]))#要知道有一半的矩阵是一样的
    j,k = np.where(pvalue_matrix < threshold)
    result3 = np.array([j,k],dtype = np.uint16)
    print("pvalue_Bonf_threshold = ", threshold)
    print("number of pairs = ",len(j))
    return result3,len(j)##返回的是具体的index
def overlap(ncbi,scRNA):
    count=0
    over_gene=[]
    for gene in ncbi:
        if gene in scRNA:
            count+=1
            over_gene.append(gene)
    return count,over_gene
def pairconvert(data, index):
    sub = np.array(data.iloc[:,index[1,:]]) - np.array(data.iloc[:,index[0,:]]) > 0    
    sub = sub*2-1
    return sub
def pair_2_pair_index(name,drop_dict,feature_num,pair_rsl):
    dele=[]
    for pair_tup in pair_rsl:
        num=pair_tup[0]
        pair_list=pair_tup[1]
        if num == feature_num:
            pair_index=np.array([[pair[0] for pair in pair_list],[pair[1] for pair in pair_list]])
            print(pair_index)
            if drop_dict[name] != None:
                drop_gene=drop_dict[name]
                for gene in drop_gene:
                    for i in range(pair_index.shape[1]):
                        if gene in pair_index[:,i]:
                            dele.append(i)
                new_index=np.delete(pair_index,dele,axis=1)
                         
                return new_index
                            
            else:
                return pair_index
def pair_2_pair_index_inner(pair_list):
    ##this will used in the final module
    pair_index=np.array([[pair[0] for pair in pair_list],[pair[1] for pair in pair_list]])
    return pair_index
def pair_index_2_train_index(pair_index,train_data):
    all_gene=list(train_data.columns)
    raw=pair_index.shape[0]
    col=pair_index.shape[1]
    index=np.array([all_gene.index(i) for i in pair_index.flatten()]).reshape((raw,col))
    return index


def pairconvert_test(data,gene_pair_index):
    sub = np.array(data.iloc[:][gene_pair_index[1,:]]) - np.array(data.iloc[:][gene_pair_index[0,:]]) > 0
    sub = sub*2-1
    return sub
def pair2gene(index):
    result = index.reshape((-1,1))
    result = np.squeeze(result)
    result = np.unique(result)
    return result
def extract(data,index,typ):
    if typ == 'rank':
        extract_data = pairconvert(data,index)#一个增加或者减少的矩阵
    if typ == 'value':
        value_index = pair2gene(index)
        extract_data = data.iloc[:,value_index]#所有涉及pair的基因矩阵
    return extract_data#这里输出的是一个告诉你增加(1)还是减少(-1)的pair
def extract_value(data,index):
    extract_data=data.loc[:,index]
    return extract_data

def train_Lasso(data,label,plot=False,to_file=None):
    # Compute paths
    print("Computing regularization path using the coordinate descent lasso...")
    t1 = time.time()
    model = linear_model.LassoCV(max_iter=1000,cv=5,random_state=5).fit(data, label)
    t_lasso_cv = time.time() - t1

    # This is to avoid division by zero while doing np.log10
    EPSILON = 1e-5

    # Display results
    if plot:
        m_log_alphas = -np.log10(model.alphas_ + EPSILON)

        plt.figure()
        ymin, ymax = 0, 0.6
        plt.plot(m_log_alphas, model.mse_path_, ':')
        plt.plot(m_log_alphas, model.mse_path_.mean(axis=-1), 'k',
                 label='Average across the folds', linewidth=2)
        plt.axvline(-np.log10(model.alpha_ + EPSILON), linestyle='--', color='k',
                    label='alpha: CV estimate')

        plt.legend()

        plt.xlabel('-log(alpha)')
        plt.ylabel('Mean square error')
        plt.title('Mean square error on each fold: coordinate descent '
                  '(train time: %.2fs)' % t_lasso_cv)
        plt.axis('tight')
        plt.ylim(ymin, ymax)
        plt.savefig(to_file,format="pdf")
        plt.show()
    num_coef = len(np.where(model.coef_)[0])
    print("num_lasso_pair = ", num_coef)
    return model, num_coef
##Plot
def plot_roc(labels, predict_prob):
    false_positive_rate,true_positive_rate,thresholds=metrics.roc_curve(labels, predict_prob)
    roc_auc = metrics.auc(false_positive_rate, true_positive_rate)
    plt.title('ROC')
    plt.plot(false_positive_rate, true_positive_rate,'b',             label = 'AUC = %0.4f'% roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0,1],[0,1],'r--')
    plt.ylabel('TPR')
    plt.xlabel('FPR')
    plt.show()
def test_Lasso(data, label, clf):
    predt = clf.predict(data)
    y_scores = np.array(predt)
    y_true = label
    # print "True: ", y_true
    # print "Predict: ", y_scores
    print("Lasso ROC AUC: ", metrics.roc_auc_score(y_true, y_scores))
#     plot_roc(y_true,y_scores)
    return metrics.roc_auc_score(y_true, y_scores)
def test_rank(data,label,index):
    # for pair, the value of pairconvert is 1 for disease, -1 for health
    rankdata = pairconvert(data,index)
    print(rankdata)
    print(rankdata.shape)
    num = len(index[0,:])
    tag = []

    for i in range(0,num):
        tag.append(c[index[0,i],index[1,i]] > d[index[0,i],index[1,i]])##大部分都是增加的
#     print(tag)
    tag = np.array(tag)
    tag = tag*2-1
    print(tag)
    print(y_test1)
    # convert into 1 and 0
    rankdata = rankdata*tag > 0##又要，又要增加的
def testmodel(clf,x_test,y_test,index,thre_type,typ,classifiers,names):
    testrsl = []
    num_pair=len(index[0,:])
    if thre_type == 'fdr':
        testrsl.append('fdr<'+'%.0e' % threshold)
    else:
        testrsl.append('rev>'+'%.2f' % threshold)
    testrsl.append("number pair: %d" % num_pair)
    etc_test = extract(x_test,index,typ)
#     model, num_coef=train_Lasso(,label,plot=False)
    lassoauc = test_Lasso(etc_test,y_test,clf)
    testrsl.append("number of lasso pair: %d " % num_lasso_pair)
    testrsl.append('Lasso AUC:%.3f' % lassoauc)
#     rank_test()
    rankauc=test_rank(x_test,y_test,index)
    testrsl.append("Rank AUC:%.3f" % rankauc)
    print(rankauc)
    for name, clf_ in zip(names, classifiers):
        clf_.fit(etc_test, y_test)
        
        predt = clf_.predict(etc_test)
        print(name, "test ROC AUC: ",metrics.roc_auc_score(y_test, predt))
        testrsl.append('%s result:%.3f' % (name,metrics.roc_auc_score(y_test, predt)))
        
    return testrsl
def gene_relative_rank(length,n_sm,x_train):
    t1 = time.time()
    result1 = np.zeros([length,length,n_sm],dtype=np.bool)
    result1_ = np.zeros([length,length,n_sm],dtype=np.bool)
    for i in range(0,n_sm):
        time_l = time.time()
        x = np.tile(x_train.iloc[i][:], (length,1))
        sub = x - x.T

        result1[:,:,i] = (sub > 0)
        result1_[:,:,i] = (sub < 0)

        print('loop {:d} : {:5f}'.format(i, time.time() - time_l))
        return result,result_
def get_index_para(x_train,y_train,pvalue):
    length=x_train.shape[1]
    n_sm=x_train.shape[0]
    index,num_pair,a,b,c,d = get_reverse_index.get_rev_index(length,n_sm,x_train,y_train,pvalue)
    return index,num_pair,a,b,c,d





    
