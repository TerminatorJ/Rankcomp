
# coding: utf-8

# In[2]:


import load_data
import utils
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
from tqdm import tqdm

# In[3]:


def gene_relative_rank(length,n_sm,x_train):
    t1 = time.time()
    result1 = np.zeros([length,length,n_sm],dtype=np.bool)
    result1_ = np.zeros([length,length,n_sm],dtype=np.bool)
    for i in tqdm(range(0,n_sm)):
        time.sleep(0.05)
        time_l = time.time()
        x = np.tile(x_train.iloc[i][:], (length,1))
        sub = x - x.T
        result1[:,:,i] = (sub > 0)
        result1_[:,:,i] = (sub < 0)
        #print('loop {:d} : {:5f}'.format(i, time.time() - time_l))
    return result1,result1_
def get_value(length,n_sm,x_train,y_train):
    result1,result1_=gene_relative_rank(length,n_sm,x_train)
    ##Figure out the number of greater or lower rank pairs among all the samples
    con_label = np.squeeze(1 - y_train)
    y_train=np.squeeze(y_train)
    a = np.sum(result1*con_label,axis = 2)
    b = np.sum(result1_*con_label,axis = 2)
    # Case
    c = np.sum(result1*y_train, axis = 2)
    d = np.sum(result1_*y_train, axis = 2)
    return a,b,c,d

def get_reverse_index(length,n_sm,x_train,y_train,rev_list=[0.95]):
    result1,result1_=gene_relative_rank(length,n_sm,x_train)
    ##Figure out the number of greater or lower rank pairs among all the samples
    print("Type of y_train: %s" % str(type(y_train)))
    con_label = np.squeeze(1 - y_train)
    print("Type of y_train: %s" % str(type(y_train)))
    y_train=np.squeeze(y_train)
    a = np.sum(result1*con_label,axis = 2)
    b = np.sum(result1_*con_label,axis = 2)
    # Case
    c = np.sum(result1*y_train, axis = 2)
    d = np.sum(result1_*y_train, axis = 2)
    ##Use reversal percentage to figure out pairs
    

    num_disease = np.sum(y_train)
    num_control = len(y_train) - num_disease

    print("num of case: ", num_disease)
    print("num of control: ", num_control)

    for rev in rev_list:
        at = ((a.astype(float)/num_control) >= rev)
        ct = ((c.astype(float)/num_disease) <= (1-rev))

        index0, index1 = np.where(at * ct)

        for i in range(0,len(index0)):
            if index0[i] > index1[i]:
                index0[i],index1[i] = index1[i],index0[i]   
        #print("reversal percentage = ", rev)
        #print("number of pairs = ", len(index0))

    index = np.stack((index0, index1))
    print("The reverse index is:%s" % str(index))
    return index,a,b,c,d
def fisher_index(a,b,c,d,length,threshold):
    a_ = a.reshape((-1,1))
    a_ = np.squeeze(a_)
    a_ = a_.astype(np.uint)
    b_ = b.reshape((-1,1))
    b_ = np.squeeze(b_)
    b_ = b_.astype(np.uint)
    c_ = c.reshape((-1,1))
    c_ = np.squeeze(c_)
    c_ = c_.astype(np.uint)
    d_ = d.reshape((-1,1))
    d_ = np.squeeze(d_)
    d_ = d_.astype(np.uint)
    # fisher exact test
    _, _, twosided = pvalue_npy(a_, b_, c_, d_)
    # oddsratio,p=stats.fisher_exact(a_, b_, c_, d_,alternative="two-sided")
    # fdr
    rejected, pvalue_Bonf, alphacSidak, alphacBonf = sm.multipletests(twosided, alpha=0.05, method='bonferroni', 
                                                                      is_sorted=False, returnsorted=False)
    pvalue_Bonf2 = pvalue_Bonf.reshape((length,length))
    #Filter the genes you can set the thresholds.
    
    # find indices of pvalue < fdr_threshold
    length_2 = len(pvalue_Bonf2[0,:])
    # add 1 in triangle matrix to remove duplicated index
    pvalue_matrix = pvalue_Bonf2 + np.triu(np.ones([length_2,length_2]))#要知道有一半的矩阵是一样的
    #print(pvalue_matrix)
    j,k = np.where(pvalue_matrix < threshold)
    #j,k = np.where(pvalue_matrix < 1)
    result3 = np.array([j,k],dtype = np.uint16)
    print("pvalue_Bonf_threshold = ", threshold)
    print("number of pairs = ",len(j))
    return result3,len(j)##返回的是具体的index
def fisher_index_out(a,b,c,d,length,threshold):
    a_ = a.reshape((-1,1))
    a_ = np.squeeze(a_)
    a_ = a_.astype(np.uint)
    b_ = b.reshape((-1,1))
    b_ = np.squeeze(b_)
    b_ = b_.astype(np.uint)
    c_ = c.reshape((-1,1))
    c_ = np.squeeze(c_)
    c_ = c_.astype(np.uint)
    d_ = d.reshape((-1,1))
    d_ = np.squeeze(d_)
    d_ = d_.astype(np.uint)
    # fisher exact test
    _, _, twosided = pvalue_npy(a_, b_, c_, d_)
    # oddsratio,p=stats.fisher_exact(a_, b_, c_, d_,alternative="two-sided")
    # fdr
    rejected, pvalue_Bonf, alphacSidak, alphacBonf = sm.multipletests(twosided, alpha=0.05, method='bonferroni',
                                                                      is_sorted=False, returnsorted=False)
    pvalue_Bonf2 = pvalue_Bonf.reshape((length,length))
    #Filter the genes you can set the thresholds.

    # find indices of pvalue < fdr_threshold
    length_2 = len(pvalue_Bonf2[0,:])
    # add 1 in triangle matrix to remove duplicated index
    pvalue_matrix = pvalue_Bonf2 + np.triu(np.ones([length_2,length_2]))#要知道有一半的矩阵是一样的
    #j,k = np.where(pvalue_matrix < threshold)
    #j,k = np.where(pvalue_matrix < 1)
    #result3 = np.array([j,k],dtype = np.uint16)
    #print("pvalue_Bonf_threshold = ", threshold)
    #print("number of pairs = ",len(j))
    print(pvalue_matrix)
    return pvalue_matrix
# In[5]:
def get_rev_index_out(length,n_sm,x_train,y_train,threshold):
    #get train
    print("Getting reverse index:")
    a,b,c,d=get_value(length,n_sm,x_train,y_train)
    ##Use fisher exact test to find out reversal pairs
    pvalue_matrix= fisher_index_out(a,b,c,d,length,threshold)
    return pvalue_matrix

def get_rev_index(length,n_sm,x_train,y_train,threshold):
    #get train
    print("Getting reverse index:")   
    a,b,c,d=get_value(length,n_sm,x_train,y_train)
    ##Use fisher exact test to find out reversal pairs
    index, num_pair = fisher_index(a,b,c,d,length,threshold)
    return index,num_pair,a,b,c,d

#main
if __name__=="__main__":
    env="win"##win or linux
    sc_dir_list_raw_train=["data//casemay10_test.txt","data//ctrlmay10_test.txt"]
    ncbi_dir="data//NCBI_leukemia_mm_gene.txt"
    sc_dir_list_raw_test=["data//casemay26_test.txt","data//ctrlmay26_test.txt"]
    rev_list = np.arange(0.99, 0.90, -0.01)
    thresholds = [1e-2]
    index,num_pair=get_rev_index(env,sc_dir_list_raw_train,sc_dir_list_raw_test,ncbi_dir,rev_list,thresholds)
    

