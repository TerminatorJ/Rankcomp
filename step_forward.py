# coding: utf-8
import time
import os
from multiprocessing import Pool
import get_reverse_index
import utils
import load_data
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.cross_validation import train_test_split
import pandas as pd
import p_selection
import feature_selection
def step_forward(train_matrix,train_label,need_gene,opt_p,env="Linux",typ="rank",thre_type="fpr",f_plot=True,test_plot=True,f_low=5,f_high=51):
    '''
    train_matrix: dataframe. should be colname=genename; rowname=sample_name.
    label: array like.Can be DataFrame Series or array like.
    opt_p: float. the optimized p value you get from the function of get_opt_p.
    typ: str default rank. the form of gene you input as the training matrix.
    thre_typ: str default fdr. the method used to filter the DEG gene in the first step.
    f_plot: bool default True. Whether plotting the processing of feature selection.
    f_low: int default 5. this is used to set the range of step forward selection. Actually, the more features(pairs) means more large value of AUC.
    f_high: int default 50. this is used to set the range of step forward selection. Actually, the more features(pairs)means more large value of AUC. if you want to get more accurate result regardless of the gene pairs numbers, you can set larger f_high,However, which will increase the complexity of calculating.
    '''
    t1=time.time()
    pvalue=opt_p
    typ=typ
    f_plot=f_plot
    f_low=f_low
    f_high=f_high
    thre_type=thre_type
    k_feature_lst=[i for i in range(f_low,f_high)]
    train_data=load_data.gene_data(train_matrix,need_gene,env)
    train_matrix=train_data.get_matrix()
    train_label=train_label
    train_x,test_x,train_y,test_y=train_test_split(train_matrix,train_label,test_size=0.2,random_state=0)
    print("Getting the training and testing data: Done")
    print("The number of sample you input is %d" % train_x.shape[0])
    index,num_raw_pair,a,b,c,d=utils.get_index_para(train_x,train_y,pvalue)
    max_pair=feature_selection.get_optimized_feature(k_feature_lst,pvalue,train_x,train_y,index,c,d,num_raw_pair,typ,test_x,test_y,thre_type,f_plot,test_plot)
    print("\nthe feature selection has been done you can see the result in the figure/opt_sfa_rsl.pdf and figure/opt_test_rsl.pdf to get the selected gene pairs")
    #print("The pair we select as follow:%s" % str(max_pair))
    t2=time.time()
    elapse=t2-t1
    print("step forward selection elapsing: %s" % str(elapse))
    return max_pair
