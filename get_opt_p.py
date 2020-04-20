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

def get_p(train_matrix,label,need_gene,train_dir_list=False,hint_list=False,need_gene_dir=None,typ="rank",thre_type="fdr",p_low=-47,p_high=-42,inter_num=10,err=100,exp=500,plot_p=True):
    '''
    train_matrix: dataframe. should be colname=genename; rowname=sample_name.
    label: array like.Can be DataFrame Series or array like.
    train_dir_list: list default False. if you just want to input the datadir, please input like:["data/case_matrix.txt","data/ctrl_matrix.txt"].note that, all the data in the list should be the same order as train_matrix.
    hint_list: list default False.This argument is set to be campatibel with the data you save in the "data/", eg. if your set list as ["data/case_matrix.txt","data/ctrl_matrix.txt"]. Then, your hint list should be [1,0], where 1 represent case, 0 represent control.
    need_gene: array like .the gene related to this disease, some may be immune related, or others.
    need_gene_dir: str default None. you can also put the need gene in to the dir in the data/
    typ: str default rank. the model type, you can also used Adaboost, DT or else.
    thre_typ: str default fdr. the method used to filter the DEG gene in the first step.
    p_low: int default -47. used to set the lower limit, if the gene pairs don't fit your expectation, please reset it.
    p_high: int default -42. used to set the upper limit.if the gene pairs don't fit your expectation, please reset it
    inter_num: int default 10. used to choose the number of p in the range between p_low and p_high.
    err: int default 100. this is the difference value between your expectance of the number of selected gene pairs and the number of pairs model give under the respective p values.  
    exp: int default 500. this is the expected pair number you want to used in the following step of step forward selection. larger number will increse the complexity of model.
    plot_b: bool default. whether plotting the plot of p value selection, you can found it in dir of ./figure/Pvalue_selection.pdf. 
    '''
    t1=time.time()
    env="linux"
###for scRNA
    bulk=False
    if train_dir_list:
        sc_dir_list_raw_train=train_dir_list
        #root_dir=os.getcwd()
        #pj=lambda *path: os.path.abspath(os.path.join(*path))
#################################################pickle loading######################################
##these data should be transposed,colname=genes,rownames=samples                                    #
#print("For the whole data we loaded:")                                                             # 
#x_train=pickle.load(open(pj(root_dir,"result/RIFpj_clst_1_train_matrix_ovlp.pickle"),"rb"))        #
#y_train=pickle.load(open(pj(root_dir,"result/RIFpj_clst_1_train_label_ovlp.pickle"),"rb"))         #
#y_train=np.array(y_train)                                                                          #
#x_test=pickle.load(open(pj(root_dir,"result/RIFpj_clst_1_test_matrix_ovlp.pickle"),"rb"))          #
#y_test=pickle.load(open(pj(root_dir,"result/RIFpj_clst_1_test_label_ovlp.pickle"),"rb"))           #
#y_test=np.array(y_test)                                                                            #
#################################################loading data########################################
        print("For the whole data we loaded:") 
        #filter_gene="data/RIF_gene.txt"
        filter_gene= need_gene_dir
        train_data=load_data.gene_data_dir(sc_dir_list_raw_train,filter_gene,env)
        train_matrix=train_data.get_matrix_app(hint_list,bulk=False)
        train_label=train_data.gene_label(bulk=bulk,bulk_dir=None)
        print("Getting the train and test data")
        train_x,test_x,train_y,test_y=train_test_split(train_matrix,train_label,test_size=0.2,random_state=0)
#####################################################################################################
        print("Train matrix shape: %s" % str(train_x.shape))
        print("Test matrix shape: %s" % str(test_x.shape))
        print("Train label shape: %s" % str(train_y.shape[0]))
        print("Test label shape: %s" % str(test_y.shape[0]))
#############################################fundamental setting######################################
    elif train_dir_list==False:
        train_data=load_data.gene_data(train_matrix,need_gene,env)
        train_matrix=train_data.get_matrix()
        train_label=label
        print("Getting the train and test data")
        train_x,test_x,train_y,test_y=train_test_split(train_matrix,train_label,test_size=0.2,random_state=0)
#####################################################################################################
        print("Train matrix shape: %s" % str(train_x.shape))
        print("Test matrix shape: %s" % str(test_x.shape))
        print("Train label shape: %s" % str(train_y.shape[0]))
        print("Test label shape: %s" % str(test_y.shape[0]))
    train_y=np.array(train_y)
    test_y=np.array(test_y)
    typ=typ
    thre_type = thre_type
    p_low=p_low
    p_high=p_high
    inter_num=inter_num
    pvalue_list=[i for i in np.logspace(p_low,p_high,inter_num)]
    err=err
    exp=exp
############################################getting the optimized pvalue##############################
    print("Getting the optimized model...")
    plot_p=plot_p
    optimized_p=p_selection.plot_optimized_pvalue(pvalue_list,train_x,train_y,plot_p,err,exp)
############################################main loop#################################################
#get_optimized_feature(k_feature_lst,threshold,x_train,y_train,index,c,d,num_raw_pair,typ,x_test,y_test,thre_type)
    t2=time.time()
    elapse=t2-t1
    print("The optimaed p is: %.0e" % optimized_p)
    print("Selecting the optimzed p value elapsing: %s" % str(elapse))
    return optimized_p

