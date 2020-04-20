#!/bin/python
import time
import os
from multiprocessing import Pool
import multiprocessing
import get_reverse_index
import utils
import test_ml_model
import load_data
import numpy as np
import matplotlib.pyplot as plt
import pickle
from tqdm import tqdm
def step_f(threshold,k_feature,x_train,y_train,index,c,d,num_raw_pair,typ,x_test,y_test,thre_type,this_name,gene_dict,feature_dict,pair_dict):
    #print("Here is %d" % k_feature)
    model_auc,selected_index,num_gene = test_ml_model.sfa(k_feature,x_train,y_train,index,c,d,num_raw_pair,typ,True,None)
    etc_train=utils.extract(x_train,selected_index,typ)
    etc_test=utils.extract(x_test,selected_index,typ)
    test_rsl=test_ml_model.testmodel_single(this_name,model_auc,k_feature,num_gene,threshold,num_raw_pair,index,thre_type)
    #auc_rsl=float(test_rsl[1].split(": ")[1]) 
    feature_dict[k_feature]=test_rsl 
    selected_pair_list=list(zip(x_train.columns[selected_index[0]],x_train.columns[selected_index[1]]))
    pair_dict[k_feature]=selected_pair_list
    selected_gene_name=x_train.columns[np.union1d(selected_index[0],selected_index[1])]
    gene_dict[k_feature]=selected_gene_name
    return feature_dict,pair_dict,gene_dict
def get_optimized_feature(k_feature_lst,threshold,x_train,y_train,index,c,d,num_raw_pair,typ,x_test,y_test,thre_type,plot,test_plot):
    manager=multiprocessing.Manager()
    feature_dict=manager.dict()
    gene_dict=manager.dict()
    pair_dict=manager.dict()
    jobs=[]
    pool_num=len(k_feature_lst)
    p=Pool(pool_num)
    for k_feature in k_feature_lst:
        
        job=p.apply_async(step_f,args=(threshold,k_feature,x_train,y_train,index,c,d,num_raw_pair,typ,x_test,y_test,thre_type,"rank",gene_dict,feature_dict,pair_dict))
        jobs.append(job)
    p.close()
    p.join()
    print("Saving the result to the direction")
    root_dir=os.getcwd()
    pj=lambda *path: os.path.abspath(os.path.join(*path))
    pickle.dump(feature_dict.items(),open(pj(root_dir,"result/feature_dict.pickle"),"wb"))
    pickle.dump(gene_dict.items(),open(pj(root_dir,"result/gene_dict.pickle"),"wb"))
    pickle.dump(pair_dict.items(),open(pj(root_dir,"result/pair_dict.pickle"),"wb"))
    x=[i for i in feature_dict.keys()]
    y_rank=[float(i[1].split(": ")[1]) for i in feature_dict.values()]
    root_dir=os.getcwd()
    pj=lambda *path: os.path.abspath(os.path.join(*path))
    if plot:
        plt.figure(figsize=(8,5))
        plt.plot(x,y_rank,"-",label="rankcomp",color="red")
        plt.xlabel("Feature number",fontsize=15)
        plt.ylabel("AUC",fontsize=15)
        plt.title("sfs results among 5~50 features",fontsize=15)
        plt.ylim((0,1))
        plt.xlim((0,55))
        auc_max=max(y_rank)
        auc_max_index=y_rank.index(auc_max)
        max_x=x[auc_max_index]
        text_x=max_x-5
        text_y=auc_max-0.05
        plt.text(text_x,text_y,"(%d , %.3f)" % (max_x,auc_max))
        plt.legend(loc=2,ncol=1)
        plt.savefig(pj(root_dir,"figure/opt_sfa_rsl.pdf"),type="pdf")
    ##getting the best performance and applied it in the test dataset
    max_pair=test_ml_model.test_inner(typ,feature_dict,pair_dict,gene_dict,x_test,y_test,c,d,x_train,test_plot)
    return max_pair

   
