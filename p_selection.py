#!/bin/python
import get_reverse_index
import os
import time
import time
import os
from multiprocessing import Pool
import get_reverse_index
import utils
import load_data
import numpy as np
import matplotlib.pyplot as plt
import pickle
import pandas as pd
def get_p_value(pair_dict,err,exp):
    left=exp-err
    right=exp+err
    for p,num_pair in pair_dict.items():
        if num_pair in range(left,right):
            need=p
            break
    print("test pair_dict:",pair_dict)
    return need
    


def plot_optimized_pvalue(p_range,x_train,y_train,plot,err,exp):
    length=x_train.shape[1]
    n_sm=x_train.shape[0]
    pair_dict={}
    pair_num=[]
    for pvalue in p_range:
        index,num_pair,a,b,c,d = get_reverse_index.get_rev_index(length,n_sm,x_train,y_train,pvalue)
        pair_dict[pvalue]=num_pair
        pair_num.append(num_pair)
    opt_p=get_p_value(pair_dict,err,exp)
    if plot:
        root_path=os.getcwd()
        pj=lambda *path: os.path.abspath(os.path.join(*path))
        figure_path=pj(root_path,"figure/Pvalue_selection.pdf")
        val_path=pj(root_path,"figure/pvalue_selection_dict.pickle")
        pickle.dump(pair_dict,open(val_path,"wb"))
    ##plot the selection process
        fig,ax=plt.subplots(1,1)
        plt.plot(p_range,pair_num,linestyle="-",label="pvalue")
        ax.set_xscale("log")
        plt.xlabel("p values",size=12)
        plt.ylabel("number of gene pairs",size=12)
        ax.set_title("Selection of the optimal pvalue")
        plt.savefig(figure_path,format="pdf")
    return opt_p


