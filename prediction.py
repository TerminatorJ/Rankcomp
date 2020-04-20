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
def pred(train_matrix,test_matrix,test_label,pair_index,c,d):
    '''
    This is used for predicting the result of newly dataset
    '''
    t1=time.time()
    test_auc=test_rank_more_new(train_matrix,test_matrix,test_label,pair_index,c,d)
    t2=time.time()
    elapse=t2-t1
    print("The prediction task elapsing: %s" % str(elapse))
    return test_auc

