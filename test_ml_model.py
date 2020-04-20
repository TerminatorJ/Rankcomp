
# coding: utf-8

# In[ ]:
import matplotlib.pyplot as plt
import numpy as np
import utils
import get_reverse_index
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn import metrics
import os
import time
from tqdm import tqdm
import pickle
from sklearn.cross_validation import train_test_split
# In[ ]:


def plot_roc(labels, predict_prob,to_file):
    false_positive_rate,true_positive_rate,thresholds=metrics.roc_curve(labels, predict_prob)
    roc_auc = metrics.auc(false_positive_rate, true_positive_rate)
    fig,axe=plt.subplots(1,1)
    plt.title('ROC')
    plt.plot(false_positive_rate, true_positive_rate,'b',label = 'AUC = %0.4f'% roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0,1],[0,1],'r--')
    plt.ylabel('TPR')
    plt.xlabel('FPR')
    plt.savefig(to_file,format="pdf")
    plt.show()

def test_rank(num_lasso_pair,clf,data,label,index,c,d,num_pair,to_file_rank_less,to_file_rank_more):
    # for pair, the value of pairconvert is 1 for disease, -1 for health
    rankdata = utils.pairconvert(data,index)
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
    #print(y_test1)
    # convert into 1 and 0
    rankdata = rankdata*tag > 0
    # Use those lasso coefficient is not zero
    filt = clf.coef_!=0
    rank = rankdata * filt
    print("number of pair whose coefficient is not zero: %s"% str(np.sum(filt)))
    # sumup, then the value represent the percentage of reversal pairs
    rankpredt_less = np.squeeze(np.sum(rank, axis = 1))/num_lasso_pair
    print("Rank less ROC AUC: ", metrics.roc_auc_score(label, rankpredt_less))
    plot_roc(label,rankpredt_less,to_file_rank_less)

    rankpredt_more = np.squeeze(np.sum(rankdata, axis = 1))/num_pair
    print("Rank more ROC AUC: ", metrics.roc_auc_score(label, rankpredt_more))
    plot_roc(label,rankpredt_more,to_file_rank_more)
    return metrics.roc_auc_score(label, rankpredt_less),metrics.roc_auc_score(label, rankpredt_more)

def train_rank_more(data,label,index,c,d):
    # for pair, the value of pairconvert is 1 for disease, -1 for health
    rankdata = utils.pairconvert(data,index)
    #print(rankdata)
    #print(rankdata.shape)
    num = len(index[0,:])
    tag = []
    for i in range(0,num):
        tag.append(c[index[0,i],index[1,i]] > d[index[0,i],index[1,i]])##大部分都是增加的
    tag = np.array(tag)
    tag = tag*2-1
    #print(tag)
    # convert into 1 and 0
    rankdata = rankdata*tag > 0
    num_pair=rankdata.shape[1]
    rankpredt_more = np.squeeze(np.sum(rankdata, axis = 1))/num_pair
    #plot_roc(label,rankpredt_more,to_file_rank_more)
    return metrics.roc_auc_score(label, rankpredt_more)

def test_rank_more(data,label,index,c,d):
    # for pair, the value of pairconvert is 1 for disease, -1 for health
    rankdata = utils.pairconvert(data,index)
    print(rankdata)
    print(rankdata.shape)
    num = len(index[0,:])
    tag = []
    for i in range(0,num):
        tag.append(c[index[0,i],index[1,i]] > d[index[0,i],index[1,i]])##大部分都是增加的
    tag = np.array(tag)
    tag = tag*2-1
    print(tag)
    # convert into 1 and 0
    rankdata = rankdata*tag > 0
    num_pair=rankdata.shape[1]
    rankpredt_more = np.squeeze(np.sum(rankdata, axis = 1))/num_pair
    print("Rank more ROC AUC: ", metrics.roc_auc_score(label, rankpredt_more))
    #plot_roc(label,rankpredt_more,to_file_rank_more)
    return metrics.roc_auc_score(label, rankpredt_more)
def test_rank_more_new(train_matrix,test_matrix,test_label,pair_index,c,d):
    # for pair, the value of pairconvert is 1 for disease, -1 for health
    rankdata = utils.pairconvert_test(test_matrix,pair_index)
    #print(rankdata)
    #print(rankdata.shape)
    num = len(pair_index[0,:])
    tag = []
    index=utils.pair_index_2_train_index(pair_index,train_matrix)
    for i in range(0,num):
        tag.append(c[index[0,i],index[1,i]] > d[index[0,i],index[1,i]])##大部分都是增加的
    tag = np.array(tag)
    tag = tag*2-1
    #print(tag)
    # convert into 1 and 0
    rankdata = rankdata*tag > 0
    num_pair=rankdata.shape[1]
    rankpredt_more = np.squeeze(np.sum(rankdata, axis = 1))/num_pair
    #print(rankpredt_more)
    #print(test_label)
    #print("Rank more ROC AUC: ", metrics.roc_auc_score(test_label, rankpredt_more))
    #plot_roc(label,rankpredt_more,to_file_rank_more)
    return metrics.roc_auc_score(test_label, rankpredt_more)




def sfa(k_feature,x_train,y_train,index,c,d,num_pair,typ,rank,clf_):
    s_index = []
    train_x,test_x,train_y,test_y=train_test_split(x_train,y_train,test_size=0.2,random_state=0)
    for k in range(0,k_feature):
        maxauc = 0
        max_i = 0
        for i in range(0,num_pair):
            if i in s_index:
                continue
            k_index = list(s_index)
            k_index.append(i)
            if rank==True:
                rankauc = train_rank_more(x_train,y_train,index[:,k_index],c,d)
            else:
                etc_train = utils.extract(train_x,index[:,k_index],typ)
                etc_test = utils.extract(test_x,index[:,k_index],typ)
                clf_.fit(etc_train,train_y)
                predt = clf_.predict(etc_test)
                rankauc=metrics.roc_auc_score(test_y,predt)
            if rankauc > maxauc:
                max_i = i
                maxauc = rankauc
        s_index.append(max_i)
    print("The best training result of %d is: %.5f" % (k_feature,maxauc))
    selected_index = index[:,s_index]
    num_gene=len(np.union1d(selected_index[0],selected_index[1]))
    return maxauc,selected_index,num_gene#return the number of pairs defined by features

def sfa_value(k_feature,x_train,y_train,clf_,num_gene):
    s_index = []
    print("Going on %d" % k_feature)
    train_x,test_x,train_y,test_y=train_test_split(x_train,y_train,test_size=0.2,random_state=0)
    for k in range(0,k_feature):
        maxauc = 0
        max_i = 0
        for i in range(0,num_gene):
            if i in s_index:
                continue
            k_index = list(s_index)
            k_index.append(i) 
            etc_train = train_x.iloc[:,k_index]
            etc_test = test_x.iloc[:,k_index]
            clf_.fit(etc_train,train_y)
            predt = clf_.predict(etc_test)
            thisauc=metrics.roc_auc_score(test_y,predt)
            if thisauc > maxauc:
                max_i = i
                maxauc = thisauc
        s_index.append(max_i)
    print("The best training result of this model is: %.5f" % maxauc)
    selected_index = s_index
    return maxauc,selected_index#return the number of pairs defined by features



def test_Lasso(data, label,clf,to_file):
    predt = clf.predict(data)
    y_scores = np.array(predt)
    y_true = label
    print("Lasso ROC AUC: ", metrics.roc_auc_score(y_true, y_scores))
    plot_roc(y_true,y_scores,to_file)
    return metrics.roc_auc_score(y_true, y_scores)







def testmodel(name,classifiers,names,model_auc,etc_train,etc_test,y_train,k_feature,num_gene,c,d,threshold,num_pair,x_test,y_test,index,thre_type):
    testrsl = []
    num_pair_tst=len(index[0,:])
    if thre_type == 'fdr':
        testrsl.append('fdr<'+'%.0e' % threshold)
    else:
        testrsl.append('rev>'+'%.2f' % threshold)
    testrsl.append("%s AUC is: %.5f" % (name,model_auc))
    testrsl.append("number of raw pair: %d" % num_pair)
    testrsl.append("number of selected features: %d" % k_feature)
    testrsl.append("number of genes: %d" % num_gene)
    #etc_test = utils.extract(x_test,index,typ)
    #lassoauc = test_Lasso(etc_test,y_test,clf,to_file_lasso)
    #testrsl.append("number of lasso pair: %d " % num_lasso_pair)
    #testrsl.append('Lasso AUC:%.3f' % lassoauc)
    #if typ=="rank":
    #    rankauc_more=test_rank_more(x_test,y_test,index,c,d,num_pair_tst) 
    #    testrsl.append("Rank_more AUC:%.3f" % rankauc_more)
   
        
    for model_name, clf_ in zip(names, classifiers):
        clf_.fit(etc_train, y_train)
        predt = clf_.predict(etc_test)
        print(model_name, "test ROC AUC: ",metrics.roc_auc_score(y_test, predt))
        testrsl.append('%s result:%.3f' % (model_name,metrics.roc_auc_score(y_test, predt)))
    return testrsl
def testmodel_single(name,model_auc,k_feature,num_gene,threshold,num_pair,index,thre_type):
    testrsl = []
    num_pair_tst=len(index[0,:])
    if thre_type == 'fdr':
        testrsl.append('fdr<'+'%.0e' % threshold)
    else:
        testrsl.append('rev>'+'%.2f' % threshold)
    testrsl.append("%s AUC is: %.5f" % (name,model_auc))
    testrsl.append("number of raw pair: %d" % num_pair)
    testrsl.append("number of selected features: %d" % k_feature)
    return testrsl
def testmodel_value(k_feature,name,maxauc,num_gene):
    testrsl = []
    testrsl.append("%s AUC is: %.5f" % (name,maxauc))
    testrsl.append("number of selected features: %d" % k_feature)
    testrsl.append("number of genes: %d" % num_gene)
    return testrsl
def get_max_pair(feature_dict,pair_dict,gene_dict):
    max_auc=0
    max_index=0
    for feature in feature_dict.keys():
        rsl_list=feature_dict[feature]
        auc_rsl=float(rsl_list[1].split(": ")[1])
        if auc_rsl > max_auc:
            max_auc=auc_rsl
            max_index=feature
    max_gene=gene_dict[feature]
    max_pair=pair_dict[feature]
    return max_index,max_gene,max_pair

def test_inner(typ,feature_dict,pair_dict,gene_dict,x_test,y_test,c,d,x_train,test_plot):
    #this is used for the inner teseting, the developer can used it to get the best gene pairs
    if typ=="rank":
        pj=lambda *path: os.path.abspath(os.path.join(*path))
        root=os.getcwd()
        new_feature_dict={}
        max_auc=0
        max_feature=0
        rsl_dict={}
        print("Doing the testing jobs")
        for feature in tqdm(feature_dict.keys()):
            time.sleep(0.05)
            pair_list=pair_dict[feature]
            pair_index=utils.pair_2_pair_index_inner(pair_list)
            num_pair=len(pair_index[0,:])
            test_auc=test_rank_more_new(x_train,x_test,y_test,pair_index,c,d)
            test_auc=float(test_auc)
            rsl_dict[feature]=test_auc
            if test_auc > max_auc:
                max_auc=test_auc
                max_feature=feature
            #print("The result of test set in the %d feature: %.5f" % (feature,test_auc))
            new_feature_dict[feature]=max_auc
        #print()
        max_rsl=max_auc
        max_gene=gene_dict[feature]
        max_pair=pair_dict[feature]
        result_str="The best performance in the testing dataset:\npairs number:%d\nauc:%.4f\ngene:%s\npair:%s " % (int(max_feature),float(max_auc),str(list(max_gene)),str(max_pair)) 
        print("\033[31m %s\033[0m" % result_str)
        pickle.dump(result_str,open(pj(root,"result/testing_result.pickle"),"wb"))
    x=[i for i in feature_dict.keys()]
    y_test=[i for i in rsl_dict.values()]
    root_dir=os.getcwd()
    pj=lambda *path: os.path.abspath(os.path.join(*path))
    if test_plot:
        plt.figure(figsize=(8,5))
        plt.plot(x,y_test,"-",label="rankcomp_test",color="red")
        plt.xlabel("Feature number",fontsize=15)
        plt.ylabel("AUC",fontsize=15)
        plt.title("sfs results among 5~50 features",fontsize=15)
        plt.ylim((0,1))
        plt.xlim((0,55))
        auc_max=max(y_test)
        auc_max_index=y_test.index(auc_max)
        max_x=x[auc_max_index]
        text_x=max_x-5
        text_y=auc_max-0.05
        plt.text(text_x,text_y,"(%d , %.3f)" % (max_x,auc_max))
        plt.legend(loc=2,ncol=1)
        plt.savefig(pj(root_dir,"figure/opt_testrsl.pdf"),type="pdf")  
    return max_pair



def test_bulk_sc(drop_dict,sc_name,train_matrix,train_label,feature_range,test_data,clf_name,c,d,selected_pair_file):
    test_rsl={}
    if clf_name=="rank":
        pj=lambda *path: os.path.abspath(os.path.join(*path))
        root=os.getcwd()
        pair_rsl=pickle.load(open(pj(root,"result/%s" % selected_pair_file),"rb"))
        for name,all_data in test_data.items():
            test_matrix=all_data[0]
            test_label=all_data[1]
            for feature in feature_range:
                pair_index=utils.pair_2_pair_index(name,drop_dict,feature,pair_rsl)##get the pair like [["a","b"],["b","c"]]
                num_pair=len(pair_index[0,:])
                test_auc=test_rank_more_new(train_matrix,test_matrix,test_label,pair_index,c,d)    
                print("The result of test set %s among %d feature: %.5f" % (name,feature,test_auc))
                test_rsl.setdefault(name,{}).setdefault(feature,test_auc)
        pickle.dump(test_rsl,open(pj(root,"result/test_bulk_and_sc_result.pickle"),"wb"))
    if clf_name=="Ada":
        pj=lambda *path: os.path.abspath(os.path.join(*path))
        root=os.getcwd()
        pair_rsl=pickle.load(open(pj(root,"result/%s" % selected_pair_file),"rb"))
        train_x,test_x,train_y,test_y=train_test_split(train_matrix,train_label,test_size=0.2,random_state=0)
        for name,all_data in test_data.items():
            test_matrix=all_data[0]
            test_label=all_data[1]
            for feature in feature_range:
                pair_index=utils.pair_2_pair_index(name,drop_dict,feature,pair_rsl)##get the pair like [["a","b"],["b","c"]]
                num_pair=len(pair_index[0,:])
                index=utils.pair_index_2_train_index(pair_index,train_matrix)
                etc_train = utils.extract(train_x,index,"rank")
                etc_test = utils.pairconvert_test(test_matrix,pair_index)
                Ada=AdaBoostClassifier(random_state=7).fit(etc_train,train_y)
                predt = Ada.predict(etc_test)
                test_auc=metrics.roc_auc_score(test_label, predt)            
                print("The result of test set %s among %d feature: %.5f" % (name,feature,test_auc))
                test_rsl.setdefault(name,{}).setdefault(feature,test_auc)
        pickle.dump(test_rsl,open(pj(root,"result/test_bulk_and_sc_result_Ada.pickle"),"wb")) 
    if clf_name=="DT":
        pj=lambda *path: os.path.abspath(os.path.join(*path))
        root=os.getcwd()
        pair_rsl=pickle.load(open(pj(root,"result/%s" % selected_pair_file),"rb"))
        train_x,test_x,train_y,test_y=train_test_split(train_matrix,train_label,test_size=0.2,random_state=0)
        for name,all_data in test_data.items():
            test_matrix=all_data[0]
            test_label=all_data[1]
            for feature in feature_range:
                pair_index=utils.pair_2_pair_index(name,drop_dict,feature,pair_rsl)##get the pair like [["a","b"],["b","c"]]
                num_pair=len(pair_index[0,:])
                index=utils.pair_index_2_train_index(pair_index,train_matrix)
                etc_train = utils.extract(train_x,index,"rank")
                etc_test = utils.pairconvert_test(test_matrix,pair_index)
                DT=DecisionTreeClassifier(random_state=7).fit(etc_train,train_y)
                predt = DT.predict(etc_test)
                test_auc=metrics.roc_auc_score(test_label, predt)
                print("The result of test set %s among %d feature: %.5f" % (name,feature,test_auc))
                test_rsl.setdefault(name,{}).setdefault(feature,test_auc)
        pickle.dump(test_rsl,open(pj(root,"result/test_bulk_and_sc_result_DT.pickle"),"wb"))


def test_bulk_sc_value(drop_dict,sc_name,train_matrix,train_label,feature_range,test_data,clf_name,c,d,gene_rsl):
    test_rsl={}
    gene_feat_dict=dict(zip([i[0] for i in gene_rsl],[list(i[1]) for i in gene_rsl]))
    if clf_name=="DT":
        pj=lambda *path: os.path.abspath(os.path.join(*path))
        root=os.getcwd()
        train_x,test_x,train_y,test_y=train_test_split(train_matrix,train_label,test_size=0.2,random_state=0)
        for name,all_data in test_data.items():
            test_matrix=all_data[0]
            test_label=all_data[1]
            for feature in feature_range:
                gene_index=list(gene_feat_dict[feature])
                #print(gene_index)
                #etc_train=train_x.iloc[:,:100]
                #etc_test=test_matrix.iloc[:,:100]
                etc_train = utils.extract_value(train_x,gene_index)
                etc_test = utils.extract_value(test_matrix,gene_index)
                DT=DecisionTreeClassifier(random_state=7).fit(etc_train,train_y)
                predt = DT.predict(etc_test)
                test_auc=metrics.roc_auc_score(test_label,predt)
                #print("test label: ",test_label)
                #print("pred label: ",predt)
                #print("train x:", train_x)
                print("The result of test set %s among %d feature: %.5f" % (name,feature,test_auc))
                test_rsl.setdefault(name,{}).setdefault(feature,test_auc)
            #print("etc_train shape:", etc_train.shape)
            #print("etc_test shape:", etc_test.shape)
            #print("train_y shape:", train_y.shape)
            #print("test_label shape:", test_label.shape)
        pickle.dump(test_rsl,open(pj(root,"result/test_bulk_and_sc_result_DT_value.pickle"),"wb"))
