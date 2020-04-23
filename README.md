# iPAGE: individualized Pair Analysis of Gene Expression

Introduction
-----------------------------
This is a learning framework to extract the DEG (Differential expression genes) by high fided gene pairs


Publications(please cite following paper): 

Installation: 

Using pip
```python
pip install rankcomp
```
packages requirement: 
python3*,fisher

Here is how you can extract DEG for your data:
For user
```python
from rankcomp import get_opt_p
from rankcomp import step_forward
import pickle
import pandas as pd

need_gene=pd.read_table("./data/RIF_gene.txt",header=[0])
need_gene=need_gene["Symbol"]
##you can use the test data saved in the ./data direction as .pickle file type
x_test=pickle.load(open("./result/RIFpj_clst_1_test_matrix_ovlp.pickle","rb"))
y_test=pickle.load(open("./result/RIFpj_clst_1_test_label_ovlp.pickle","rb"))
  
##The first step is to select the optimized p value as input for the next step.
opt_p=get_opt_p.get_p(x_test,y_test,need_gene,inter_num=10,p_high=-2,inter_num=5)#the matrix and label should include two types of case. default 0and 1.
gene_pair=step_forward.step_forward(x_test,y_test,need_gene,opt_p)#this scripts can help you to extract different genes pairs with strong significance.
##If you want to customize the running process, please see the following guidline of parameter setting.
```
output:

e.g: [(geneA,geneB),(geneC,geneD)]



# Parameters list:

##   For get_p
train_matrix: dataframe. Should be set like colname=genename; rowname=sample_name.

label: array like. Can be DataFrame Series or array like.

train_dir_list: list default False. If you just want to input the datadir, please input like:     ["data/case_matrix.txt","data/ctrl_matrix.txt"].note that, all the data in the list should be the same order as train_matrix.

hint_list: list default False.This argument is set to be campatibel with the data you save in the "data/", eg. if your set list as ["data/case_matrix.txt","data/ctrl_matrix.txt"]. Then, your hint list should be [1,0], where 1 represent case, 0 represent control.

need_gene: array like .The gene related to this disease, some may be immune related, or others.

need_gene_dir: str default None. You can also put the need gene in to the dir in the data/

typ: str default rank. The model type, you can also used Adaboost, DT or else.

thre_typ: str default fdr. The method used to filter the DEG gene in the first step.

p_low: int default -47. Used to set the lower limit, if the gene pairs don't fit your expectation, please reset it.

p_high: int default -42. Used to set the upper limit.if the gene pairs don't fit your expectation, please reset it.

inter_num: int default 10. Used to choose the number of p in the range between p_low and p_high.

err: int default 100. This is the difference value between your expectance of the number of selected gene pairs and the number of pairs model give under the respective p values.

exp: int default 500. This is the expected pair number you want to used in the following step of step forward selection. larger number will increse the complexity of model.

plot_b: bool default. Whether plotting the plot of p value selection, you can found it in dir of ./figure/Pvalue_selection.pdf.


##   For step_forward
train_matrix: dataframe. Should be colname=genename; rowname=sample_name.

label: array like.Can be DataFrame Series or array like.

opt_p: float. The optimized p value you get from the function of get_opt_p.

typ: str default rank. The form of gene you input as the training matrix.

thre_typ: str default fdr. The method used to filter the DEG gene in the first step.

f_plot: bool default True. Whether plotting the processing of feature selection.

f_low: int default 5. This is used to set the range of step forward selection. Actually, the more features(pairs) means more large value of AUC.

f_high: int default 50. This is used to set the range of step forward selection. Actually, the more features(pairs)means more large value of AUC. if you want to get more accurate result regardless of the gene pairs numbers, you can set larger f_high,However, which will increase the complexity of calculating.





