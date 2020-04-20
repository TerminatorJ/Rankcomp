# Rankcomp
This is a learning framework to extract the DEG (Differential expression genes) by high fided gene pairs
Introduction
-----------------------------
Publications(please cite following paper): 

Installation: 

Using pip
```python
pip install rankcomp
```

Here is how you can extract DEG for your data:
For user
```python
from rankcomp import get_opt_p
from rankcomp import step_forward
import pickle
##you can use the test data saved in the ./data direction as .pickle file type
x_test=pickle.load(open("./data/RIFpj_clst_1_test_matrix_ovlp.pickle","rb"))
y_test=pickle.load(open("./data/RIFpj_clst_1_test_label_ovlp.pickle","rb"))
##The first step is to select the optimized p value as input for the next step.
opt_p=get_opt_p.get_p(x_test,y_test,need_gene,inter_num=10,p_high=-2)#the matrix and label should include two types of case. default 0and 1.
gene_pair=step_forward.step_forward(x_test,y_test,need_gene,opt_p)#this scripts can help you to extract different genes pairs with strong significance.
```
output:
e.g: [(geneA,geneB),(geneC,geneD)]
