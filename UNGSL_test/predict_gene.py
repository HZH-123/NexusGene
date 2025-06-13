import pandas as pd

import data_loader
from UNGSL_test.data_h5_loader import read_h5file

#df1=pd.read_csv(r"/root/autodl-tmp/UNGSL/UNGSL_test/gene_names/topk_gene_name1.csv")
#df1=pd.read_csv(r"/root/autodl-tmp/UNGSL/UNGSL_test/gene_names/topk_gene_name2.csv")
# df3=pd.read_csv(r"/root/autodl-tmp/UNGSL/UNGSL_test/gene_names/topk_gene_name3.csv")
# df4=pd.read_csv(r"/root/autodl-tmp/UNGSL/UNGSL_test/gene_names/topk_gene_name4.csv")
# df5=pd.read_csv(r"/root/autodl-tmp/UNGSL/UNGSL_test/gene_names/topk_gene_name5.csv")
# df6=pd.read_csv(r"/root/autodl-tmp/UNGSL/UNGSL_test/gene_names/topk_gene_name6.csv")
# df7=pd.read_csv(r"/root/autodl-tmp/UNGSL/UNGSL_test/gene_names/topk_gene_name7.csv")
# gene=[]
# df=[df1,df2,df3,df4,df5,df6,df7]
# for dataframe in df:
#     for name in dataframe["gene_name"]:
#         name=name[2:-1]
#         if name not in gene:
#             gene.append(name)
# #df_all=pd.DataFrame(gene,columns=["gene_name"])
# #df_all.to_csv(r'/root/autodl-tmp/UNGSL/UNGSL_test/gene_names/gene_name_all.csv',encoding='utf-8')
# print(gene)
# print(len(gene))

#df1=pd.read_csv(f"/root/autodl-tmp/UNGSL/UNGSL_test/topk_gene/topk_gene_GGNet.csv")
# data = data_loader.load_graph_data('/root/autodl-tmp/UNGSL/UNGSL_test/GGNet/dataset_GGNet.pkl')
# data = data_loader.load_graph_data('/root/autodl-tmp/UNGSL/UNGSL_test/PPNet/dataset_PPNet.pkl')
#df1=pd.read_csv(r"/root/autodl-tmp/UNGSL/UNGSL_test/topk_gene/topk_gene_PPNet.csv")
#df1=pd.read_csv(r'/root/autodl-tmp/UNGSL/UNGSL_test/topk_gene/gene_name_all.csv')
# data = data_loader.load_graph_data('/root/autodl-tmp/UNGSL/UNGSL_test/PathNet/dataset_PathNet.pkl')
df1=pd.read_csv(r"/root/autodl-tmp/UNGSL/UNGSL_test/topk_gene/topk_gene_PathNet_pred.csv")
df2=pd.read_csv(r"/root/autodl-tmp/UNGSL/UNGSL_test/topk_gene/topk_gene_PPNet_pred.csv")
df3=pd.read_csv(r"/root/autodl-tmp/UNGSL/UNGSL_test/topk_gene/topk_gene_GGNet_pred.csv")
df=[df1,df2,df3]
all=[]
num=0
for dataframe in df:
    for name in dataframe["gene_name"]:
        if name not in all:
            all.append(name)
            num=num+1
print(f"num:{num}")
# idx1=(data1.y==1).nonzero(as_tuple=True)[0]
# idx2=(data2.y==1).nonzero(as_tuple=True)[0]
# idx3=(data3.y==1).nonzero(as_tuple=True)[0]
# d1=[]
# d2=[]
# d3=[]
# for idx in idx1:
#     d1.append(data1.gene_names[idx])
# for idx in idx2:
#     d2.append(data2.gene_names[idx])
# for idx in idx3:
#     d3.append(data3.gene_names[idx])
# data=[d1,d2,d3]
# gene=[]
# gene_name=[]
# for dataframe in data:
#     for name in dataframe:
#         if name not in gene_name:
#             gene_name.append(name)
data=pd.read_csv(r"/root/autodl-tmp/UNGSL/UNGSL_test/topk_gene/gene_name.csv")
known_gene=data.values[:,1]
# df_all=pd.DataFrame(gene_name,columns=["gene_name"])
# df_all.to_csv(r'/root/autodl-tmp/UNGSL/UNGSL_test/topk_gene/gene_name.csv',encoding='utf-8')
# known_idx=(data.y==1).nonzero(as_tuple=True)[0]
# known_gene=[]
# for idx in known_idx:
#     known_gene.append(data.gene_names[idx])
#df1=pd.read_csv(r'/root/autodl-tmp/UNGSL/UNGSL_test/topk_gene/gene_name_all.csv')
# for name in all:
#     if name in known_gene:
#         count+=1
#         a.append(name)
a=pd.DataFrame(all,columns=["gene_name"])
a.to_csv(r'/root/autodl-tmp/UNGSL/UNGSL_test/topk_gene/all.csv',encoding='utf-8')


