from collections import defaultdict
from pickle import FALSE
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.modules.activation import PReLU
from torch_geometric.nn import GCNConv, GATConv,RGCNConv
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import glorot, uniform
from torch_geometric.utils import softmax
import math

from torch_sparse.tensor import to
from utils import *
from torch_sparse import SparseTensor, set_diag
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax, degree, to_undirected
from torch.nn import Sequential, Linear, ReLU, Dropout
from torch_scatter import scatter
import numpy as np

class HeteGNN(MessagePassing):
    def __init__(self, input_dim,output_dim,rel_num,negative_slope=0.2,num_company_rel=7,num_person_rel=3,
    aggr = "add", flow= "source_to_target", node_dim = -2):
        super(HeteGNN,self).__init__(aggr=aggr, flow=flow, node_dim=node_dim)
        self.input_dim=input_dim
        self.output_dim=output_dim
        self.rel_num=rel_num
        self.negative_slope=negative_slope

        self.proj_com=nn.Linear(input_dim,output_dim,bias=False)
        self.proj_per=nn.Linear(input_dim,output_dim,bias=False)

        self.ck_linears   = nn.ModuleList()
        self.cq_linears   = nn.ModuleList()

        for t in range(rel_num):
            self.ck_linears.append(nn.Linear(output_dim,   output_dim))
            self.cq_linears.append(nn.Linear(output_dim,   output_dim))

        self.cv_linear=nn.Linear(output_dim,   output_dim)
        self.crelation_pri   = nn.Parameter(torch.ones(rel_num))

        self.rel_wi=nn.ModuleList()
        for i in range(rel_num):
            if i in [6,7,8,9]:
                self.rel_wi.append(nn.Linear(output_dim,output_dim,bias=False))
            else:
                self.rel_wi.append(nn.Linear(output_dim*2,output_dim,bias=False))


        self.skip = nn.Parameter(torch.ones(1))
        self.bn=nn.BatchNorm1d(output_dim)

    def forward(self,company_emb,person_emb,edge_index,edge_type,edge_weight,company_num,person_num):
        company_emb=self.proj_com(company_emb)
        person_emb=self.proj_per(person_emb)
        emb=torch.cat((company_emb,person_emb),dim=0)
        emb=self.bn(emb)
        edge_index= torch.LongTensor(edge_index).transpose(0,1)

        edge_type=torch.LongTensor(edge_type)
        edge_weight=torch.FloatTensor(edge_weight).unsqueeze(1)

        rs_list=[]
        rel_type=[]

        for i in range(self.rel_num):
            mask = (edge_type == i)
            sub_edge_index = edge_index[:, mask]
            sub_edge_weight=edge_weight[mask]
            if mask.sum() !=0:
                rs=F.leaky_relu((self.propagate(sub_edge_index, x=emb,edge_weight=sub_edge_weight,edge_type=i)),self.negative_slope)
                rs_list+=[rs]
                rel_type+=[i]
        com_att=[]

        for ser,i in enumerate(rel_type):

            rel_emb=rs_list[ser]
            q_mat = self.cq_linears[i](emb)
            k_mat = self.ck_linears[i](rel_emb)
            res_att = ((q_mat * k_mat).sum(dim=-1) * self.crelation_pri[i] / math.sqrt(self.output_dim)).unsqueeze(1)

            com_att+=[res_att]

        com_attscore=torch.cat(com_att,dim=1)
        com_attscore=F.softmax(com_attscore,dim=1)
        res=0
        for i in range(len(com_att)):
            res+= com_attscore[:,i].unsqueeze(1) * self.cv_linear(rs_list[i])
        alpha=torch.sigmoid(self.skip)
        res=(res+alpha*F.gelu(emb))
        res_c,res_p=res[:company_num],res[company_num:]

        return res_c,res_p

    def message(self,edge_index, x_i,x_j, edge_weight, edge_type):
        if torch.sum(edge_weight)!=edge_index.shape[1]:
            x_j=self.rel_wi[edge_type](x_j)
            edge_weight=softmax(edge_weight,edge_index[1])
            rs=x_j*edge_weight
        else:
            node_f = torch.cat((x_i, x_j), 1)                                       #nx2d
            temp = self.rel_wi[edge_type](node_f).to(x_i.device)      #nx1

            alpha=softmax(temp,edge_index[1])

            rs=x_j*alpha
        return rs

    def update(self, inputs):
        return super().update(inputs)
 

class HyperGNN(nn.Module):
    def __init__(self,input_dim,output_dim,hyper_edge_num=3,num_layer=1,negative_slope=0.2):
        super(HyperGNN,self).__init__()
        self.negative_slope=negative_slope
        
        self.proj=nn.Linear(input_dim,output_dim,bias=False)
       
        self.alpha=nn.Parameter(torch.ones(hyper_edge_num,1))
        
        glorot(self.alpha)

    def forward(self,company_emb,hyp_graph):
        outlist=[]
        for i in range(len(hyp_graph)):
            laplacian=scipy_sparse_mat_to_torch_sparse_tensor(hyp_graph[i].laplacian())
            rs= laplacian@self.proj(company_emb)
            outlist+=[rs]
           
        res=0
       
        alpha=torch.sigmoid(self.alpha)
       
        for i in range(len(outlist)):
            res+=outlist[i]*alpha[i]
        return res

#risk data: dict-->{company_index:[[cause type, court type, result category, time(months),time_label],...] }
class RiskInfo(nn.Module):
    def __init__(self,input_dim,company_num,cause_type_num,court_type_num,res_num,time_label_num):
        super(RiskInfo,self).__init__()
        self.input_dim=input_dim
        self.company_num=company_num
        self.time_lable_num=2


        self.ca_emb=nn.Embedding(cause_type_num,12)
        self.court_emb=nn.Embedding(court_type_num,4)
        self.cate_emb=nn.Embedding(res_num,4)
        self.lstm_hidden=20

        self.proj=nn.Linear(20,20,bias=False)


        self.time_decay=Decayer()


    def forward(self,risk_data):

        com_emb=torch.zeros((self.company_num,self.lstm_hidden))

        for index in risk_data:
            cause=self.ca_emb(torch.LongTensor(risk_data[index])[:,0]) 
            court=self.court_emb(torch.LongTensor(risk_data[index])[:,1])
            cate=self.cate_emb(torch.LongTensor(risk_data[index])[:,2])
            risk=torch.cat((cause,court,cate),dim=1)

            time_interval=torch.FloatTensor(risk_data[index])[:,3]

            time_interval=self.time_decay(time_interval).unsqueeze(1)

            risk=self.proj(time_interval*risk)

            com_emb[index]=risk.sum(0)
        return com_emb


class RiskGNN(nn.Module):
    def __init__(self,input_dim,output_dim,
    company_num,person_num,rel_num,cause_type_num,
     device,com_initial_emb,person_initial_emb,
     court_type_num=4,category_num=4,time_label_num=5,num_heads=1,dropout=0.2,norm=True,
     ):
        super(RiskGNN,self).__init__()
        self.input_dim=input_dim
        self.output_dim=output_dim
        self.company_num=company_num
        self.person_num=person_num
        self.rel_num=rel_num
        self.cause_type=cause_type_num
        self.device=device
        self.court_type=court_type_num
        self.category_=category_num
        self.num_heads=num_heads
        self.dropout=dropout
        self.norm=norm
        self.company_emb=torch.FloatTensor(com_initial_emb)
        self.person_emb=torch.FloatTensor(person_initial_emb)

        self.riskinfo=RiskInfo(input_dim,company_num,cause_type_num,court_type_num,category_num,time_label_num=time_label_num)
        self.hypergnn=HyperGNN(input_dim,output_dim,num_layer=1)
        self.hetegnn=nn.ModuleList()
        for i in range(5):
            if i==0:
                self.hetegnn.append(HeteGNN(input_dim,output_dim,rel_num))
            else:
                self.hetegnn.append(HeteGNN(output_dim,output_dim,rel_num))

        self.company_proj=nn.Linear(32,input_dim,bias=False)
        self.person_proj=nn.Linear(32,input_dim,bias=False)

        self.risk_proj=nn.Linear(input_dim+23,input_dim,bias=False)
        self.info_proj=nn.Linear(output_dim,output_dim,bias=False)

        self.final_proj=nn.Sequential(nn.Linear(input_dim,output_dim,bias=False),nn.ReLU(),nn.Linear(output_dim,output_dim,bias=False))
        self.alpha=torch.ones((1))


    # risk data: dict-->{company_index:[[cause type, court type, category, time(months),time_label],...] }
    # company attribute information: np.array()-->[[register_captial, paid_captial, set up time(months)]]
    # graph: edge index:[sour,tar].T -->2xN; edge type: [,,...,] -->N; edge weight:[,,...,]-->N
    # hyper graph: dict:{industry:{ind1:[...],ind2:[...],...},area:{area1:[...],area2:[...],...},qualify:{qua1:[...],qua2:[...],...}}
    def forward(self,risk_data,company_attr,hete_graph,hyp_graph,idx,x):
        company_emb=self.company_proj(self.company_emb)
        person_emb=self.person_proj(self.person_emb)
        company_basic_info=torch.zeros((self.company_num,len(company_attr[0])))
       
        company_basic_info[idx]=torch.Tensor(company_attr)
        
        company_emb=torch.cat((company_emb,company_basic_info),dim=1)
        risk_info=self.riskinfo(risk_data)
        company_emb_info=self.risk_proj(torch.cat((company_emb,risk_info),dim=1))

        company_emb_hyper=self.hypergnn(company_emb_info,hyp_graph)

        edge_index,edge_type,edge_weight=hete_graph

        for i in range(5):
            if i==0:
                company_emb_hete,person_emb=self.hetegnn[i](company_emb_info,person_emb,edge_index,edge_type,edge_weight,self.company_num,self.person_num)
            else:
                company_emb_hete,person_emb=self.hetegnn[i](company_emb_hete,person_emb,edge_index,edge_type,edge_weight,self.company_num,self.person_num)
        company_emb_final=self.info_proj(company_emb_hyper+company_emb_hete)

        alpha=torch.sigmoid(self.alpha)
        company_emb_final=alpha*F.gelu(company_emb_final)+(1-alpha)*self.final_proj(company_emb_info)

        return company_emb_final[idx]
