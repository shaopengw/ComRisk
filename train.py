# coding: utf-8
from os import times
import pandas as pd
import torch
from tqdm.utils import SimpleTextIOWrapper

from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier


from gnn import RiskGNN
from utils import *
from sklearn.metrics import accuracy_score as acc
from sklearn.metrics import recall_score as rec
from sklearn.metrics import precision_score as pre
from sklearn.metrics import f1_score as f1
from sklearn.metrics import roc_auc_score as roc

import time
import argparse

parser = argparse.ArgumentParser(description='Training GNN')
'''
    Dataset arguments
'''
parser.add_argument('--data_dir', type=str, default='./data',
                    help='The address of preprocessed graph.')
parser.add_argument('--model_dir', type=str, default='.\model_save',
                    help='The address for storing the models and optimization results.')
parser.add_argument('--cuda', type=int, default=0,
                    help='Avaiable GPU ID')

'''
   Model arguments
'''
parser.add_argument('--conv_name', type=str, default='riskgnn',
                    choices=['riskgnn'],
                    help='The name of GNN filter.')
parser.add_argument('--input_dim', type=int, default=16,
                    help='Number of input dimension')
parser.add_argument('--output_dim', type=int, default=12,
                    help='Number of output dimension')
parser.add_argument('--n_heads', type=int, default=1,
                    help='Number of attention head')
parser.add_argument('--n_layers', type=int, default=1,
                    help='Number of HeteGAT layers')
parser.add_argument('--dropout', type=float, default=0.2,
                    help='Dropout ratio')

'''
    Optimization arguments
'''
parser.add_argument('--optimizer', type=str, default='adam',
                    choices=['adamw', 'adam', 'sgd', 'adagrad'],
                    help='optimizer to use.')

parser.add_argument('--n_epoch', type=int, default=500,
                    help='Number of epoch to run')
parser.add_argument('--clip', type=float, default=0.25,
                    help='Gradient Norm Clipping')
parser.add_argument('--weight_decay', type=float, default=1e-4,
                    help='weight decay of adamw ')

args = parser.parse_args()

device=torch.device("cpu")
set_random_seed(14)

criterion = torch.nn.CrossEntropyLoss()
#load risk data
##load justification information:  dict-->{company_index:[[cause type, court type, result category, time(months),time_label],...] }


# ##load company attribute information: np.array()-->[[register_captial, paid_captial, set up time(months)]] ------------same for train/valid/test (only set up time are different)---not sure


# #load heterogeneous graph
# ## graph: edge index:[sour,tar].T -->2xN; edge type: [,,...,] -->N; edge weight:[,,...,]-->N; noted that company and person index are in same sequence, company first, then person 


# #load hyper graph
# ##hyper graph: dict:{industry:{ind1:[...],ind2:[...],...},area:{area1:[...],area2:[...],...},qualify:{qua1:[...],qua2:[...],...}} ------------same for train/valid/test



# #load label
# ##label: [conpany_index,laebl]    company_index:[idx1,idx2,...]   label:[1,0,1,...], both are np.array()


###justification dict
total_company_num=3976
person_num=2405
court_type=4
category=4
time_label_num=5

train_company_num=2816
valid_company_num=721
test_company_num=491


rel_num=12
cause_type_num=11


path='data_graph'
train_data=pd.read_pickle('./data/train_data.pkl')
valid_data=pd.read_pickle('./data/validate_data.pkl')
test_data=pd.read_pickle('./data/test_data.pkl')
split_data_idx=pd.read_pickle('./data/split_data_idx.pkl')

train_risk_data,train_company_attr,train_hete_graph,train_hyp_graph,train_label=train_data
valid_risk_data,valid_company_attr,valid_hete_graph,valid_hyp_graph,valid_label=valid_data
test_risk_data,test_company_attr,test_hete_graph,test_hyp_graph,test_label=test_data
train_idx,valid_idx,test_idx=split_data_idx


com_initial_emb,person_initial_emb=pd.read_pickle('./data/meta_emb.pkl')


x_train=initializae_company_info(train_risk_data,train_company_attr,train_company_num,cause_type_num,court_type,category,train_idx)
x_valid=initializae_company_info(valid_risk_data,valid_company_attr,valid_company_num,cause_type_num,court_type,category,valid_idx)
x_test=initializae_company_info(test_risk_data,test_company_attr,test_company_num,cause_type_num,court_type,category,test_idx)


gnn=RiskGNN(args.input_dim,args.output_dim,
    total_company_num,person_num,rel_num,
    cause_type_num, device,com_initial_emb,person_initial_emb,
    court_type,category,time_label_num,
    num_heads=1,dropout=0.2,norm=True)

classifier = Classifier(args.output_dim, 2).to(device)
model = nn.Sequential(gnn, classifier)

if args.optimizer == 'adamw':
    optimizer = torch.optim.AdamW(model.parameters(),weight_decay=args.weight_decay)
elif args.optimizer == 'adam':
    optimizer = torch.optim.Adam(model.parameters(),lr=0.01)
elif args.optimizer == 'sgd':
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
elif args.optimizer == 'adagrad':
    optimizer = torch.optim.Adagrad(model.parameters())

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 20, eta_min=1e-6)

['industry', 'area', 'qualify']
train_hyp=[]
for i in ['industry', 'area', 'qualify']:
    train_hyp+=[gen_attribute_hg(total_company_num, train_hyp_graph[i], X=None)]
valid_hyp=[]
for i in ['industry', 'area', 'qualify']:
    valid_hyp+=[gen_attribute_hg(total_company_num, valid_hyp_graph[i], X=None)]
test_hyp=[]
for i in ['industry', 'area', 'qualify']:
    test_hyp+=[gen_attribute_hg(total_company_num, test_hyp_graph[i], X=None)]


def test():
    best_model = torch.load('./model_save/%s.pkl'%(args.conv_name))
    best_model.eval()
    gnn, classifier = best_model
    with torch.no_grad():
        company_emb=gnn.forward(test_risk_data,test_company_attr,test_hete_graph,test_hyp,test_idx,x_test)

        res = classifier.forward(company_emb)

        pred=res.argmax(dim=1)
        ac=acc(test_label,pred)
        pr=pre(test_label,pred)
        re=rec(test_label,pred)
        f=f1(test_label,pred)
        rc=roc(test_label,res[:,1])

        print('Best Test Acc: %.4f Best Test Pre: %.4f Best Test Recall: %.4f Best Test F1: %.4f Best Test ROC: %.4f' % (ac,pr,re,f,rc))
        print(sum(pred))


best_acc=0
best_f1=0


for epoch in np.arange(args.n_epoch):

    st=time.time()

    '''
        Train 
    '''
    model.train()
    train_losses = []
    # torch.cuda.empty_cache()
    company_emb=gnn.forward(train_risk_data,train_company_attr,train_hete_graph,train_hyp,train_idx,x_train)

    
    res = classifier.forward(company_emb)

    loss = criterion(res, torch.LongTensor(train_label))
    optimizer.zero_grad()
    loss.backward()

    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
    optimizer.step()

    train_losses += [loss.cpu().detach().tolist()]
    # train_step += 1
    scheduler.step()
    del res, loss

    '''
        Valid 
    '''
    model.eval()
    with torch.no_grad():
        company_emb=gnn.forward(valid_risk_data,valid_company_attr,valid_hete_graph,valid_hyp,valid_idx,x_valid)

        res = classifier.forward(company_emb)
        loss = criterion(res,torch.LongTensor(valid_label) )

        pred=res.argmax(dim=1)
        ac=acc(valid_label,pred)
        pr=pre(valid_label,pred)
        re=rec(valid_label,pred)
        f=f1(valid_label,pred)
        rc=roc(valid_label,res[:,1])

        if ac > best_acc and f>best_f1:
            best_acc = ac
            best_f1=f
            torch.save(model, './model_save/%s.pkl'%(args.conv_name))

            print('UPDATE!!!')


        et = time.time()
        print(("Epoch: %d (%.1fs)  LR: %.5f Train Loss: %.2f  Valid Loss: %.2f  Valid Acc: %.4f Valid Pre: %.4f  Valid Recall: %.4f Valid F1: %.4f  Valid Roc: %.4f"  ) % \
              (epoch, (et - st), optimizer.param_groups[0]['lr'], np.average(train_losses), \
               loss.cpu().detach().tolist(), ac,pr,re,f,rc))

        del res, loss

        if epoch+1==args.n_epoch:
            company_emb=gnn.forward(test_risk_data,test_company_attr,test_hete_graph,test_hyp,test_idx,x_test)

            res = classifier.forward(company_emb)

            pred=res.argmax(dim=1)
            ac=acc(test_label,pred)
            pr=pre(test_label,pred)
            re=rec(test_label,pred)
            f=f1(test_label,pred)
            rc=roc(test_label,res[:,1])
            
            print('Last Test Acc: %.4f Last Test Pre: %.4f Last Test Recall: %.4f Last Test F1: %.4f Last Test ROC: %.4f' % (ac,pr,re,f,rc))


'''
    Evaluate 
'''
# best_model = torch.load(os.path.join(args.model_dir, args.conv_name))
def test():
    best_model = torch.load('./model_save/%s.pkl'%(args.conv_name))
    best_model.eval()
    gnn, classifier = best_model
    with torch.no_grad():

        company_emb=gnn.forward(test_risk_data,test_company_attr,test_hete_graph,test_hyp,test_idx,x_test)

        res = classifier.forward(company_emb)

        pred=res.argmax(dim=1)
        ac=acc(test_label,pred)
        pr=pre(test_label,pred)
        re=rec(test_label,pred)
        f=f1(test_label,pred)
        rc=roc(test_label,res[:,1])

        print('Best Test Acc: %.4f Best Test Pre: %.4f Best Test Recall: %.4f Best Test F1: %.4f Best Test ROC: %.4f' % (ac,pr,re,f,rc))
        print(sum(pred))

test()
