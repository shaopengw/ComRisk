# SMEsD (V1.0) (update in 2022-07-01)
## Introduction
The dataset is used in paper " Combining Enterprise Intra-Risk with Contagion Risk for Bankruptcy Prediction via Graph Neural Networks" [ArXiv](https://arxiv.org/abs/2202.03874)

The SMEsD consists of 3,976 SMEs and related persons in China from 2014 to 2021, which constitutes a multiplex enterprise knowledge graph. All enterprises are associated with their basic business information and lawsuit events spanning from 2000 to 2021. Specifically, the enterprise business information includes registered capital, paid-in capital and established time. Each lawsuit consists of the associated plaintiff, defendant, subjects, court level, result and timestamp.

We split the SMEsD into train, valid and test dataset according to the enterprise bankruptcy time, i.e., 2014-2018 for train dataset, 2019 for valid dataset and 2020-2021 for test dataset.

## FieldName Description
### Load risk data
justification information:
dict-->{company_index:[[cause type, court type, result category, time (months),time_label],...] }

#### Cause Type
We divide all the causes into 11 catetories, considering both the similarity of various casue type and their frequency in our dataset.

    0:"金融借款合同纠纷","借款合同纠纷","小额借款合同纠纷","借款合同","金融不良债权追偿纠纷","企业借贷纠纷","民间借贷纠纷","追偿权纠纷",
    1:"买卖合同纠纷","买卖合同",
    2:"租赁合同纠纷",
    3:"工程合同纠纷","建设工程施工合同纠纷","建设工程分包合同纠纷","加工合同纠纷","运输合同纠纷",
    4:"债权纠纷","债权转让合同纠纷","债务转移合同纠纷","债权人代位权纠纷",
    5:"其他合同","合同纠纷","不当得利纠纷","商品房预售合同纠纷","劳务合同纠纷","建设工程合同纠纷","物业服务合同纠纷","商品房销售合同纠纷","旅游合同纠纷","委托合同纠纷","劳动合同纠纷",
    6:"知识产权与竞争纠纷","知识产权合同纠纷","知识产权权属、侵权纠纷","侵权纠纷","不正当竞争纠纷","侵害商标权纠纷",
    7:"劳动争议、人事争议","劳动争议","人事争议","追索劳动报酬纠纷",
    8:"侵权责任纠纷","事故责任纠纷",
    9:"与公司、证券、保险、票据等有关的民事纠纷","与企业有关的纠纷","与公司有关的纠纷","与破产有关的纠纷","期货交易纠纷","信托纠纷","保险纠纷","票据纠纷","信用证纠纷","股东损害公司债权人利益责任纠纷","职工破产债权确认纠纷","股权转让纠纷","破产债权确认纠纷","破产撤销权纠纷","债权人撤销权纠纷","普通破产债权确认纠纷","提供劳务者受害责任纠纷","追收未缴出资纠纷",
    10:"海事海商纠纷","通海水域污染损害责任纠纷","物权保护纠纷","所有权纠纷","用益物权纠纷","用益物权纠","担保物权纠纷","物权纠纷","损害赔偿纠纷","婚姻家庭纠纷","继承纠纷","人格权纠纷"


#### Court Type
There are four court types in China, indexs of which are shown as follows:

    "Grassroots people’ court": 0
    "Intermediate people’s court": 1
    "Higher people’s  court": 2
    "Supreme court": 3


#### Result Category
There are four types of results for an enterprise in a lawsuit,indexs of which are shown as follows:


    "plaintiff won": 0
    "plaintiff failed": 1
    "defendant won": 2
    "defendant failed": 3


#### Time
The time is counted by months, which denotes the interval from observation time.Specifically, for bankruptcy enterprises, the time means how long it is from a lawsuit sentence time to the enterprise's bankruptcy. For survival enterprises, the observation time are set December,2021.




### Load company attribute information: 
np.array()-->[[register_captial, paid_captial, set up time (months)]] 

Note: Similar to the meaning in justification information, the set up time here means how long it is from the set up time of an enterprise to the observation time.

### Load heterogeneous graph
graph: 

edge index:[sour,tar].T -->2xN;

edge type: [,,...,] -->N;

edge weight:[,,...,]-->N;

#### Edge Index
Company and person index are in same sequence, company first, then person. The indexs and corresponding original names of nodes can be got though:

    pandas.read_pickle('node2index.pkl')


#### Edge type
There are 20 types of relations in the dataset. We just utilize 0-11 in our paper.

    "supervised": 0,  # c->p
    "supervise": 1,  # p->c
    "executed": 2,  # c->p
    "execute": 3,  # p->c
    "stakeholderd": 4,  # c->p
    "stakeholder": 5,  # p->c
    "invest_CC": 6,  # c->C
    "invested_CC": 7,  # c->C
    "invest_CP": 8,  # p->c
    "invested_CP": 9,  # c->p
    "branch": 10,  # c->C
    "branched": 11,  # c->C
    "loan": 12,  # c->C
    "deal": 13,  # c->C
    "rent": 14,  # c->C
    "creditor": 15,  # c->C
    "loaned": 16,  # c->C
    "dealed": 17,  # c->C
    "rented": 18,  # c->C
    "creditored": 19,  # c->C



### Load hyper graph
hyper graph:

 dict:{industry:{ind1:[...],ind2:[...],...},
 area:{area1:[...],area2:[...],...},qualify:{qua1:[...],qua2:[...],...}}

Note: same for train/valid/test dataset



### load label
Label:
[1,0,1,...]

Note: 1 means enterprise bankruptcy, 0 means surviving.

## Contact
If you have any questions about the dataset or the paper, feel free to contact me via:

    weishaopeng1997@gmail.com


## Citation
If you use find this dataset useful, please cite it in your work as follow:

    @article{zhao2022fisrebp,
    title={FisrEbp: Enterprise Bankruptcy Prediction via Fusing its Intra-risk and Spillover-Risk},
    author={Zhao, Yu and Wei, Shaopeng and Guo, Yu and Yang, Qing and Kou, Gang},
    journal={arXiv preprint arXiv:2202.03874},
    year={2022}
}
