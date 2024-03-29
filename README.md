# ComRisk
This repository is for paper "Combining intra-risk and contagion risk for enterprise bankruptcy prediction using graph neural networks" [Paper](https://www.sciencedirect.com/science/article/abs/pii/S0020025523016675?dgcid=author)

## Reference

If you find this work useful, please cite the following in your manuscript:



    @article{wei2024combining,
      title={Combining Intra-Risk and Contagion Risk for Enterprise Bankruptcy Prediction Using Graph Neural Networks},
      author={Wei, Shaopeng and Lv, Jia and Guo, Yu and Yang, Qing and Chen, Xingyan and Zhao, Yu and Li, Qing and Zhuang, Fuzhen and Kou, Gang},
      journal={Information Sciences},
      pages={120081},
      year={2024},
      issn = {0020-0255},
      doi = {https://doi.org/10.1016/j.ins.2023.120081},
      url = {https://www.sciencedirect.com/science/article/pii/S0020025523016675},
      publisher={Elsevier}
    }

    

## SMEsD
The SMEsD consists of 3,976 SMEs and related persons in China from 2014 to 2021, which constitutes a multiplex enterprise knowledge graph. All enterprises are associated with their basic business information and lawsuit events spanning from 2000 to 2021. Specifically, the enterprise business information includes registered capital,  paid-in capital and established time. Each lawsuit consists of the associated plaintiff, defendant, subjects, court level, result and timestamp.

We split the SMEsD into train, valid and test dataset according to the enterprise bankruptcy time, i.e., 2014-2018 for train dataset, 2019 for valid dataset and 2020-2021 for test dataset.

For more information about the dataset, please refer to "SMEsD.md".

## Code
To reproduce the result in the paper, you can run

    python train.py 

If you want to reproduce pre-trained embedding, you can run

    python metapath2vec.py
 
  noted that you should firstly substitute the `samply.py` in PyG with the modified version in this repository.

All the experiments are under the enviroment with:

python 3.7.10

pytorch 1.8.1 (cpu version)

torch-geometric 1.7.0





