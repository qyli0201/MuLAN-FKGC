# MuLAN: Multi-Level Attention-Enhanced Matching Network for Few-Shot Knowledge Graph Completion
This is the source code release of the following paper:
Qianyu Li, Bozheng Feng, Xiaoli Tang, Han Yu and Hengjie Song. MuLAN: Multi-Level Attention-Enhanced Matching Network for Few-Shot Knowledge Graph Completion

Recent years have witnessed increasing interest in the few-shot knowledge graph completion due to its potential to augment the coverage of few-shot relations in knowledge graphs. Existing methods often use the one-hop neighbor of entities to enhance the entity embeddings, and match the query instance and support set at the instance level. However, such methods cannot handle inter-neighbor interaction, local entity matching and the varying significance of feature dimensions. To bridge this gap, we propose the Multi-Level Attention-enhanced matching Network (MuLAN) for few-shot knowledge graph completion. In MuLAN, a multi-head self-attention neighbor encoder is designed to capture the inter-neighbor interaction and learn the entity embeddings. Then, entity-level attention and instance-level attention are responsible for matching the query instance and support set from the local and global perspectives, respectively, while feature-level attention is utilized to calculate the weights of the feature dimensions. Furthermore, we design a consistency constraint to ensure the support instance embeddings are close to each other. Extensive experiments based on two well-known datasets (i.e., NELL-One and Wiki-One) demonstrate significant advantages of MuLAN over 11 state-of-the-art competitors. Compared to the best-performing baseline, MuLAN achieves 14.5\% higher MRR and 13.3\% higher Hits@K on average.

![](https://github.com/qyli0201/MuLAN-FKGC/blob/main/model.pdf)


## Datasets
- [NELL-One](https://drive.google.com/file/d/1XXvYpTSTyCnN-PBdUkWBXwXBI99Chbps/view?usp=sharing): Unzip it to the directory ./MuLAN-FKGC/data/.

- [Wiki-One](https://drive.google.com/file/d/1_3HBJde2KVMhBgJeGN1-wyvW88gRU1iL/view?usp=sharing): Unzip it to the directory ./MuLAN-FKGC/data/.

## Pre-trained embeddings
- [NELL-One](https://drive.google.com/file/d/1XXvYpTSTyCnN-PBdUkWBXwXBI99Chbps/view?usp=sharing): Unzip it to the directory ./MuLAN-FKGC/data/NELL/embed/.

- [Wiki-One](https://drive.google.com/file/d/1_3HBJde2KVMhBgJeGN1-wyvW88gRU1iL/view?usp=sharing): Unzip it to the directory ./MuLAN-FKGC/data/Wiki/embed/.

## Run the code
` `` 
./run_nell.sh
./run_wiki.sh
` ``
