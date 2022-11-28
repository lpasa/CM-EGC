# Compact Graph Neural Network Models for Node Classification

Luca Pasa, Nicolò Navarin, Alessandro Sperduti

## Abstract
Recent research on graph convolutional networks tend to increase the complexity and non-linearity of graph convolution operators.
Many of these operators result in models exploiting a huge number of learnable parameters, that have to be tuned during the training phase. This aspect makes the application of these approaches on huge datasets challenging due to the considerable computational time required by the training phase. On the other hand, having a huge number of parameters limits the applicability of the models in many node classification problems, in particular in the semi-supervised setting where the number of samples for which the target is known is limited to a few dozen nodes.

In this paper, we propose a simple and efficient operator dubbed Compact Multi-head Exponential Graph Convolution (CM-EGC).  To limit the number of learnable parameters, the proposed model exploits a compact and structurally meaningful representation of the input node features resorting to truncated SVD matrix decomposition, while the expressiveness of the model is ensured by adopting a multi-head attention-based gating mechanism. We evaluated the CM-EGC on semi-supervised and fully-supervised node classification tasks considering 3 well-known benchmark datasets. The results show that even using an extremely compact model, the classification performance of the proposed approach is comparable and sometimes better than the state of the art.

Paper: https://dl.acm.org/doi/10.1145/3477314.3507100

If you find this code useful, please cite the following:

>@inproceedings{Pasa2022Compact,
author = {Pasa, Luca and Navarin, Nicol\`{o} and Sperduti, Alessandro},
title = {Compact Graph Neural Network Models for Node Classification},
year = {2022},
isbn = {9781450387132},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3477314.3507100},
doi = {10.1145/3477314.3507100},
booktitle = {Proceedings of the 37th ACM/SIGAPP Symposium on Applied Computing},
pages = {592–599},
numpages = {8},
keywords = {graph neural network, machine learning on graphs, deep learning, structured data, graph convolution},
location = {Virtual Event},
series = {SAC '22}
}
