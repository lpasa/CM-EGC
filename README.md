# Compact Graph Neural Network Models for Node Classification

Luca Pasa, Nicolò Navarin, Alessandro Sperduti

## Abstract
Recent research on graph convolutional networks tend to increase the complexity and non-linearity of graph convolution operators.
Many of these operators result in models exploiting a huge number of learnable parameters, that have to be tuned during the training phase. This aspect makes the application of these approaches on huge datasets challenging due to the considerable computational time required by the training phase. On the other hand, having a huge number of parameters limits the applicability of the models in many node classification problems, in particular in the semi-supervised setting where the number of samples for which the target is known is limited to a few dozen nodes.

In this paper, we propose a simple and efficient operator dubbed Compact Multi-head Exponential Graph Convolution (CM-EGC).  To limit the number of learnable parameters, the proposed model exploits a compact and structurally meaningful representation of the input node features resorting to truncated SVD matrix decomposition, while the expressiveness of the model is ensured by adopting a multi-head attention-based gating mechanism. We evaluated the CM-EGC on semi-supervised and fully-supervised node classification tasks considering 3 well-known benchmark datasets. The results show that even using an extremely compact model, the classification performance of the proposed approach is comparable and sometimes better than the state of the art.
