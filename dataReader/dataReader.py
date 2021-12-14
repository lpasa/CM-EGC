import torch
import dgl
from dgl import DGLGraph
from dgl.data import citation_graph as citegrh
from sklearn.model_selection import train_test_split

def DGLDatasetReader(dataset_name,self_loops,device=None):

    data = load_data(dataset_name,self_loops)
    if dataset_name == 'reddit':
        g = data.graph.int().to(device)
        # normalization
        degs = g.in_degrees().float()
        norm = torch.pow(degs, -0.5)
        norm[torch.isinf(norm)] = 0
        g.ndata['norm'] = norm.unsqueeze(1)


    else:
        g = DGLGraph(data.graph).to(device)
        # normalization
        degs = g.in_degrees().float()
        norm = torch.pow(degs, -0.5)
        norm[torch.isinf(norm)] = 0
        norm = norm.to(device)
        g.ndata['norm'] = norm.unsqueeze(1)
    # add self loop
    if self_loops:
        g = dgl.remove_self_loop(g)
        g = dgl.add_self_loop(g)
    return g,torch.FloatTensor(data.features),torch.LongTensor(data.labels),data.num_labels,\
           torch.ByteTensor(data.train_mask),torch.ByteTensor(data.test_mask),torch.ByteTensor(data.val_mask)

def load_data(dataset_name,self_loops=False):
    if dataset_name == 'cora':
        return citegrh.load_cora()
    elif dataset_name == 'citeseer':
        return citegrh.load_citeseer()
    elif dataset_name== 'pubmed':
        return citegrh.load_pubmed()
    else:
        raise ValueError('Unknown dataset: {}'.format(dataset_name))


def get_rnd_full_masks(n_sample, test_valid_perc, rnd_seed=0):
    data_indices=range(n_sample)
    idx_train, idx_test = train_test_split(data_indices, test_size=test_valid_perc, random_state=rnd_seed)  #
    idx_train, idx_val = train_test_split(idx_train, test_size=len(idx_test), random_state=rnd_seed)
    train_mask=torch.zeros(n_sample).long()
    test_mask = torch.zeros(n_sample).long()
    val_mask = torch.zeros(n_sample).long()
    return train_mask.scatter_(0,torch.LongTensor(idx_train),1).bool(),\
           test_mask.scatter_(0,torch.LongTensor(idx_test),1).bool(),\
           val_mask.scatter_(0,torch.LongTensor(idx_val),1).bool()
