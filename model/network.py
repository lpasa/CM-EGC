from torch import nn
import torch

class GCNetwork(nn.Module):
    def __init__(self,
                 g,
                 in_feats,
                 n_classes,
                 dropout,
                 k,
                 convLayer,
                 n_pc=1,
                 n_heads=1,
                 n_layers_head=1,
                 out_fun = nn.Softmax(dim=1),
                 device=None,
                 norm=None,
                 bias=False):
        super(GCNetwork, self).__init__()
        self.g=g


        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device=device
        self.layers = nn.ModuleList()#TODO:Cancella
        self.out_fun=out_fun
        self.dropout = nn.Dropout(p=dropout)
        self.layer = convLayer(in_feats, n_classes, k, n_pc=n_pc, n_heads=n_heads, n_layers_head=n_layers_head, bias=bias, norm=norm)

    def init_layer(self,features):
        self.layer.init_conv(self.g,features)

    def init_layer_from_file(self,features, lambda_input_sizes):
        self.layer.init_from_mem(self.g, features ,lambda_input_sizes)

    def init_layer_from_file_noPCA(self,features, lambda_input_sizes):
        self.layer.init_from_mem_no_PCA(self.g, features ,lambda_input_sizes)

    def forward(self,features):
        h = features
        h = self.dropout(h)
        h = self.layer(self.g,h)
        return self.out_fun(h),h

    def get_f(self,features):
        h = features
        h = self.dropout(h)
        f_exp = self.layer.get_hyper_par(self.g,h)
        return f_exp


