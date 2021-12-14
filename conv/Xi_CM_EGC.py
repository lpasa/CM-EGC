import torch
from scipy.special import factorial
from dgl import function as fn
from dgl.base import DGLError
from utils.utils_method import SimpleMultiLayerNN
from torch.nn.init import ones_

class XiCMEGC(torch.nn.Module):
    def __init__(self,
                 in_feats,
                 out_feats,
                 k=1,
                 n_pc=1,
                 n_heads=1,
                 n_layers_head=1,
                 bias=True,
                 norm=None,
                 allow_zero_in_degree=False):
        super(XiCMEGC, self).__init__()
        self._in_feats = in_feats
        self._out_feats = out_feats
        self._k =k
        self._n_pc=n_pc
        self._n_heads=n_heads
        self._n_layers_heads=n_layers_head
        self._cached_h = None
        self._cached_VS = None
        self._norm = norm
        self._allow_zero_in_degree = allow_zero_in_degree

        self.fc = torch.nn.Linear(in_feats, out_feats, bias=bias)
        self.heads_fun= torch.nn.ModuleList()

    def init_conv(self, graph, feat):
        with graph.local_scope():
            if not self._allow_zero_in_degree:
                if (graph.in_degrees() == 0).any():
                    raise DGLError('There are 0-in-degree nodes in the graph, '
                                   'output for those nodes will be invalid. '
                                   'This is harmful for some applications, '
                                   'causing silent performance regression. '
                                   'Adding self-loop on the input graph by '
                                   'calling `g = dgl.add_self_loop(g)` will resolve '
                                   'the issue. Setting ``allow_zero_in_degree`` '
                                   'to be `True` when constructing this module will '
                                   'suppress the check and let the code run.')
            if self._cached_h is None:
                feat_list = []

                # compute normalization
                degs = graph.in_degrees().float().clamp(min=1)
                norm = torch.pow(degs, -0.5)
                norm = norm.to(feat.device).unsqueeze(1)

                feat_list.append(feat.float())
                for i in range(self._k):
                    feat = feat * norm
                    feat = feat.float()
                    graph.ndata['h'] = feat
                    graph.update_all(fn.copy_u('h', 'm'),
                                     fn.sum('m', 'h'))
                    feat = graph.ndata.pop('h')
                    feat = feat * norm
                    feat_list.append(feat)

                self._cached_h = feat_list
            else:
                feat_list=self._cached_h
            if self._cached_VS is None:
                feat_0tok = torch.cat(feat_list, dim=1)
                V, S, _ = torch.svd(feat_0tok)
                S = S[0:self._n_pc]
                V = V[:,0:self._n_pc]

                cumsum = torch.cumsum(S ** 2, dim=0)
                avg_head_pc = cumsum[-1] / self._n_heads
                last_index = 0
                self.head_index = []
                for head in range(self._n_heads):


                    lambda_input_size = torch.nonzero(cumsum >= avg_head_pc * (head + 1))

                    if len(lambda_input_size)>0:
                        lambda_input_size=lambda_input_size[0]
                    else:
                        lambda_input_size = cumsum.shape[0] -1

                    if lambda_input_size-last_index <= 0:
                        lambda_input_size =last_index + 1

                    self.head_index.append((last_index, lambda_input_size))
                    if self._n_layers_heads <=1:
                        self.heads_fun.append(
                            torch.nn.Linear(lambda_input_size - last_index, 1).to(feat_list[0].device))
                    else:
                        self.heads_fun.append(
                            SimpleMultiLayerNN(lambda_input_size-last_index, [self._k]*self._n_layers_heads, 1,
                                               hidden_act_fun=torch.nn.Tanh()).to(feat_list[0].device))
                    last_index = lambda_input_size
                self._cached_head_input = torch.mm(V,torch.diag(S)).to(feat_list[0].device)
        torch.cuda.empty_cache()

    def reset_parameters(self):
        self.fc.reset_parameters()
        for head_fun in self.heads_fun:
            head_fun.reset_parameters()

    def forward(self, graph, feat):

        with graph.local_scope():
            if not self._allow_zero_in_degree:
                if (graph.in_degrees() == 0).any():
                    raise DGLError('There are 0-in-degree nodes in the graph, '
                                   'output for those nodes will be invalid. '
                                   'This is harmful for some applications, '
                                   'causing silent performance regression. '
                                   'Adding self-loop on the input graph by '
                                   'calling `g = dgl.add_self_loop(g)` will resolve '
                                   'the issue. Setting ``allow_zero_in_degree`` '
                                   'to be `True` when constructing this module will '
                                   'suppress the check and let the code run.')


            feat_list = self._cached_h
            head_input = self._cached_head_input
            result = torch.zeros(feat_list[0].shape[0], self._out_feats).to(feat_list[0].device)
            beta = torch.zeros(feat_list[0].shape[0], 1).to(feat_list[0].device)


            for head, (start_index, end_index) in zip(self.heads_fun, self.head_index):
                beta += head(head_input[:, start_index:end_index])

            for i, k_feat in enumerate(feat_list):
                result += self.fc(k_feat * (beta.pow(i) / factorial(i)))
            if self._norm is not None:
                result = self._norm(result)

        torch.cuda.empty_cache()
        return result


class XiCMEGC_Tanh(XiCMEGC):
    def __init__(self,
                 in_feats,
                 out_feats,
                 k=1,
                 n_pc=1,
                 n_heads=1,
                 n_layers_head=1,
                 bias=True,
                 norm=None,
                 allow_zero_in_degree=False):
        super(XiCMEGC_Tanh, self).__init__(
            in_feats,
            out_feats,
            k,
            n_pc,
            n_heads,
            n_layers_head,
            bias,
            norm,
            allow_zero_in_degree)

        self.layer_act=torch.nn.Tanh()
        self.beta = []
        self.head_act_fun = torch.nn.Tanh()


    def init_conv(self, graph, feat):
        with graph.local_scope():
            if not self._allow_zero_in_degree:
                if (graph.in_degrees() == 0).any():
                    raise DGLError('There are 0-in-degree nodes in the graph, '
                                   'output for those nodes will be invalid. '
                                   'This is harmful for some applications, '
                                   'causing silent performance regression. '
                                   'Adding self-loop on the input graph by '
                                   'calling `g = dgl.add_self_loop(g)` will resolve '
                                   'the issue. Setting ``allow_zero_in_degree`` '
                                   'to be `True` when constructing this module will '
                                   'suppress the check and let the code run.')
            if self._cached_h is None:
                feat_list = []

                # compute normalization
                degs = graph.in_degrees().float().clamp(min=1)
                norm = torch.pow(degs, -0.5)
                norm = norm.to(feat.device).unsqueeze(1)

                feat_list.append(feat.float())
                for i in range(self._k):
                    feat = feat * norm
                    feat = feat.float()
                    graph.ndata['h'] = feat
                    graph.update_all(fn.copy_u('h', 'm'),
                                     fn.sum('m', 'h'))
                    feat = graph.ndata.pop('h')
                    feat = feat * norm
                    feat_list.append(feat)

                self._cached_h = feat_list
            else:
                feat_list = self._cached_h
            if self._cached_VS is None:
                feat_0tok = torch.cat(feat_list, dim=1)
                V, S, _ = torch.svd(feat_0tok)
                S = S[0:self._n_pc]
                V = V[:, 0:self._n_pc]

                cumsum = torch.cumsum(S ** 2, dim=0)
                avg_head_pc = cumsum[-1] / self._n_heads
                last_index = 0
                self.head_index = []
                for head in range(self._n_heads):

                    lambda_input_size = torch.nonzero(cumsum >= avg_head_pc * (head + 1))

                    if len(lambda_input_size) > 0:
                        lambda_input_size = lambda_input_size[0]
                    else:
                        lambda_input_size = cumsum.shape[0] - 1

                    if lambda_input_size - last_index <= 0:
                        lambda_input_size = last_index + 1

                    self.head_index.append((last_index, lambda_input_size))
                    if self._n_layers_heads <= 1:

                        self.heads_fun.append(
                            torch.nn.Linear(lambda_input_size - last_index, 1).to(feat_list[0].device))
                    else:
                        self.heads_fun.append(
                            SimpleMultiLayerNN(lambda_input_size - last_index, [self._k] * self._n_layers_heads, 1,
                                               hidden_act_fun=torch.nn.Tanh()).to(feat_list[0].device))
                    last_index = lambda_input_size
                    self.beta.append(torch.nn.Parameter(torch.FloatTensor(1)).to(feat_list[0].device))
                self._cached_head_input = torch.mm(V, torch.diag(S)).to(feat_list[0].device)
        self.reset_parameters()
        torch.cuda.empty_cache()

    def reset_parameters(self):
        self.fc.reset_parameters()
        for head_fun in self.heads_fun:
            head_fun.reset_parameters()
        for beta in self.beta:
            ones_(beta)


    def forward(self, graph, feat):

        with graph.local_scope():
            if not self._allow_zero_in_degree:
                if (graph.in_degrees() == 0).any():
                    raise DGLError('There are 0-in-degree nodes in the graph, '
                                   'output for those nodes will be invalid. '
                                   'This is harmful for some applications, '
                                   'causing silent performance regression. '
                                   'Adding self-loop on the input graph by '
                                   'calling `g = dgl.add_self_loop(g)` will resolve '
                                   'the issue. Setting ``allow_zero_in_degree`` '
                                   'to be `True` when constructing this module will '
                                   'suppress the check and let the code run.')

            feat_list = self._cached_h
            head_input = self._cached_head_input
            result = torch.zeros(feat_list[0].shape[0], self._out_feats).to(feat_list[0].device)
            beta = torch.zeros(feat_list[0].shape[0], 1).to(feat_list[0].device)

            for beta_j,( head, (start_index, end_index)) in zip(self.beta, zip(self.heads_fun, self.head_index)):
                beta += beta_j * self.head_act_fun(head(head_input[:, start_index:end_index]))

            for i, k_feat in enumerate(feat_list):
                result += self.fc(k_feat * (beta.pow(i) / factorial(i)))
            if self._norm is not None:
                result = self._norm(result)

        torch.cuda.empty_cache()
        return result


class XiCMEGC_Tanh_inc_deep(XiCMEGC_Tanh):
    def init_conv(self, graph, feat):
        with graph.local_scope():
            if not self._allow_zero_in_degree:
                if (graph.in_degrees() == 0).any():
                    raise DGLError('There are 0-in-degree nodes in the graph, '
                                   'output for those nodes will be invalid. '
                                   'This is harmful for some applications, '
                                   'causing silent performance regression. '
                                   'Adding self-loop on the input graph by '
                                   'calling `g = dgl.add_self_loop(g)` will resolve '
                                   'the issue. Setting ``allow_zero_in_degree`` '
                                   'to be `True` when constructing this module will '
                                   'suppress the check and let the code run.')
            if self._cached_h is None:
                feat_list = []

                # compute normalization
                degs = graph.in_degrees().float().clamp(min=1)
                norm = torch.pow(degs, -0.5)
                norm = norm.to(feat.device).unsqueeze(1)

                feat_list.append(feat.float())
                for i in range(self._k):
                    feat = feat * norm
                    feat = feat.float()
                    graph.ndata['h'] = feat
                    graph.update_all(fn.copy_u('h', 'm'),
                                     fn.sum('m', 'h'))
                    feat = graph.ndata.pop('h')
                    feat = feat * norm
                    feat_list.append(feat)

                self._cached_h = feat_list
            else:
                feat_list = self._cached_h
            if self._cached_VS is None:
                feat_0tok = torch.cat(feat_list, dim=1)
                V, S, _ = torch.svd(feat_0tok)
                S = S[0:self._n_pc]
                V = V[:, 0:self._n_pc]

                cumsum = torch.cumsum(S ** 2, dim=0)
                avg_head_pc = cumsum[-1] / self._n_heads
                last_index = 0
                self.head_index = []
                for head in range(self._n_heads):

                    lambda_input_size = torch.nonzero(cumsum >= avg_head_pc * (head + 1))

                    if len(lambda_input_size) > 0:
                        lambda_input_size = lambda_input_size[0]
                    else:
                        lambda_input_size = cumsum.shape[0] - 1

                    if lambda_input_size - last_index <= 0:
                        lambda_input_size = last_index + 1

                    self.head_index.append((last_index, lambda_input_size))
                    if head == 0 :

                        self.heads_fun.append(
                            torch.nn.Linear(lambda_input_size - last_index, 1).to(feat_list[0].device))
                    else:
                        self.heads_fun.append(
                            SimpleMultiLayerNN(lambda_input_size - last_index, [self._k] * (head), 1,
                                               hidden_act_fun=torch.nn.Tanh()).to(feat_list[0].device))
                    last_index = lambda_input_size
                    self.beta.append(torch.nn.Parameter(torch.FloatTensor(1)).to(feat_list[0].device))
                self._cached_head_input = torch.mm(V, torch.diag(S)).to(feat_list[0].device)
        self.reset_parameters()
        torch.cuda.empty_cache()

class XiCMEGC_Tanh_same_input(XiCMEGC_Tanh):

    def init_conv(self, graph, feat):
        with graph.local_scope():
            if not self._allow_zero_in_degree:
                if (graph.in_degrees() == 0).any():
                    raise DGLError('There are 0-in-degree nodes in the graph, '
                                   'output for those nodes will be invalid. '
                                   'This is harmful for some applications, '
                                   'causing silent performance regression. '
                                   'Adding self-loop on the input graph by '
                                   'calling `g = dgl.add_self_loop(g)` will resolve '
                                   'the issue. Setting ``allow_zero_in_degree`` '
                                   'to be `True` when constructing this module will '
                                   'suppress the check and let the code run.')
            if self._cached_h is None:
                feat_list = []

                # compute normalization
                degs = graph.in_degrees().float().clamp(min=1)
                norm = torch.pow(degs, -0.5)
                norm = norm.to(feat.device).unsqueeze(1)

                feat_list.append(feat.float())
                for i in range(self._k):
                    # norm = D^-1/2
                    feat = feat * norm
                    feat = feat.float()
                    graph.ndata['h'] = feat
                    graph.update_all(fn.copy_u('h', 'm'),
                                     fn.sum('m', 'h'))
                    feat = graph.ndata.pop('h')
                    feat = feat * norm
                    feat_list.append(feat)

                self._cached_h = feat_list
            else:
                feat_list = self._cached_h
            if self._cached_VS is None:
                feat_0tok = torch.cat(feat_list, dim=1)
                V, S, _ = torch.svd(feat_0tok)
                S = S[0:self._n_pc]
                V = V[:, 0:self._n_pc]

                last_index = 0
                self.head_index = []
                for head in range(self._n_heads):

                    if self._n_layers_heads <= 1:

                        self.heads_fun.append(
                            torch.nn.Linear(self._n_pc, 1).to(feat_list[0].device))
                    else:
                        self.heads_fun.append(
                            SimpleMultiLayerNN(self._n_pc, [self._k] * self._n_layers_heads, 1,
                                               hidden_act_fun=torch.nn.Tanh()).to(feat_list[0].device))
                    self.beta.append(torch.nn.Parameter(torch.FloatTensor(1)).to(feat_list[0].device))
                self._cached_head_input = torch.mm(V, torch.diag(S)).to(feat_list[0].device)
        self.reset_parameters()
        torch.cuda.empty_cache()

    def forward(self, graph, feat):

        with graph.local_scope():
            if not self._allow_zero_in_degree:
                if (graph.in_degrees() == 0).any():
                    raise DGLError('There are 0-in-degree nodes in the graph, '
                                   'output for those nodes will be invalid. '
                                   'This is harmful for some applications, '
                                   'causing silent performance regression. '
                                   'Adding self-loop on the input graph by '
                                   'calling `g = dgl.add_self_loop(g)` will resolve '
                                   'the issue. Setting ``allow_zero_in_degree`` '
                                   'to be `True` when constructing this module will '
                                   'suppress the check and let the code run.')

            feat_list = self._cached_h
            head_input = self._cached_head_input
            result = torch.zeros(feat_list[0].shape[0], self._out_feats).to(feat_list[0].device)
            beta = torch.zeros(feat_list[0].shape[0], 1).to(feat_list[0].device)

            for beta_j,head in zip(self.beta, self.heads_fun):
                beta += beta_j * self.head_act_fun(head(head_input))




            for i, k_feat in enumerate(feat_list):
                result += self.fc(k_feat * (beta.pow(i) / factorial(i)))
            if self._norm is not None:
                result = self._norm(result)

        torch.cuda.empty_cache()
        return result
