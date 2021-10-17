import dgl
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.multiprocessing as mp
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
import dgl.function as fn
import dgl.nn.pytorch as dglnn
import time
import argparse
from dgl.utils import expand_as_pair, check_eq_shape, dgl_warning
import os
from dgl.data.utils import _get_dgl_url, generate_mask_tensor, load_graphs, save_graphs, deprecate_property
from dgl.data.dgl_dataset import DGLBuiltinDataset
from _thread import start_new_thread
from functools import wraps
from dgl.data import register_data_args, load_data
import tqdm
import json
import traceback
import matplotlib.pyplot as plt
import scipy
import pdb
from dgl import backend as F1
from sklearn.preprocessing import StandardScaler  # !!!!!!!!!!
from dgl.convert import from_scipy
from sklearn.metrics import f1_score
import math
from pyinstrument import Profiler

epsilon = 1 - math.log(2)

def create_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return mask


def calc_f1(pred, batch_labels):
    if len(batch_labels.shape) > 1:
        # pred_labels = (pred.detach().cpu() > 0)

        pred_labels = nn.Sigmoid()(pred.detach().cpu())
        pred_labels[pred_labels > 0.5] = 1
        pred_labels[pred_labels <= 0.5] = 0

        # pred_labels = pred.detach().cpu()
        # pred_labels[pred_labels >= 0] = 1
        # pred_labels[pred_labels < 0] = 0
    else:
        pred_labels = pred.detach().cpu().argmax(dim=1)

    score = f1_score(batch_labels.cpu(), pred_labels, average="micro")

    return score


def process_graph_data(adj_full, adj_train, feats, class_map, role):
    """
    setup vertex property map for output classes, train/val/test masks, and feats
    """
    num_vertices = adj_full.shape[0]
    if isinstance(list(class_map.values())[0], list):
        num_classes = len(list(class_map.values())[0])
        class_arr = np.zeros((num_vertices, num_classes))
        for k, v in class_map.items():
            class_arr[k] = v
    else:
        num_classes = max(class_map.values()) - min(class_map.values()) + 1
        class_arr = np.zeros((num_vertices, num_classes))
        offset = min(class_map.values())
        for k, v in class_map.items():
            class_arr[k][v - offset] = 1
    return adj_full, adj_train, feats, class_arr, role


def load_data(prefix, normalize=True):
    """
    Load the various data files residing in the `prefix` directory.
    Files to be loaded:
        adj_full.npz        sparse matrix in CSR format, stored as scipy.sparse.csr_matrix
                            The shape is N by N. Non-zeros in the matrix correspond to all
                            the edges in the full graph. It doesn't matter if the two nodes
                            connected by an edge are training, validation or test nodes.
                            For unweighted graph, the non-zeros are all 1.
        adj_train.npz       sparse matrix in CSR format, stored as a scipy.sparse.csr_matrix
                            The shape is also N by N. However, non-zeros in the matrix only
                            correspond to edges connecting two training nodes. The graph
                            sampler only picks nodes/edges from this adj_train, not adj_full.
                            Therefore, neither the attribute information nor the structural
                            information are revealed during training. Also, note that only
                            a x N rows and cols of adj_train contains non-zeros. For
                            unweighted graph, the non-zeros are all 1.
        role.json           a dict of three keys. Key 'tr' corresponds to the list of all
                              'tr':     list of all training node indices
                              'va':     list of all validation node indices
                              'te':     list of all test node indices
                            Note that in the raw data, nodes may have string-type ID. You
                            need to re-assign numerical ID (0 to N-1) to the nodes, so that
                            you can index into the matrices of adj, features and class labels.
        class_map.json      a dict of length N. Each key is a node index, and each value is
                            either a length C binary list (for multi-class classification)
                            or an integer scalar (0 to C-1, for single-class classification).
        feats.npz           a numpy array of shape N by F. Row i corresponds to the attribute
                            vector of node i.
    Inputs:
        prefix              string, directory containing the above graph related files
        normalize           bool, whether or not to normalize the node features
    Outputs:
        adj_full            scipy sparse CSR (shape N x N, |E| non-zeros), the adj matrix of
                            the full graph, with N being total num of train + val + test nodes.
        adj_train           scipy sparse CSR (shape N x N, |E'| non-zeros), the adj matrix of
                            the training graph. While the shape is the same as adj_full, the
                            rows/cols corresponding to val/test nodes in adj_train are all-zero.
        feats               np array (shape N x f), the node feature matrix, with f being the
                            length of each node feature vector.
        class_map           dict, where key is the node ID and value is the classes this node
                            belongs to.
        role                dict, where keys are: 'tr' for train, 'va' for validation and 'te'
                            for test nodes. The value is the list of IDs of nodes belonging to
                            the train/val/test sets.
    """
    adj_full = scipy.sparse.load_npz('/home/ubuntu/workspace/{}/adj_full.npz'.format(prefix)).astype(np.bool)
#    adj_full = scipy.sparse.load_npz('/home/ubuntu/workspace/amazon 2/adj_full.npz').astype(np.bool)
    adj_train = scipy.sparse.load_npz('/home/ubuntu/workspace/{}/adj_train.npz'.format(prefix)).astype(np.bool)
    role = json.load(open('/home/ubuntu/workspace/{}/role.json'.format(prefix)))
#    pdb.set_trace()
    feats = np.load('/home/ubuntu/workspace/{}/feats.npy'.format(prefix))
    class_map = json.load(open('/home/ubuntu/workspace/{}/class_map.json'.format(prefix)))
    class_map = {int(k): v for k, v in class_map.items()}
    assert len(class_map) == feats.shape[0]
    # ---- normalize feats ----
    train_nodes = np.array(list(set(adj_train.nonzero()[0])))
    train_feats = feats[train_nodes]
    scaler = StandardScaler()
    scaler.fit(train_feats)
    feats = scaler.transform(feats)
    # -------------------------
    return adj_full, adj_train, feats, class_map, role


from load_graph import load_reddit, load_ogb, inductive_split


class SAGEConv(nn.Module):
    r"""
    Description
    -----------
    GraphSAGE layer from paper `Inductive Representation Learning on
    Large Graphs <https://arxiv.org/pdf/1706.02216.pdf>`__.
    .. math::
        h_{\mathcal{N}(i)}^{(l+1)} &= \mathrm{aggregate}
        \left(\{h_{j}^{l}, \forall j \in \mathcal{N}(i) \}\right)
        h_{i}^{(l+1)} &= \sigma \left(W \cdot \mathrm{concat}
        (h_{i}^{l}, h_{\mathcal{N}(i)}^{l+1}) \right)
        h_{i}^{(l+1)} &= \mathrm{norm}(h_{i}^{l})
    If a weight tensor on each edge is provided, the aggregation becomes:
    .. math::
        h_{\mathcal{N}(i)}^{(l+1)} = \mathrm{aggregate}
        \left(\{e_{ji} h_{j}^{l}, \forall j \in \mathcal{N}(i) \}\right)
    where :math:`e_{ji}` is the scalar weight on the edge from node :math:`j` to node :math:`i`.
    Please make sure that :math:`e_{ji}` is broadcastable with :math:`h_j^{l}`.
    Parameters
    ----------
    in_feats : int, or pair of ints
        Input feature size; i.e, the number of dimensions of :math:`h_i^{(l)}`.
        SAGEConv can be applied on homogeneous graph and unidirectional
        `bipartite graph <https://docs.dgl.ai/generated/dgl.bipartite.html?highlight=bipartite>`__.
        If the layer applies on a unidirectional bipartite graph, ``in_feats``
        specifies the input feature size on both the source and destination nodes.  If
        a scalar is given, the source and destination node feature size would take the
        same value.
        If aggregator type is ``gcn``, the feature size of source and destination nodes
        are required to be the same.
    out_feats : int
        Output feature size; i.e, the number of dimensions of :math:`h_i^{(l+1)}`.
    feat_drop : float
        Dropout rate on features, default: ``0``.
    aggregator_type : str
        Aggregator type to use (``mean``, ``gcn``, ``pool``, ``lstm``).
    bias : bool
        If True, adds a learnable bias to the output. Default: ``True``.
    norm : callable activation function/layer or None, optional
        If not None, applies normalization to the updated node features.
    activation : callable activation function/layer or None, optional
        If not None, applies an activation function to the updated node features.
        Default: ``None``.
    Examples
    --------
    >>> import dgl
    >>> import numpy as np
    >>> import torch as th
    >>> from dgl.nn import SAGEConv
    >>> # Case 1: Homogeneous graph
    >>> g = dgl.graph(([0,1,2,3,2,5], [1,2,3,4,0,3]))
    >>> g = dgl.add_self_loop(g)
    >>> feat = th.ones(6, 10)
    >>> conv = SAGEConv(10, 2, 'pool')
    >>> res = conv(g, feat)
    >>> res
    tensor([[-1.0888, -2.1099],
            [-1.0888, -2.1099],
            [-1.0888, -2.1099],
            [-1.0888, -2.1099],
            [-1.0888, -2.1099],
            [-1.0888, -2.1099]], grad_fn=<AddBackward0>)
    >>> # Case 2: Unidirectional bipartite graph
    >>> u = [0, 1, 0, 0, 1]
    >>> v = [0, 1, 2, 3, 2]
    >>> g = dgl.bipartite((u, v))
    >>> u_fea = th.rand(2, 5)
    >>> v_fea = th.rand(4, 10)
    >>> conv = SAGEConv((5, 10), 2, 'mean')
    >>> res = conv(g, (u_fea, v_fea))
    >>> res
    tensor([[ 0.3163,  3.1166],
            [ 0.3866,  2.5398],
            [ 0.5873,  1.6597],
            [-0.2502,  2.8068]], grad_fn=<AddBackward0>)
    """
    def __init__(self,
                 in_feats,
                 out_feats,
                 aggregator_type,
                 feat_drop=0.,
                 bias=True,
                 norm=None,
                 activation=None):
        super(SAGEConv, self).__init__()

        self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)
        self._out_feats = out_feats
        self._aggre_type = aggregator_type
        self.norm = norm
        self.feat_drop = nn.Dropout(feat_drop)
        self.activation = activation
        # aggregator type: mean/pool/lstm/gcn
        if aggregator_type == 'pool':
            self.fc_pool = nn.Linear(self._in_src_feats, self._in_src_feats)
        if aggregator_type == 'lstm':
            self.lstm = nn.LSTM(self._in_src_feats, self._in_src_feats, batch_first=True)
        if aggregator_type != 'gcn':
            self.fc_self = nn.Linear(self._in_dst_feats, out_feats, bias=False)
        self.fc_neigh = nn.Linear(self._in_src_feats, out_feats, bias=False)
        if bias:
            self.bias = nn.parameter.Parameter(th.zeros(self._out_feats))
        else:
            self.register_buffer('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        r"""
        Description
        -----------
        Reinitialize learnable parameters.
        Note
        ----
        The linear weights :math:`W^{(l)}` are initialized using Glorot uniform initialization.
        The LSTM module is using xavier initialization method for its weights.
        """
        gain = nn.init.calculate_gain('relu')
        if self._aggre_type == 'pool':
            nn.init.xavier_uniform_(self.fc_pool.weight, gain=gain)
        if self._aggre_type == 'lstm':
            self.lstm.reset_parameters()
        if self._aggre_type != 'gcn':
            nn.init.xavier_uniform_(self.fc_self.weight, gain=gain)
        nn.init.xavier_uniform_(self.fc_neigh.weight, gain=gain)

    def _compatibility_check(self):
        """Address the backward compatibility issue brought by #2747"""
        if not hasattr(self, 'bias'):
            dgl_warning("You are loading a GraphSAGE model trained from a old version of DGL, "
                        "DGL automatically convert it to be compatible with latest version.")
            bias = self.fc_neigh.bias
            self.fc_neigh.bias = None
            if hasattr(self, 'fc_self'):
                if bias is not None:
                    bias = bias + self.fc_self.bias
                    self.fc_self.bias = None
            self.bias = bias

    def _lstm_reducer(self, nodes):
        """LSTM reducer
        NOTE(zihao): lstm reducer with default schedule (degree bucketing)
        is slow, we could accelerate this with degree padding in the future.
        """
        m = nodes.mailbox['m'] # (B, L, D)
        batch_size = m.shape[0]
        h = (m.new_zeros((1, batch_size, self._in_src_feats)),
             m.new_zeros((1, batch_size, self._in_src_feats)))
        _, (rst, _) = self.lstm(m, h)
        return {'neigh': rst.squeeze(0)}

    def forward(self, graph, feat, edge_weight=None):
        r"""
        Description
        -----------
        Compute GraphSAGE layer.
        Parameters
        ----------
        graph : DGLGraph
            The graph.
        feat : torch.Tensor or pair of torch.Tensor
            If a torch.Tensor is given, it represents the input feature of shape
            :math:`(N, D_{in})`
            where :math:`D_{in}` is size of input feature, :math:`N` is the number of nodes.
            If a pair of torch.Tensor is given, the pair must contain two tensors of shape
            :math:`(N_{in}, D_{in_{src}})` and :math:`(N_{out}, D_{in_{dst}})`.
        edge_weight : torch.Tensor, optional
            Optional tensor on the edge. If given, the convolution will weight
            with regard to the message.
        Returns
        -------
        torch.Tensor
            The output feature of shape :math:`(N, D_{out})` where :math:`D_{out}`
            is size of output feature.
        """
        self._compatibility_check()
        with graph.local_scope():
            if isinstance(feat, tuple):
                feat_src = self.feat_drop(feat[0])
                feat_dst = self.feat_drop(feat[1])
            else:
                feat_src = feat_dst = self.feat_drop(feat)
                if graph.is_block:
                    feat_dst = feat_src[:graph.number_of_dst_nodes()]
            msg_fn = fn.copy_src('h', 'm')
            if edge_weight is not None:
                assert edge_weight.shape[0] == graph.number_of_edges()
                graph.edata['_edge_weight'] = edge_weight
                msg_fn = fn.u_mul_e('h', '_edge_weight', 'm')

            h_self = feat_dst

            # Handle the case of graphs without edges
            if graph.number_of_edges() == 0:
                graph.dstdata['neigh'] = th.zeros(
                    feat_dst.shape[0], self._in_src_feats).to(feat_dst)

            # Determine whether to apply linear transformation before message passing A(XW)
            lin_before_mp = self._in_src_feats > self._out_feats

            # Message Passing
            if self._aggre_type == 'mean':
                graph.srcdata['h'] = self.fc_neigh(feat_src) if lin_before_mp else feat_src
                reduce_fn = fn.mean('m', 'neigh') if edge_weight is None else fn.sum('m', 'neigh')
                graph.update_all(msg_fn, reduce_fn)
                h_neigh = graph.dstdata['neigh']
                if not lin_before_mp:
                    h_neigh = self.fc_neigh(h_neigh)
            elif self._aggre_type == 'gcn':
                check_eq_shape(feat)
                graph.srcdata['h'] = self.fc_neigh(feat_src) if lin_before_mp else feat_src
                if isinstance(feat, tuple):  # heterogeneous
                    graph.dstdata['h'] = self.fc_neigh(feat_dst) if lin_before_mp else feat_dst
                else:
                    graph.dstdata['h'] = graph.srcdata['h']
                graph.update_all(msg_fn, fn.sum('m', 'neigh'))
                # divide in_degrees
                degs = graph.in_degrees().to(feat_dst)
                h_neigh = (graph.dstdata['neigh'] + graph.dstdata['h']) / (degs.unsqueeze(-1) + 1)
                if not lin_before_mp:
                    h_neigh = self.fc_neigh(h_neigh)
            elif self._aggre_type == 'pool':
                graph.srcdata['h'] = F.relu(self.fc_pool(feat_src))
                graph.update_all(msg_fn, fn.max('m', 'neigh'))
                h_neigh = self.fc_neigh(graph.dstdata['neigh'])
            elif self._aggre_type == 'lstm':
                graph.srcdata['h'] = feat_src
                graph.update_all(msg_fn, self._lstm_reducer)
                h_neigh = self.fc_neigh(graph.dstdata['neigh'])
            else:
                raise KeyError('Aggregator type {} not recognized.'.format(self._aggre_type))

            # GraphSAGE GCN does not require fc_self.
            if self._aggre_type == 'gcn':
                rst = h_neigh
            else:
                rst = self.fc_self(h_self) + h_neigh

            # bias term
            if self.bias is not None:
                rst = rst + self.bias

            # activation
            if self.activation is not None:
                rst = self.activation(rst)
            # normalization
            if self.norm is not None:
                rst = self.norm(rst)
            return rst

# more parameters are needed to contruct SAGE
class SAGE(nn.Module):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout,
                 fanout,
                 buffer_size,
                 batch_size,
                 IS):

        super().__init__()
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.layers = nn.ModuleList()
        self.layers.append(SAGEConv(in_feats, n_hidden, 'mean'))
        for i in range(1, n_layers - 1):
            self.layers.append(SAGEConv(n_hidden, n_hidden, 'mean'))
        self.layers.append(SAGEConv(n_hidden, n_classes, 'mean'))
        self.dropout = nn.Dropout(dropout)
        self.activation = activation
        self.fanout = fanout
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.IS = IS

    # for importance sampling, use the fanout and dst_in_degrees
    def forward(self, blocks, x, IS=None):
        h = x

        for l_num, (layer, block) in enumerate(zip(self.layers, blocks)):

            if IS and l_num == 0:

                h_dst = h[:block.number_of_dst_nodes()]
                h = layer(block, (h, h_dst),edge_weight=block.edata['prob'])


            else:
                h_dst = h[:block.number_of_dst_nodes()]
                h = layer(block, (h, h_dst))

            if l_num != len(self.layers) - 1:
                h = self.activation(h)
                h = self.dropout(h)
        return h
    def inference(self, g, x, batch_size, device):
        """
        Inference with the GraphSAGE model on full neighbors (i.e. without neighbor sampling).
        g : the entire graph.
        x : the input of entire node set.
        The inference code is written in a fashion that it could handle any number of nodes and
        layers.
        """
        # During inference with sampling, multi-layer blocks are very inefficient because
        # lots of computations in the first few layers are repeated.
        # Therefore, we compute the representation of all nodes layer by layer.  The nodes
        # on each layer are of course splitted in batches.
        # TODO: can we standardize this?
        for l, layer in enumerate(self.layers):
            y = th.zeros(g.number_of_nodes(), self.n_hidden if l != len(self.layers) - 1 else self.n_classes)

            sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1)
            dataloader = dgl.dataloading.NodeDataLoader(
                g,
                th.arange(g.number_of_nodes()),
                sampler,
                batch_size=args.batch_size,
                shuffle=True,
                drop_last=False,
                num_workers=args.num_workers)

            for input_nodes, output_nodes, blocks in tqdm.tqdm(dataloader):
                block = blocks[0]

                block = block.int().to(device)
                h = x[input_nodes].to(device)
                h = layer(block, h)
                if l != len(self.layers) - 1:
                    h = self.activation(h)
                    h = self.dropout(h)

                y[output_nodes] = h.cpu()

            x = y
        return y
class FindBestEvaluator:
    def __init__(self, fanout):
        self.fanout = fanout
        self.best_val_acc = 0
        self.best_test_acc = 0

    def full_evaluate(self, model, g, labels, val_nid, test_nid, batch_size, device):
        """
        Evaluate the model on the validation set specified by ``val_mask``.
        g : The entire graph.
        inputs : The features of all the nodes.
        labels : The labels of all the nodes.
        val_mask : A 0-1 mask indicating which nodes do we actually compute the accuracy for.
        batch_size : Number of nodes to compute at the same time.
        device : The GPU device to evaluate on.
        """
        model.eval()
        with th.no_grad():
            inputs = g.ndata['feat']
            pred = model.inference(g, inputs, batch_size, device)
        model.train()
        return compute_acc(pred[val_nid], labels[val_nid]), compute_acc(pred[test_nid], labels[test_nid])

    def sample_evaluate(self, model, g, labels, val_nid, test_nid, batch_size, fanout, device):
        model.eval()
        multilabel = len(labels.shape) > 1 and labels.shape[-1] > 1
        feats = g.ndata['feat']

        val_accs = []
        sampler = dgl.dataloading.MultiLayerNeighborSampler(fanout, fanout, None, 0, g)
        dataloader = dgl.dataloading.NodeDataLoader(g, val_nid,
                                                    sampler,
                                                    batch_size=args.eval_batch_size,
                                                    shuffle=False,
                                                    drop_last=False,
                                                    num_workers=4)
        with th.no_grad():
            for step, (input_nodes, seeds, blocks) in enumerate(dataloader):
                # copy block to gpu
                blocks = [blk.int().to(device) for blk in blocks]
                # Load the input features as well as output labels
                batch_inputs, batch_labels = load_subtensor(feats, labels, seeds, input_nodes, device)
                batch_labels = batch_labels.float() if multilabel else batch_labels.long()
                batch_pred = model(blocks, batch_inputs)
                acc = compute_acc(batch_pred, batch_labels)
                val_accs.append(acc)

        test_accs = []
        sampler = dgl.dataloading.MultiLayerNeighborSampler(fanout, fanout, None, 0, g)
        dataloader = dgl.dataloading.NodeDataLoader(g, test_nid,
                                                    sampler,
                                                    batch_size=args.eval_batch_size,
                                                    shuffle=False,
                                                    drop_last=False,
                                                    num_workers=4)
        with th.no_grad():
            for input_nodes, seeds, blocks in tqdm.tqdm(dataloader):
                # copy block to gpu
                blocks = [blk.int().to(device) for blk in blocks]
                # Load the input features as well as output labels
                batch_inputs, batch_labels = load_subtensor(feats, labels, seeds, input_nodes, device)
                batch_labels = batch_labels.float() if multilabel else batch_labels.long()
                batch_pred = model(blocks, batch_inputs)
                acc = compute_acc(batch_pred, batch_labels)
                test_accs.append(acc)

        model.train()
        return np.mean(val_accs), np.mean(test_accs)

    def __call__(self, model, g, labels, val_nid, test_nid, batch_size, device):
        start = time.time()
        if isinstance(self.fanout, list) and self.fanout[0] > 0:
            val_acc, test_acc = self.sample_evaluate(model, g, labels, val_nid, test_nid, batch_size, self.fanout, device)
        else:
            val_acc, test_acc = self.full_evaluate(model, g, labels, val_nid, test_nid, batch_size, device)

        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            self.best_test_acc = test_acc
        print('Best Eval Acc {:.4f} Test Acc {:.4f}'.format(self.best_val_acc, self.best_test_acc))


def compute_acc(pred, labels):
    """
    Compute the accuracy of prediction given the labels.
    """
    #    labels = labels.long()
    return calc_f1(pred, labels)


def evaluate(model, g, inputs, labels, val_nid, batch_size, device):
    """
    Evaluate the model on the validation set specified by ``val_nid``.
    g : The entire graph.
    inputs : The features of all the nodes.
    labels : The labels of all the nodes.
    val_nid : the node Ids for validation.
    batch_size : Number of nodes to compute at the same time.
    device : The GPU device to evaluate on.
    """
    model.eval()
    with th.no_grad():
        pred = model.inference(g, inputs, batch_size, device)
    model.train()
    output = pred[val_nid]
    return compute_acc(pred[val_nid], labels[val_nid]), compute_acc(pred[val_nid], labels[val_nid])


def load_subtensor(g, seeds, input_nodes, device):
    """
    Copys features and labels of a set of nodes onto GPU.
    """
    batch_inputs = g.ndata['features'][input_nodes].to(device)
    batch_labels = g.ndata['labels'][seeds].to(device)
    return batch_inputs, batch_labels

def cross_entropy(x, labels):
    y = F.cross_entropy(x, labels[:, 0], reduction="none")
    y = th.log(epsilon + y) - math.log(epsilon)
    return th.mean(y)
class CachedData:
    '''Cache part of the data
    This class caches a small portion of the data and uses the cached data
    to accelerate the data movement to GPU.
    Parameters
    ----------
    node_data : tensor
        The actual node data we want to move to GPU.
    buffer_nodes : tensor
        The node IDs that we like to cache in GPU.
    device : device
        The device where we store cached data.
    '''
    def __init__(self, node_data, buffer_nodes, device):
        num_nodes = node_data.shape[0]
        # Let's construct a vector that stores the location of the cached data.
        # If a node is cached, the corresponding element in the vector stores the location in the cache.
        # If a node is not cached, the element points to the end of the cache.
        self.cached_locs = th.ones(num_nodes, dtype=th.int32, device=device) * len(buffer_nodes)
        self.cached_locs[buffer_nodes] = th.arange(len(buffer_nodes), dtype=th.int32, device=device)
        # Let's construct the cache. The last row in the cache doesn't contain valid data.
        self.cache_data = th.zeros(len(buffer_nodes) + 1, node_data.shape[1], dtype=node_data.dtype, device=device)
        self.cache_data[:len(buffer_nodes)] = node_data[buffer_nodes].to(device)
        self.invalid_loc = len(buffer_nodes)
        self.node_data = node_data

    def __getitem__(self, nids):
        locs = self.cached_locs[nids].long()
        data = self.cache_data[locs]
        out_cache_nids = nids[locs == self.invalid_loc]
        data[locs == self.invalid_loc] = self.node_data[out_cache_nids].to(self.cache_data.device)
        return data


#### Entry point
def run(args, device, data):
    # Unpack data
    in_feats, n_classes, train_g, val_g, test_g, number_of_nodes, in_degree_all_nodes = data
    labels = train_g.ndata['label']
    train_nid = th.nonzero(train_g.ndata['train_mask'], as_tuple=True)[0]
    val_nid = th.nonzero(val_g.ndata['val_mask'], as_tuple=True)[0]
    test_nid = th.nonzero(~(test_g.ndata['train_mask'] | test_g.ndata['val_mask']), as_tuple=True)[0]
    number_of_nodes = g.number_of_nodes()
    in_degree_all_nodes = g.in_degrees()

    print('get cache sampling probability')
    # Compute the node sampling probability.
    prob = np.divide(in_degree_all_nodes, sum(in_degree_all_nodes))
    prob_gpu = th.tensor(prob).to(device)


    print('create the model')
    avd = int(sum(in_degree_all_nodes)//number_of_nodes)
    # Define model and optimizer
    in_degree_all_nodes = in_degree_all_nodes.to(device)
    model = SAGE(in_feats, args.num_hidden, n_classes, args.num_layers, F.relu, args.dropout,
                 [int(fanout) for fanout in args.fan_out.split(',')], args.buffer_size,args.batch_size / number_of_nodes,
                 args.IS)
    model = model.to(device)
    loss_fcn = nn.BCEWithLogitsLoss()
    loss_fcn = loss_fcn.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    eval_fanout = [int(fanout) if int(fanout) > 0 else -1 for fanout in args.eval_fan_out.split(',')]
    evaluate = FindBestEvaluator(eval_fanout)
    # Training loop
    avg = 0
    iter_tput = []
    keys = ['Loss', 'Train Acc', 'Test Acc', 'Eval Acc', 'Test F1', 'Eval F1']
    history = dict(zip(keys, ([] for _ in keys)))
    fanout= [int(fanout) for fanout in args.fan_out.split(',')]
    min_fanout = [f for f in fanout]
    max_fanout = [f for f in fanout]
    # We want to keep all nodes in the cache in the input layer.
    if args.buffer_size != 0:
        min_fanout[0] = 0
        max_fanout[0] = 10
#    min_fanout = [1,5,5]
    feats = g.ndata['feat']
    buffer_nodes = None
    print('start training')

    for epoch in range(args.num_epochs):
        profiler = Profiler()
        profiler.start()
        #num_input_nodes = 0
        #num_cached_nodes = 0
        #cache_bool_idx = th.zeros(g.number_of_nodes())
        if epoch % args.buffer_rs_every == 0:
            if args.buffer_size != 0:
                # initial the buffer
                num_nodes = g.num_nodes()
                num_sample_nodes = int(args.buffer_size * num_nodes)
                buffer_nodes = np.random.choice(num_nodes, num_sample_nodes, replace=True, p=prob)
                buffer_nodes = np.unique(buffer_nodes)
                cached_data = CachedData(feats, buffer_nodes, device)
                cached_g = dgl.out_subgraph(g, buffer_nodes)
                cached_in_degree = cached_g.in_degrees().to(device)
            sampler = dgl.dataloading.MultiLayerNeighborSampler(min_fanout, max_fanout, buffer_nodes, args.buffer_size, g)
            dataloader = dgl.dataloading.NodeDataLoader(
                train_g,
                train_nid,
                sampler,
                batch_size=args.batch_size,
                shuffle=True,
                drop_last=False,
                num_workers=args.num_workers)

        tic = time.time()

        # Loop over the dataloader to sample the computation dependency graph as a list of
        # blocks.
        tic_step = time.time()
        for step, (input_nodes, seeds, blocks) in enumerate(dataloader):
            # Load the input features as well as output labels

            blocks = [block.int().to(device) for block in blocks]
            #num_input_nodes += len(input_nodes)
            #num_cached_nodes += th.sum(cache_bool_idx[input_nodes])
            batch_inputs = g.ndata['feat'][input_nodes].to(device)
            batch_labels = labels[seeds].to(device)
            if args.buffer_size != 0 and args.IS == 1:
                for l_num, block in enumerate(blocks):
                    src, dst = block.edges()
                    dst = block.dstdata[dgl.NID][dst.long()]
                    src = block.srcdata[dgl.NID][src.long()]

                    N = in_degree_all_nodes[dst]
                    cached_N = cached_in_degree[dst]
                    sample_prob = 1 - th.pow(1 - prob_gpu[src], num_sample_nodes)
                    prob2 = max_fanout[l_num] / th.minimum(cached_N,
                                                           th.ones(len(cached_N), device=device) * max_fanout[l_num])
                    sample_prob = sample_prob;# * prob2

                    block.edata['prob'] = (1 / (sample_prob * N)).float()
                    coefficient = block.edata['prob']
                    #                    coefficient[coefficient>5] = 5
                    block.edata['prob'] = coefficient

                batch_pred = model(blocks, batch_inputs, args.IS)
            else:
                batch_pred = model(blocks, batch_inputs)

            batch_labels = batch_labels.type_as(batch_pred)
            loss = loss_fcn(batch_pred, batch_labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            history['Loss'].append(loss.item())
            iter_tput.append(len(seeds) / (time.time() - tic_step))
            if step % args.log_every == 0:
                acc = compute_acc(batch_pred, batch_labels)
                gpu_mem_alloc = th.cuda.max_memory_allocated() / 1000000 if th.cuda.is_available() else 0
                print(
                    'Epoch {:05d} | Step {:05d} | Loss {:.4f} | Train Acc {:.4f} | Speed (samples/sec) {:.4f} | GPU {:.1f} MiB'.format(
                        epoch, step, loss.item(), acc.item(), np.mean(iter_tput[3:]), gpu_mem_alloc))
            tic_step = time.time()

        history['Train Acc'].append(acc.item())
        #print('inputs: {:.3f}, cached: {:.3f}'.format(num_input_nodes / (step + 1), num_cached_nodes / (step + 1)))

        toc = time.time()
        print('Epoch Time(s): {:.4f}'.format(toc - tic))
        if epoch >= 5:
            avg += toc - tic
        if epoch % args.eval_every == 0:# and epoch != 0:
            fanout = [60, 60, 60]
            sampler_test = dgl.dataloading.MultiLayerNeighborSampler(fanout, fanout, None, 0, g)

            test_dataloader = dgl.dataloading.NodeDataLoader(
                g,
                test_nid,
                sampler_test,
                batch_size=args.batch_size,
                shuffle=True,
                drop_last=False,
                num_workers=args.num_workers)
            model.eval()
            test_acc = []
            for input_nodes, seeds, blocks in tqdm.tqdm(test_dataloader):
                blocks = [blk.int().to(device) for blk in blocks]
                batch_inputs = g.ndata['feat'][input_nodes].to(device)
                batch_labels = blocks[-1].dstdata['label'].to(device)
                batch_pred = model(blocks, batch_inputs)

                test_acc.append(calc_f1(batch_pred, batch_labels))

            print('Test Acc {:.4f}'.format(np.mean(test_acc)))
    #            history['Eval F1'].append(eval_f1)
    #    print('Avg epoch time: {}'.format(avg / (epoch - 4)))
    #    epochs = range(len(history['Eval F1']))
    #    plt.figure(1)
    #    plt.plot(epochs, history['Eval F1'], label='Eval F1')
    #    plt.title('Eval F1')
    #    plt.xlabel('Epochs')
    #    plt.ylabel('Eval F1')
    #    plt.legend()
    #    plt.savefig('results1/Eval F1_' + args.dataset + '.png')
    #    plt.show()
    #
    #
    #    epochs = range(len(history['Test F1']))
    #    plt.figure(1)
    #    plt.plot(epochs, history['Test F1'], label='Test F1')
    #    plt.title('Test F1')
    #    plt.xlabel('Epochs')
    #    plt.ylabel('Test F1')
    #    plt.legend()
    #    plt.savefig('results1/Test F1_' + args.dataset + '.png')
    #    plt.show()

    json_r = json.dumps(history)
    f = open('results1/history_' + args.dataset + '_' + str(args.buffer_size) + '_' + str(
        args.buffer_rs_every) + '_IS_' + str(args.IS) + '.json', "w")
    f.write(json_r)
    f.close()
    # with open('results1/dict.json') as f:
    #     data = json.load(f)
    return test_acc

if __name__ == '__main__':
    argparser = argparse.ArgumentParser("multi-gpu training")
    argparser.add_argument('--gpu', type=int, default=0,
                           help="GPU device ID. Use -1 for CPU training")
    argparser.add_argument('--IS', type=int, default=1)
    argparser.add_argument('--dataset', type=str, default='yelp')
    argparser.add_argument('--num-epochs', type=int, default=10)
    argparser.add_argument('--num-hidden', type=int, default=512)
    argparser.add_argument('--num-layers', type=int, default=3)
    argparser.add_argument('--fan-out', type=str, default='5,10,15')
    argparser.add_argument('--eval-fan-out', type=str, default='15,15,15')
    argparser.add_argument('--buffer-size', type=float, default=0.01)
    argparser.add_argument('--eval-batch-size', type=int, default=10000)
    argparser.add_argument('--buffer', type=np.ndarray, default=None)
    argparser.add_argument('--batch-size', type=int, default=1000)
    argparser.add_argument('--log-every', type=int, default=100)
    argparser.add_argument('--eval-every', type=int, default=9)
    argparser.add_argument('--buffer_rs-every', type=int, default=1)
    argparser.add_argument('--lr', type=float, default=0.003)
    argparser.add_argument('--dropout', type=float, default=0.5)
    argparser.add_argument('--num-workers', type=int, default=4,
                           help="Number of sampling processes. Use 0 for no extra process.")
    argparser.add_argument('--inductive', action='store_true',
                           help="Inductive learning setting")
    argparser.add_argument("--weight-decay", type=float, default=5e-4,
                           help="Weight for L2 loss")
    argparser.add_argument("--aggregator-type", type=str, default="gcn",
                           help="Aggregator type: mean/gcn/pool/lstm")
    args, unknown = argparser.parse_known_args()

    if args.gpu >= 0:
        device = th.device('cuda:%d' % args.gpu)
    else:
        device = th.device('cpu')

    #    data = load_data(args)
    print('loading graph')
    adj_full, adj_train, feats, class_map, role = load_data(args.dataset)
    adj_full, adj_train, feats, class_arr, role = process_graph_data(adj_full, adj_train, feats, class_map, role)

    g = from_scipy(adj_full)
    print('processing graph')

    features = feats
    labels = class_arr
    train_mask = create_mask(role['tr'], labels.shape[0])
    val_mask = create_mask(role['va'], labels.shape[0])
    test_mask = create_mask(role['te'], labels.shape[0])
    g.ndata['train_mask'] = generate_mask_tensor(train_mask)
    g.ndata['val_mask'] = generate_mask_tensor(val_mask)
    g.ndata['test_mask'] = generate_mask_tensor(test_mask)
    g.ndata['feat'] = F1.tensor(features, dtype=F1.data_type_dict['float32'])
    g.ndata['label'] = F1.tensor(labels, dtype=F1.data_type_dict['int64'])
    g.ndata['labels'] = F1.tensor(labels, dtype=F1.data_type_dict['int64'])
    in_feats = features.shape[1]

    n_classes = labels.shape[1]
    n_edges = g.number_of_edges()
    n_nodes = g.number_of_nodes()
    if args.inductive:
        train_g, val_g, test_g = inductive_split(g)
    else:
        train_g = val_g = test_g = g


    print('Pack graph')
    data = in_feats, n_classes, train_g, val_g, test_g, g.number_of_nodes(), g.in_degrees()

    test_accs = []
    for i in range(10):
        test_accs.append(run(args, device, data))
        print('Average test accuracy:', np.mean(test_accs), '±', np.std(test_accs))
