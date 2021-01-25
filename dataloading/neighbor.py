"""Data loading components for neighbor sampling"""
from .dataloader import BlockSampler
from .. import sampling, subgraph, distributed
import numpy as np
import torch as th
from collections import Counter

class MultiLayerNeighborSampler(BlockSampler):
    """Sampler that builds computational dependency of node representations via
    neighbor sampling for multilayer GNN.

    This sampler will make every node gather messages from a fixed number of neighbors
    per edge type.  The neighbors are picked uniformly.

    Parameters
    ----------
    fanouts : list[int] or list[dict[etype, int] or None]
        List of neighbors to sample per edge type for each GNN layer, starting from the
        first layer.

        If the graph is homogeneous, only an integer is needed for each layer.

        If None is provided for one layer, all neighbors will be included regardless of
        edge types.

        If -1 is provided for one edge type on one layer, then all inbound edges
        of that edge type will be included.
    replace : bool, default True
        Whether to sample with replacement
    return_eids : bool, default False
        Whether to return the edge IDs involved in message passing in the block.
        If True, the edge IDs will be stored as an edge feature named ``dgl.EID``.

    Examples
    --------
    To train a 3-layer GNN for node classification on a set of nodes ``train_nid`` on
    a homogeneous graph where each node takes messages from 5, 10, 15 neighbors for
    the first, second, and third layer respectively (assuming the backend is PyTorch):

    >>> sampler = dgl.dataloading.MultiLayerNeighborSampler([5, 10, 15])
    >>> collator = dgl.dataloading.NodeCollator(g, train_nid, sampler)
    >>> dataloader = torch.utils.data.DataLoader(
    ...     collator.dataset, collate_fn=collator.collate,
    ...     batch_size=1024, shuffle=True, drop_last=False, num_workers=4)
    >>> for blocks in dataloader:
    ...     train_on(blocks)

    If training on a heterogeneous graph and you want different number of neighbors for each
    edge type, one should instead provide a list of dicts.  Each dict would specify the
    number of neighbors to pick per edge type.

    >>> sampler = dgl.dataloading.MultiLayerNeighborSampler([
    ...     {('user', 'follows', 'user'): 5,
    ...      ('user', 'plays', 'game'): 4,
    ...      ('game', 'played-by', 'user'): 3}] * 3)
    """
    def __init__(self, fanouts,buffer,buffer_size, replace=False, return_eids=False):
        super().__init__(len(fanouts),return_eids)

        self.fanouts = fanouts
        self.replace = replace
        self.buffer = buffer
        self.buffer_size = buffer_size

    def sample_frontier(self, block_id, g, seed_nodes):

        fanout = self.fanouts[block_id]
        buffer_size = self.buffer_size
        buffer  = self.buffer
        if isinstance(g, distributed.DistGraph):
            if fanout is None:
                # TODO(zhengda) There is a bug in the distributed version of in_subgraph.
                # let's use sample_neighbors to replace in_subgraph for now.
                frontier = distributed.sample_neighbors(g, seed_nodes, -1, replace=False)
            else:
                frontier = distributed.sample_neighbors(g, seed_nodes, fanout, replace=self.replace)
        else:
            if fanout is None:
                frontier = subgraph.in_subgraph(g, seed_nodes)
            else:
                ## added by jialin
                if buffer_size==0:
                    frontier = sampling.sample_neighbors(g, seed_nodes, fanout, replace=self.replace)
                else:
                    if block_id == 0:
                        # num_nodes = g.num_nodes()
                        # num_sample_nodes = int(buffer_size*num_nodes)  # a parameter that to tune
                        # sample_nodes = np.random.permutation(num_nodes)[:num_sample_nodes]
                        sample_nodes = buffer
                        x_x, x_neighbor, x_eid = g.out_edges(sample_nodes, form='all')
                        E_id = x_eid[np.isin(x_neighbor.numpy(), seed_nodes)]
                        # x_x1,x_neighbor1, x_eid1 = g.out_edges(sample_nodes, form='all')
                        # E_id = x_eid1[np.isin(x_neighbor1.numpy(), seed_nodes['_U'])]
                        # index_original = np.isin(x_neighbor1.numpy(), seed_nodes['_U'])
                        # x_nei= x_neighbor1[index_original]
                        # seed_rest = np.setdiff1d(seed_nodes['_U'].numpy(),np.unique(x_nei))
                        # x_neighbor2, x_x2, x_eid2 = g.in_edges(seed_rest, form='all')
                        # E_id = th.cat((E_id, x_eid2), 0)



                        # x_xid = x_x[index_original]
                        # counts = Counter(x_xid.numpy())
                        # temp=list(filter(lambda a: counts[a] < fanout+1,counts))
                        # sample_neig = [np.where(x_xid == i)[0] for i in temp]
                        # e_index1 = np.concatenate(sample_neig).ravel()
                        #
                        #
                        # temp = list(filter(lambda a: counts[a] > fanout,counts))
                        # if temp is not None:
                        #     sample_neig=[np.where(x_xid == i)[0][0:fanout] for i in temp]
                        #     e_index2=np.concatenate(sample_neig).ravel()
                        #     index = np.concatenate([e_index1, e_index2]).ravel()
                        # else:
                        #     index = e_index1
                        # in_o = np.asarray(index_original.nonzero())
                        # in_o = np.reshape(in_o,(in_o.size,))
                        # E_id = x_eid[in_o[index]]
                        frontier = subgraph.edge_subgraph(g, E_id, preserve_nodes=True)
                    else:
                          frontier = sampling.sample_neighbors(g, seed_nodes, fanout, replace=self.replace)

        return frontier

class MultiLayerFullNeighborSampler(MultiLayerNeighborSampler):
    """Sampler that builds computational dependency of node representations by taking messages
    from all neighbors for multilayer GNN.

    This sampler will make every node gather messages from every single neighbor per edge type.

    Parameters
    ----------
    n_layers : int
        The number of GNN layers to sample.
    return_eids : bool, default False
        Whether to return the edge IDs involved in message passing in the block.
        If True, the edge IDs will be stored as an edge feature named ``dgl.EID``.

    Examples
    --------
    To train a 3-layer GNN for node classification on a set of nodes ``train_nid`` on
    a homogeneous graph where each node takes messages from all neighbors for the first,
    second, and third layer respectively (assuming the backend is PyTorch):

    >>> sampler = dgl.dataloading.MultiLayerFullNeighborSampler(3)
    >>> collator = dgl.dataloading.NodeCollator(g, train_nid, sampler)
    >>> dataloader = torch.utils.data.DataLoader(
    ...     collator.dataset, collate_fn=collator.collate,
    ...     batch_size=1024, shuffle=True, drop_last=False, num_workers=4)
    >>> for blocks in dataloader:
    ...     train_on(blocks)
    """
    def __init__(self, n_layers, return_eids=False):
        super().__init__([None] * (n_layers),[None],[None], return_eids=return_eids)
