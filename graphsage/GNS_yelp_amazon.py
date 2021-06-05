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
        self.layers.append(dglnn.SAGEConv(in_feats, n_hidden, 'mean'))
        for i in range(1, n_layers - 1):
            self.layers.append(dglnn.SAGEConv(n_hidden, n_hidden, 'mean'))
        self.layers.append(dglnn.SAGEConv(n_hidden, n_classes, 'mean'))
        self.dropout = nn.Dropout(dropout)
        self.activation = activation
        self.fanout = fanout
        self.buffer_size = buffer_size
        self.IS = IS
        self.batch_size = batch_size

    # for importance sampling, use the fanout and dst_in_degrees
    def forward(self, blocks, x, A=None):
        h = x
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            h = layer(block, h)
            if self.buffer_size != 0 and self.IS == 1 and layer == 0:
                h = th.mul(A.repeat(len(h.T), 1).reshape(len(h), len(h.T)), h)

            if l != len(self.layers) - 1:
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
    if 'prob' not in g.ndata:
        prob = np.divide(in_degree_all_nodes, sum(in_degree_all_nodes))
    else:
        prob = g.ndata['prob'].numpy()

    print('create the model')
    avd = int(sum(in_degree_all_nodes)//number_of_nodes)
    # Define model and optimizer

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
    feats = g.ndata['feat']
    buffer_nodes = None
    print('start training')

    for epoch in range(args.num_epochs):
        profiler = Profiler()
        profiler.start()
        if epoch % args.buffer_rs_every == 0:
            if args.buffer_size != 0:
                # initial the buffer
                num_nodes = g.num_nodes()
                num_sample_nodes = int(args.buffer_size * num_nodes)  # a parameter that to tune
                #            args.buffer = np.random.permutation(num_nodes)[:num_sample_nodes]
                buffer_nodes = np.random.choice(num_nodes, num_sample_nodes, replace=False,
                                               p=prob)
                cached_data = CachedData(feats, buffer_nodes, device)
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
        model.train()
        tic_step = time.time()
        for step, (input_nodes, seeds, blocks) in enumerate(dataloader):
            # Load the input features as well as output labels

            blocks = [block.int().to(device) for block in blocks]

            if args.buffer_size != 0 and args.IS == 1:
                fanouts = [int(fanout) for fanout in args.fan_out.split(',')]
                batchsize = blocks[-1].num_dst_nodes()
                A = th.ones(batchsize) * (batchsize / number_of_nodes) / args.buffer_size
                A = th.reshape(A.type(th.FloatTensor), (batchsize, 1))
                for layer, block in reversed(list(enumerate(blocks))):
                    if layer == 0:
                        break
                    dst_id = block.dstdata.__getitem__('_ID')

                    srcnode_id, dstnode_id = block.find_edges(
                        th.tensor([i_dex for i_dex in range(block.num_edges())], dtype=th.int32).to(device))
                    i_i = th.LongTensor([srcnode_id.cpu().numpy(), dstnode_id.cpu().numpy()])
                    value = th.div(fanouts[layer] * th.ones(block.num_edges()),
                                   in_degree_all_nodes[dst_id[dstnode_id.cpu().numpy()].cpu().numpy()])
                    value[value > 1] = 1
                    v = th.FloatTensor(value.numpy())
                    A_temp = th.sparse.FloatTensor(i_i, v, th.Size([block.num_src_nodes(), block.num_dst_nodes()]))
                    A = th.sparse.mm(A_temp, A)

                    A[A > 1 / args.buffer_size] = 1 / args.buffer_size
                    A[A == 0] = 1


            batch_inputs = g.ndata['feat'][input_nodes].to(device)
            batch_labels = labels[seeds].to(device)

            # Compute loss and prediction
            if args.buffer_size != 0 and args.IS == 1:

                batch_pred = model(blocks, batch_inputs, A.to(device))
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

        toc = time.time()
        print('Epoch Time(s): {:.4f}'.format(toc - tic))
        if epoch >= 5:
            avg += toc - tic
        if epoch % args.eval_every == 0 and epoch >= 5:
            sampler_test = dgl.dataloading.MultiLayerNeighborSampler(
                [60, 60, 60], args.buffer, args.buffer_size, g)

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
            #            history['Test Acc'].append(test_acc.item())
            #            history['Eval Acc'].append(eval_acc.item())
            history['Test F1'].append(test_acc)
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


if __name__ == '__main__':
    argparser = argparse.ArgumentParser("multi-gpu training")
    argparser.add_argument('--gpu', type=int, default=0,
                           help="GPU device ID. Use -1 for CPU training")
    argparser.add_argument('--IS', type=int, default=0)
    argparser.add_argument('--dataset', type=str, default='amazon')
    argparser.add_argument('--num-epochs', type=int, default=10)
    argparser.add_argument('--num-hidden', type=int, default=256)
    argparser.add_argument('--num-layers', type=int, default=3)
    argparser.add_argument('--fan-out', type=str, default='5,10,15')
    argparser.add_argument('--eval-fan-out', type=str, default='15,15,15')
    argparser.add_argument('--buffer-size', type=float, default=0.01)
    argparser.add_argument('--eval-batch-size', type=int, default=10000)
    argparser.add_argument('--buffer', type=np.ndarray, default=None)
    argparser.add_argument('--batch-size', type=int, default=1000)
    argparser.add_argument('--log-every', type=int, default=100)
    argparser.add_argument('--eval-every', type=int, default=1)
    argparser.add_argument('--buffer_rs-every', type=int, default=1)
    argparser.add_argument('--lr', type=float, default=0.003)
    argparser.add_argument('--dropout', type=float, default=0)
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
    adj_full, adj_train, feats, class_map, role = load_data('yelp')
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

    run(args, device, data)
