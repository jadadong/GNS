import dgl
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
import dgl.function as fn
import dgl.nn.pytorch as dglnn
import time
import math
import argparse
from _thread import start_new_thread
from functools import wraps
from dgl.data import RedditDataset
import tqdm
import traceback
from ogb.nodeproppred import DglNodePropPredDataset, Evaluator
import json
import matplotlib.pyplot as plt
from pyinstrument import Profiler
from sklearn.metrics import f1_score

epsilon = 1 - math.log(2)

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
        self.layers.append(dglnn.SAGEConv(in_feats, n_hidden, 'mean'))
        for i in range(1, n_layers - 1):
            self.layers.append(dglnn.SAGEConv(n_hidden, n_hidden, 'mean'))
        self.layers.append(dglnn.SAGEConv(n_hidden, n_classes, 'mean'))
        self.dropout = nn.Dropout(dropout)
        self.activation = activation
        self.fanout = fanout
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.IS = IS
    # for importance sampling, use the fanout and dst_in_degrees
    def forward(self, blocks, x,A=None):
        h = x

        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
          

            h_dst = h[:block.number_of_dst_nodes()]

         
            h = layer(block, (h, h_dst))
            if self.buffer_size !=0 and self.IS==1 and layer == 0:
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
        nodes = th.arange(g.number_of_nodes())
        for l, layer in enumerate(self.layers):
            y = th.zeros(g.number_of_nodes(), self.n_hidden if l != len(self.layers) - 1 else self.n_classes)

            sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1)
            dataloader = dgl.dataloading.NodeDataLoader(
                g,
                th.arange(g.number_of_nodes()),
                sampler,
                batch_size=args.eval_batch_size,
                shuffle=False,
                drop_last=False,
                num_workers=args.num_workers)

            for input_nodes, output_nodes, blocks in tqdm.tqdm(dataloader):
                block = blocks[0].int().to(device)

                h = x[input_nodes].to(device)
                h_dst = h[:block.number_of_dst_nodes()]
                h = layer(block, (h, h_dst))
                if l != len(self.layers) - 1:
                    h = self.activation(h)
                    h = self.dropout(h)

                y[output_nodes] = h.cpu()

            x = y
        return y


def compute_acc1(pred, labels):
    """
    Compute the accuracy of prediction given the labels.
    """
    evaluator = Evaluator(name="ogbn-arxiv")

    return evaluator.eval({"y_pred": pred.argmax(dim=-1, keepdim=True), "y_true": labels})["acc"]

def compute_acc(logits, labels):
    multilabel = len(labels.shape) > 1 and labels.shape[-1] > 1
    if multilabel:
        pred_labels = nn.Sigmoid()(logits.detach())
        pred_labels[pred_labels > 0.5] = 1
        pred_labels[pred_labels <= 0.5] = 0
        return f1_score(labels.cpu().numpy(), pred_labels.cpu().numpy(), average="micro")
    else:
        return compute_acc1(logits, labels)

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
            for step, (input_nodes, seeds, blocks) in enumerate(dataloader):
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

def load_subtensor(feats, labels, seeds, input_nodes, device):
    """
    Copys features and labels of a set of nodes onto GPU.
    """
    batch_inputs = feats[input_nodes].to(device)
    batch_labels = labels[seeds].to(device)
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
    # added by jialin, additional paramenters
    train_nid, val_nid, test_nid, in_feats, labels, n_classes, g = data
    number_of_nodes = g.number_of_nodes()
    in_degree_all_nodes = g.in_degrees()

    print('get cache sampling probability')
    if 'prob' not in g.ndata:
        prob = np.divide(in_degree_all_nodes, sum(in_degree_all_nodes))
    else:
        prob = g.ndata['prob'].numpy()

    # Init
    #prob = th.zeros(number_of_nodes, 1)
    #prob[train_nid] = 1 / len(train_nid)
    #deg = g.in_degrees().reshape(number_of_nodes, 1)
    #for _ in range(args.num_layers):
    #    g.ndata['p'] = prob / deg
    #    g.update_all(fn.copy_u('p', 'm'), fn.sum('m', 'p'))
    #    prob = g.ndata['p'] + prob
    #    prob = prob / th.sum(prob)
    #prob = np.squeeze(prob.numpy())

    print('create the model')
    avd = int(sum(in_degree_all_nodes)//number_of_nodes)
    in_degree_all_nodes = in_degree_all_nodes.to(device)
    # Define model and optimizer
    # added by jialin, additional paramenters
    model = SAGE(in_feats, args.num_hidden, n_classes, args.num_layers, F.relu, args.dropout,
                 [int(fanout) for fanout in args.fan_out.split(',')], args.buffer_size,
                 args.batch_size / number_of_nodes,args.IS)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    eval_fanout = [int(fanout) if int(fanout) > 0 else -1 for fanout in args.eval_fan_out.split(',')]
    evaluate = FindBestEvaluator(eval_fanout)

    multilabel = len(labels.shape) > 1 and labels.shape[-1] > 1
    if multilabel:
        loss_func = nn.BCEWithLogitsLoss()
        loss_func.to(device)
    else:
        loss_func = cross_entropy
        #loss_func = nn.CrossEntropyLoss()

    # Training loop
    avg = 0
    iter_tput = []
    keys = ['Loss', 'Train Acc', 'Test Acc', 'Eval Acc']
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
                num_sample_nodes = int(args.buffer_size * num_nodes)
                buffer_nodes = np.random.choice(num_nodes, num_sample_nodes, replace=False, p=prob)
                cached_data = CachedData(feats, buffer_nodes, device)
            sampler = dgl.dataloading.MultiLayerNeighborSampler(min_fanout, max_fanout, buffer_nodes, args.buffer_size, g)
            dataloader = dgl.dataloading.NodeDataLoader(
                g,
                train_nid,
                sampler,
                batch_size=args.batch_size,
                shuffle=True,
                drop_last=False,
                num_workers=args.num_workers)

        tic = time.time()

        # Loop over the dataloader to sample the computation dependency graph as a list of
        # blocks.
        num_batches = 0
        for step, (input_nodes, seeds, blocks) in enumerate(dataloader):
            num_batches += 1
            tic_step = time.time()

            # copy block to gpu
            blocks = [blk.int().to(device) for blk in blocks]
            dst_in_dgrees =[]

            if args.buffer_size != 0 and args.IS == 1:
                fanouts = [int(fanout) for fanout in args.fan_out.split(',')]
                batchsize = blocks[-1].num_dst_nodes()
                A = th.ones(batchsize, 1, dtype=th.float32, device=device) * (batchsize/number_of_nodes)/args.buffer_size
                for layer, block in reversed(list(enumerate(blocks))):
                    if layer ==0:
                        break
                    dst_id = block.dstdata.__getitem__('_ID').long()
                    srcnode_id, dstnode_id = block.edges()
                    i_i = th.stack([srcnode_id.long(), dstnode_id.long()])
                    value = th.div(fanouts[layer] * th.ones(block.num_edges(), device=device),
                                   in_degree_all_nodes[dst_id[dstnode_id.long()]])
                    value[value > 1] = 1
                    v = value
                    A_temp = th.sparse.FloatTensor(i_i, v, th.Size([block.num_src_nodes(), block.num_dst_nodes()]))
                    A = th.sparse.mm(A_temp, A)
                    A[A>1/(avd*args.buffer_size)]=1/(avd*args.buffer_size)
                    A[A==0]=1
  
            # Load the input features as well as output labels
            batch_inputs, batch_labels = load_subtensor(cached_data if args.buffer_size > 0 else feats,
                                                        labels, seeds, input_nodes, device)
            batch_labels = batch_labels.float() if multilabel else batch_labels.long()
            if args.buffer_size != 0 and args.IS == 1:
                batch_pred  = model(blocks, batch_inputs,A.to(device))
            else:
                batch_pred = model(blocks, batch_inputs)

            loss = loss_func(batch_pred, batch_labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            iter_tput.append(len(seeds) / (time.time() - tic_step))
            if step % args.log_every == 0:
                acc = compute_acc(batch_pred, batch_labels)
                gpu_mem_alloc = th.cuda.max_memory_allocated() / 1000000 if th.cuda.is_available() else 0
                print(
                    'Epoch {:05d} | Step {:05d} | Loss {:.4f} | Train Acc {:.4f} | Speed (samples/sec) {:.4f} | GPU {:.1f} MiB'.format(
                        epoch, step, loss.item(), acc, np.mean(iter_tput[3:]), gpu_mem_alloc))
        profiler.stop()
        print(profiler.output_text(unicode=True, color=True))

        history['Loss'].append(loss.item())
        history['Train Acc'].append(acc)

        toc = time.time()
        print('Epoch Time(s): {:.4f}, #iterations: {}'.format(toc - tic, num_batches))
        if epoch >= 5:
            avg += toc - tic
        if epoch % args.eval_every == 0 and epoch != 0:
            evaluate(model, g, labels, val_nid, test_nid, args.eval_batch_size, device)

    print('Avg epoch time: {}'.format(avg / (epoch - 4)))
    epochs = range(args.num_epochs)
    plt.figure(1)
    plt.plot(epochs, history['Loss'], label='Training loss')
    plt.title('Training loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(
        'results/loss_' + args.dataset + '_' + str(args.buffer_size) + '_' + str(args.buffer_rs_every) + '.png')
    plt.show()
    plt.figure(2)
    plt.plot(epochs, history['Train Acc'], label='Training accuracy')
    plt.title('Training accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(
        'results/acc_' + args.dataset + '_' + str(args.buffer_size) + '_' + str(args.buffer_rs_every) + '.png')
    plt.show()

    json_r = json.dumps(history)
    f = open('results/history_' + args.dataset + '_' + str(args.buffer_size) + '_' + str(
        args.buffer_rs_every) +'.json', "w")
    f.write(json_r)
    f.close()
    return evaluate.best_test_acc


if __name__ == '__main__':
    argparser = argparse.ArgumentParser("multi-gpu training")
    argparser.add_argument('--gpu', type=int, default=0,
                           help="GPU device ID. Use -1 for CPU training")
    argparser.add_argument('--dataset', type=str, default='ogbn-products')
    argparser.add_argument('--num-epochs', type=int, default=10)
    argparser.add_argument('--num-hidden', type=int, default=256)
    argparser.add_argument('--num-layers', type=int, default=3)
    argparser.add_argument('--fan-out', type=str, default='5,10,15')
    argparser.add_argument('--eval-fan-out', type=str, default='15,15,15')
    argparser.add_argument('--batch-size', type=int, default=1000)
    argparser.add_argument('--eval-batch-size', type=int, default=10000)
    argparser.add_argument('--log-every', type=int, default=20)
    argparser.add_argument('--eval-every', type=int, default=1)
    argparser.add_argument('--lr', type=float, default=0.003)
    argparser.add_argument('--dropout', type=float, default=0.5)
    argparser.add_argument('--num-workers', type=int, default=4,
                           help="Number of sampling processes. Use 0 for no extra process.")
    argparser.add_argument('--save-pred', type=str, default='')
    argparser.add_argument('--wd', type=float, default=0)
    argparser.add_argument('--buffer-size', type=float, default=0.01)
    argparser.add_argument('--buffer_rs-every', type=int, default=1)
    argparser.add_argument('--IS', type=int, default=1)
    args = argparser.parse_args()

    if args.gpu >= 0:
        device = th.device('cuda:%d' % args.gpu)
    else:
        device = th.device('cpu')

    # load graph
    if '.dgl' not in args.dataset:
        print('load graph from OGB.')
        data = DglNodePropPredDataset(name=args.dataset)
        splitted_idx = data.get_idx_split()
        train_idx, val_idx, test_idx = splitted_idx['train'], splitted_idx['valid'], splitted_idx['test']
        graph, labels = data[0]
        graph.create_formats_()
        n_classes = len(th.unique(labels[th.logical_not(th.isnan(labels))]))
    else:
        print('load a prepared graph')
        data = dgl.load_graphs(args.dataset)[0]
        graph = data[0]
        graph = graph.formats(['csr', 'csc'])
        print('create csr and csc')
        if 'oag' in args.dataset:
            labels = graph.ndata['field']
            graph.ndata['feat'] = graph.ndata['emb']

            # Split the dataset into training, validation and testing
            label_sum = labels.sum(1)
            valid_labal_idx = th.nonzero(label_sum > 0, as_tuple=True)[0]
            train_size = int(len(valid_labal_idx) * 0.80)
            val_size = int(len(valid_labal_idx) * 0.10)
            test_size = len(valid_labal_idx) - train_size - val_size
            train_idx, val_idx, test_idx = valid_labal_idx[th.randperm(len(valid_labal_idx))].split([train_size, val_size, test_size])

            # Remove infrequent labels. Otherwise, some of the labels will not have instances
            # in the training, validation or test set.
            label_filter = labels[train_idx].sum(0) > 100
            label_filter = th.logical_and(label_filter, labels[val_idx].sum(0) > 100)
            label_filter = th.logical_and(label_filter, labels[test_idx].sum(0) > 100)
            labels = labels[:,label_filter]
            n_classes = labels.shape[1]

            # Adjust training, validation and testing set to make sure all paper nodes
            # in these sets have labels.
            train_idx = train_idx[labels[train_idx].sum(1) > 0]
            val_idx = val_idx[labels[val_idx].sum(1) > 0]
            test_idx = test_idx[labels[test_idx].sum(1) > 0]
            # All labels have instances.
            assert np.all(labels[train_idx].sum(0).numpy() > 0)
            assert np.all(labels[val_idx].sum(0).numpy() > 0)
            assert np.all(labels[test_idx].sum(0).numpy() > 0)
            # All instances have labels.
            assert np.all(labels[train_idx].sum(1).numpy() > 0)
            assert np.all(labels[val_idx].sum(1).numpy() > 0)
            assert np.all(labels[test_idx].sum(1).numpy() > 0)
        else:
            labels = graph.ndata['label']
            train_idx = th.nonzero(graph.ndata['train_mask'], as_tuple=True)[0]
            val_idx = th.nonzero(graph.ndata['val_mask'], as_tuple=True)[0]
            test_idx = th.nonzero(graph.ndata['test_mask'], as_tuple=True)[0]
            n_classes = len(th.unique(labels[th.logical_not(th.isnan(labels))]))

    in_feats = graph.ndata['feat'].shape[1]
    # Pack data
    data = train_idx, val_idx, test_idx, in_feats, labels, n_classes, graph
    print('|V|: {}, |E|: {}, #train: {}, #val: {}, #test: {}, #classes: {}'.format(
        graph.number_of_nodes(), graph.number_of_edges(), len(train_idx), len(val_idx), len(test_idx),
        n_classes))

    # Run 1 times
    test_accs = []
    for i in range(1):
        test_accs.append(run(args, device, data))
        print('Average test accuracy:', np.mean(test_accs), '±', np.std(test_accs))
