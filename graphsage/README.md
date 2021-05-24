
Global Neighbor Sampling for Mixed CPU-GPU Training on Giant Graphs (GNS)
============

## Datasets


All datasets used in our papers are available for download.

Available in https://github.com/GraphSAINT/GraphSAINT
* Yelp
* Amazon

Available in [DGL](https://github.com/dmlc/dgl) library
* OAG-paper
* OGBN-products
* OGBN-Papers100M


Results
-------

### Training

Check out a customized DGL from [here](https://github.com/zheng-da/dgl/tree/new_sampling).

```
git clone https://github.com/zheng-da/dgl.git
cd dgl
git checkout new_sampling
```

Follow the instruction [here](https://doc.dgl.ai/install/index.html) to install DGL from source.

The following commands train GraphSage with GNS.

```bash
python3 GNS_sampling_prob.py --dataset ogbn-products    # training on OGBN-products
python3 GNS_sampling_prob.py --dataset oag_max_paper.dgl     # training on OAG-paper, OGBN-products and OGBN-Papers100M
python3 GNS_sampling_prob.py --dataset ogbn-papers100M   # training on OGBN-Papers100M
python3 GNS_yelp_amazon.py --dataset yelp   # training on Yelp
python3 GNS_yelp_amazon.py --dataset amazon   # training on Amazon
```

