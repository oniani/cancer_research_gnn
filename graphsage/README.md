# Hyper-parameters

GraphSAGE

| Hyper-uarameter                                   | Value        |
| ------------------------------------------------- | ------------ |
| Dropout probability                               | 0.25         |
| Learning rate                                     | 1e-2 (0.01)  |
| Number of training epochs                         | 800          |
| Number of hidden gcn units  (if gcn is specified) | 16           |
| Number of hidden gcn layers (if gcn is specified) | 1            |
| Weight for L2 loss                                | 5e-4 (0.005) |
| Aggregator type                                   | mean         |


Average statistics:

- Accuracy: 0.901
- Precision: 0.926
- Recall: 0.910
- F-Score: 0.915


## Errors

If one gets an error saying `IndexError: index X is out of bounds for axis 0 with size X`,
below is the solution:

1. Find the path of the `dgl` module

```python
import dgl
print(dgl.__path__)
```

2. Find the file `citation_graph.py`. It is usually located under `site-packages/dgl/data/citation_graph.py` in the python lib folder
   obtained by the previous step.

3. Open `citation_graph.py` and find where `# build symmetric adjacency matrix` comment is located.
   Then edit `train_mask, val_mask, and test_mask` so that the ranges do not exceed the index number.
   Example is shown below.

```python
# build symmetric adjacency matrix
adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
self.graph = nx.from_scipy_sparse_matrix(adj, create_using=nx.DiGraph())

features = _normalize(features)
self.features = np.asarray(features.todense())
self.labels = np.where(labels)[1]

# self.train_mask = _sample_mask(range(140), labels.shape[0])
# self.val_mask = _sample_mask(range(200, 500), labels.shape[0])
# self.test_mask = _sample_mask(range(500, 1500), labels.shape[0])

self.train_mask = _sample_mask(range(555), labels.shape[0])
self.val_mask = _sample_mask(range(555, 713), labels.shape[0])
self.test_mask = _sample_mask(range(713, 794), labels.shape[0])
```



Inductive Representation Learning on Large Graphs (GraphSAGE)
============

- Paper link: [http://papers.nips.cc/paper/6703-inductive-representation-learning-on-large-graphs.pdf](http://papers.nips.cc/paper/6703-inductive-representation-learning-on-large-graphs.pdf)
- Author's code repo: [https://github.com/williamleif/graphsage-simple](https://github.com/williamleif/graphsage-simple). Note that the original code is 
simple reference implementation of GraphSAGE.

Requirements
------------
- requests

``bash
pip install requests
``


Results
-------

### Full graph training

Run with following (available dataset: "cora", "citeseer", "pubmed")
```bash
python3 train_full.py --dataset cora --gpu 0    # full graph
```

* cora: ~0.8330 
* citeseer: ~0.7110
* pubmed: ~0.7830

### Minibatch training

Train w/ mini-batch sampling (on the Reddit dataset)
```bash
python3 train_sampling.py --num-epochs 30       # neighbor sampling
python3 train_sampling.py --num-epochs 30 --inductive  # inductive learning with neighbor sampling
python3 train_sampling_multi_gpu.py --num-epochs 30    # neighbor sampling with multi GPU
python3 train_sampling_multi_gpu.py --num-epochs 30 --inductive  # inductive learning with neighbor sampling, multi GPU
python3 train_cv.py --num-epochs 30             # control variate sampling
python3 train_cv_multi_gpu.py --num-epochs 30   # control variate sampling with multi GPU
```

Accuracy:

| Model                 | Accuracy |
|:---------------------:|:--------:|
| Full Graph            | 0.9504   |
| Neighbor Sampling     | 0.9495   |
| N.S. (Inductive)      | 0.9460   |
| Control Variate       | 0.9490   |

### Unsupervised training

Train w/ mini-batch sampling in an unsupervised fashion (on the Reddit dataset)
```bash
python3 train_sampling_unsupervised.py
```

Notably,

* The loss function is defined by predicting whether an edge exists between two nodes or not.  This matches the official
  implementation, and is equivalent to the loss defined in the paper with 1-hop random walks.
* When computing the score of `(u, v)`, the connections between node `u` and `v` are removed from neighbor sampling.
  This trick increases the F1-micro score on test set by 0.02.
* The performance of the learned embeddings are measured by training a softmax regression with scikit-learn, as described
  in the paper.

Micro F1 score reaches 0.9212 on test set.
