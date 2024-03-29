import argparse
import time
import numpy as np
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl import DGLGraph
from dgl.data import register_data_args, load_data
from dgl.nn.pytorch.conv import GMMConv

from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import classification_report as report

torch.manual_seed(2)


class MoNet(nn.Module):
    def __init__(
        self,
        g,
        in_feats,
        n_hidden,
        out_feats,
        n_layers,
        dim,
        n_kernels,
        dropout,
    ):
        super(MoNet, self).__init__()
        self.g = g
        self.layers = nn.ModuleList()
        self.pseudo_proj = nn.ModuleList()

        # Input layer
        self.layers.append(GMMConv(in_feats, n_hidden, dim, n_kernels))
        self.pseudo_proj.append(nn.Sequential(nn.Linear(2, dim), nn.Tanh()))

        # Hidden layer
        for _ in range(n_layers - 1):
            self.layers.append(GMMConv(n_hidden, n_hidden, dim, n_kernels))
            self.pseudo_proj.append(
                nn.Sequential(nn.Linear(2, dim), nn.Tanh())
            )

        # Output layer
        self.layers.append(GMMConv(n_hidden, out_feats, dim, n_kernels))
        self.pseudo_proj.append(nn.Sequential(nn.Linear(2, dim), nn.Tanh()))
        self.dropout = nn.Dropout(dropout)

    def forward(self, feat, pseudo):
        h = feat
        for i in range(len(self.layers)):
            if i != 0:
                h = self.dropout(h)
            h = self.layers[i](self.g, h, self.pseudo_proj[i](pseudo))
        return h


def evaluate(model, features, pseudo, labels, mask):
    model.eval()
    with torch.no_grad():
        logits = model(features, pseudo)
        logits = logits[mask]
        labels = labels[mask]

        # Statistics
        _, indices = torch.max(logits, dim=1)
        prediction = indices.long().cpu().numpy()
        correct = torch.sum(indices == labels)
        accuracy = correct.item() * 1.0 / len(labels)
        precision, recall, fscore, _ = score(
            labels, prediction, average="macro"
        )

        class_based_report = report(labels, prediction)

        return accuracy, precision, recall, fscore, class_based_report


def main(args):
    # load and preprocess dataset
    data = load_data(args)
    features = torch.FloatTensor(data.features)
    labels = torch.LongTensor(data.labels)
    if False:  # hasattr(torch, 'BoolTensor'):
        train_mask = torch.BoolTensor(data.train_mask)
        val_mask = torch.BoolTensor(data.val_mask)
        test_mask = torch.BoolTensor(data.test_mask)
    else:
        train_mask = torch.ByteTensor(data.train_mask)
        val_mask = torch.ByteTensor(data.val_mask)
        test_mask = torch.ByteTensor(data.test_mask)
    in_feats = features.shape[1]
    n_classes = data.num_labels
    n_edges = data.graph.number_of_edges()
    print(
        """----Data statistics------'
      #Edges %d
      #Classes %d
      #Train samples %d
      #Val samples %d
      #Test samples %d"""
        % (
            n_edges,
            n_classes,
            train_mask.sum().item(),
            val_mask.sum().item(),
            test_mask.sum().item(),
        )
    )

    if args.gpu < 0:
        cuda = False
    else:
        cuda = True
        torch.cuda.set_device(args.gpu)
        features = features.cuda()
        labels = labels.cuda()
        train_mask = train_mask.cuda()
        val_mask = val_mask.cuda()
        test_mask = test_mask.cuda()
        print("use cuda:", args.gpu)

    # graph preprocess and calculate normalization factor
    g = data.graph
    g.remove_edges_from(nx.selfloop_edges(g))
    g = DGLGraph(g)
    n_edges = g.number_of_edges()
    us, vs = g.edges()
    pseudo = []
    for i in range(g.number_of_edges()):
        pseudo.append(
            [1 / np.sqrt(g.in_degree(us[i])), 1 / np.sqrt(g.in_degree(vs[i]))]
        )
    pseudo = torch.Tensor(pseudo)
    if cuda:
        pseudo = pseudo.cuda()

    # create GraphSAGE model
    model = MoNet(
        g,
        in_feats,
        args.n_hidden,
        n_classes,
        args.n_layers,
        args.pseudo_dim,
        args.n_kernels,
        args.dropout,
    )

    if cuda:
        model.cuda()
    loss_fcn = torch.nn.CrossEntropyLoss()

    # use optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    # initialize graph
    dur = []
    for epoch in range(args.n_epochs):
        model.train()
        if epoch >= 3:
            t0 = time.time()
        # forward
        logits = model(features, pseudo)
        loss = loss_fcn(logits[train_mask], labels[train_mask])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch >= 3:
            dur.append(time.time() - t0)

        accuracy, precision, recall, fscore, _ = evaluate(
            model, features, pseudo, labels, val_mask
        )
        print("Epoch:", epoch)
        print("Loss:", loss.item())
        print("Accuracy:", accuracy)
        print("Precision:", precision)
        print("Recall:", recall)
        print("F-Score:", fscore)
        print()
        print("=" * 80)
        print()

    accuracy, precision, recall, fscore, class_based_report = evaluate(
        model, features, pseudo, labels, test_mask
    )
    print("=" * 80)
    print(" " * 28 + "Final Statistics")
    print("=" * 80)
    print("Accuracy", accuracy)
    print("Precision", precision)
    print("Recall", recall)
    print("F-Score", fscore)
    print(class_based_report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MoNet on citation network")
    register_data_args(parser)
    parser.add_argument(
        "--dropout", type=float, default=0.25, help="dropout probability"
    )
    parser.add_argument("--gpu", type=int, default=-1, help="gpu")
    parser.add_argument("--lr", type=float, default=1e-1, help="learning rate")
    parser.add_argument(
        "--n-epochs", type=int, default=800, help="number of training epochs"
    )
    parser.add_argument(
        "--n-hidden", type=int, default=2, help="number of hidden gcn units"
    )
    parser.add_argument(
        "--n-layers", type=int, default=1, help="number of hidden gcn layers"
    )
    parser.add_argument(
        "--pseudo-dim",
        type=int,
        default=2,
        help="Pseudo coordinate dimensions in GMMConv, 2 for cora and 3 for pubmed",
    )
    parser.add_argument(
        "--n-kernels",
        type=int,
        default=3,
        help="Number of kernels in GMMConv layer",
    )
    parser.add_argument(
        "--weight-decay", type=float, default=5e-4, help="Weight for L2 loss"
    )
    args = parser.parse_args()
    print(args)

    main(args)
