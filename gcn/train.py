import argparse, time
import numpy as np
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl import DGLGraph
from dgl.data import register_data_args, load_data

from gcn import GCN

# from gcn_mp import GCN
# from gcn_spmv import GCN

from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import classification_report as report

torch.manual_seed(2)


def evaluate(model, features, labels, mask):
    """Gives accuracy."""
    model.eval()
    with torch.no_grad():
        logits = model(features)
        logits = logits[mask]
        labels = labels[mask].cpu().numpy()

        # Statistics
        _, indices = torch.max(logits, dim=1)
        prediction = indices.long().cpu().numpy()
        accuracy = (prediction == labels).sum() / len(prediction)
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
    if hasattr(torch, "BoolTensor"):
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
            train_mask.int().sum().item(),
            val_mask.int().sum().item(),
            test_mask.int().sum().item(),
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

    # graph preprocess and calculate normalization factor
    g = data.graph
    # add self loop
    if args.self_loop:
        g.remove_edges_from(nx.selfloop_edges(g))
        g.add_edges_from(zip(g.nodes(), g.nodes()))
    g = DGLGraph(g)
    n_edges = g.number_of_edges()
    # normalization
    degs = g.in_degrees().float()
    norm = torch.pow(degs, -0.5)
    norm[torch.isinf(norm)] = 0
    if cuda:
        norm = norm.cuda()
    g.ndata["norm"] = norm.unsqueeze(1)

    # create GCN model
    model = GCN(
        g,
        in_feats,
        args.n_hidden,
        n_classes,
        args.n_layers,
        F.relu,
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
        logits = model(features)
        loss = loss_fcn(logits[train_mask], labels[train_mask])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch >= 3:
            dur.append(time.time() - t0)

        accuracy, precision, recall, fscore, _ = evaluate(
            model, features, labels, val_mask
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
        model, features, labels, test_mask
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
    parser = argparse.ArgumentParser(description="GCN")
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
        "--n-hidden", type=int, default=4, help="number of hidden gcn units"
    )
    parser.add_argument(
        "--n-layers", type=int, default=1, help="number of hidden gcn layers"
    )
    parser.add_argument(
        "--weight-decay", type=float, default=5e-4, help="Weight for L2 loss"
    )
    parser.add_argument(
        "--self-loop",
        action="store_true",
        help="graph self-loop (default=False)",
    )
    parser.set_defaults(self_loop=False)
    args = parser.parse_args()
    print(args)

    main(args)
