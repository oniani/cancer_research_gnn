# Results

## GraphSAGE

### Mean Aggregator

| Statistic | Value              |
| ----------| ------------------ |
| Accuracy  | 0.9012345679012346 |
| Precision | 0.9261904761904762 |
| Recall    | 0.9102607709750566 |
| F-Score   | 0.9147043432757718 |

| Hyper-parameter             | Value         |
| --------------------------- | ------------- |
| Dropout probability         | 0.25          |
| Learning rate               | 1e-2 (0.01)   |
| Number of training epochs   | 800           |
| Number of hidden gcn units  | 16            |
| Number of hidden gcn layers | 1             |
| Weight for L2 loss          | 5e-4 (0.0005) |
| Aggregator type             | mean          |

### GCN Aggregator

| Statistic | Value               |
| ----------| ------------------- |
| Accuracy  | 0.5185185185185185  |
| Precision | 0.34514790764790765 |
| Recall    | 0.4333333333333333  |
| F-Score   | 0.3604218362282878  |

| Hyper-parameter              | Value         |
| -----------------------------| ------------- |
| Dropout probability          | 0.25          |
| Learning rate                | 1e-1 (0.1)    |
| Number of training epochs    | 800           |
| Number of hidden gcn units   | 2             |
| Number of hidden gcn layers  | 1             |
| Weight for L2 loss           | 5e-4 (0.0005) |
| Aggregator type              | gcn           |


## Monet

|Statistic | Value                | 
| -------- | -------------------- |
| Accuracy | 0.2222222222222222   |
| Precision| 0.031746031746031744 |
| Recall   | 0.14285714285714285  |
| F-Score  | 0.051948051948051945 |

| Hyper-parameter                                                      | Value        |
| -------------------------------------------------------------------- | ------------ |
| Dropout probability                                                  | 0.25         |
| Learning rate                                                        | 1e-1 (0.1)   |
| Number of training epochs                                            | 800          |
| Number of hidden gcn units                                           | 2            |
| Number of hidden gcn layers                                          | 1            |
| Pseudo coordinate dimensions in GMMConv, 2 for cora and 3 for pubmed | 2            |
| Number of kernels in GMMConv layer                                   | 3            |
| Weight for L2 loss                                                   | 5e-4 (0.0005)|


## Feature Engineering

```sh
python generate_data.py --num_classes=6
python feature_engineering.py
```
