# Results

- [GraphSAGE](#graphsage)
  - [Mean Aggregator](#mean-aggregator)
  - [GCN Aggregator](#gcn-aggregator)
- [MoNet](#monet)
- [GAT](#gat)
- [GCN](#gcn)
- [APPNP](#appnp)
- [GIN](#gin)
- [TAGCN](#tagcn)
- [SGC](#sgc)
- [AGNN](#agnn)
- [ChebNet](#chebnet)

## GraphSAGE

### Mean Aggregator

<table>
<tr><th>Statistics</th><th>Hyperparameters</th></tr>
<tr><td><table></table>

| Statistic | Value              |
| ----------| ------------------ |
| Accuracy  | 0.9012345679012346 |
| Precision | 0.9261904761904762 |
| Recall    | 0.9102607709750566 |
| F-Score   | 0.9147043432757718 |

</td><td>

| Hyperparameter              | Value         |
| --------------------------- | ------------- |
| Dropout probability         | 0.25          |
| Learning rate               | 1e-2 (0.01)   |
| Number of training epochs   | 800           |
| Number of hidden gcn units  | 16            |
| Number of hidden gcn layers | 1             |
| Weight for L2 loss          | 5e-4 (0.0005) |
| Aggregator type             | mean          |

</td></tr> </table>

### GCN Aggregator

<table>
<tr><th>Statistics</th><th>Hyperparameters</th></tr>
<tr><td>

| Statistic | Value                |
| ----------| -------------------- |
| Accuracy  | 0.2222222222222222   |
| Precision | 0.031746031746031744 |
| Recall    | 0.14285714285714285  |
| F-Score   | 0.051948051948051945 |

</td><td>

| Hyperparameter              | Value         |
| --------------------------- | ------------- |
| Dropout probability         | 0.25          |
| Learning rate               | 1e-1 (0.1)    |
| Number of training epochs   | 800           |
| Number of hidden gcn units  | 2             |
| Number of hidden gcn layers | 1             |
| Weight for L2 loss          | 5e-4 (0.0005) |
| Aggregator type             | gcn           |

</td></tr> </table>

## MoNet

<table>
<tr><th>Statistics</th><th>Hyperparameters</th></tr>
<tr><td>

|Statistic | Value                | 
| -------- | -------------------- |
| Accuracy | 0.2222222222222222   |
| Precision| 0.031746031746031744 |
| Recall   | 0.14285714285714285  |
| F-Score  | 0.051948051948051945 |

</td><td>

| Hyperparameter                                                       | Value         |
| -------------------------------------------------------------------- | ------------- |
| Dropout probability                                                  | 0.25          |
| Learning rate                                                        | 1e-1 (0.1)    |
| Number of training epochs                                            | 800           |
| Number of hidden gcn units                                           | 2             |
| Number of hidden gcn layers                                          | 1             |
| Pseudo coordinate dimensions in GMMConv, 2 for cora and 3 for pubmed | 2             |
| Number of kernels in GMMConv layer                                   | 3             |
| Weight for L2 loss                                                   | 5e-4 (0.0005) |

</td></tr> </table>

## GAT

<table>
<tr><th>Statistics</th><th>Hyperparameters</th></tr>
<tr><td>

| Statistic | Value               |
| --------- | ------------------- |
| Accuracy  | 0.3333333333333333  |
| Precision | 0.13022060398372243 |
| Recall    | 0.2777777777777778  |
| F-Score   | 0.16048370567739292 |

</td><td>

| Hyperparameter                             | Value         |
| ------------------------------------------ | ------------- |
| Number of training epochs                  | 800           |
| Number of hidden attention heads           | 4             |
| Uumber of output attention heads           | 1             |
| Number of hidden layers                    | 1             |
| Number of hidden units                     | 8             |
| Use residual connection                    | False         |
| Input feature dropout                      | 0.4           |
| Attention dropout                          | 0.25          |
| Learning rate                              | 1e-2 (0.01)   |
| Weight decay                               | 5e-4 (0.0005) |
| The negative slope of leaky relu           | 0.2           |
| Indicates whether to use early stop or not | False         |
| Skip re-evaluate the validation set        | False         |

</td></tr> </table>

## GCN

<table>
<tr><th>Statistics</th><th>Hyperparameters</th></tr>
<tr><td>

| Statistic | Value                |
| --------- | -------------------- |
| Accuracy  | 0.8024691358024691   |
| Precision | 0.8550170307357392 |
| Recall    | 0.786734693877551  |
| F-Score   | 0.8035058227176454 |

</td><td>

| Hyperparameter              | Value         |
| --------------------------- | ------------- |
| Dropout probability         | 0          |
| Learning rate               | 1e-2 (0.01)    |
| Number of training epochs   | 4000           |
| Number of hidden gcn units  | 500             |
| Number of hidden gcn layers | 1             |
| Weight for L2 loss          | 0 |

</td></tr> </table>

## APPNP

<table>
<tr><th>Statistics</th><th>Hyperparameters</th></tr>
<tr><td>

| Statistic | Value              |
| --------- | ------------------ |
| Accuracy  | 0.8888888888888888 |
| Precision | 0.9251082251082252 |
| Recall    | 0.8943877551020407 |
| F-Score   | 0.9049666689418242 |

</td><td>

| Hyperparameter              | Value         |
| --------------------------- | ------------- |
| Input feature dropout       | 0.25          |
| Edge propagation dropout    | 0.5           |
| Learning rate               | 1e-1 (0.1)    |
| Number of training epochs   | 800           |
| Hidden unit sizes for appnp | [64]          |
| Number of propagation steps | 10            |
| Teleport Probability        | 0.4           |
| Weight for L2 loss          | 5e-4 (0.0005) |

</td></tr> </table>

## GIN

<table>
<tr><th>Statistics</th><th>Hyperparameters</th></tr>
<tr><td>

| Statistic | Value              |
| --------- | ------------------ |
| Accuracy  | 0.8271604938271605 |
| Precision | 0.8205627705627706 |
| Recall    | 0.8091836734693878 |
| F-Score   | 0.8045525902668759 |

</td><td>

| Hyperparameter            | Value            |
| ------------------------- | ---------------- |
| Extra args                | [16, 1, 0, True] |
| Learning rate             | 1e-2 (0.01)      |
| Weight decay              | 5e-6 (0.000005)  |
| Number of training epochs | 800              |

</td></tr> </table>

## TAGCN

<table>
<tr><th>Statistics</th><th>Hyperparameters</th></tr>
<tr><td>

| Statistic | Value              |
| --------- | ------------------ |
| Accuracy  | 0.9012345679012346 |
| Precision | 0.9070381998953428 |
| Recall    | 0.9102607709750566 |
| F-Score   | 0.9041060526774812 |

</td><td>

| Hyperparameter            | Value                 |
| ------------------------- | --------------------- |
| Extra args                | [16, 1, F.relu, 0.5]  |
| Learning rate             | 1e-2 (0.01)           |
| Weight decay              | 5e-4 (0.0005)         |
| Number of training epochs | 800                   |

</td></tr> </table>

## SGC

<table>
<tr><th>Statistics</th><th>Hyperparameters</th></tr>
<tr><td>

| Statistic | Value               |
| --------- | ------------------- |
| Accuracy  | 0.2345679012345679  |
| Precision | 0.17500000000000002 |
| Recall    | 0.15714285714285717 | 
| F-Score   | 0.07845216008481315 |

</td><td>

| Hyperparameter            | Value            |
| ------------------------- | ---------------- |
| Extra args                | [None, 2, False] |
| Learning rate             | 1e-1 (0.1)       |
| Weight decay              | 5e-6 (0.000005)  |
| Number of training epochs | 800              |

</td></tr> </table>

## AGNN

<table>
<tr><th>Statistics</th><th>Hyperparameters</th></tr>
<tr><td>

| Statistic | Value                |
| --------- | -------------------- |
| Accuracy  | 0.2222222222222222   |
| Precision | 0.031746031746031744 |
| Recall    | 0.14285714285714285  |
| F-Score   | 0.051948051948051945 |

</td><td>

| Hyperparameter            | Value                   |
| ------------------------- | ----------------------- |
| Extra args                | [32, 2, 1.0, True, 0.5] |
| Learning rate             | 1e-1 (0.1)              |
| Weight decay              | 5e-4 (0.0005)           |
| Number of training epochs | 800                     |

</td></tr> </table>

## ChebNet

<table>
<tr><th>Statistics</th><th>Hyperparameters</th></tr>
<tr><td>

| Statistic | Value              |
| --------- | -------------------|
| Accuracy  | 0.9012345679012346 |
| Precision | 0.9022735409953455 |
| Recall    | 0.9201814058956915 |
| F-Score   | 0.9073651359365645 |

</td><td>

| Hyperparameter            | Value            |
| ------------------------- | ---------------- |
| Extra args                | [32, 1, 2, True] |
| Learning rate             | 1e-2 (0.001)     |
| Weight decay              | 5e-4 (0.0005)    |
| Number of training epochs | 800              |

</td></tr> </table>

## Feature Engineering

```sh
python generate_data.py --num_classes=6
python feature_engineering.py
```
