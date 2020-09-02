# Results

## GraphSAGE

### Mean Aggregator

| Hyper-parameter                                   | Value        |
| ------------------------------------------------- | ------------ |
| Dropout probability                               | 0.25         |
| Learning rate                                     | 1e-2 (0.01)  |
| Number of training epochs                         | 800          |
| Number of hidden gcn units  (if gcn is specified) | 16           |
| Number of hidden gcn layers (if gcn is specified) | 1            |
| Weight for L2 loss                                | 5e-4 (0.005) |
| Aggregator type                                   | mean         |

- Accuracy: 0.9012345679012346
- Precision: 0.9261904761904762
- Recall: 0.9102607709750566
- F-Score: 0.9147043432757718

### GCN Aggregator

| Hyper-parameter                                   | Value        |
| ------------------------------------------------- | ------------ |
| Dropout probability                               | 0.25         |
| Learning rate                                     | 1e-1 (0.01)  |
| Number of training epochs                         | 800          |
| Number of hidden gcn units  (if gcn is specified) | 2            |
| Number of hidden gcn layers (if gcn is specified) | 1            |
| Weight for L2 loss                                | 5e-4 (0.005) |
| Aggregator type                                   | gcn          |

- Accuracy: 0.5185185185185185
- Precision: 0.34514790764790765
- Recall: 0.4333333333333333
- F-Score: 0.3604218362282878


## Feature Engineering

```sh
python generate_data.py --num_classes=6
python feature_engineering.py
```
