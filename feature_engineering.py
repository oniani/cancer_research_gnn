# Feature engineering
import csv
import random

from collections import Counter  

import pandas as pd
import numpy as np

from sklearn.ensemble.forest import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support as score


random.seed(729)


def main():
    header = ["id"]
    feats = []
    df = pd.read_csv("cora.content", sep="\t")
    for i in range(df.shape[1]):
        feat = "feat_" + str(i)
        header.append(feat)
        feats.append(feat)
    header.append("class")

    feats = np.array(feats)

    df.columns = header

    x_train, x_test, y_train, y_test = train_test_split(df[feats], df["class"], test_size=0.3)

    clf = RandomForestClassifier(n_estimators = 200)
    clf.fit(x_train, y_train)

    importances = clf.feature_importances_
    sorted_idx = np.argsort(importances)

    x = list(zip(feats[sorted_idx], importances[sorted_idx]))
    x_sorted = sorted(x, key=lambda x: -x[1])

    # Statistics
    y_pred = clf.predict(x_test)
    precision, recall, fscore, _ = score(y_test, y_pred, average="macro")

    print("Precision:", round(precision, 3))
    print("Recall:   ", round(recall, 3))
    print("F-Score:  ", round(fscore, 3))
    print("Accuracy: ", round((y_pred == y_test).sum() / len(y_pred), 3))

    selected_feats = [key for key, val in x_sorted[:20]]
    print(selected_feats)

    # content = list(csv.reader(open("temp.content"), delimiter="\t"))

    # A hack, but works (:
    # idxs = [int(feat.split("_")[1]) for feat in selected_feats]

    # with open("cora.content", "w") as file:
    #    content_writer = csv.writer(file, delimiter="\t")
    #    for row in content:
    #        lst = [row[0]]
    #        lst.extend([val for idx, val in enumerate(row) if idx in idxs])
    #        lst.extend([row[-1]])
    #        content_writer.writerow(lst)


if __name__ == "__main__":
    main()
