#!/usr/bin/env python

# Description: Generate new data based on select few classes
# Author: David Oniani
# Date: 17-07-2020

import csv
import random

from collections import Counter


def get_class(data, classname):
    """Gets a list of patients based on a class."""

    return [row for row in data if row[-1] == classname]


def main():
    """The main function."""

    cites = list(csv.reader(open("temp.cites"), delimiter="\t"))
    content = list(csv.reader(open("temp.content"), delimiter="\t"))

    classes = [row[-1] for row in Counter(content)]

    limit = min(classes.values())

    data = []
    for classname in classes:
        data.extend(random.sample(get_class(content, classname), limit))
        # data.extend(get_class(content, classname))

    ids = []
    with open("cora.content", "w") as file:
       content_writer = csv.writer(file, delimiter="\t")
       for row in data:
           ids.append(row[0])
           content_writer.writerow(row)


    with open("cora.cites", "w") as file:
        cites_writer = csv.writer(file, delimiter="\t")
        for row in cites:
            if row[0] in ids and row[1] in ids:
                cites_writer.writerow(row)


if __name__ == "__main__":
    main()
