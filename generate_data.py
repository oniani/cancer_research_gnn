# Generate new data based on select few classes

import argparse
import csv
import random

from collections import Counter


random.seed(729)


def get_class(data, classname):
    """Gets a list of patients based on a class."""

    return [row for row in data if row[-1] == classname]


def main():
    """The main function."""

    parser = argparse.ArgumentParser(description="Process the arguments.")
    parser.add_argument("--num_classes", type=int, default=5, help="Retain a specified number of classes")
    args = parser.parse_args()

    # cites = list(csv.reader(open("temp.cites"), delimiter="\t"))
    content = list(csv.reader(open("temp.content"), delimiter="\t"))

    classes = dict(Counter([row[-1] for row in content]).most_common(args.num_classes))

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


    # with open("cora.cites", "w") as file:
    #    cites_writer = csv.writer(file, delimiter="\t")
    #    for row in cites:
    #        if row[0] in ids and row[1] in ids:
    #            cites_writer.writerow(row)


if __name__ == "__main__":
    main()
