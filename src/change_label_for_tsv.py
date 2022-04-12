import csv
import pandas
import argparse
from tqdm import tqdm
import os
from shutil import copyfile


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file",
                        type=str,
                        required=True,
                        help="Input path for abnormal data.")
    parser.add_argument("--output_file",
                        type=str,
                        required=True,
                        help="Output path for combined data.")
    parser.add_argument("--change_to", type=str,required = True)
    args = parser.parse_args()

    sents = []
    labels = []
    IDs = []
    f = open(args.input_file, "r", encoding="utf-8-sig")
    data = csv.reader(f, delimiter="\t")
    for (i, row) in enumerate(data):
        if i == 0:
            continue
        id = row[0]
        sent = row[1]
        label = args.change_to
        sents.append(sent)
        labels.append(label)
        IDs.append(id)
    f.close()

    dataframe = pandas.DataFrame({'': IDs, 'sentence': sents, 'label': labels})
    dataframe.to_csv(args.output_file, sep='\t', index=False)

if __name__ == "__main__":
    main()