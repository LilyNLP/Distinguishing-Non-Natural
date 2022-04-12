import argparse
import csv
import pandas
from tqdm import tqdm
import os
from shutil import copyfile


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file_1",
                        type=str,
                        required=True,
                        help="Input path for abnormal data.")
    parser.add_argument("--input_file_2",
                        type=str,
                        required=True,
                        help="Input path for abnormal data.")
    parser.add_argument("--output_file",
                        type=str,
                        required=True,
                        help="Output path for combined data.")
    args = parser.parse_args()

    import random

    f_1 = open(args.input_file_1, "r", encoding="utf-8-sig")
    f_2 = open(args.input_file_2, "r", encoding="utf-8-sig")
    import random
    out = open(args.output_file, 'w', encoding='utf-8')
    lines = []
    with open(args.input_file_1, 'r', encoding='utf-8') as infile:
        for (i, line) in enumerate(infile):
            if i == 0:
                out.write(line)
                continue
            lines.append(line)

    with open(args.input_file_2, 'r', encoding='utf-8') as infile:
        for (i, line) in enumerate(infile):
            if i == 0:
                continue
            lines.append(line)

    random.shuffle(lines)
    random.shuffle(lines)
    random.shuffle(lines)
    for line in lines:
        out.write(line)
    out.close()


if __name__ == "__main__":
    main()
