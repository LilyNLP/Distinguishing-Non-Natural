import argparse
import csv
import pandas
from tqdm import tqdm
import os
from shutil import copyfile


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--abnormal_file",
                        type=str,
                        required=True,
                        help="Input path for abnormal data.")
    parser.add_argument("--abnormal_file_type",
                        type=str,
                        default='txt',
                        help="tsv or txt.")
    parser.add_argument("--abnormal_number",
                        type=int,
                        default=-1,
                        help="How many abnormal examples to extract.")
    parser.add_argument("--normal_file",
                        type=str,
                        required=True,
                        help="Input path for normal data.")
    parser.add_argument("--normal_file_type",
                        type=str,
                        default='tsv',
                        help="tsv or txt.")
    parser.add_argument("--normal_number",
                        type=int,
                        default=-1,
                        help = "How many normal examples to extract.")
    parser.add_argument("--output_file",
                        type=str,
                        required=True,
                        help="Output path for combined data.")
    parser.add_argument("--output_file_type",
                        type=str,
                        default='tsv',
                        help="tsv or txt.")
    args = parser.parse_args()


    # read in normal file
    IDs = []
    sents = []
    labels = []
    i = 0
    if args.normal_file_type == 'tsv':
        with open(args.normal_file, "r", encoding="utf-8-sig") as f:
            data = csv.reader(f, delimiter="\t")
            for row in data:
                if row[1] == 'sentence' or row[1] == 'query':
                    continue
                sents.append(row[1])
                labels.append('0')
                i += 1
                IDs.append(i)
                if i >= args.normal_number and args.normal_number != -1:
                    break
    elif args.normal_file_type == 'txt':
        with open(args.normal_file) as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                line = line.replace('\n ', ' ').replace('\t', ' ')
                if line[0] == '0':
                    sents.append(line.replace('0 ', ''))
                if line[0] == '1':
                    sents.append(line.replace('1 ', ''))
                labels.append('0')
                i += 1
                IDs.append(i)
                if i >= args.normal_number and args.normal_number != -1:
                    break

    
    # read in abnormal file
    if args.abnormal_file_type == 'tsv':
        with open(args.abnormal_file, "r", encoding="utf-8-sig") as f:
            data = csv.reader(f, delimiter="\t")
            for row in data:
                if row[1] == 'sentence' or row[1] == 'query':
                    continue
                sents.append(row[1])
                labels.append('1')
                i += 1
                IDs.append(i)
                if i >= args.normal_number + args.abnormal_number and args.abnormal_number != -1:
                    break
    elif args.abnormal_file_type == 'txt':
        with open(args.abnormal_file) as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                line = line.replace('\n ', ' ').replace('\t', ' ')
                if line[0] == '0':
                    sents.append(line.replace('0 ', '') + ' .')
                if line[0] == '1':
                    sents.append(line.replace('1 ', '')+ ' .')
                labels.append('1')
                i += 1
                IDs.append(i)
                if i >= args.normal_number + args.abnormal_number and args.abnormal_number != -1:
                    break


    # write to output file
    if args.output_file_type == 'tsv':
        dataframe = pandas.DataFrame({'': IDs, 'sentence': sents, 'label': labels})
        dataframe.to_csv('./temp/temp.tsv', sep='\t', index=False)
        print("output file in form of tsv built")
        import random
        out = open(args.output_file, 'w', encoding='utf-8')
        lines = []
        with open('temp/temp.tsv', 'r', encoding='utf-8') as infile:
            for (i, line) in enumerate(infile):
                if i == 0:
                    out.write(line)
                    continue
                lines.append(line)
            random.shuffle(lines)
            random.shuffle(lines)
            random.shuffle(lines)
            for line in lines:
                out.write(line)
            out.close()
    elif args.output_file_type == 'txt':
        out = open(args.output_file, 'w')
        lines = []
        for (label, sent) in zip(labels, sents):
            lines.append(label + ' ' + sent + '\n')
        random.shuffle(lines)
        random.shuffle(lines)
        random.shuffle(lines)
        for line in lines:
            out.write(line)
        out.close()

if __name__ == "__main__":
    main()