import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import argparse
import simplification
import csv
import pandas
from tqdm import tqdm


import numpy as np
import TextFooler.dataloader as dataloader
from TextFooler.train_classifier import Model
from simplification import Simplifier
import TextFooler.criteria as criteria
import random
import sys
import tensorflow.compat.v1 as tf
from tqdm import tqdm
tf.disable_v2_behavior()
import tensorflow_hub as hub

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, SequentialSampler, TensorDataset

#from TextFooler.BERT.tokenization import BertTokenizer
#from TextFooler.BERT.modeling import BertForSequenceClassification, BertConfig
from transformers import AutoTokenizer,AutoModelForSequenceClassification
from transformers import  BertForSequenceClassification


class NLI_infer_BERT(nn.Module):
    def __init__(self,
                 pretrained_dir,
                 nclasses,
                 max_seq_length=128,
                 batch_size=32):
        super(NLI_infer_BERT, self).__init__()
        self.model = AutoModelForSequenceClassification.from_pretrained(pretrained_dir, num_labels=nclasses).cuda()

        # construct dataset loader
        self.dataset = NLIDataset_BERT(pretrained_dir, max_seq_length=max_seq_length, batch_size=batch_size)

    def text_pred(self, text_data,simlifier, batch_size=32):
        # Switch the model to eval mode.
        self.model.eval()

        # transform text data into indices and create batches
        dataloader = self.dataset.transform_text(text_data, simlifier,batch_size=batch_size)

        probs_all = []
        #         for input_ids, input_mask, segment_ids in tqdm(dataloader, desc="Evaluating"):
        for input_ids, input_mask, segment_ids in dataloader:
            input_ids = input_ids.cuda()
            input_mask = input_mask.cuda()
            segment_ids = segment_ids.cuda()

            with torch.no_grad():
                logits = self.model(input_ids, input_mask, segment_ids)
                probs = nn.functional.softmax(logits[0], dim=-1)
                probs_all.append(probs)

        return torch.cat(probs_all, dim=0)


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids


class NLIDataset_BERT(Dataset):
    """
    Dataset class for Natural Language Inference datasets.

    The class can be used to read preprocessed datasets where the premises,
    hypotheses and labels have been transformed to unique integer indices
    (this can be done with the 'preprocess_data' script in the 'scripts'
    folder of this repository).
    """

    def __init__(self,
                 pretrained_dir,
                 max_seq_length=128,
                 batch_size=32):
        """
        Args:
            data: A dictionary containing the preprocessed premises,
                hypotheses and labels of some dataset.
            padding_idx: An integer indicating the index being used for the
                padding token in the preprocessed data. Defaults to 0.
            max_premise_length: An integer indicating the maximum length
                accepted for the sequences in the premises. If set to None,
                the length of the longest premise in 'data' is used.
                Defaults to None.
            max_hypothesis_length: An integer indicating the maximum length
                accepted for the sequences in the hypotheses. If set to None,
                the length of the longest hypothesis in 'data' is used.
                Defaults to None.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_dir) #"bert-base-uncased"
        self.max_seq_length = max_seq_length
        self.batch_size = batch_size

    def convert_examples_to_features(self, examples, max_seq_length, tokenizer,simplify):
        """Loads a data file into a list of `InputBatch`s."""

        features = []
        for (ex_index, text_a) in enumerate(examples):
            if simplify != None:
                text_a = simplify(' '.join(text_a))
            else:
                text_a = ' '.join(text_a)
            #print(text_a)
            tokens_a = tokenizer.tokenize(text_a)

            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[:(max_seq_length - 2)]

            tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
            segment_ids = [0] * len(tokens)

            input_ids = tokenizer.convert_tokens_to_ids(tokens)

            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            input_mask = [1] * len(input_ids)

            # Zero-pad up to the sequence length.
            padding = [0] * (max_seq_length - len(input_ids))
            input_ids += padding
            input_mask += padding
            segment_ids += padding

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length

            features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids))
        return features

    def transform_text(self, data, simplify,batch_size=32):
        # transform data into seq of embeddings
        eval_features = self.convert_examples_to_features(data,
                                                          self.max_seq_length, self.tokenizer,simplify)

        all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
        eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids)

        # Run prediction for full data
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=batch_size)

        return eval_dataloader


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--complex_threshold",
                        type=float,
                        default=0.1,
                        help="Threshold is defined differently with different simplifiy_version.")
    parser.add_argument("--simplify_version",
                        type=str,
                        default='v1',
                        help="Choose from v1 to v6.")
    parser.add_argument("--file_to_simplify",
                        type=str,
                        help="Input path for original train.tsv.")
    parser.add_argument("--output_path",
                        type=str,
                        required=True,
                        help="Output path for simplified train.tsv.")
    parser.add_argument("--ratio",
                        type=float,
                        default=0.2,
                        help="ratio of words to be changed.")
    parser.add_argument("--syn_num",
                        type=int,
                        default=10,
                        help="ratio of words to be changed.")
    parser.add_argument("--most_freq_num",
                        type=int,
                        default=5,
                        help="ratio of words to be changed.")
    parser.add_argument("--detector_model_path",
                        type=str)

    args = parser.parse_args()
    simplifier = simplification.Simplifier(threshold=args.complex_threshold, ratio=args.ratio, syn_num =args.syn_num, most_freq_num =args.most_freq_num)
    simplify_dict = {'v2': simplifier.simplify_v2,
                     'random_freq_v1': simplifier.random_freq_v1,
                     'random_freq_v2': simplifier.random_freq_v2}
    simplify = simplify_dict[args.simplify_version]

    detector_model = NLI_infer_BERT(args.detector_model_path, nclasses=2, batch_size=1)
    detector = detector_model.text_pred

    # read data
    IDs = []
    sents = []
    labels = []
    with open(args.file_to_simplify, "r", encoding="utf-8-sig") as f:
        data = csv.reader(f, delimiter="\t")
        for row in data:
            if row[1] == 'sentence':
                continue
            IDs.append(row[0])
            sents.append(row[1])
            labels.append(row[2])
    # simplify sentences
    simp_sents = []
    simp_IDs = []
    simp_labels = []
    for ID, sent, label in tqdm(zip(IDs, sents, labels)):
        for i in range(20):
            #print(i)
            simp_sent = simplify(sent)
            detect_probs = detector([simp_sent.split()], None, batch_size=1)
            #print(detect_probs, torch.argmax(detect_probs))
            if torch.argmax(detect_probs) == 1:
                simp_sents.append(simp_sent)
                simp_IDs.append(ID)
                simp_labels.append(label)
                break
            elif i == 19:
                #print("oh no!\n")
                simp_sents.append(simp_sent)
                simp_IDs.append(ID)
                simp_labels.append(label)


    # store simplified results
    simp_dataframe = pandas.DataFrame({'': simp_IDs, 'sentence': simp_sents, 'label': simp_labels})
    simp_dataframe.to_csv(args.output_path, sep='\t', index=False)

if __name__ == "__main__":
    main()