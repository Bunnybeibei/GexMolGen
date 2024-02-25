import pandas as pd
import numpy as np

from typing import Dict

import torch
from torch.utils.data import Dataset, DataLoader

from scGPT.scgpt.tokenizer.gene_tokenizer import GeneVocab
from scGPT.scgpt.tokenizer import tokenize_and_pad_batch


def normalize_total(X, target_sum=None):
    X = np.round(X,3)
    # Calculate the total expression level of each sample.
    total = np.sum(X, axis=1, keepdims=True)
    # Calculate normalization factors.
    if target_sum is None:
        target_sum = 1e3
    scale_factor = target_sum / total
    # Normalize each sample.
    normed = np.round(X * scale_factor, 3)
    return normed


def log1p(X):
    # Perform log(1+x) transformation on each element.
    transformed = np.round(np.log1p(X),3)
    return transformed


def preprocess_bulk(data):
    return log1p(normalize_total(data))


class SeqDataset(Dataset):
    def __init__(self, data: Dict[str, torch.Tensor]):
        self.data = data

    def __len__(self):
        return self.data["genes"].shape[0]

    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.data.items()}


def prepare_dataloader(
    data_pt: Dict[str, torch.Tensor],
    batch_size: int,
    shuffle: bool = False,
    drop_last: bool = False,
    num_workers: int = 0,
    ) -> DataLoader:
    dataset = SeqDataset(data_pt)

    data_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
        pin_memory=True,
    )
    return data_loader


class dataPrepare(object):
    def __init__(self,
                 genetic_vocab_file,
                 batch_size=100,):
        """
        Initializing data processing function.
        Args:
            vocab_file: Pretrained gene categories
            batch_size: Batch size
            max_len: The sequence length for inputting into FlashAttention.
        """
        self.vocab = GeneVocab.from_file(genetic_vocab_file)
        for s in ["<pad>", "<cls>", "<eoc>"]:
            if s not in self.vocab:
                self.vocab.append_token(s)
        self.batch_size = batch_size


    def get_Gene_id(self, control, pert):
        """
        Remove genes that are not in the pre-trained gene list
        and output the numerical representation of genes in the pre-trained gene list.
        Additionally, combine the control group and experimental group into a single dataset.

        Args:
            control: Gene expression data from the control groups.
            pert: Gene expression data from the pert groups.

        Returns:
            gene_ids: The numerical representation of genes in the pre-trained gene list.
            combined_data: The combined dataset of the control group and experimental group.
        """
        idInVocab = [
            1 if gene in self.vocab else -1 for gene in control.columns
        ]
        gene_ids_in_vocab = np.array(idInVocab)
        control = control.loc[:, gene_ids_in_vocab >= 0]
        pert = pert.loc[:, gene_ids_in_vocab >= 0]

        gene_ids = np.array(self.vocab(pert.columns.tolist()), dtype=int)
        gene_ids = np.concatenate([gene_ids, gene_ids])

        combined_data = pd.concat([control,pert],axis=1)
        return gene_ids, combined_data


    def main(self, control, pert):

        gene_ids, test_data = self.get_Gene_id(control=control, pert=pert)
        test_data.iloc[:,:] = preprocess_bulk(test_data.values)
        vals = test_data.values.astype(float)

        tokenized_test = tokenize_and_pad_batch(
            vals,
            gene_ids=gene_ids,
            max_len = test_data.shape[1],
            vocab=self.vocab,
            pad_token="<pad>",
            pad_value=-2,
            append_cls=False,  # append <cls> token at the beginning
            include_zero_gene=True,
        )

        test_loader = prepare_dataloader(
            tokenized_test,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
        )

        return test_loader