from torch.utils.data import Dataset
import torch
import pandas as pd
import numpy as np


class NaverMovieCorpus(Dataset):

    def __init__(self, filepath, tokenizer, padder):
        self.df = pd.read_csv(filepath,  sep='\t')
        self.tokenizer = tokenizer
        self.padder = padder

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx) -> torch.Tensor:
        document = self.df.iloc[idx]['document']
        label = self.df.iloc[idx]['label']
        tokenized_document = self.padder(self.tokenizer.tokenize_and_transform(document))
        data_sample = (torch.tensor(tokenized_document), torch.tensor(label))
        return data_sample