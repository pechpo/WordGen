import torch
from torch.utils.data import Dataset
import pandas as pd
import tiktoken
enc = tiktoken.get_encoding("cl100k_base")

class ROCStories_dataset(Dataset):
    def __init__(self, path):
        self.data = pd.read_csv(path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        seq_in = self.data["storytitle"][idx] + "\n" + self.data["sentence1"][idx] + " "
        seq_out = ""
        for i in range(2, 6):
            seq_out += self.data["sentence" + str(i)][idx] + " "

        seq_in = enc.encode(seq_in)
        seq_out = [1] + enc.encode(seq_out)

        seq_in = torch.tensor(seq_in)
        seq_out = torch.tensor(seq_out)

        return seq_in, seq_out