import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import tiktoken
from setting import *

enc = tiktoken.get_encoding("cl100k_base")


class ROCStories_dataset(Dataset):
    def __init__(self, path):
        self.data = pd.read_csv(path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        seq_in = self.data["storytitle"][idx] + \
            "\n" + self.data["sentence1"][idx] + " "
        seq_out = ""
        for i in range(2, 6):
            seq_out += self.data["sentence" + str(i)][idx] + " "

        seq_in = enc.encode(seq_in)
        seq_out = [begin_token] + enc.encode(seq_out) + [end_token]

        seq_in = torch.tensor(seq_in)
        seq_out = torch.tensor(seq_out)

        return seq_in, seq_out


def collate(seq_list):  # 给同一个batch内的序列补齐，并生成对应的batch_mask
    seq_in = [x[0] for x in seq_list]
    seq_out = [x[1] for x in seq_list]
    maxlen_in = max([len(x) for x in seq_in])
    maxlen_out = max([len(x) for x in seq_out])
    mask_in = torch.BoolTensor(len(seq_list), maxlen_in)
    mask_out = torch.BoolTensor(len(seq_list), maxlen_out-1)
    for i in range(len(seq_in)):
        pad = (0, maxlen_in - len(seq_in[i]))
        mask = [False] * len(seq_in[i]) + [True] * (maxlen_in - len(seq_in[i]))
        seq_in[i] = nn.functional.pad(
            seq_in[i], pad, mode="constant", value=padding_token
        )
        mask_in[i, :] = torch.tensor(mask)
    for i in range(len(seq_out)):
        pad = (0, maxlen_out - len(seq_out[i]))
        mask = [False] * (len(seq_out[i])-1) + [True] * \
            (maxlen_out - len(seq_out[i])+1)  # 最后一个token，即[end]，不需要考虑其生成
        seq_out[i] = nn.functional.pad(
            seq_out[i], pad, mode="constant", value=padding_token
        )
        mask_out[i, :] = torch.tensor(mask[:-1])  # 输出序列的最后一个token是被丢掉的
    # print(seq_in, seq_out, sep="\n")
    # print(mask_in.shape, mask_out.shape)
    return torch.stack(seq_in), torch.stack(seq_out), mask_in, mask_out
