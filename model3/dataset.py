from datasets import load_dataset
import pandas as pd
from transformers import AutoTokenizer
from transformers import DataCollatorForLanguageModeling

def data_handle():

    # 获得数据集
    path_train = "../story_generation_dataset/ROCStories_train.csv"
    path_val = "../story_generation_dataset/ROCStories_val.csv"
    path_test = "../story_generation_dataset/ROCStories_test.csv"
    data_files = {"train": path_train, "val": path_val, "test":path_test}

    datasets = load_dataset("csv", data_files=data_files)

    # 处理数据集
    tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
    tokenizer.pad_token = tokenizer.eos_token

    def preprocess_function(item):
        seq = item["storytitle"] + "\n"
        for i in range(1, 6):
            seq += item["sentence" + str(i)] + " "
        return tokenizer(seq, padding="max_length", max_length=128)

    tokenized_datasets = datasets.map(preprocess_function
        ,remove_columns=datasets["train"].column_names)

    def preprocess_function2(item):
        item["labels"] = item["input_ids"].copy()
        return item

    lm_datasets = tokenized_datasets.map(preprocess_function2)

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    return lm_datasets, data_collator