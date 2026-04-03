import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer


class TextClassificationDataset(Dataset):
    """CSV格式文本分类数据集。

    CSV文件应包含至少两列：文本列和标签列。
    标签可以是整数（0~num_labels-1）或字符串（自动映射为整数）。
    """

    def __init__(self, csv_path, tokenizer, max_length, text_column="text",
                 label_column="label", label2id=None):
        self.df = pd.read_csv(csv_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.text_column = text_column
        self.label_column = label_column
        self.label2id = label2id

        self.texts = self.df[self.text_column].astype(str).tolist()
        raw_labels = self.df[self.label_column].tolist()

        if self.label2id is not None:
            self.labels = [self.label2id[str(l)] for l in raw_labels]
        else:
            self.labels = [int(l) for l in raw_labels]

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "token_type_ids": encoding["token_type_ids"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.long),
        }


def build_dataloader(csv_path, tokenizer, config, shuffle=True):
    """构建 DataLoader。"""
    label2id = None
    if any(isinstance(l, str) and not l.isdigit()
           for l in pd.read_csv(csv_path)[config.label_column].head(10)):
        label2id = config.label2id()

    dataset = TextClassificationDataset(
        csv_path=csv_path,
        tokenizer=tokenizer,
        max_length=config.max_length,
        text_column=config.text_column,
        label_column=config.label_column,
        label2id=label2id,
    )

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=True,
    )
