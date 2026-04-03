import logging

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

logger = logging.getLogger(__name__)


class TextClassificationDataset(Dataset):
    """CSV格式文本分类数据集。

    在 __init__ 阶段对全部文本进行 tokenize，
    __getitem__ 只做张量切片，大幅减少训练时的开销。
    """

    def __init__(self, csv_path, tokenizer, max_length, text_column="text",
                 label_column="label", label2id=None):
        self.csv_path = csv_path
        df = pd.read_csv(csv_path)

        if text_column not in df.columns:
            raise ValueError(f"CSV 缺少文本列 '{text_column}'，可用列: {list(df.columns)}")
        if label_column not in df.columns:
            raise ValueError(f"CSV 缺少标签列 '{label_column}'，可用列: {list(df.columns)}")

        texts = df[text_column].fillna("").astype(str).tolist()
        raw_labels = df[label_column].tolist()

        if label2id is not None:
            self.labels = [label2id[str(l)] for l in raw_labels]
        else:
            self.labels = [int(l) for l in raw_labels]

        num_classes = len(set(self.labels))
        logger.info("加载 %s: %d 条样本, %d 个类别", csv_path, len(texts), num_classes)

        self.encodings = tokenizer(
            texts,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "token_type_ids": self.encodings["token_type_ids"][idx],
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
        }


def _detect_string_labels(df, label_column, sample_size=20):
    """检测标签列是否包含非数字字符串。"""
    sample = df[label_column].head(sample_size)
    return any(isinstance(l, str) and not l.isdigit() for l in sample)


def build_dataloader(csv_path, tokenizer, config, shuffle=True):
    """构建 DataLoader，自动检测标签类型。"""
    df = pd.read_csv(csv_path)
    label2id = config.label2id() if _detect_string_labels(df, config.label_column) else None

    dataset = TextClassificationDataset(
        csv_path=csv_path,
        tokenizer=tokenizer,
        max_length=config.max_length,
        text_column=config.text_column,
        label_column=config.label_column,
        label2id=label2id,
    )

    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        num_workers=getattr(config, "num_workers", 0),
        pin_memory=True,
        drop_last=False,
    )
