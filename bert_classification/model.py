import torch
import torch.nn as nn
from transformers import BertModel


class BertClassifier(nn.Module):
    """基于 BERT 的文本分类模型。

    支持三种池化策略获取句子表示:
      - "cls":    取 last_hidden_state[:, 0, :]（CLS 位置）
      - "mean":   对非 padding token 的隐层做均值池化
      - "pooler": 使用 BERT 自带的 pooler_output
    """

    def __init__(self, pretrained_model_name, num_labels, dropout=0.1,
                 pooling="cls"):
        super().__init__()
        self.bert = BertModel.from_pretrained(pretrained_model_name)
        self.pooling = pooling
        hidden_size = self.bert.config.hidden_size

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)
        self._init_classifier()

    def _init_classifier(self):
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def _pool(self, outputs, attention_mask):
        if self.pooling == "pooler":
            return outputs.pooler_output
        last_hidden = outputs.last_hidden_state
        if self.pooling == "mean":
            mask = attention_mask.unsqueeze(-1).float()
            return (last_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        return last_hidden[:, 0, :]

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        pooled = self._pool(outputs, attention_mask)
        pooled = self.dropout(pooled)
        return self.classifier(pooled)

    def freeze_bert_layers(self, num_layers_to_freeze=0):
        """冻结 BERT 前 N 层编码器参数，仅微调后面的层和分类头。"""
        if num_layers_to_freeze <= 0:
            return
        modules_to_freeze = [self.bert.embeddings]
        for i in range(min(num_layers_to_freeze, len(self.bert.encoder.layer))):
            modules_to_freeze.append(self.bert.encoder.layer[i])
        for module in modules_to_freeze:
            for param in module.parameters():
                param.requires_grad = False

    def count_parameters(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable
