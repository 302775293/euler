import torch
import torch.nn as nn
from transformers import BertModel, BertConfig


class BertClassifier(nn.Module):
    """基于BERT的文本分类模型。

    在BERT编码器之上加一个Dropout + 全连接分类头。
    """

    def __init__(self, pretrained_model_name, num_labels, dropout=0.1):
        super().__init__()
        self.bert = BertModel.from_pretrained(pretrained_model_name)
        hidden_size = self.bert.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        pooled_output = outputs.pooler_output  # [CLS] 向量
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits

    def freeze_bert_layers(self, num_layers_to_freeze=0):
        """冻结BERT前 N 层编码器参数，仅微调后面的层。"""
        if num_layers_to_freeze <= 0:
            return
        modules_to_freeze = [self.bert.embeddings]
        for i in range(min(num_layers_to_freeze, len(self.bert.encoder.layer))):
            modules_to_freeze.append(self.bert.encoder.layer[i])
        for module in modules_to_freeze:
            for param in module.parameters():
                param.requires_grad = False
