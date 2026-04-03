"""
BERT 13分类 fine-tune 训练脚本。

使用传统的 PyTorch 训练循环，不使用 Trainer 等高阶库。

用法:
    python train.py [--config_override key=value ...]

示例:
    python train.py
    python train.py --train_file data/train.csv --val_file data/val.csv --num_epochs 3
"""

import argparse
import json
import os
import time

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from transformers import BertTokenizer
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm

from config import Config
from dataset import build_dataloader
from model import BertClassifier


def parse_args():
    parser = argparse.ArgumentParser(description="BERT 13分类训练脚本")
    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--train_file", type=str, default=None)
    parser.add_argument("--val_file", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--num_epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--max_length", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def apply_args(config, args):
    """将命令行参数覆盖到 Config 中。"""
    for key in ["model_name", "train_file", "val_file", "output_dir",
                "num_epochs", "batch_size", "learning_rate", "max_length", "device"]:
        value = getattr(args, key, None)
        if value is not None:
            setattr(config, key, value)


def train_one_epoch(model, dataloader, optimizer, scheduler, criterion, device,
                    log_interval=50):
    model.train()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    step = 0

    progress = tqdm(dataloader, desc="Training", leave=False)
    for batch in progress:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        token_type_ids = batch["token_type_ids"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        logits = model(input_ids, attention_mask, token_type_ids)
        loss = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), Config.max_grad_norm)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        preds = torch.argmax(logits, dim=-1)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())
        step += 1

        if step % log_interval == 0:
            avg_loss = total_loss / step
            acc = accuracy_score(all_labels, all_preds)
            progress.set_postfix(loss=f"{avg_loss:.4f}", acc=f"{acc:.4f}")

    avg_loss = total_loss / max(step, 1)
    acc = accuracy_score(all_labels, all_preds)
    return avg_loss, acc


@torch.no_grad()
def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    step = 0

    for batch in tqdm(dataloader, desc="Evaluating", leave=False):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        token_type_ids = batch["token_type_ids"].to(device)
        labels = batch["label"].to(device)

        logits = model(input_ids, attention_mask, token_type_ids)
        loss = criterion(logits, labels)

        total_loss += loss.item()
        preds = torch.argmax(logits, dim=-1)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())
        step += 1

    avg_loss = total_loss / max(step, 1)
    acc = accuracy_score(all_labels, all_preds)
    return avg_loss, acc, all_preds, all_labels


def main():
    args = parse_args()
    config = Config()
    apply_args(config, args)

    os.makedirs(config.output_dir, exist_ok=True)

    device = torch.device(config.device)
    print(f"使用设备: {device}")

    # 加载分词器
    print(f"加载分词器: {config.model_name}")
    tokenizer = BertTokenizer.from_pretrained(config.model_name)

    # 构建数据加载器
    print(f"加载训练集: {config.train_file}")
    train_loader = build_dataloader(config.train_file, tokenizer, config, shuffle=True)
    print(f"训练集样本数: {len(train_loader.dataset)}")

    val_loader = None
    if config.val_file and os.path.exists(config.val_file):
        print(f"加载验证集: {config.val_file}")
        val_loader = build_dataloader(config.val_file, tokenizer, config, shuffle=False)
        print(f"验证集样本数: {len(val_loader.dataset)}")

    # 构建模型
    print(f"加载预训练模型: {config.model_name}")
    model = BertClassifier(
        pretrained_model_name=config.model_name,
        num_labels=config.num_labels,
        dropout=config.hidden_dropout_prob,
    )
    model.to(device)

    # 优化器：对BERT参数和分类头使用不同学习率
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters()
                       if not any(nd in n for nd in no_decay)],
            "weight_decay": config.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters()
                       if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=config.learning_rate)

    total_steps = len(train_loader) * config.num_epochs
    scheduler = OneCycleLR(
        optimizer,
        max_lr=config.learning_rate,
        total_steps=total_steps,
        pct_start=config.warmup_ratio,
        anneal_strategy="linear",
    )

    criterion = nn.CrossEntropyLoss()

    # 训练循环
    best_val_acc = 0.0
    training_log = []

    print("=" * 60)
    print("开始训练")
    print(f"  Epochs: {config.num_epochs}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Learning rate: {config.learning_rate}")
    print(f"  Total steps: {total_steps}")
    print("=" * 60)

    for epoch in range(1, config.num_epochs + 1):
        start_time = time.time()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, scheduler, criterion,
            device, log_interval=config.log_interval,
        )

        epoch_time = time.time() - start_time
        log_entry = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "epoch_time": epoch_time,
        }

        print(f"\nEpoch {epoch}/{config.num_epochs}  "
              f"Train Loss: {train_loss:.4f}  Train Acc: {train_acc:.4f}  "
              f"Time: {epoch_time:.1f}s")

        if val_loader is not None:
            val_loss, val_acc, val_preds, val_labels = evaluate(
                model, val_loader, criterion, device,
            )
            log_entry["val_loss"] = val_loss
            log_entry["val_acc"] = val_acc
            print(f"           Val   Loss: {val_loss:.4f}  Val   Acc: {val_acc:.4f}")

            if config.save_best_model and val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_path = os.path.join(config.output_dir, "best_model.pt")
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_acc": val_acc,
                    "val_loss": val_loss,
                    "config": {
                        "model_name": config.model_name,
                        "num_labels": config.num_labels,
                        "max_length": config.max_length,
                        "label_names": config.label_names,
                    },
                }, best_model_path)
                print(f"           ** 保存最优模型 (val_acc={val_acc:.4f}) -> {best_model_path}")

                # 最后一个 epoch 的验证集分类报告
                if epoch == config.num_epochs:
                    id2label = config.id2label()
                    target_names = [id2label[i] for i in range(config.num_labels)]
                    report = classification_report(
                        val_labels, val_preds, target_names=target_names, digits=4,
                    )
                    print("\n验证集分类报告:")
                    print(report)

        training_log.append(log_entry)

    # 保存最终模型
    final_model_path = os.path.join(config.output_dir, "final_model.pt")
    torch.save({
        "epoch": config.num_epochs,
        "model_state_dict": model.state_dict(),
        "config": {
            "model_name": config.model_name,
            "num_labels": config.num_labels,
            "max_length": config.max_length,
            "label_names": config.label_names,
        },
    }, final_model_path)
    print(f"\n最终模型已保存: {final_model_path}")

    # 保存训练日志
    log_path = os.path.join(config.output_dir, "training_log.json")
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(training_log, f, ensure_ascii=False, indent=2)
    print(f"训练日志已保存: {log_path}")

    # 保存分词器（方便预测时加载）
    tokenizer.save_pretrained(os.path.join(config.output_dir, "tokenizer"))
    print("分词器已保存")

    print("\n训练完成!")


if __name__ == "__main__":
    main()
