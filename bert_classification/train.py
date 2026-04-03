"""
BERT 13分类 fine-tune 训练脚本。

使用传统的 PyTorch 训练循环，不使用 Trainer 等高阶库。

用法:
    python train.py
    python train.py --train_file data/train.csv --val_file data/val.csv --num_epochs 3
"""

import argparse
import json
import logging
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from transformers import BertTokenizer
from sklearn.metrics import accuracy_score, classification_report, f1_score
from tqdm import tqdm

from config import Config
from dataset import build_dataloader
from model import BertClassifier

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--no_amp", action="store_true",
                        help="禁用混合精度训练")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=None)
    parser.add_argument("--early_stopping_patience", type=int, default=None)
    parser.add_argument("--pooling", type=str, default=None,
                        choices=["cls", "mean", "pooler"])
    parser.add_argument("--freeze_layers", type=int, default=0,
                        help="冻结 BERT 前 N 层编码器")
    return parser.parse_args()


def apply_args(config, args):
    """将命令行参数覆盖到 Config 中。"""
    arg_map = {
        "model_name": "model_name", "train_file": "train_file",
        "val_file": "val_file", "output_dir": "output_dir",
        "num_epochs": "num_epochs", "batch_size": "batch_size",
        "learning_rate": "learning_rate", "max_length": "max_length",
        "device": "device", "seed": "seed",
        "gradient_accumulation_steps": "gradient_accumulation_steps",
        "early_stopping_patience": "early_stopping_patience",
        "pooling": "pooling_strategy",
    }
    for arg_key, cfg_key in arg_map.items():
        value = getattr(args, arg_key, None)
        if value is not None:
            setattr(config, cfg_key, value)
    if args.no_amp:
        config.use_amp = False


def build_optimizer(model, config):
    """构建 AdamW 优化器，BERT 主干和分类头使用不同学习率。"""
    no_decay = {"bias", "LayerNorm.weight", "LayerNorm.bias"}
    classifier_params = {"classifier.weight", "classifier.bias"}

    groups = {
        "bert_decay": [], "bert_no_decay": [],
        "cls_decay": [], "cls_no_decay": [],
    }
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        is_classifier = name in classifier_params
        is_no_decay = any(nd in name for nd in no_decay)
        if is_classifier:
            key = "cls_no_decay" if is_no_decay else "cls_decay"
        else:
            key = "bert_no_decay" if is_no_decay else "bert_decay"
        groups[key].append(param)

    optimizer_grouped_parameters = [
        {"params": groups["bert_decay"], "lr": config.learning_rate,
         "weight_decay": config.weight_decay},
        {"params": groups["bert_no_decay"], "lr": config.learning_rate,
         "weight_decay": 0.0},
        {"params": groups["cls_decay"], "lr": config.classifier_lr,
         "weight_decay": config.weight_decay},
        {"params": groups["cls_no_decay"], "lr": config.classifier_lr,
         "weight_decay": 0.0},
    ]
    optimizer_grouped_parameters = [g for g in optimizer_grouped_parameters if g["params"]]
    return AdamW(optimizer_grouped_parameters)


class EarlyStopping:
    """当验证指标连续 patience 个 epoch 没有改善时停止训练。"""

    def __init__(self, patience=3, mode="max"):
        self.patience = patience
        self.mode = mode
        self.best_score = None
        self.counter = 0
        self.should_stop = False

    def step(self, score):
        if self.best_score is None:
            self.best_score = score
            return
        improved = (score > self.best_score) if self.mode == "max" else (score < self.best_score)
        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True


def train_one_epoch(model, dataloader, optimizer, scheduler, criterion, device,
                    scaler, use_amp, accumulation_steps=1, log_interval=50):
    model.train()
    total_loss = 0.0
    num_samples = 0
    correct = 0
    step = 0

    optimizer.zero_grad()
    progress = tqdm(dataloader, desc="Training", leave=False)
    for batch_idx, batch in enumerate(progress):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        token_type_ids = batch["token_type_ids"].to(device)
        labels = batch["label"].to(device)

        with torch.autocast(device_type=device.type, enabled=use_amp):
            logits = model(input_ids, attention_mask, token_type_ids)
            loss = criterion(logits, labels)
            loss = loss / accumulation_steps

        scaler.scale(loss).backward()

        if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(dataloader):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), Config.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()
            step += 1

        batch_size = labels.size(0)
        total_loss += loss.item() * accumulation_steps * batch_size
        preds = torch.argmax(logits, dim=-1)
        correct += (preds == labels).sum().item()
        num_samples += batch_size

        if step > 0 and step % log_interval == 0:
            avg_loss = total_loss / num_samples
            acc = correct / num_samples
            lr = scheduler.get_last_lr()[0]
            progress.set_postfix(
                loss=f"{avg_loss:.4f}", acc=f"{acc:.4f}", lr=f"{lr:.2e}")

    avg_loss = total_loss / max(num_samples, 1)
    acc = correct / max(num_samples, 1)
    return avg_loss, acc


@torch.no_grad()
def evaluate(model, dataloader, criterion, device, use_amp=False):
    model.eval()
    total_loss = 0.0
    num_samples = 0
    all_preds = []
    all_labels = []

    for batch in tqdm(dataloader, desc="Evaluating", leave=False):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        token_type_ids = batch["token_type_ids"].to(device)
        labels = batch["label"].to(device)

        with torch.autocast(device_type=device.type, enabled=use_amp):
            logits = model(input_ids, attention_mask, token_type_ids)
            loss = criterion(logits, labels)

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        preds = torch.argmax(logits, dim=-1)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())
        num_samples += batch_size

    avg_loss = total_loss / max(num_samples, 1)
    acc = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro")
    return avg_loss, acc, macro_f1, all_preds, all_labels


def main():
    args = parse_args()
    config = Config()
    apply_args(config, args)

    set_seed(config.seed)
    os.makedirs(config.output_dir, exist_ok=True)

    device = torch.device(config.device)
    use_amp = config.use_amp and device.type == "cuda"
    logger.info("设备: %s | AMP: %s", device, use_amp)
    logger.info("配置:\n%s", config)

    logger.info("加载分词器: %s", config.model_name)
    tokenizer = BertTokenizer.from_pretrained(config.model_name)

    logger.info("加载训练集: %s", config.train_file)
    train_loader = build_dataloader(config.train_file, tokenizer, config, shuffle=True)
    logger.info("训练集样本数: %d", len(train_loader.dataset))

    val_loader = None
    if config.val_file and os.path.exists(config.val_file):
        logger.info("加载验证集: %s", config.val_file)
        val_loader = build_dataloader(config.val_file, tokenizer, config, shuffle=False)
        logger.info("验证集样本数: %d", len(val_loader.dataset))

    logger.info("加载预训练模型: %s", config.model_name)
    model = BertClassifier(
        pretrained_model_name=config.model_name,
        num_labels=config.num_labels,
        dropout=config.hidden_dropout_prob,
        pooling=getattr(config, "pooling_strategy", "cls"),
    )
    if args.freeze_layers > 0:
        model.freeze_bert_layers(args.freeze_layers)
        logger.info("冻结 BERT 前 %d 层", args.freeze_layers)

    model.to(device)
    total_params, trainable_params = model.count_parameters()
    logger.info("参数量: 总计 %s, 可训练 %s",
                f"{total_params:,}", f"{trainable_params:,}")

    optimizer = build_optimizer(model, config)

    effective_batch = config.batch_size * config.gradient_accumulation_steps
    steps_per_epoch = (len(train_loader) + config.gradient_accumulation_steps - 1) // config.gradient_accumulation_steps
    total_steps = steps_per_epoch * config.num_epochs
    scheduler = OneCycleLR(
        optimizer,
        max_lr=[config.learning_rate, config.learning_rate,
                config.classifier_lr, config.classifier_lr][:len(optimizer.param_groups)],
        total_steps=total_steps,
        pct_start=config.warmup_ratio,
        anneal_strategy="linear",
    )

    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler(enabled=use_amp)

    early_stopper = None
    if val_loader and config.early_stopping_patience > 0:
        early_stopper = EarlyStopping(patience=config.early_stopping_patience, mode="max")

    best_val_acc = 0.0
    training_log = []

    logger.info("=" * 60)
    logger.info("开始训练")
    logger.info("  Epochs: %d", config.num_epochs)
    logger.info("  Batch size: %d (effective: %d)", config.batch_size, effective_batch)
    logger.info("  BERT LR: %s, Classifier LR: %s", config.learning_rate, config.classifier_lr)
    logger.info("  Total optimizer steps: %d", total_steps)
    logger.info("  Early stopping patience: %s",
                config.early_stopping_patience if early_stopper else "disabled")
    logger.info("=" * 60)

    for epoch in range(1, config.num_epochs + 1):
        start_time = time.time()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, scheduler, criterion,
            device, scaler, use_amp,
            accumulation_steps=config.gradient_accumulation_steps,
            log_interval=config.log_interval,
        )

        epoch_time = time.time() - start_time
        log_entry = {
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "train_acc": round(train_acc, 6),
            "epoch_time": round(epoch_time, 2),
            "lr": scheduler.get_last_lr()[0],
        }

        logger.info(
            "Epoch %d/%d | Train Loss: %.4f | Train Acc: %.4f | Time: %.1fs",
            epoch, config.num_epochs, train_loss, train_acc, epoch_time,
        )

        if val_loader is not None:
            val_loss, val_acc, val_f1, val_preds, val_labels = evaluate(
                model, val_loader, criterion, device, use_amp=use_amp,
            )
            log_entry.update({
                "val_loss": round(val_loss, 6),
                "val_acc": round(val_acc, 6),
                "val_macro_f1": round(val_f1, 6),
            })
            logger.info(
                "         | Val   Loss: %.4f | Val   Acc: %.4f | Val F1: %.4f",
                val_loss, val_acc, val_f1,
            )

            if config.save_best_model and val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_path = os.path.join(config.output_dir, "best_model.pt")
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "val_acc": val_acc,
                    "val_f1": val_f1,
                    "val_loss": val_loss,
                    "config": {
                        "model_name": config.model_name,
                        "num_labels": config.num_labels,
                        "max_length": config.max_length,
                        "label_names": config.label_names,
                        "pooling_strategy": getattr(config, "pooling_strategy", "cls"),
                    },
                }, best_model_path)
                logger.info("         ** 保存最优模型 (val_acc=%.4f) -> %s",
                            val_acc, best_model_path)

            id2label = config.id2label()
            target_names = [id2label[i] for i in range(config.num_labels)]
            report = classification_report(
                val_labels, val_preds, target_names=target_names, digits=4,
            )
            logger.info("\n验证集分类报告:\n%s", report)

            if early_stopper:
                early_stopper.step(val_acc)
                if early_stopper.should_stop:
                    logger.info("早停触发: 验证集准确率连续 %d 个 epoch 未提升",
                                config.early_stopping_patience)
                    training_log.append(log_entry)
                    break

        training_log.append(log_entry)

    final_model_path = os.path.join(config.output_dir, "final_model.pt")
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "config": {
            "model_name": config.model_name,
            "num_labels": config.num_labels,
            "max_length": config.max_length,
            "label_names": config.label_names,
            "pooling_strategy": getattr(config, "pooling_strategy", "cls"),
        },
    }, final_model_path)
    logger.info("最终模型已保存: %s", final_model_path)

    log_path = os.path.join(config.output_dir, "training_log.json")
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(training_log, f, ensure_ascii=False, indent=2)
    logger.info("训练日志已保存: %s", log_path)

    tokenizer.save_pretrained(os.path.join(config.output_dir, "tokenizer"))
    logger.info("分词器已保存")

    logger.info("训练完成! 最优验证准确率: %.4f", best_val_acc)


if __name__ == "__main__":
    main()
