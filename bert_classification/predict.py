"""
BERT 13分类预测脚本。

支持以下预测模式：
1. 对CSV文件进行批量预测，输出带预测结果的CSV
2. 对单条文本进行交互式预测

用法:
    # 批量预测
    python predict.py --model_path output/best_model.pt --input_file data/test.csv --output_file output/predictions.csv

    # 单条预测
    python predict.py --model_path output/best_model.pt --text "今天股市大涨"

    # 交互式预测
    python predict.py --model_path output/best_model.pt --interactive
"""

import argparse
import os

import pandas as pd
import torch
import torch.nn.functional as F
from transformers import BertTokenizer
from tqdm import tqdm

from model import BertClassifier


def load_model(model_path, device):
    """加载训练好的模型。"""
    checkpoint = torch.load(model_path, map_location=device)
    model_config = checkpoint["config"]

    model = BertClassifier(
        pretrained_model_name=model_config["model_name"],
        num_labels=model_config["num_labels"],
        dropout=0.0,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model, model_config


def predict_single(text, model, tokenizer, max_length, device, id2label):
    """对单条文本进行预测。"""
    encoding = tokenizer(
        text,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)
    token_type_ids = encoding["token_type_ids"].to(device)

    with torch.no_grad():
        logits = model(input_ids, attention_mask, token_type_ids)
        probs = F.softmax(logits, dim=-1)
        pred_id = torch.argmax(probs, dim=-1).item()
        confidence = probs[0][pred_id].item()

    pred_label = id2label.get(pred_id, str(pred_id))

    top_k = min(5, len(id2label))
    top_probs, top_indices = torch.topk(probs[0], top_k)
    top_results = [
        (id2label.get(idx.item(), str(idx.item())), prob.item())
        for idx, prob in zip(top_indices, top_probs)
    ]

    return {
        "text": text,
        "pred_label": pred_label,
        "pred_id": pred_id,
        "confidence": confidence,
        "top_k": top_results,
    }


def predict_batch(csv_path, model, tokenizer, max_length, device, id2label,
                  text_column="text", batch_size=64):
    """对CSV文件进行批量预测。"""
    df = pd.read_csv(csv_path)
    texts = df[text_column].astype(str).tolist()

    all_preds = []
    all_confidences = []

    for i in tqdm(range(0, len(texts), batch_size), desc="Predicting"):
        batch_texts = texts[i:i + batch_size]
        encodings = tokenizer(
            batch_texts,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        input_ids = encodings["input_ids"].to(device)
        attention_mask = encodings["attention_mask"].to(device)
        token_type_ids = encodings["token_type_ids"].to(device)

        with torch.no_grad():
            logits = model(input_ids, attention_mask, token_type_ids)
            probs = F.softmax(logits, dim=-1)
            pred_ids = torch.argmax(probs, dim=-1)
            confidences = probs.gather(1, pred_ids.unsqueeze(1)).squeeze(1)

        all_preds.extend(pred_ids.cpu().tolist())
        all_confidences.extend(confidences.cpu().tolist())

    df["pred_id"] = all_preds
    df["pred_label"] = [id2label.get(p, str(p)) for p in all_preds]
    df["confidence"] = all_confidences

    return df


def main():
    parser = argparse.ArgumentParser(description="BERT 13分类预测脚本")
    parser.add_argument("--model_path", type=str, required=True,
                        help="模型checkpoint路径")
    parser.add_argument("--input_file", type=str, default=None,
                        help="待预测的CSV文件路径")
    parser.add_argument("--output_file", type=str, default=None,
                        help="预测结果输出路径")
    parser.add_argument("--text", type=str, default=None,
                        help="单条待预测文本")
    parser.add_argument("--text_column", type=str, default="text",
                        help="CSV中文本列名")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="批量预测的batch大小")
    parser.add_argument("--device", type=str, default=None,
                        help="推理设备 (cpu/cuda)")
    parser.add_argument("--interactive", action="store_true",
                        help="交互式预测模式")
    args = parser.parse_args()

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    print(f"加载模型: {args.model_path}")
    model, model_config = load_model(args.model_path, device)

    label_names = model_config.get("label_names", None)
    if label_names:
        id2label = {i: name for i, name in enumerate(label_names)}
    else:
        id2label = {i: str(i) for i in range(model_config["num_labels"])}

    max_length = model_config.get("max_length", 128)

    tokenizer_path = os.path.join(os.path.dirname(args.model_path), "tokenizer")
    if os.path.exists(tokenizer_path):
        tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
    else:
        tokenizer = BertTokenizer.from_pretrained(model_config["model_name"])
    print("分词器加载完成")

    if args.text:
        result = predict_single(args.text, model, tokenizer, max_length,
                                device, id2label)
        print(f"\n文本: {result['text']}")
        print(f"预测类别: {result['pred_label']} (id={result['pred_id']})")
        print(f"置信度: {result['confidence']:.4f}")
        print(f"\nTop-{len(result['top_k'])} 预测:")
        for label, prob in result["top_k"]:
            bar = "█" * int(prob * 30)
            print(f"  {label:>6s}: {prob:.4f} {bar}")

    elif args.input_file:
        print(f"\n批量预测: {args.input_file}")
        df = predict_batch(
            args.input_file, model, tokenizer, max_length, device, id2label,
            text_column=args.text_column, batch_size=args.batch_size,
        )
        output_file = args.output_file or args.input_file.replace(".csv", "_predictions.csv")
        df.to_csv(output_file, index=False, encoding="utf-8-sig")
        print(f"预测结果已保存: {output_file}")
        print(f"共预测 {len(df)} 条样本")

        print("\n预测分布:")
        print(df["pred_label"].value_counts().to_string())

        if "label" in df.columns:
            from sklearn.metrics import accuracy_score, classification_report
            acc = accuracy_score(df["label"], df["pred_id"])
            print(f"\n准确率: {acc:.4f}")

    elif args.interactive:
        print("\n进入交互式预测模式 (输入 'quit' 退出):")
        while True:
            text = input("\n请输入文本: ").strip()
            if text.lower() in ("quit", "exit", "q"):
                print("退出预测。")
                break
            if not text:
                continue
            result = predict_single(text, model, tokenizer, max_length,
                                    device, id2label)
            print(f"预测类别: {result['pred_label']} (置信度: {result['confidence']:.4f})")
            for label, prob in result["top_k"]:
                bar = "█" * int(prob * 30)
                print(f"  {label:>6s}: {prob:.4f} {bar}")
    else:
        print("请指定 --text, --input_file 或 --interactive 参数")


if __name__ == "__main__":
    main()
