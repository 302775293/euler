# BERT 13分类 Fine-tune

基于 PyTorch 和 HuggingFace Transformers 的 BERT 文本分类微调项目，使用传统 PyTorch 训练循环（不依赖 Trainer）。

## 项目结构

```
bert_classification/
├── config.py                  # 配置参数
├── dataset.py                 # 数据集加载（CSV格式）
├── model.py                   # BERT分类模型
├── train.py                   # 训练脚本
├── predict.py                 # 预测脚本
├── generate_sample_data.py    # 示例数据生成
├── requirements.txt           # 依赖包
└── data/                      # 数据目录
    ├── train.csv
    ├── val.csv
    └── test.csv
```

## 环境安装

```bash
pip install -r requirements.txt
```

## 数据格式

CSV 文件需包含以下两列：

| 列名   | 说明                         |
| ------ | ---------------------------- |
| text   | 待分类文本                    |
| label  | 标签（整数 0~12 或类别名称）   |

示例：

```csv
text,label
中国男足在世界杯预选赛中取得关键胜利,0
央行宣布降准释放长期资金约万亿元,1
```

## 分类类别（13类）

| ID | 类别 | ID | 类别 | ID | 类别 |
|----|------|----|------|----|------|
| 0  | 体育 | 5  | 科技 | 10 | 社会 |
| 1  | 财经 | 6  | 时尚 | 11 | 汽车 |
| 2  | 房产 | 7  | 时政 | 12 | 国际 |
| 3  | 家居 | 8  | 游戏 |    |      |
| 4  | 教育 | 9  | 娱乐 |    |      |

## 快速开始

### 1. 生成示例数据

```bash
cd bert_classification
python generate_sample_data.py
```

### 2. 训练模型

```bash
python train.py
```

支持命令行参数覆盖配置：

```bash
python train.py \
  --model_name bert-base-chinese \
  --train_file data/train.csv \
  --val_file data/val.csv \
  --num_epochs 5 \
  --batch_size 32 \
  --learning_rate 2e-5 \
  --device cuda
```

### 3. 预测

**单条文本预测：**

```bash
python predict.py --model_path output/best_model.pt --text "今天股市大涨"
```

**批量预测（CSV文件）：**

```bash
python predict.py \
  --model_path output/best_model.pt \
  --input_file data/test.csv \
  --output_file output/predictions.csv
```

**交互式预测：**

```bash
python predict.py --model_path output/best_model.pt --interactive
```

## 训练输出

训练完成后，`output/` 目录下包含：

- `best_model.pt` — 验证集最优模型
- `final_model.pt` — 最终模型
- `training_log.json` — 训练过程指标日志
- `tokenizer/` — 保存的分词器

## 自定义配置

修改 `config.py` 中的参数即可调整：

- `model_name`：预训练模型名称（支持任意 HuggingFace BERT 模型）
- `num_labels`：分类类别数
- `max_length`：最大序列长度
- `batch_size`、`learning_rate`、`num_epochs` 等训练超参数
- `label_names`：类别名称列表
