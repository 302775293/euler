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

## 核心特性

- **传统 PyTorch 训练循环**：手写 train/eval loop，不依赖 Trainer
- **AMP 混合精度训练**：自动在 GPU 上启用 FP16 加速
- **差异化学习率**：BERT 主干和分类头分别设置学习率
- **梯度累积**：支持小显存下模拟大 batch 训练
- **早停机制**：验证集指标连续若干 epoch 未提升自动停止
- **多种池化策略**：支持 `cls`、`mean`、`pooler` 三种句子表示方式
- **分类头 Xavier 初始化**：改善随机初始化分类头的收敛速度
- **预分词优化**：数据集初始化时完成全部 tokenize，训练时零开销
- **可复现性**：全局随机种子控制
- **结构化日志**：使用 logging 模块统一日志格式

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

支持丰富的命令行参数：

```bash
python train.py \
  --model_name bert-base-chinese \
  --train_file data/train.csv \
  --val_file data/val.csv \
  --num_epochs 5 \
  --batch_size 32 \
  --learning_rate 2e-5 \
  --device cuda \
  --pooling mean \
  --freeze_layers 4 \
  --gradient_accumulation_steps 2 \
  --early_stopping_patience 3 \
  --seed 42
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

- `best_model.pt` — 验证集最优模型（含完整恢复信息）
- `final_model.pt` — 最终模型
- `training_log.json` — 训练过程指标日志（含 loss/acc/f1/lr）
- `tokenizer/` — 保存的分词器

## 自定义配置

修改 `config.py` 中的参数即可调整：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `model_name` | bert-base-chinese | 预训练模型名称 |
| `num_labels` | 13 | 分类类别数 |
| `max_length` | 128 | 最大序列长度 |
| `batch_size` | 32 | 批大小 |
| `learning_rate` | 2e-5 | BERT 主干学习率 |
| `classifier_lr` | 5e-5 | 分类头学习率 |
| `num_epochs` | 5 | 训练轮数 |
| `use_amp` | True | 是否启用混合精度 |
| `gradient_accumulation_steps` | 1 | 梯度累积步数 |
| `early_stopping_patience` | 3 | 早停耐心值 |
| `pooling_strategy` | cls | 池化策略 (cls/mean/pooler) |
| `seed` | 42 | 随机种子 |
