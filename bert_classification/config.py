import os


class Config:
    # 模型相关
    model_name = "bert-base-chinese"
    num_labels = 13
    max_length = 128
    hidden_dropout_prob = 0.1

    # 训练相关
    batch_size = 32
    learning_rate = 2e-5
    weight_decay = 0.01
    num_epochs = 5
    warmup_ratio = 0.1
    max_grad_norm = 1.0

    # 数据相关
    train_file = "data/train.csv"
    val_file = "data/val.csv"
    test_file = "data/test.csv"
    text_column = "text"
    label_column = "label"

    # 输出相关
    output_dir = "output"
    save_best_model = True
    log_interval = 50

    # 设备
    device = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu"

    # 标签映射（根据实际任务修改）
    label_names = [
        "体育", "财经", "房产", "家居", "教育",
        "科技", "时尚", "时政", "游戏", "娱乐",
        "社会", "汽车", "国际",
    ]

    @classmethod
    def label2id(cls):
        return {name: idx for idx, name in enumerate(cls.label_names)}

    @classmethod
    def id2label(cls):
        return {idx: name for idx, name in enumerate(cls.label_names)}
