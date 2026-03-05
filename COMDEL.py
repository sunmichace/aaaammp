# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import (
    precision_score, recall_score, f1_score, matthews_corrcoef,
    accuracy_score, roc_auc_score, precision_recall_curve, auc
)
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import seaborn as sns
from transformers import AutoModel, get_linear_schedule_with_warmup
import torch.serialization
from sklearn.preprocessing import LabelEncoder
import time
import warnings
import sys
import psutil
from collections import defaultdict


warnings.filterwarnings('ignore')

torch.serialization.add_safe_globals([LabelEncoder])

# 根目录路径配置
ROOT_DIR = "./"

# 训练/验证/测试集数据文件路径
train_csv = os.path.join(ROOT_DIR, "mydata/newdata/rawdata_train.csv")
val0_csv = os.path.join(ROOT_DIR, "mydata/newdata/rawdata_val.csv")
t0_csv = os.path.join(ROOT_DIR, "mydata/newdata/rawdata_test.csv")
t1_csv = os.path.join(ROOT_DIR, "test_data/bagel4_1_val.csv")
t2_csv = os.path.join(ROOT_DIR, "test_data/bagel4_2_val.csv")
t3_csv = os.path.join(ROOT_DIR, "test_data/bagel4_3_val.csv")


esm2_backbone_path = os.path.join(ROOT_DIR, "model/esm2_t6_8M_UR50D")

# 各数据集预测结果输出路径
val0_output = os.path.join(ROOT_DIR, "results_ultra_light_scratch/val0_val_test_predictions.csv")
t0_output = os.path.join(ROOT_DIR, "results_ultra_light_scratch/t0_rawdata_val_predictions.csv")
t1_output = os.path.join(ROOT_DIR, "results_ultra_light_scratch/t1_bagel4_1_val_predictions.csv")
t2_output = os.path.join(ROOT_DIR, "results_ultra_light_scratch/t2_bagel4_2_val_predictions.csv")
t3_output = os.path.join(ROOT_DIR, "results_ultra_light_scratch/t3_bagel4_3_val_predictions.csv")
# 模型评估指标汇总保存路径
metrics_summary_output = os.path.join(ROOT_DIR, "results_ultra_light_scratch/metrics_summary.csv")
# 曲线保存目录(备用)
cumulative_curve_dir = os.path.join(ROOT_DIR, "results_ultra_light_scratch/cumulative_curves")

# 训练好的最优模型保存路径
comdel_model_path = r"results_ultra_light_scratch/cnn_scratch_best.pt"

# 设备配置：优先使用GPU，无GPU则使用CPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"使用设备: {device}")

# 自定义Lookahead优化器，封装基础优化器，提升优化稳定性和收敛效果
class Lookahead(torch.optim.Optimizer):
    def __init__(self, optimizer, k=5, alpha=0.5):
        self.optimizer = optimizer    # 基础优化器
        self.k = k                    # 慢更新权重的步长间隔
        self.alpha = alpha            # 慢更新的插值系数
        self.param_groups = self.optimizer.param_groups
        self.state = defaultdict(dict)
        self.fast_state = self.optimizer.state
        # 初始化每个参数组的步数计数器
        for group in self.param_groups:
            group["step_counter"] = 0

    # 更新单个参数组的慢权重
    def update(self, group):
        for fast in group["params"]:
            param_state = self.state[fast]
            # 初始化慢权重参数
            if "slow_param" not in param_state:
                param_state["slow_param"] = torch.clone(fast.data).detach()
            slow = param_state["slow_param"]
            # 慢权重融合更新
            slow += (fast.data - slow) * self.alpha
            fast.data.copy_(slow)

    # 批量更新所有参数组的慢权重
    def update_lookahead(self):
        for group in self.param_groups:
            self.update(group)

    # 重写step方法，执行一次梯度更新
    def step(self, closure=None):
        loss = self.optimizer.step(closure)
        # 每隔k步执行一次慢权重更新
        for group in self.param_groups:
            if group["step_counter"] % self.k == 0:
                self.update(group)
            group["step_counter"] += 1
        return loss

    # 重写状态保存方法，兼容基础优化器和慢权重
    def state_dict(self):
        fast_state_dict = self.optimizer.state_dict()
        slow_state = {
            (id(k) if isinstance(k, torch.Tensor) else k): v
            for k, v in self.state.items()
        }
        fast_state = fast_state_dict["state"]
        param_groups = fast_state_dict["param_groups"]
        return {
            "fast_state": fast_state,
            "slow_state": slow_state,
            "param_groups": param_groups,
        }

    # 重写状态加载方法，恢复优化器状态
    def load_state_dict(self, state_dict):
        slow_state_dict = {
            "state": state_dict["slow_state"],
            "param_groups": state_dict["param_groups"],
        }
        fast_state_dict = {
            "state": state_dict["fast_state"],
            "param_groups": state_dict["param_groups"],
        }
        super(Lookahead, self).load_state_dict(slow_state_dict)
        self.optimizer.load_state_dict(fast_state_dict)
        self.fast_state = self.optimizer.state

    # 梯度清零
    def zero_grad(self):
        self.optimizer.zero_grad()

    # 打印优化器信息
    def __repr__(self):
        return f"Lookahead({self.optimizer.__repr__()}, k={self.k}, alpha={self.alpha})"

# 氨基酸序列分词器：将氨基酸字符转为模型可识别的数字ID
class AminoAcidTokenizer:
    def __init__(self, max_seq_len=256):
        # 氨基酸词表，包含20种常见氨基酸+特殊标记
        self.vocab = {
            '<pad>': 0, 'A': 1, '<cls>': 2, '<sep>': 3, 'R': 4, 'N': 5, 'D': 6,
            'C': 7, 'Q': 8, 'E': 9, 'G': 10, 'H': 11, 'I': 12, 'L': 13, 'K': 14,
            'M': 15, 'F': 16, 'P': 17, 'S': 18, 'T': 19, 'W': 20, 'Y': 21, 'V': 22
        }
        self.id2token = {v: k for k, v in self.vocab.items()}  # id到字符的反向映射
        self.max_seq_len = max_seq_len                        # 序列最大长度限制
        self.pad_token_id = 0                                 # padding对应的id
        self.vocab_size = len(self.vocab)                     # 词表总大小

    # 获取词表
    def get_vocab(self):
        return self.vocab

    # 对单条序列进行编码，字符转id，支持截断和填充
    def encode_single(self, sequence, padding="do_not_pad", truncation=True):
        sequence = sequence.strip().upper()
        input_ids = [self.vocab.get(char, self.pad_token_id) for char in sequence]

        # 序列截断：超过最大长度则截取前max_seq_len个字符
        if truncation and len(input_ids) > self.max_seq_len:
            input_ids = input_ids[:self.max_seq_len]

        # 序列填充：不足最大长度则补0
        if padding != "do_not_pad":
            padding_len = self.max_seq_len - len(input_ids)
            if padding_len > 0:
                input_ids += [self.pad_token_id] * padding_len

        return {
            "input_ids": input_ids,
            "attention_mask": [1] * len(input_ids) if padding == "do_not_pad" else [1] * self.max_seq_len
        }

    # 对批量序列进行编码，返回tensor格式的输入id
    def batch_encode(self, sequences, padding='max_length', truncation=True, max_length=256):
        batch_input_ids = []
        for seq in sequences:
            seq = seq.strip().upper()
            input_ids = [self.vocab.get(char, self.pad_token_id) for char in seq]

            # 序列截断
            if truncation and len(input_ids) > max_length:
                input_ids = input_ids[:max_length]

            # 序列填充到指定长度
            if padding == 'max_length':
                input_ids += [self.pad_token_id] * (max_length - len(input_ids))

            batch_input_ids.append(input_ids)

        return torch.tensor(batch_input_ids, dtype=torch.long)

# 计算模型参数量和模型文件大小
def calculate_model_metrics(model):
    param_count = sum(p.numel() for p in model.parameters())
    param_count_k = param_count / 1000

    # 临时保存模型计算文件大小
    temp_path = "temp_model.pt"
    torch.save(model.state_dict(), temp_path)
    model_size_mb = os.path.getsize(temp_path) / (1024 * 1024)
    os.remove(temp_path)

    return {
        "Param Count (k)": round(param_count_k, 3),
        "Model Size (MB)": round(model_size_mb, 4)
    }

# 自定义肽序列数据集类，继承torch Dataset
class PeptideDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = sequences  # 肽序列列表
        self.labels = labels        # 对应的标签列表(0/1)
        # 校验序列和标签数量一致
        if len(self.sequences) != len(self.labels):
            raise ValueError("序列数量与标签数量不匹配")

    # 返回数据集总长度
    def __len__(self):
        return len(self.sequences)

    # 根据索引获取单条数据：序列、标签、序列长度
    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx], len(self.sequences[idx])

# 轻量级CNN分类模型，肽序列二分类任务，纯从头训练
class LightweightCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=128, max_seq_len=256):
        super().__init__()
        self.max_seq_len = max_seq_len

        # 氨基酸嵌入层，将id转为稠密向量
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.embed_dropout = nn.Dropout(0.1)  # 嵌入层dropout防止过拟合

        # 特征提取器：三层一维卷积+池化，提取序列特征
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(embed_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(hidden_dim, hidden_dim * 2, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(hidden_dim * 2, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(1)  # 自适应池化，输出固定维度
        )

        # 分类器：两层全连接层，输出二分类结果
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 2)
        )

        self._init_weights()  # 初始化模型权重

    # 模型权重初始化函数
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Embedding):
                nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.Conv1d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # 模型前向传播逻辑
    def forward(self, input_ids):
        x = self.embedding(input_ids)        # [batch, seq_len, embed_dim]
        x = self.embed_dropout(x)             # 嵌入层dropout
        x = x.permute(0, 2, 1)                # 维度转换适配卷积层 [batch, embed_dim, seq_len]
        x = self.feature_extractor(x)         # 特征提取
        x = x.squeeze(-1)                     # 压缩维度
        logits = self.classifier(x)           # 分类输出
        return logits

# 模型训练器类，封装模型初始化、训练、评估、预测逻辑
class ScratchTrainer:
    def __init__(self, device=device):
        self.device = device
        self.cnn_model = None  # CNN模型实例
        self.tokenizer = AminoAcidTokenizer(max_seq_len=256)  # 初始化分词器
        # 训练过程指标记录
        self.metrics = {
            'train_loss': [], 'val_loss': [],
            'best_train_loss': 0.0, 'best_val_loss': 0.0, 'best_val_f1': 0.0
        }
        self.cnn_model_metrics = None  # CNN模型参数量和大小指标

    # 初始化CNN模型并加载到指定设备
    def init_cnn(self):
        vocab_size = len(self.tokenizer.get_vocab())
        self.cnn_model = LightweightCNN(
            vocab_size=vocab_size,
            embed_dim=128,
            hidden_dim=128
        ).to(self.device)

        # 计算并打印模型指标
        self.cnn_model_metrics = calculate_model_metrics(self.cnn_model)
        print(f"\n===== CNN模型指标 =====")
        print(f"CNN模型参数量: {self.cnn_model_metrics['Param Count (k)']:.1f}k")
        print(f"CNN模型大小: {self.cnn_model_metrics['Model Size (MB)']:.4f}MB")

    # 私有方法：批量序列编码并加载到指定设备
    def _tokenize(self, sequences):
        return self.tokenizer.batch_encode(
            sequences,
            padding='max_length',
            truncation=True,
            max_length=256
        ).to(self.device)

    # 私有方法：计算模型评估的各项指标
    def _compute_metrics(self, y_true, y_pred, y_prob, inference_time, sample_count):
        try:
            # 计算ROC-AUC值，标签唯一时返回0
            roc_auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.0
            # 计算PR-AUC值
            precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_prob)
            pr_auc = auc(recall_curve, precision_curve)
        except Exception as e:
            print(f"计算AUC时出错: {e}")
            roc_auc = 0.0
            pr_auc = 0.0

        # 计算单样本平均推理时间
        avg_inference_time = inference_time / sample_count if sample_count > 0 else 0.0

        # 整理所有评估指标
        metrics = {
            'Accuracy': round(accuracy_score(y_true, y_pred), 4),
            'F1 Score': round(f1_score(y_true, y_pred, zero_division=0), 4),
            'Precision': round(precision_score(y_true, y_pred, zero_division=0), 4),
            'Recall': round(recall_score(y_true, y_pred, zero_division=0), 4),
            'MCC': round(matthews_corrcoef(y_true, y_pred), 4),
            'ROC-AUC': round(roc_auc, 4),
            'PR-AUC': round(pr_auc, 4),
            'Inference Time (s)': round(avg_inference_time, 4),
            **self.cnn_model_metrics
        }

        return metrics

    # 私有方法：计算验证集的平均损失值
    def _compute_val_loss(self, model, dataloader):
        model.eval()
        total_loss = 0.0
        ce_loss_fn = nn.CrossEntropyLoss()

        with torch.no_grad():
            for input_ids, labels in dataloader:
                logits = model(input_ids)
                loss = ce_loss_fn(logits, labels)
                total_loss += loss.item() * len(input_ids)

        avg_loss = total_loss / len(dataloader.dataset)
        return avg_loss

    # CNN模型从头训练主逻辑
    def train_scratch(self, train_seqs, train_labels, val_seqs, val_labels,
                      epochs=50, lr=5e-4, patience=10):
        # 初始化数据集
        train_dataset = PeptideDataset(train_seqs, train_labels)
        val_dataset = PeptideDataset(val_seqs, val_labels)

        # 自定义数据加载的整理函数
        def collate_fn(batch):
            seqs, labels, _ = zip(*batch)
            input_ids = self._tokenize(seqs)
            labels = torch.LongTensor(labels).to(self.device)
            return input_ids, labels

        # 初始化数据加载器
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, collate_fn=collate_fn)
        val_loader = DataLoader(val_dataset, batch_size=64, collate_fn=collate_fn)

        # 初始化优化器：AdamW + Lookahead
        base_optimizer = optim.AdamW(self.cnn_model.parameters(), lr=lr, weight_decay=5e-5)
        optimizer = Lookahead(base_optimizer, k=5, alpha=0.5)
        # 总训练步数
        total_steps = len(train_loader) * epochs
        # 预热学习率调度器
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(total_steps * 0.1),
            num_training_steps=total_steps
        )
        ce_loss = nn.CrossEntropyLoss()  # 交叉熵损失函数

        # 早停相关参数初始化
        best_val_f1 = 0.0
        best_val_loss = float('inf')
        early_stop = 0

        # 开始训练循环
        for epoch in range(epochs):
            self.cnn_model.train()
            total_loss = 0
            # 遍历训练集数据
            for input_ids, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}"):
                optimizer.zero_grad()               # 梯度清零
                logits = self.cnn_model(input_ids)  # 前向传播
                loss = ce_loss(logits, labels)      # 计算损失
                loss.backward()                     # 反向传播计算梯度
                torch.nn.utils.clip_grad_norm_(self.cnn_model.parameters(), 1.0)  # 梯度裁剪防止爆炸
                optimizer.step()                    # 更新权重
                scheduler.step()                    # 更新学习率
                total_loss += loss.item() * len(input_ids)

            # 计算平均训练损失
            avg_train_loss = total_loss / len(train_loader.dataset)
            self.metrics['train_loss'].append(avg_train_loss)

            # 验证集评估
            self.cnn_model.eval()
            val_preds, val_probs, val_true = [], [], []
            with torch.no_grad():
                for input_ids, labels in val_loader:
                    logits = self.cnn_model(input_ids)
                    preds = torch.argmax(logits, dim=1)
                    probs = F.softmax(logits, dim=1)[:, 1]
                    val_preds.extend(preds.cpu().numpy())
                    val_probs.extend(probs.cpu().numpy())
                    val_true.extend(labels.cpu().numpy())

            # 计算验证集损失和核心指标
            val_loss = self._compute_val_loss(self.cnn_model, val_loader)
            self.metrics['val_loss'].append(val_loss)
            val_f1 = f1_score(val_true, val_preds, zero_division=0)
            val_auc = roc_auc_score(val_true, val_probs) if len(np.unique(val_true)) > 1 else 0.0

            # 打印当前epoch训练结果
            print(
                f"Epoch {epoch + 1} | 训练损失={avg_train_loss:.4f} | 验证损失={val_loss:.4f} | F1={val_f1:.4f} | AUC={val_auc:.4f}")

            # 保存最优模型：F1提升或损失下降时更新
            if val_f1 > best_val_f1 + 1e-4 or val_loss < best_val_loss - 1e-4:
                best_val_f1 = val_f1
                best_val_loss = val_loss
                self.metrics['best_train_loss'] = avg_train_loss
                self.metrics['best_val_loss'] = val_loss
                self.metrics['best_val_f1'] = best_val_f1
                early_stop = 0
                os.makedirs(os.path.dirname(comdel_model_path), exist_ok=True)
                torch.save(self.cnn_model.state_dict(), comdel_model_path)
                print(f"保存最佳CNN模型到: {comdel_model_path} | Val F1={best_val_f1:.4f} | Val Loss={best_val_loss:.4f}")
            else:
                early_stop += 1
                # 触发早停机制
                if early_stop >= patience:
                    print(f"早停！最佳Val F1={best_val_f1:.4f} | 最佳Val Loss={best_val_loss:.4f}")
                    break

        # 加载训练好的最优CNN模型
        self.cnn_model.load_state_dict(torch.load(comdel_model_path, map_location=self.device))
        print("CNN模型从头训练完成！")
        print(
            f"最佳指标 | Train Loss={self.metrics['best_train_loss']:.4f} | Val Loss={self.metrics['best_val_loss']:.4f} | "
            f"Val F1={self.metrics['best_val_f1']:.4f}")

    # CNN模型评估函数，对指定数据集进行评估并返回结果
    def evaluate(self, seqs, labels, ids, dataset_name):
        print(f"\n========== 评估 {dataset_name} 集 (CNN模型) ==========")
        sample_count = len(seqs)
        dataset = PeptideDataset(seqs, labels)

        # 数据整理函数
        def collate_fn(batch):
            seqs, labels, _ = zip(*batch)
            input_ids = self._tokenize(seqs)
            labels = torch.LongTensor(labels).to(self.device)
            return input_ids, labels

        loader = DataLoader(dataset, batch_size=64, collate_fn=collate_fn)

        self.cnn_model.eval()
        preds, probs, true_labels = [], [], []
        start = time.time()
        # 推理预测
        with torch.no_grad():
            for input_ids, labels in loader:
                logits = self.cnn_model(input_ids)
                pred = torch.argmax(logits, dim=1)
                prob = F.softmax(logits, dim=1)[:, 1]
                preds.extend(pred.cpu().numpy())
                probs.extend(prob.cpu().numpy())
                true_labels.extend(labels.cpu().numpy())
        inference_time = time.time() - start
        # 计算评估指标
        metrics = self._compute_metrics(true_labels, preds, probs, inference_time, sample_count)

        # 打印评估指标
        print(f"\n【{dataset_name} 集 - CNN模型评估指标】")
        metrics_list = [
            'Accuracy', 'F1 Score', 'Precision', 'Recall', 'MCC', 'ROC-AUC', 'PR-AUC',
            'Inference Time (s)', 'Param Count (k)', 'Model Size (MB)'
        ]
        print(f"{'指标':<18} | {'数值':<12}")
        print("-" * 35)
        for metric in metrics_list:
            val = metrics[metric]
            print(f"{metric:<18} | {val:.4f}")

        # 返回评估结果
        return {
            'dataset': dataset_name,
            'ids': ids,
            'seqs': seqs,
            'true_labels': true_labels,
            'preds': preds,
            'probs': probs,
            'metrics': metrics
        }

    # 单条序列预测函数 (CNN模型)
    def predict_single(self, sequence):
        self.cnn_model.eval()
        with torch.no_grad():
            input_ids = self._tokenize([sequence])
            logits = self.cnn_model(input_ids)
            pred = torch.argmax(logits, dim=1).item()
            prob = F.softmax(logits, dim=1)[:, 1].item()
        return {
            'sequence': sequence,
            'label': pred,
            'prob': round(prob, 4),
            'label_name': '阳性' if pred == 1 else '阴性'
        }

# 保存所有数据集的CNN模型评估指标汇总到csv文件
def save_metrics_summary(all_results, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    rows = []
    for res in all_results:
        row = {'Dataset': res['dataset'], **res['metrics']}
        rows.append(row)

    df = pd.DataFrame(rows)
    cols_order = [
        'Dataset', 'Accuracy', 'F1 Score', 'Precision', 'Recall',
        'MCC', 'ROC-AUC', 'PR-AUC', 'Inference Time (s)',
        'Param Count (k)', 'Model Size (MB)'
    ]
    df = df[cols_order]
    df = df.round(4)
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\nCNN模型指标汇总已保存至: {output_path}")

# 保存单数据集的CNN模型预测结果到csv文件
def save_prediction_results(result, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df = pd.DataFrame({
        'id': result['ids'],
        'sequence': result['seqs'],
        'true_label': result['true_labels'],
        'CNN_pred_label': result['preds'],
        'CNN_pred_prob': result['probs']
    })
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"{result['dataset']} 集 CNN模型预测结果已保存至: {output_path}")

# 加载csv格式的肽序列数据，做数据清洗和校验
def load_csv_data(csv_path):
    if not os.path.exists(csv_path): raise FileNotFoundError(f"CSV文件不存在: {csv_path}")
    df = pd.read_csv(csv_path)
    required_cols = ['id', 'seq', 'label']
    # 校验必要列是否存在
    if not all(col in df.columns for col in required_cols):
        missing = [col for col in required_cols if col not in df.columns]
        raise ValueError(f"CSV缺少必要列: {missing}")

    ids, sequences, labels = [], [], []
    # 遍历数据并清洗
    for _, row in df.iterrows():
        id_val, seq = row['id'], str(row['seq']).strip().upper()
        # 过滤空序列
        if len(seq) == 0:
            print(f"过滤空序列（id: {id_val}）")
            continue
        # 过滤无效标签
        try:
            label = int(row['label'])
            if label not in (0, 1): raise ValueError(f"标签必须为0或1")
        except Exception as e:
            print(f"过滤无效标签（{e}）: {row['label']}（id: {id_val}）")
            continue
        ids.append(id_val)
        sequences.append(seq)
        labels.append(label)

    # 打印数据加载信息
    total, positive = len(labels), sum(labels)
    print(f"加载完成 {os.path.basename(csv_path)}：共{total}条（阳性{positive}条，占比{positive / total * 100:.2f}%）")
    return ids, sequences, labels

# 程序主函数
def main():
    print("=== 从头学习：肽序列分类（CNN模型 + Lookahead+Warmup优化）===")

    print("\n【1/4】加载数据集...")
    train_ids, train_seqs, train_labels = load_csv_data(train_csv)
    val0_ids, val0_seqs, val0_labels = load_csv_data(val0_csv)
    t0_ids, t0_seqs, t0_labels = load_csv_data(t0_csv)
    t1_ids, t1_seqs, t1_labels = load_csv_data(t1_csv)
    t2_ids, t2_seqs, t2_labels = load_csv_data(t2_csv)
    t3_ids, t3_seqs, t3_labels = load_csv_data(t3_csv)

    print("\n【2/4】初始化CNN模型...")
    trainer = ScratchTrainer(device)
    trainer.init_cnn()

    print("\n【3/4】训练从头学习的CNN模型...")
    trainer.train_scratch(train_seqs, train_labels, val0_seqs, val0_labels,
                          epochs=50, lr=5e-4, patience=10)

    print("\n【4/4】评估CNN模型效果...")
    # 评估各数据集
    val0_res = trainer.evaluate(val0_seqs, val0_labels, val0_ids, "val_test_early_stop")
    t0_res = trainer.evaluate(t0_seqs, t0_labels, t0_ids, "rawdata_val")
    t1_res = trainer.evaluate(t1_seqs, t1_labels, t1_ids, "bagel4_1_val")
    t2_res = trainer.evaluate(t2_seqs, t2_labels, t2_ids, "bagel4_2_val")
    t3_res = trainer.evaluate(t3_seqs, t3_labels, t3_ids, "bagel4_3_val")

    # 保存评估结果
    all_results = [val0_res, t0_res, t1_res, t2_res, t3_res]
    save_metrics_summary(all_results, metrics_summary_output)

    save_prediction_results(val0_res, val0_output)
    save_prediction_results(t0_res, t0_output)
    save_prediction_results(t1_res, t1_output)
    save_prediction_results(t2_res, t2_output)
    save_prediction_results(t3_res, t3_output)

    print("\n=== 所有流程完成（CNN模型训练与评估）===")

if __name__ == "__main__":
    main()