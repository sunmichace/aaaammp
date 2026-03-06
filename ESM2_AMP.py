# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    classification_report, confusion_matrix,
    precision_score, recall_score, f1_score,
    roc_curve, auc, precision_recall_curve, matthews_corrcoef, accuracy_score
)
import numpy as np
import pandas as pd
import os
import shutil
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, AutoModel
import torch.serialization
from sklearn.preprocessing import LabelEncoder


torch.serialization.add_safe_globals([LabelEncoder])

plt.rcParams["axes.unicode_minus"] = False


def extract_state_dict(checkpoint_obj):
    """
    兼容常见checkpoint格式:
    - {"model_state_dict": ...}
    - {"state_dict": ...}
    - 直接保存的state_dict
    """
    if isinstance(checkpoint_obj, dict) and "model_state_dict" in checkpoint_obj:
        return checkpoint_obj["model_state_dict"]
    if isinstance(checkpoint_obj, dict) and "state_dict" in checkpoint_obj:
        return checkpoint_obj["state_dict"]
    if isinstance(checkpoint_obj, dict):
        tensor_keys = [k for k, v in checkpoint_obj.items() if isinstance(v, torch.Tensor)]
        if tensor_keys:
            return {k: checkpoint_obj[k] for k in tensor_keys}
    raise ValueError("无法从checkpoint中解析state_dict")


# ===================== 自定义数据集类 =====================
class PeptideDataset(Dataset):
    """
    自定义Pytorch数据集类，适配肽段序列数据加载
    输入：序列列表、标签列表
    输出：单条序列、对应标签、序列长度
    """
    def __init__(self, sequences, labels):
        self.sequences = sequences  # 肽段序列列表
        self.labels = labels        # 序列对应的标签列表 0/1

    def __len__(self):
        """返回数据集总条数"""
        return len(self.sequences)

    def __getitem__(self, idx):
        """根据索引获取单条数据"""
        return self.sequences[idx], self.labels[idx], len(self.sequences[idx])


# ===================== 核心模型训练预测类 =====================
class PeptidePredictor:
    """
    肽段二分类预测器核心类，封装ESM2模型加载、训练、验证、测试、预测、可视化全流程
    """
    def __init__(self, esm2_model_name_or_path, device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        初始化函数
        :param esm2_model_name_or_path: ESM2预训练模型的本地路径/模型名
        :param device: 训练设备 cuda(显卡)/cpu(处理器)
        """
        self.device = device  # 训练设备
        self.esm2_model_name_or_path = esm2_model_name_or_path  # ESM2模型路径
        self.model = None     # 初始化模型对象
        self.tokenizer = None # 初始化分词器对象

        self.all_folds_metrics = []  # 存储五折交叉验证的所有测试集指标

        # 初始化训练/验证/测试的指标存储字典，存储每轮epoch的指标值
        self.train_metrics = {
            "loss": [], "accuracy": [], "precision": [], "recall": [], "f1": [], "mcc": []
        }
        self.val_metrics = {
            "loss": [], "accuracy": [], "precision": [], "recall": [], "f1": [], "mcc": []
        }
        self.test_metrics = {
            "loss": [], "accuracy": [], "precision": [], "recall": [], "f1": [], "mcc": []
        }

    def load_pretrained_model(self, freeze_layers=True):
        """
        加载ESM2预训练模型，并构建二分类下游任务模型
        :param freeze_layers: 是否冻结ESM2的底层网络层，默认True，仅训练高层，防止过拟合
        :return: 加载好的带分类头的ESM2模型
        """
        print(f"正在加载ESM2 8M模型: {self.esm2_model_name_or_path}")
        try:
            # 加载ESM2对应的分词器，用于序列编码
            self.tokenizer = AutoTokenizer.from_pretrained(self.esm2_model_name_or_path)
            # 加载ESM2预训练骨干模型
            esm2_backbone = AutoModel.from_pretrained(self.esm2_model_name_or_path)
            print(f"ESM2 8M隐藏维度: {esm2_backbone.config.hidden_size}")

            # 定义ESM2下游二分类模型：ESM2骨干 + 自定义分类头
            class ESM2Classifier(nn.Module):
                def __init__(self, esm2_backbone):
                    super().__init__()
                    self.esm2_backbone = esm2_backbone  # ESM2预训练骨干
                    self.hidden_dim = esm2_backbone.config.hidden_size  # ESM2的隐藏层维度
                    # 定义分类头：三层全连接+ReLU激活+Dropout正则化，最后输出2分类
                    self.classifier = nn.Sequential(
                        nn.Linear(self.hidden_dim, 512),
                        nn.ReLU(),
                        nn.Dropout(0.6),
                        nn.Linear(512, 256),
                        nn.ReLU(),
                        nn.Dropout(0.6),
                        nn.Linear(256, 2)  # 二分类，输出维度为2
                    )

                def forward(self, input_ids, attention_mask):
                    """
                    前向传播函数，核心推理逻辑
                    :param input_ids: 分词后的序列编码ID
                    :param attention_mask: 注意力掩码，标记有效序列位置
                    :return: 模型预测的二分类logits值
                    """
                    # ESM2骨干模型输出特征
                    outputs = self.esm2_backbone(input_ids=input_ids, attention_mask=attention_mask)
                    # 取CLS_TOKEN的特征向量：[batch_size, seq_len, hidden_dim] -> [batch_size, hidden_dim]
                    cls_repr = outputs.last_hidden_state[:, 0, :]
                    # 计算每条序列的有效长度，用于CLS特征归一化，缓解序列长度差异的影响
                    seq_lengths = attention_mask.sum(dim=1, keepdim=True).float()
                    cls_repr = cls_repr / torch.sqrt(seq_lengths + 1e-8)  # 加极小值防止除零
                    # 分类头做最终预测
                    return self.classifier(cls_repr)

            # 分层冻结策略：冻结底层网络，只训练高层网络，节省显存+防止过拟合
            if freeze_layers:
                # ESM2的encoder层从第4层开始训练，前4层冻结
                for i, layer in enumerate(esm2_backbone.encoder.layer):
                    for param in layer.parameters():
                        param.requires_grad = (i >= 4)
                # 冻结词嵌入层，不训练
                for param in esm2_backbone.embeddings.parameters():
                    param.requires_grad = False

            # 将模型加载到指定设备，并赋值给全局模型对象
            self.model = ESM2Classifier(esm2_backbone).to(self.device)
            print("ESM2 8M二分类模型（CLS归一化+增强建模）加载完成！")
        except Exception as e:
            print(f"模型加载失败: {e}")
            raise

    def _compute_binary_metrics(self, y_true, y_pred):
        """
        私有函数：计算二分类任务的核心评估指标，封装复用
        :param y_true: 真实标签列表
        :param y_pred: 预测标签列表
        :return: 准确率、精确率、召回率、F1分数、马修斯相关系数MCC
        """
        precision = precision_score(y_true, y_pred, zero_division=0)  # 精确率，零分情况置0
        recall = recall_score(y_true, y_pred, zero_division=0)        # 召回率
        f1 = f1_score(y_true, y_pred, zero_division=0)                # F1分数，精准率和召回率的调和平均
        mcc = matthews_corrcoef(y_true, y_pred)                       # MCC，不平衡数据的核心评估指标
        accuracy = accuracy_score(y_true, y_pred)                     # 准确率
        return accuracy, precision, recall, f1, mcc

    def _plot_pr_curve(self, y_true, y_probs, save_dir):
        """绘制精确率-召回率曲线(PR曲线)，不平衡数据比ROC更可靠"""
        precision, recall, _ = precision_recall_curve(y_true, y_probs)
        pr_auc = auc(recall, precision)
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, lw=2, label=f'PR Curve (PR-AUC = {pr_auc:.3f})')
        plt.axhline(y=np.mean(y_true), color='r', linestyle='--',label=f'Random Guess (PR-AUC = {np.mean(y_true):.3f})')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Binary Classification PR Curve (More Reliable for Imbalanced Data)')
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(f"{save_dir}/pr_curve.pdf", dpi=300)
        plt.close()
        print(f"PR曲线已保存至: {save_dir}/pr_curve.pdf")

    def _plot_length_sensitivity(self, sequences, y_true, y_pred, save_dir):
        """绘制不同序列长度的模型性能对比图，评估模型的长度鲁棒性"""
        length_groups = {
            "Short sequences (≤30aa)": [],
            "Medium sequences (30-100aa)": [],
            "Long sequences (>100aa)": []
        }
        # 按序列长度分组统计
        for seq, true, pred in zip(sequences, y_true, y_pred):
            l = len(seq)
            if l <= 30:
                length_groups["Short sequences (≤30aa)"].append((true, pred))
            elif 30 < l <= 100:
                length_groups["Medium sequences (30-100aa)"].append((true, pred))
            else:
                length_groups["Long sequences (>100aa)"].append((true, pred))

        metrics = {"f1": [], "accuracy": []}
        labels = []
        for group_name, data in length_groups.items():
            if not data:
                continue
            labels.append(group_name)
            yt, yp = zip(*data)
            metrics["f1"].append(f1_score(yt, yp, zero_division=0))
            metrics["accuracy"].append(accuracy_score(yt, yp))

        x = np.arange(len(labels))
        width = 0.35
        plt.figure(figsize=(10, 6))
        plt.bar(x - width / 2, metrics["accuracy"], width, label='Accuracy')
        plt.bar(x + width / 2, metrics["f1"], width, label='F1 Score')
        plt.ylim(0, 1.05)
        plt.xlabel('Sequence Length Groups')
        plt.ylabel('Score')
        plt.title('Model Performance on Different Sequence Lengths (Length Sensitivity)')
        plt.xticks(x, labels, rotation=30)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{save_dir}/length_sensitivity.pdf", dpi=300)
        plt.close()
        print(f"长度敏感性分析图已保存至: {save_dir}/length_sensitivity.pdf")

    def _plot_loss_curve(self, save_dir):
        """绘制训练/验证集损失曲线，观察模型收敛情况"""
        epochs = range(1, len(self.train_metrics["loss"]) + 1)
        plt.figure(figsize=(8, 5))
        plt.plot(epochs, self.train_metrics["loss"], 'b-', label='Training Loss')
        plt.plot(epochs, self.val_metrics["loss"], 'g-', label='Validation Loss')
        plt.title('Training/Validation Loss Curve')
        plt.xlabel('Epoch')
        plt.ylabel('Loss Value')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{save_dir}/loss_curve.pdf", dpi=300)
        plt.close()

    def _plot_accuracy_curve(self, save_dir):
        """绘制训练/验证集准确率曲线"""
        epochs = range(1, len(self.train_metrics["accuracy"]) + 1)
        plt.figure(figsize=(8, 5))
        plt.plot(epochs, self.train_metrics["accuracy"], 'b-', label='Training Accuracy')
        plt.plot(epochs, self.val_metrics["accuracy"], 'g-', label='Validation Accuracy')
        plt.title('Training/Validation Accuracy Curve')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.ylim(0, 100)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{save_dir}/accuracy_curve.pdf", dpi=300)
        plt.close()

    def _plot_mcc_curve(self, save_dir):
        """绘制训练/验证集MCC曲线，评估不平衡数据的拟合效果"""
        epochs = range(1, len(self.train_metrics["mcc"]) + 1)
        plt.figure(figsize=(8, 5))
        plt.plot(epochs, self.train_metrics["mcc"], 'b-', label='Training MCC')
        plt.plot(epochs, self.val_metrics["mcc"], 'g-', label='Validation MCC')
        plt.ylim(-1, 1)
        plt.title('Training/Validation MCC Curve (Matthews Correlation Coefficient)')
        plt.xlabel('Epoch')
        plt.ylabel('MCC Value')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{save_dir}/mcc_curve.pdf", dpi=300)
        plt.close()
        print(f"MCC曲线已保存至: {save_dir}/mcc_curve.pdf")

    def _plot_roc_curve(self, y_true, y_probs, save_dir):
        """绘制ROC曲线，计算AUC值，评估二分类区分能力"""
        fpr, tpr, _ = roc_curve(y_true, y_probs)
        roc_auc = auc(fpr, tpr)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, lw=2, label=f'ROC Curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], 'k--', lw=2)  # 随机猜测基线
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate (FPR)')
        plt.ylabel('True Positive Rate (TPR)')
        plt.title('Binary Classification ROC Curve')
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(f"{save_dir}/roc_curve.pdf", dpi=300)
        plt.close()

    def _plot_confusion_matrix(self, y_true, y_pred, save_dir):
        """绘制混淆矩阵，直观展示模型预测的真假阳性/阴性"""
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['0 (Negative)', '1 (Positive)'],
                    yticklabels=['0 (Negative)', '1 (Positive)'])
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title('Binary Classification Confusion Matrix')
        plt.tight_layout()
        plt.savefig(f"{save_dir}/confusion_matrix.pdf", dpi=300)
        plt.close()

    def _reset_metrics(self):
        """重置训练和验证的指标字典，每折交叉验证前执行"""
        self.train_metrics = {key: [] for key in self.train_metrics}
        self.val_metrics = {key: [] for key in self.val_metrics}

    def train_model(self, train_sequences, train_labels, val_sequences, val_labels,
                    test_sequences, test_labels, fold_num,
                    save_metrics_dir="model_metrics",
                    batch_size=16, epochs=10, lr=1e-4,
                    early_stop_patience=3, min_delta=0.001):
        """
        模型核心训练函数，单折交叉验证的完整训练+验证+测试流程
        :param train_sequences: 训练集序列
        :param train_labels: 训练集标签
        :param val_sequences: 验证集序列
        :param val_labels: 验证集标签
        :param test_sequences: 测试集序列
        :param test_labels: 测试集标签
        :param fold_num: 当前是第几折交叉验证
        :param save_metrics_dir: 指标和绘图的保存根目录
        :param batch_size: 批次大小
        :param epochs: 训练轮数
        :param lr: 学习率
        :param early_stop_patience: 早停耐心值，验证集指标不提升的轮数
        :param min_delta: 指标最小提升阈值
        :return: 训练完成的模型
        """
        # 重置本轮的训练验证指标
        self._reset_metrics()

        # 创建当前折的结果保存目录
        fold_save_dir = os.path.join(save_metrics_dir, f"fold_{fold_num}")
        os.makedirs(fold_save_dir, exist_ok=True)

        # 赋值数据集
        X_train, y_train = train_sequences, train_labels
        X_val, y_val = val_sequences, val_labels
        X_test, y_test = test_sequences, test_labels
        print(f"\n--- Fold {fold_num} ---")
        print(f"训练集: {len(X_train)}条（阳性{sum(y_train)}条）")
        print(f"验证集: {len(X_val)}条（阳性{sum(y_val)}条）")

        # 加载数据集
        train_dataset = PeptideDataset(X_train, y_train)
        val_dataset = PeptideDataset(X_val, y_val)
        test_dataset = PeptideDataset(X_test, y_test)

        # 自定义批处理函数：解决序列长度不一致的问题，动态padding到当前批次的最大长度
        def collate_fn(batch):
            sequences, labels, seq_lengths = zip(*batch)
            # 序列编码：不自动padding，截断最大长度256
            encoded = self.tokenizer(
                sequences,
                return_tensors=None,
                padding="do_not_pad",
                truncation=True,
                max_length=256
            )
            input_ids_list = encoded["input_ids"]
            attention_mask_list = encoded["attention_mask"]
            # 获取当前批次的最大序列长度
            batch_max_len = max(len(ids) for ids in input_ids_list)
            pad_token_id = self.tokenizer.pad_token_id
            padded_input_ids = []
            padded_attention_mask = []
            # 对每条序列进行padding补0
            for ids, mask in zip(input_ids_list, attention_mask_list):
                pad_len = batch_max_len - len(ids)
                padded_ids = ids + [pad_token_id] * pad_len
                padded_mask = mask + [0] * pad_len
                padded_input_ids.append(padded_ids)
                padded_attention_mask.append(padded_mask)
            # 返回tensor格式的输入数据
            return (torch.tensor(padded_input_ids, dtype=torch.long),
                    torch.tensor(padded_attention_mask, dtype=torch.long),
                    torch.tensor(labels, dtype=torch.long))

        # 创建数据加载器，实现批量加载和打乱
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=collate_fn)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=collate_fn)

        # 定义损失函数和优化器
        criterion = nn.CrossEntropyLoss()  # 交叉熵损失，适配二分类
        optimizer = optim.Adam(
            [p for p in self.model.parameters() if p.requires_grad],  # 只优化可训练的参数
            lr=lr,
            weight_decay=1e-4  # L2正则化，防止过拟合
        )

        # 早停策略初始化：保存验证集最优F1的模型
        best_val_f1 = -1.0
        early_stop_count = 0

        # 开始训练循环
        for epoch in range(epochs):
            self.model.train()  # 模型切换到训练模式：启用Dropout、BatchNorm训练
            running_loss = 0.0
            correct = 0
            total = 0
            train_y_true = []
            train_y_pred = []

            # 进度条可视化训练过程
            progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), leave=False)
            for batch_idx, (input_ids, attention_mask, targets) in progress_bar:
                # 将数据加载到指定设备
                input_ids = input_ids.to(self.device)
                attention_mask = attention_mask.to(self.device)
                targets = targets.to(self.device)

                # 前向传播：模型预测
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                loss = criterion(outputs, targets)  # 计算损失

                # 反向传播+梯度更新
                optimizer.zero_grad()  # 梯度清零
                loss.backward()        # 反向传播计算梯度
                optimizer.step()       # 更新参数

                # 统计训练指标
                running_loss += loss.item()
                _, predicted = outputs.max(1)  # 取logits最大值作为预测标签
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                train_y_true.extend(targets.cpu().numpy())
                train_y_pred.extend(predicted.cpu().numpy())
                # 更新进度条显示
                progress_bar.set_description(
                    f"Fold {fold_num} Epoch {epoch + 1} - Loss: {running_loss / (batch_idx + 1):.4f}, "
                    f"Acc: {100. * correct / total:.2f}%"
                )

            # 计算本轮训练集的所有指标并保存
            train_loss = running_loss / len(train_loader)
            train_acc, train_precision, train_recall, train_f1, train_mcc = self._compute_binary_metrics(train_y_true, train_y_pred)
            self.train_metrics["loss"].append(train_loss)
            self.train_metrics["accuracy"].append(train_acc * 100)
            self.train_metrics["precision"].append(train_precision)
            self.train_metrics["recall"].append(train_recall)
            self.train_metrics["f1"].append(train_f1)
            self.train_metrics["mcc"].append(train_mcc)

            # 验证集评估：无梯度计算，节省显存
            val_loss, val_acc, val_y_true, val_y_pred = self.evaluate_model(val_loader, criterion)
            val_acc, val_precision, val_recall, val_f1, val_mcc = self._compute_binary_metrics(val_y_true, val_y_pred)
            self.val_metrics["loss"].append(val_loss)
            self.val_metrics["accuracy"].append(val_acc * 100)
            self.val_metrics["precision"].append(val_precision)
            self.val_metrics["recall"].append(val_recall)
            self.val_metrics["f1"].append(val_f1)
            self.val_metrics["mcc"].append(val_mcc)

            print(f"Fold {fold_num} Epoch {epoch + 1} - Val Loss: {val_loss:.4f}, Val Acc: {val_acc * 100:.2f}%, Val F1: {val_f1:.4f}")

            # 早停逻辑：验证集F1提升则保存最优模型，否则计数
            if val_f1 > best_val_f1 + min_delta:
                best_val_f1 = val_f1
                early_stop_count = 0
                # 保存最优模型的权重和相关信息
                model_filename = os.path.join(fold_save_dir, "esm2_8m_binary_best.pt")
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'epoch': epoch + 1,
                    'val_f1': val_f1,
                    'fold_num': fold_num,
                }, model_filename)
            else:
                early_stop_count += 1
                if early_stop_count >= early_stop_patience:
                    print(f"Fold {fold_num} - Early stopping triggered after {epoch + 1} epochs.")
                    break

        # 加载当前折的最优模型，在测试集上做最终评估
        print(f"--- Evaluating Fold {fold_num} best model on Test Set ---")
        best_model_path = os.path.join(fold_save_dir, "esm2_8m_binary_best.pt")
        checkpoint = torch.load(best_model_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(extract_state_dict(checkpoint))

        # 绘制并保存所有训练相关的曲线
        self._plot_loss_curve(fold_save_dir)
        self._plot_accuracy_curve(fold_save_dir)
        self._plot_mcc_curve(fold_save_dir)

        # 测试集评估
        test_loss, test_acc, test_y_true, test_y_pred = self.evaluate_model(test_loader, criterion)
        test_acc, test_precision, test_recall, test_f1, test_mcc = self._compute_binary_metrics(test_y_true, test_y_pred)

        # 保存当前折的测试集指标
        self.all_folds_metrics.append({
            'fold': fold_num,
            'accuracy': test_acc,
            'precision': test_precision,
            'recall': test_recall,
            'f1': test_f1,
            'mcc': test_mcc,
            'model_path': best_model_path
        })

        # 绘制并保存测试集的所有评估图
        self._plot_confusion_matrix(test_y_true, test_y_pred, fold_save_dir)
        test_probs = self._get_test_probabilities(test_loader)
        self._plot_roc_curve(test_y_true, test_probs, fold_save_dir)
        self._plot_pr_curve(test_y_true, test_probs, fold_save_dir)
        self._plot_length_sensitivity(X_test, test_y_true, test_y_pred, fold_save_dir)

        # 生成分类报告并保存
        report = classification_report(
            test_y_true, test_y_pred,
            target_names=['阴性（0）', '阳性（1）'],
            zero_division=0,
            output_dict=True
        )
        report_str = classification_report(
            test_y_true, test_y_pred,
            target_names=['阴性（0）', '阳性（1）'],
            zero_division=0
        )
        report_str += f"\n马修斯相关系数（MCC）: {test_mcc:.4f}"
        print(f"Fold {fold_num} Test Set Classification Report:\n", report_str)

        with open(os.path.join(fold_save_dir, "classification_report.txt"), "w", encoding="utf-8") as f:
            f.write(report_str)

        return self.model

    def _get_test_probabilities(self, dataloader):
        """获取模型对测试集的预测概率（阳性概率）"""
        self.model.eval()
        all_probs = []
        with torch.no_grad():
            for input_ids, attention_mask, _ in dataloader:
                input_ids = input_ids.to(self.device)
                attention_mask = attention_mask.to(self.device)
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()  # 取阳性类别的概率
                all_probs.extend(probs)
        return np.array(all_probs)

    def evaluate_model(self, dataloader, criterion):
        """
        模型评估函数：无梯度计算，返回损失、准确率、真实标签、预测标签
        :param dataloader: 评估数据集的加载器
        :param criterion: 损失函数
        :return: 平均损失、准确率、真实标签列表、预测标签列表
        """
        self.model.eval()  # 模型切换到评估模式：关闭Dropout、固定BatchNorm
        test_loss = 0
        correct = 0
        total = 0
        y_true = []
        y_pred = []
        with torch.no_grad():  # 禁用梯度计算，大幅节省显存和计算时间
            for input_ids, attention_mask, targets in dataloader:
                input_ids = input_ids.to(self.device)
                attention_mask = attention_mask.to(self.device)
                targets = targets.to(self.device)
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                loss = criterion(outputs, targets)
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                y_true.extend(targets.cpu().numpy())
                y_pred.extend(predicted.cpu().numpy())
        test_loss /= len(dataloader)
        test_acc = correct / total
        return test_loss, test_acc, y_true, y_pred

    def predict(self, sequences, batch_size=16, model_path=None):
        """
        批量预测函数：输入肽段序列，返回预测标签和阳性概率
        :param sequences: 待预测的肽段序列列表
        :param batch_size: 预测批次大小
        :param model_path: 预训练模型权重路径
        :return: 预测标签列表、阳性概率列表
        """
        if not self.model and not model_path:
            raise ValueError("请加载模型或指定模型路径")
        if model_path:
            print(f"加载ESM2 8M模型: {model_path}")
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(extract_state_dict(checkpoint))
        self.model.eval()
        predictions = []
        probabilities = []
        # 构造预测数据集（标签无意义，填充0即可）
        dataset = PeptideDataset(sequences, [0] * len(sequences))

        # 自定义批处理函数，同训练阶段
        def collate_fn(batch):
            sequences, _, _ = zip(*batch)
            encoded = self.tokenizer(
                sequences, return_tensors=None, padding="do_not_pad", truncation=True, max_length=256
            )
            input_ids_list = encoded["input_ids"]
            attention_mask_list = encoded["attention_mask"]
            batch_max_len = max(len(ids) for ids in input_ids_list)
            pad_token_id = self.tokenizer.pad_token_id
            padded_input_ids = []
            padded_attention_mask = []
            for ids, mask in zip(input_ids_list, attention_mask_list):
                pad_len = batch_max_len - len(ids)
                padded_ids = ids + [pad_token_id] * pad_len
                padded_mask = mask + [0] * pad_len
                padded_input_ids.append(padded_ids)
                padded_attention_mask.append(padded_mask)
            return torch.tensor(padded_input_ids, dtype=torch.long), torch.tensor(padded_attention_mask, dtype=torch.long)

        dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)
        print(f"开始预测 {len(sequences)} 条序列...")
        with torch.no_grad():
            for input_ids, attention_mask in tqdm(dataloader, total=len(dataloader)):
                input_ids = input_ids.to(self.device)
                attention_mask = attention_mask.to(self.device)
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                probs = torch.softmax(outputs, dim=1).cpu().numpy()
                predictions.extend(np.argmax(probs, axis=1))  # 取概率最大的类别作为预测标签
                probabilities.extend(probs[:, 1])  # 取阳性类别的概率
        return predictions, probabilities

    def save_predictions(self, ids, sequences, predictions, probabilities, output_file, true_labels=None):
        """将预测结果保存为CSV文件，方便后续分析"""
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        results = []
        for i, (id_val, seq, pred, prob) in enumerate(zip(ids, sequences, predictions, probabilities)):
            true_label = true_labels[i] if (true_labels and i < len(true_labels)) else "未知"
            results.append({
                "id": id_val, "sequence": seq, "sequence_length": len(seq),
                "true_label": true_label, "predicted_label": pred,
                "predicted_label_name": "阳性（1）" if pred == 1 else "阴性（0）",
                "positive_probability": round(float(prob), 4)
            })
        pd.DataFrame(results).to_csv(output_file, index=False, encoding="utf-8-sig")
        print(f"预测结果已保存至: {output_file}")

    def generate_cross_validation_summary(self, save_dir):
        """生成五折交叉验证的综合指标报告和箱线图，统计平均值和标准差"""
        if not self.all_folds_metrics:
            print("没有可用的交叉验证指标来生成报告。")
            return

        df_metrics = pd.DataFrame(self.all_folds_metrics)

        summary_report = "五折交叉验证测试集性能综合报告\n"
        summary_report += "=" * 50 + "\n\n"

        summary_report += df_metrics[['fold', 'accuracy', 'precision', 'recall', 'f1', 'mcc']].to_string(index=False) + "\n\n"

        # 计算各指标的平均值和标准差
        mean_metrics = df_metrics[['accuracy', 'precision', 'recall', 'f1', 'mcc']].mean()
        std_metrics = df_metrics[['accuracy', 'precision', 'recall', 'f1', 'mcc']].std()

        summary_report += "各指标平均值 ± 标准差:\n"
        summary_report += f"  Accuracy: {mean_metrics['accuracy']:.4f} ± {std_metrics['accuracy']:.4f}\n"
        summary_report += f"  Precision: {mean_metrics['precision']:.4f} ± {std_metrics['precision']:.4f}\n"
        summary_report += f"  Recall: {mean_metrics['recall']:.4f} ± {std_metrics['recall']:.4f}\n"
        summary_report += f"  F1-Score: {mean_metrics['f1']:.4f} ± {std_metrics['f1']:.4f}\n"
        summary_report += f"  MCC: {mean_metrics['mcc']:.4f} ± {std_metrics['mcc']:.4f}\n"

        print("\n" + "=" * 50)
        print(summary_report)

        # 保存报告
        with open(os.path.join(save_dir, "cross_validation_summary.txt"), "w", encoding="utf-8") as f:
            f.write(summary_report)

        # 绘制箱线图展示指标分布
        plt.figure(figsize=(12, 8))
        metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1', 'mcc']
        df_metrics[metrics_to_plot].boxplot()
        plt.title('五折交叉验证各指标分布')
        plt.ylabel('分数')
        plt.xticks(rotation=45)
        plt.grid(False)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "cross_validation_boxplot.pdf"), dpi=300)
        plt.close()
        print(f"\n交叉验证综合报告已保存至: {os.path.join(save_dir, 'cross_validation_summary.txt')}")
        print(f"交叉验证箱线图已保存至: {os.path.join(save_dir, 'cross_validation_boxplot.pdf')}")

    def select_best_model(self, save_dir):
        """
        从五折交叉验证的模型中选择最优模型
        选择标准：测试集F1分数最高，F1相同则选MCC最高的模型
        :return: 最优模型的保存路径
        """
        if not self.all_folds_metrics:
            print("没有可用的模型指标，无法选择最佳模型。")
            return None

        df_metrics = pd.DataFrame(self.all_folds_metrics)
        # 按F1降序、MCC降序排序
        df_metrics_sorted = df_metrics.sort_values(by=['f1', 'mcc'], ascending=[False, False])
        best_model_info = df_metrics_sorted.iloc[0]

        best_fold = int(best_model_info['fold'])
        best_f1 = best_model_info['f1']
        best_mcc = best_model_info['mcc']
        best_accuracy = best_model_info['accuracy']
        best_model_path = best_model_info['model_path']
        if not os.path.exists(best_model_path):
            raise FileNotFoundError(f"最佳模型文件不存在: {best_model_path}")

        # 创建最优模型保存目录
        best_model_dir = os.path.join(save_dir, "best_model_amp")
        os.makedirs(best_model_dir, exist_ok=True)

        # 复制最优模型到指定目录
        dest_model_path = os.path.join(best_model_dir, "esm2_8m_binary_best.pt")
        shutil.copy2(best_model_path, dest_model_path)

        # 生成最优模型报告
        best_model_info_str = f"最佳模型选择报告\n"
        best_model_info_str += "=" * 50 + "\n\n"
        best_model_info_str += f"选择标准：测试集F1分数最高（F1相同则MCC最高）\n\n"
        best_model_info_str += f"最佳模型所属折数：Fold {best_fold}\n"
        best_model_info_str += f"模型原始路径：{best_model_path}\n"
        best_model_info_str += f"模型保存路径：{dest_model_path}\n\n"
        best_model_info_str += f"测试集性能指标：\n"
        best_model_info_str += f"  Accuracy: {best_accuracy:.4f}\n"
        best_model_info_str += f"  Precision: {best_model_info['precision']:.4f}\n"
        best_model_info_str += f"  Recall: {best_model_info['recall']:.4f}\n"
        best_model_info_str += f"  F1-Score: {best_f1:.4f}\n"
        best_model_info_str += f"  MCC: {best_mcc:.4f}\n\n"
        best_model_info_str += f"五折交叉验证平均指标（参考）：\n"
        mean_metrics = df_metrics[['accuracy', 'precision', 'recall', 'f1', 'mcc']].mean()
        best_model_info_str += f"  平均Accuracy: {mean_metrics['accuracy']:.4f}\n"
        best_model_info_str += f"  平均Precision: {mean_metrics['precision']:.4f}\n"
        best_model_info_str += f"  平均Recall: {mean_metrics['recall']:.4f}\n"
        best_model_info_str += f"  平均F1-Score: {mean_metrics['f1']:.4f}\n"
        best_model_info_str += f"  平均MCC: {mean_metrics['mcc']:.4f}\n"

        # 保存报告
        info_file_path = os.path.join(best_model_dir, "best_model_info.txt")
        with open(info_file_path, "w", encoding="utf-8") as f:
            f.write(best_model_info_str)

        print("\n" + "=" * 60)
        print("最佳模型选择完成！")
        print(best_model_info_str)

        return dest_model_path


# ===================== 工具函数 =====================
def load_csv_data(csv_path):
    """
    加载CSV格式的肽段数据集，数据清洗和校验
    CSV要求包含列：id, seq, label
    :param csv_path: CSV文件路径
    :return: 清洗后的id列表、序列列表、标签列表
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV文件不存在: {csv_path}")
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        raise IOError(f"读取CSV失败: {str(e)}")
    if "seq" in df.columns:
        seq_col = "seq"
    elif "sequence" in df.columns:
        seq_col = "sequence"
    else:
        raise ValueError("CSV缺少必要列: ['seq' 或 'sequence']")
    if "label" not in df.columns:
        raise ValueError("CSV缺少必要列: ['label']")
    if "id" not in df.columns:
        df = df.copy()
        df["id"] = np.arange(1, len(df) + 1)

    ids, sequences, labels = [], [], []
    # 逐行读取并清洗数据
    for _, row in df.iterrows():
        id_val = row['id']
        seq = str(row[seq_col]).strip().upper()  # 序列转大写，去除首尾空格
        if len(seq) == 0:
            print(f"过滤空序列（id: {id_val}）")
            continue
        # 校验标签是否为0/1
        try:
            label = int(row['label'])
            if label not in (0, 1):
                raise ValueError(f"标签必须为0或1，当前为{label}")
        except Exception as e:
            print(f"过滤无效标签（{e}）: {row['label']}（id: {id_val}）")
            continue
        ids.append(id_val)
        sequences.append(seq)
        labels.append(label)
    if len(sequences) == 0:
        raise ValueError(f"数据清洗后无有效样本: {csv_path}")
    print(f"从 {os.path.basename(csv_path)} 加载完成：共{len(sequences)}条有效序列（阳性{sum(labels)}条）")
    return ids, sequences, labels


# ===================== 主函数入口 =====================
if __name__ == "__main__":
    # 配置文件路径和超参数
    ROOT_DIR = "./"
    base_fold_path = os.path.join(ROOT_DIR, "mydata/newdata/fold_")  # 交叉验证的训练/验证集路径
    test_csv = os.path.join(ROOT_DIR, "mydata/newdata/rawdata_test.csv")  # 独立测试集路径
    esm2_model_path = os.path.join(ROOT_DIR, "model/esm2_t6_8M_UR50D")  # ESM2预训练模型路径
    save_metrics_dir = os.path.join(ROOT_DIR, "model/model_AMP_ESM2_8M_256_v3")  # 结果保存根目录

    # 训练超参数
    batch_size = 8
    epochs = 10
    learning_rate = 1e-4
    early_stop_patience = 3
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")

    # 加载独立测试集
    try:
        print("\n加载最终测试集...")
        test_ids, test_sequences, test_labels = load_csv_data(test_csv)
        if len(test_sequences) == 0:
            print("测试集无有效数据，无法训练")
            exit(1)
    except Exception as e:
        print(f"数据加载失败: {e}")
        exit(1)

    # 初始化模型预测器
    try:
        predictor = PeptidePredictor(esm2_model_name_or_path=esm2_model_path, device=device)
    except Exception as e:
        print(f"模型初始化失败: {e}")
        exit(1)

    # 五折交叉验证训练循环
    num_folds = 5
    for fold in range(1, num_folds + 1):
        train_csv_path = f"{base_fold_path}{fold}_train.csv"
        val_csv_path = f"{base_fold_path}{fold}_val.csv"

        print(f"\n{'=' * 60}")
        print(f"加载 Fold {fold} 的训练集和验证集...")
        try:
            train_ids, train_sequences, train_labels = load_csv_data(train_csv_path)
            val_ids, val_sequences, val_labels = load_csv_data(val_csv_path)
            if len(train_sequences) == 0 or len(val_sequences) == 0:
                print(f"Fold {fold} 训练集/验证集无有效数据，跳过此折。")
                continue
        except Exception as e:
            print(f"Fold {fold} 数据加载失败: {e}，跳过此折。")
            continue

        # 加载预训练模型
        predictor.load_pretrained_model(freeze_layers=True)

        # 训练当前折的模型
        predictor.train_model(
            train_sequences=train_sequences, train_labels=train_labels,
            val_sequences=val_sequences, val_labels=val_labels,
            test_sequences=test_sequences, test_labels=test_labels,
            fold_num=fold,
            save_metrics_dir=save_metrics_dir,
            batch_size=batch_size, epochs=epochs, lr=learning_rate,
            early_stop_patience=early_stop_patience, min_delta=0.001
        )

        # 清理CUDA显存，防止显存溢出
        if device == 'cuda':
            torch.cuda.empty_cache()

    # 生成交叉验证综合报告
    print("\n" + "=" * 60)
    print("五折交叉验证全部完成！")
    predictor.generate_cross_validation_summary(save_metrics_dir)

    # 选择最优模型并保存
    best_model_path = predictor.select_best_model(save_metrics_dir)

    # 打印最终结果保存路径
    print(f"\n所有模型和指标文件已保存至: {save_metrics_dir}")
    if best_model_path is None:
        print("未选出最佳模型：请检查各折训练是否成功生成有效指标。")
    else:
        print(f"最佳模型已保存至: {best_model_path}")
        print(f"最佳模型说明文件: {os.path.join(os.path.dirname(best_model_path), 'best_model_info.txt')}")
