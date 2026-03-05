# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import (
    classification_report, confusion_matrix, precision_score, recall_score,
    f1_score, roc_curve, auc, precision_recall_curve, matthews_corrcoef, accuracy_score, roc_auc_score
)

import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import seaborn as sns

from transformers import AutoTokenizer, AutoModel
import torch.serialization
from sklearn.preprocessing import LabelEncoder
import time

# 图神经网络：GCN卷积层、全局均值池化、批次处理、图数据结构
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Batch, Data

# 解决LabelEncoder序列化警告问题
torch.serialization.add_safe_globals([LabelEncoder])

# ===================== 全局配置参数区 (所有路径/设备配置，可根据需求修改) =====================
# 根目录
ROOT_DIR = "./"

# 训练/验证/测试集数据路径
train_csv = os.path.join(ROOT_DIR, "mydata/newdata/rawdata_train.csv")  # 训练集
val0_csv = os.path.join(ROOT_DIR, "mydata/newdata/rawdata_val.csv")  # 验证集(val_test)
t0_csv = os.path.join(ROOT_DIR, "mydata/newdata/rawdata_test.csv")  # 测试集t0
t1_csv = os.path.join(ROOT_DIR, "test_data/bagel4_1_val.csv")  # 测试集t1
t2_csv = os.path.join(ROOT_DIR, "test_data/bagel4_2_val.csv")  # 测试集t2
t3_csv = os.path.join(ROOT_DIR, "test_data/bagel4_3_val.csv")  # 测试集t3

# ESM2预训练模型和微调ESM模型路径
esm2_backbone_path = os.path.join(ROOT_DIR, "model/esm2_t6_8M_UR50D")  # ESM2 8M backbone路径
finetuned_esm_path = os.path.join(ROOT_DIR,
                                  "model/model_AMP_ESM2_8M_256_v3/best_model_amp/esm2_8m_binary_best.pt")  # 微调后的ESM模型权重

# 各数据集预测结果输出路径
val0_output = os.path.join(ROOT_DIR, "results_ultra_light_gnn/val0_val_test_early_stop_predictions.csv")
t0_output = os.path.join(ROOT_DIR, "results_ultra_light_gnn/t0_rawdata_val_predictions.csv")
t1_output = os.path.join(ROOT_DIR, "results_ultra_light_gnn/t1_bagel4_1_val_predictions.csv")
t2_output = os.path.join(ROOT_DIR, "results_ultra_light_gnn/t2_bagel4_2_val_predictions.csv")
t3_output = os.path.join(ROOT_DIR, "results_ultra_light_gnn/t3_bagel4_3_val_predictions.csv")
metrics_summary_output = os.path.join(ROOT_DIR, "results_ultra_light_gnn/metrics_summary.csv")  # 所有模型指标汇总表

# 设备选择：优先使用GPU(cuda)，无GPU则使用CPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'


# ===================== 自定义数据集类 =====================
class PeptideDataset(Dataset):
    """
    肽序列数据集类，继承torch.utils.data.Dataset
    适配肽序列+标签的数据格式，用于模型训练/推理的数据加载
    """

    def __init__(self, sequences, labels):
        """
        初始化数据集
        :param sequences: 肽序列列表
        :param labels: 对应标签列表 (0/1 二分类)
        """
        self.sequences = sequences
        self.labels = labels
        # 校验：序列数量必须和标签数量一致
        if len(self.sequences) != len(self.labels):
            raise ValueError("序列数量与标签数量不匹配")

    def __len__(self):
        """返回数据集总长度"""
        return len(self.sequences)

    def __getitem__(self, idx):
        """
        按索引获取单条数据
        :param idx: 数据索引
        :return: 单条序列、对应标签、序列长度
        """
        return self.sequences[idx], self.labels[idx], len(self.sequences[idx])


# ===================== ESM2特征提取器类 =====================
class PeptideFeatureExtractor:
    """
    基于ESM2预训练模型的肽序列特征提取器
    核心功能：加载ESM2模型、对肽序列编码、提取CLS特征用于下游GNN模型训练
    """

    def __init__(self, esm2_model_name_or_path, device=device):
        """
        初始化特征提取器
        :param esm2_model_name_or_path: ESM2模型路径/名称
        :param device: 运行设备 cuda/cpu
        """
        self.device = device
        self.esm2_model_name_or_path = esm2_model_name_or_path
        self.model = None  # ESM2特征提取模型
        self.tokenizer = None  # ESM2分词器
        self.hidden_size = None  # ESM2模型的隐藏层维度
        self.total_params = 0  # ESM2模型总参数量

    def load_pretrained_model(self, finetuned_model_path=None):
        """
        加载预训练的ESM2模型+微调后的权重，并构建特征提取器
        :param finetuned_model_path: 微调后的ESM模型权重路径
        """
        print(f"\n正在加载特征提取模型: {finetuned_model_path}")
        try:
            # 加载ESM2分词器和骨干模型
            self.tokenizer = AutoTokenizer.from_pretrained(self.esm2_model_name_or_path)
            esm2_backbone = AutoModel.from_pretrained(self.esm2_model_name_or_path)
            self.hidden_size = esm2_backbone.config.hidden_size  # 获取ESM2隐藏层维度
            print(f"ESM2 8M隐藏维度: {self.hidden_size}")

            # 定义内部ESM2特征提取子类：仅保留骨干+提取CLS特征，无分类头
            class ESM2FeatureExtractor(nn.Module):
                def __init__(self, esm2_backbone):
                    super().__init__()
                    self.esm2_backbone = esm2_backbone
                    self.hidden_dim = esm2_backbone.config.hidden_size

                def forward(self, input_ids, attention_mask):
                    # 前向传播获取模型输出
                    outputs = self.esm2_backbone(input_ids=input_ids, attention_mask=attention_mask)
                    cls_repr = outputs.last_hidden_state[:, 0, :]  # 提取<cls>位置的特征 (batch, hidden_dim)
                    seq_lengths = attention_mask.sum(dim=1, keepdim=True).float()  # 序列有效长度
                    cls_repr = cls_repr / torch.sqrt(seq_lengths + 1e-8)  # 特征归一化，防止梯度爆炸
                    return cls_repr

            # 实例化特征提取模型并部署到指定设备
            self.model = ESM2FeatureExtractor(esm2_backbone).to(self.device)
            # 校验权重文件是否存在
            if not os.path.exists(finetuned_model_path):
                raise FileNotFoundError(f"微调模型文件不存在: {finetuned_model_path}")
            # 加载微调后的权重
            checkpoint = torch.load(finetuned_model_path, map_location=self.device, weights_only=False)

            # 提取骨干模型权重（过滤分类头权重）
            model_state_dict = checkpoint['model_state_dict']
            feature_extractor_state_dict = {}
            for k, v in model_state_dict.items():
                if k.startswith('esm2_backbone.'):
                    feature_extractor_state_dict[k] = v
            self.model.load_state_dict(feature_extractor_state_dict, strict=False)

            # 冻结所有层：特征提取阶段不训练，仅做推理
            for param in self.model.parameters():
                param.requires_grad = False
            print("已冻结特征提取模型所有层")

            # 计算模型总参数量
            self.total_params = sum(p.numel() for p in self.model.parameters())
            print(f"特征提取模型总参数量: {self.total_params:,}")
            print("特征提取模型加载完成！")
        except Exception as e:
            print(f"模型加载失败: {e}")
            raise

    def get_cls_features(self, sequences, batch_size=16):
        """
        对肽序列批量提取CLS特征
        :param sequences: 肽序列列表
        :param batch_size: 批次大小
        :return: 特征数组 (n_samples, hidden_dim)
        """
        if not self.model: raise ValueError("请先加载特征提取模型")
        self.model.eval()  # 模型切换到推理模式
        features = []
        dataset = PeptideDataset(sequences, [0] * len(sequences))  # 构造数据集，标签占位符无意义

        # 自定义批次整理函数：对序列做变长padding，适配ESM2输入格式
        def collate_fn(batch):
            sequences, _, _ = zip(*batch)
            encoded = self.tokenizer(
                sequences, return_tensors=None, padding="do_not_pad", truncation=True, max_length=256
            )
            input_ids_list, attention_mask_list = encoded["input_ids"], encoded["attention_mask"]
            batch_max_len = max(len(ids) for ids in input_ids_list)
            pad_token_id = self.tokenizer.pad_token_id
            # 对每个序列padding到当前批次最大长度
            padded_input_ids = [ids + [pad_token_id] * (batch_max_len - len(ids)) for ids in input_ids_list]
            padded_attention_mask = [mask + [0] * (batch_max_len - len(mask)) for mask in attention_mask_list]
            return torch.tensor(padded_input_ids, dtype=torch.long), torch.tensor(padded_attention_mask,
                                                                                  dtype=torch.long)

        # 加载数据加载器
        dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)
        print(f"提取CLS特征...")
        # 推理模式，关闭梯度计算，提速+省显存
        with torch.no_grad():
            for input_ids, attention_mask in tqdm(dataloader, total=len(dataloader)):
                input_ids, attention_mask = input_ids.to(self.device), attention_mask.to(self.device)
                cls_repr = self.model(input_ids=input_ids, attention_mask=attention_mask)
                features.extend(cls_repr.cpu().numpy())
        return np.array(features)

    def predict(self, sequences, batch_size=16):
        """
        调用ESM模型做预测（备用方法），返回预测标签、概率、推理时间
        :param sequences: 肽序列列表
        :param batch_size: 批次大小
        :return: predictions, probabilities, inference_time
        """
        if not self.model: raise ValueError("请先加载特征提取模型")
        self.model.eval()
        predictions = []
        probabilities = []
        dataset = PeptideDataset(sequences, [0] * len(sequences))

        # 自定义批次整理函数，同特征提取
        def collate_fn(batch):
            sequences, _, _ = zip(*batch)
            encoded = self.tokenizer(
                sequences, return_tensors=None, padding="do_not_pad", truncation=True, max_length=256
            )
            input_ids_list, attention_mask_list = encoded["input_ids"], encoded["attention_mask"]
            batch_max_len = max(len(ids) for ids in input_ids_list)
            pad_token_id = self.tokenizer.pad_token_id
            padded_input_ids = [ids + [pad_token_id] * (batch_max_len - len(ids)) for ids in input_ids_list]
            padded_attention_mask = [mask + [0] * (batch_max_len - len(mask)) for mask in attention_mask_list]
            return torch.tensor(padded_input_ids, dtype=torch.long), torch.tensor(padded_attention_mask,
                                                                                  dtype=torch.long)

        # 实例化临时预测器并加载权重
        temp_predictor = PeptidePredictor(self.esm2_model_name_or_path, self.device)
        temp_predictor.load_pretrained_model(freeze_layers=True, finetuned_model_path=finetuned_esm_path)
        predictions, probabilities, inference_time = temp_predictor.predict(sequences, batch_size)
        return predictions, probabilities, inference_time


# ===================== ESM模型预测器类 =====================
class PeptidePredictor:
    """
    基于ESM2的肽序列预测器，用于ESM模型的推理预测
    包含完整的ESM2骨干+分类头，用于对比GNN模型性能
    """

    def __init__(self, esm2_model_name_or_path, device=device):
        """
        初始化预测器
        :param esm2_model_name_or_path: ESM2模型路径/名称
        :param device: 运行设备 cuda/cpu
        """
        self.device = device
        self.esm2_model_name_or_path = esm2_model_name_or_path
        self.model = None
        self.tokenizer = None
        self.hidden_size = None

    def load_pretrained_model(self, freeze_layers=True, finetuned_model_path=None):
        """
        加载带分类头的ESM2预测模型
        :param freeze_layers: 是否冻结骨干层
        :param finetuned_model_path: 微调权重路径
        """
        print(f"\n加载临时预测模型: {finetuned_model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.esm2_model_name_or_path)
        esm2_backbone = AutoModel.from_pretrained(self.esm2_model_name_or_path)
        self.hidden_size = esm2_backbone.config.hidden_size

        # 定义带分类头的ESM2分类器
        class ESM2Classifier(nn.Module):
            def __init__(self, esm2_backbone):
                super().__init__()
                self.esm2_backbone = esm2_backbone
                self.hidden_dim = esm2_backbone.config.hidden_size
                # 分类头：三层全连接+dropout，输出二分类结果
                self.classifier = nn.Sequential(
                    nn.Linear(self.hidden_dim, 512),
                    nn.ReLU(),
                    nn.Dropout(0.6),
                    nn.Linear(512, 256),
                    nn.ReLU(),
                    nn.Dropout(0.6),
                    nn.Linear(256, 2)
                )

            def forward(self, input_ids, attention_mask):
                outputs = self.esm2_backbone(input_ids=input_ids, attention_mask=attention_mask)
                cls_repr = outputs.last_hidden_state[:, 0, :]
                seq_lengths = attention_mask.sum(dim=1, keepdim=True).float()
                cls_repr = cls_repr / torch.sqrt(seq_lengths + 1e-8)
                return self.classifier(cls_repr)

        # 选择性冻结层：仅训练后几层，提速训练
        if freeze_layers:
            for i, layer in enumerate(esm2_backbone.encoder.layer):
                for param in layer.parameters():
                    param.requires_grad = (i >= 4)
            for param in esm2_backbone.embeddings.parameters():
                param.requires_grad = False

        # 加载模型和权重
        self.model = ESM2Classifier(esm2_backbone).to(self.device)
        checkpoint = torch.load(finetuned_model_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        # 推理阶段冻结所有参数
        for param in self.model.parameters():
            param.requires_grad = False

    def predict(self, sequences, batch_size=16):
        """
        批量预测肽序列标签和概率
        :param sequences: 肽序列列表
        :param batch_size: 批次大小
        :return: 预测标签列表、阳性概率列表、推理耗时
        """
        self.model.eval()
        predictions = []
        probabilities = []
        dataset = PeptideDataset(sequences, [0] * len(sequences))

        # 自定义批次整理函数
        def collate_fn(batch):
            sequences, _, _ = zip(*batch)
            encoded = self.tokenizer(
                sequences, return_tensors=None, padding="do_not_pad", truncation=True, max_length=256
            )
            input_ids_list, attention_mask_list = encoded["input_ids"], encoded["attention_mask"]
            batch_max_len = max(len(ids) for ids in input_ids_list)
            pad_token_id = self.tokenizer.pad_token_id
            padded_input_ids = [ids + [pad_token_id] * (batch_max_len - len(ids)) for ids in input_ids_list]
            padded_attention_mask = [mask + [0] * (batch_max_len - len(mask)) for mask in attention_mask_list]
            return torch.tensor(padded_input_ids, dtype=torch.long), torch.tensor(padded_attention_mask,
                                                                                  dtype=torch.long)

        dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)
        start_time = time.time()
        with torch.no_grad():
            for input_ids, attention_mask in tqdm(dataloader, total=len(dataloader)):
                input_ids, attention_mask = input_ids.to(self.device), attention_mask.to(self.device)
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                probs = torch.softmax(outputs, dim=1).cpu().numpy()  # 转概率分布
                predictions.extend(np.argmax(probs, axis=1))  # 取最大概率为预测标签
                probabilities.extend(probs[:, 1])  # 提取阳性类(1)的概率
        inference_time = time.time() - start_time
        return predictions, probabilities, inference_time


# ===================== 数据加载工具函数 =====================
def load_csv_data(csv_path):
    """
    加载csv格式的肽序列数据集，做数据清洗和校验
    :param csv_path: csv文件路径，要求必须包含id/seq/label三列
    :return: 清洗后的id列表、序列列表、标签列表
    """
    # 校验文件是否存在
    if not os.path.exists(csv_path): raise FileNotFoundError(f"CSV文件不存在: {csv_path}")
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        raise IOError(f"读取CSV失败: {str(e)}")

    # 校验必要列是否存在
    required_cols = ['id', 'seq', 'label']
    if not all(col in df.columns for col in required_cols):
        missing = [col for col in required_cols if col not in df.columns]
        raise ValueError(f"CSV缺少必要列: {missing}")

    ids, sequences, labels = [], [], []
    # 逐行读取并清洗数据
    for _, row in df.iterrows():
        id_val, seq = row['id'], str(row['seq']).strip().upper()  # 序列去空格+转大写
        # 过滤空序列
        if len(seq) == 0:
            print(f"过滤空序列（id: {id_val}）")
            continue
        # 过滤无效标签，仅保留0/1
        try:
            label = int(row['label'])
            if label not in (0, 1): raise ValueError(f"标签必须为0或1")
        except Exception as e:
            print(f"过滤无效标签（{e}）: {row['label']}（id: {id_val}）")
            continue
        ids.append(id_val)
        sequences.append(seq)
        labels.append(label)

    # 输出数据统计信息
    total, positive = len(labels), sum(labels)
    print(
        f"加载完成 {os.path.basename(csv_path)}：共{total}条有效序列（阳性{positive}条，阴性{total - positive}条，阳性占比{positive / total * 100:.2f}%）")
    return ids, sequences, labels


# ===================== 核心GNN模型类 =====================
class EnhancedGNNClassifier(nn.Module):
    """
    轻量级GNN模型（无蒸馏版），基于GCN卷积层构建
    输入：ESM2提取的CLS特征 (batch, 320)
    输出：二分类预测结果 (batch, 2)
    """

    def __init__(self, input_dim=320, hidden_dim=128, num_classes=2, num_layers=2):
        """
        初始化GNN模型
        :param input_dim: 输入特征维度，ESM2-8M为320
        :param hidden_dim: GCN隐藏层维度
        :param num_classes: 分类类别数，二分类为2
        :param num_layers: GCN卷积层数
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # 输入投影层：将320维特征映射到隐藏层维度
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # 构建GCN卷积层列表
        self.gcn_layers = nn.ModuleList()
        self.gcn_layers.append(GCNConv(hidden_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.gcn_layers.append(GCNConv(hidden_dim, hidden_dim))

        # 层归一化+dropout：防止过拟合，提升泛化能力
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(0.2)

        # 分类头：隐藏层特征映射到分类结果
        self.classifier = nn.Linear(hidden_dim, num_classes)

        # 权重初始化：提升模型收敛速度和稳定性
        self._init_weights()

    def _init_weights(self):
        """模型权重初始化函数，不同层使用不同初始化策略"""
        for gcn in self.gcn_layers:
            nn.init.xavier_uniform_(gcn.lin.weight)
            if gcn.bias is not None:
                nn.init.zeros_(gcn.bias)
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.zeros_(self.input_proj.bias)
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)
        nn.init.ones_(self.layer_norm.weight)
        nn.init.zeros_(self.layer_norm.bias)

    def forward(self, x):
        """
        模型前向传播逻辑
        :param x: 输入特征 (batch, input_dim)
        :return: 分类logits (batch, num_classes)
        """
        batch_size = x.shape[0]

        # 输入特征投影到隐藏层维度
        x_proj = self.input_proj(x)

        # 构建图数据：单样本为单节点，无连边，空边索引
        node_feat = x_proj
        edge_index = torch.empty((2, 0), dtype=torch.long, device=x.device)
        batch_idx = torch.arange(batch_size, device=x.device)

        # GCN层前向传播
        gcn_out = node_feat
        for gcn in self.gcn_layers:
            gcn_out = gcn(gcn_out, edge_index)
            gcn_out = F.relu(gcn_out)  # ReLU激活函数

        # 全局均值池化：聚合图特征
        global_feat = global_mean_pool(gcn_out, batch_idx)

        # 归一化+dropout+分类
        out = self.layer_norm(global_feat)
        out = self.dropout(out)
        logits = self.classifier(out)
        return logits


# ===================== 模型训练器类 =====================
class StandardGNNTrainer:
    """
    GNN模型的标准训练器类
    包含：模型初始化、数据准备、训练、评估、指标计算等全流程功能
    """

    def __init__(self, feature_extractor, feature_extractor_total_params, device=device):
        """
        初始化训练器
        :param feature_extractor: 特征提取器实例
        :param feature_extractor_total_params: 特征提取器参数量
        :param device: 运行设备
        """
        self.device = device
        self.feature_extractor = feature_extractor
        self.feature_extractor_total_params = feature_extractor_total_params
        self.gnn_model = None  # GNN模型实例
        self.optimizer = None  # 优化器
        # 训练/验证指标存储字典
        self.train_metrics = {"total_loss": [], "accuracy": [], "f1": [], "recall": [], "mcc": []}
        self.val_metrics = {"accuracy": [], "f1": [], "recall": [], "mcc": [], "auc": []}
        # 模型参数量和体积统计
        self.esm_params_k = self.feature_extractor_total_params / 1000
        self.esm_size_mb = (self.feature_extractor_total_params * 4) / (1024 * 1024)
        self.gnn_params_k = 0
        self.gnn_size_mb = 0

    def init_gnn_model(self, input_dim=320, hidden_dim=128):
        """
        初始化GNN模型
        :param input_dim: 输入特征维度
        :param hidden_dim: GCN隐藏层维度
        :return: 初始化后的GNN模型
        """
        self.gnn_model = EnhancedGNNClassifier(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=2
        ).to(self.device)
        # 计算模型总参数量
        total_params = sum(p.numel() for p in self.gnn_model.parameters())
        self.gnn_params_k = total_params / 1000
        print(f"\nGNN模型初始化完成:")
        print(f"   - 输入维度: {input_dim} | GCN隐藏层维度: {hidden_dim}")
        print(f"   - GCN层数: 2")
        print(f"   - GNN模型总参数量: {total_params:,} ({self.gnn_params_k:.2f}k)")
        return self.gnn_model

    def prepare_training_data(self, sequences, labels, batch_size=32, save_dir="results_ultra_light_gnn"):
        """
        准备训练数据，提取特征并缓存到本地，避免重复提取
        :param sequences: 训练序列列表
        :param labels: 训练标签列表
        :param batch_size: 特征提取批次大小
        :param save_dir: 缓存文件保存目录
        :return: 训练特征、标签、序列字典
        """
        os.makedirs(save_dir, exist_ok=True)
        cache_path = os.path.join(save_dir, "training_data_no_distill.npz")

        # 优先加载缓存数据，提速
        if os.path.exists(cache_path):
            print(f"加载缓存的训练数据: {cache_path}")
            data = np.load(cache_path)
            return {
                'features': data['features'],
                'labels': data['labels'],
                'sequences': sequences
            }

        # 无缓存则提取特征并保存
        features = self.feature_extractor.get_cls_features(sequences, batch_size=batch_size)
        labels_np = np.array(labels)

        np.savez_compressed(
            cache_path,
            features=features,
            labels=labels_np
        )
        print(f"训练特征已保存，维度: {features.shape}")
        return {
            'features': features,
            'labels': labels_np,
            'sequences': sequences
        }

    def prepare_validation_data(self, val_sequences, val_labels, batch_size=32, save_dir="results_ultra_light_gnn"):
        """
        准备验证数据，逻辑同训练数据，提取特征并缓存
        :param val_sequences: 验证序列列表
        :param val_labels: 验证标签列表
        :param batch_size: 特征提取批次大小
        :param save_dir: 缓存文件保存目录
        :return: 验证特征、标签字典
        """
        os.makedirs(save_dir, exist_ok=True)
        cache_path = os.path.join(save_dir, f"validation_data_no_distill.npz")

        if os.path.exists(cache_path):
            print(f"加载缓存的验证数据: {cache_path}")
            data = np.load(cache_path)
            return {'val_features': data['val_features'], 'val_labels': data['val_labels']}

        val_features = self.feature_extractor.get_cls_features(val_sequences, batch_size=batch_size)
        val_labels_np = np.array(val_labels)

        np.savez_compressed(cache_path, val_features=val_features, val_labels=val_labels_np)
        print(f"验证集特征已保存，维度: {val_features.shape}")
        return {'val_features': val_features, 'val_labels': val_labels_np}

    def _compute_metrics(self, y_true, y_pred, y_prob):
        """
        私有方法：计算二分类任务的全量评估指标
        :param y_true: 真实标签
        :param y_pred: 预测标签
        :param y_prob: 预测阳性概率
        :return: 指标字典
        """
        # 计算ROC-AUC，处理单类别异常
        try:
            roc_auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.0
        except:
            roc_auc = 0.0

        # 计算PR-AUC，处理单类别异常
        try:
            precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_prob)
            pr_auc = auc(recall_curve, precision_curve)
        except:
            pr_auc = 0.0

        # 返回所有指标
        return {
            'Accuracy': accuracy_score(y_true, y_pred),
            'F1 Score': f1_score(y_true, y_pred, zero_division=0),
            'Precision': precision_score(y_true, y_pred, zero_division=0),
            'Recall': recall_score(y_true, y_pred, zero_division=0),
            'MCC': matthews_corrcoef(y_true, y_pred),
            'ROC-AUC': roc_auc,
            'PR-AUC': pr_auc
        }

    def train_gnn(self, training_data, val0_data, save_dir="results_ultra_light_gnn",
                  batch_size=512, epochs=50, lr=1e-3, early_stop_patience=10):
        """
        核心训练函数：训练GNN模型，带早停机制防止过拟合
        :param training_data: 训练数据字典
        :param val0_data: 验证数据字典
        :param save_dir: 模型保存目录
        :param batch_size: 训练批次大小
        :param epochs: 最大训练轮次
        :param lr: 学习率
        :param early_stop_patience: 早停耐心值
        :return: 训练后的模型、总训练耗时
        """
        os.makedirs(save_dir, exist_ok=True)

        # 数据转tensor并部署到指定设备
        train_features = torch.FloatTensor(training_data['features']).to(self.device)
        train_labels = torch.LongTensor(training_data['labels']).to(self.device)
        val0_features = torch.FloatTensor(val0_data['val_features']).to(self.device)
        val0_labels = torch.LongTensor(val0_data['val_labels']).to(self.device)

        # 构建数据加载器
        train_dataset = torch.utils.data.TensorDataset(train_features, train_labels)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val0_dataset = torch.utils.data.TensorDataset(val0_features, val0_labels)
        val0_loader = DataLoader(val0_dataset, batch_size=batch_size)

        # 定义优化器和损失函数
        self.optimizer = optim.AdamW(self.gnn_model.parameters(), lr=lr, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()
        best_val_f1, early_stop_count = 0.0, 0  # 早停相关初始化

        print(f"\n开始训练GNN模型（无蒸馏）...")
        print(f"超参数配置: 批大小={batch_size}, 轮次={epochs}, 学习率={lr}")
        start_time = time.time()

        # 训练主循环
        for epoch in range(epochs):
            self.gnn_model.train()  # 模型切换到训练模式
            total_loss, correct, total = 0, 0, 0
            train_preds, train_true = [], []

            # 批次训练
            for batch_idx, (features, labels) in enumerate(
                    tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs} - Training")):
                self.optimizer.zero_grad()  # 梯度清零
                logits = self.gnn_model(features)  # 前向传播
                loss = criterion(logits, labels)  # 计算损失
                loss.backward()  # 反向传播求梯度
                torch.nn.utils.clip_grad_norm_(self.gnn_model.parameters(), max_norm=1.0)  # 梯度裁剪防爆炸
                self.optimizer.step()  # 更新权重

                # 统计训练指标
                total_loss += loss.item()
                _, predicted = logits.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                train_preds.extend(predicted.cpu().numpy())
                train_true.extend(labels.cpu().numpy())

            # 计算训练集指标
            train_acc = 100. * correct / total
            avg_loss = total_loss / len(train_loader)
            train_metrics = self._compute_metrics(np.array(train_true), np.array(train_preds), None)
            self.train_metrics["total_loss"].append(avg_loss)
            self.train_metrics["accuracy"].append(train_acc)
            self.train_metrics["f1"].append(train_metrics['F1 Score'])
            self.train_metrics["recall"].append(train_metrics['Recall'])
            self.train_metrics["mcc"].append(train_metrics['MCC'])

            # 验证集评估
            self.gnn_model.eval()  # 模型切换到推理模式
            val0_preds, val0_true, val0_probs = [], [], []
            with torch.no_grad():
                for features, labels in val0_loader:
                    logits = self.gnn_model(features)
                    _, predicted = logits.max(1)
                    val0_preds.extend(predicted.cpu().numpy())
                    val0_true.extend(labels.cpu().numpy())
                    val0_probs.extend(F.softmax(logits, dim=1)[:, 1].cpu().numpy())

            # 计算验证集指标
            val0_metrics = self._compute_metrics(np.array(val0_true), np.array(val0_preds), np.array(val0_probs))
            self.val_metrics["accuracy"].append(val0_metrics['Accuracy'] * 100)
            self.val_metrics["f1"].append(val0_metrics['F1 Score'])
            self.val_metrics["recall"].append(val0_metrics['Recall'])
            self.val_metrics["mcc"].append(val0_metrics['MCC'])
            self.val_metrics["auc"].append(val0_metrics['ROC-AUC'])

            # 打印本轮训练/验证指标
            print(f'Epoch {epoch + 1}: 训练准确率={train_acc:.2f}% | 训练F1={train_metrics["F1 Score"]:.4f} | '
                  f'验证集 F1={val0_metrics["F1 Score"]:.4f} | 验证集 AUC={val0_metrics["ROC-AUC"]:.4f}')

            # 早停机制：验证集F1提升则保存最佳模型，否则计数+1
            if val0_metrics['F1 Score'] > best_val_f1 + 1e-4:
                best_val_f1 = val0_metrics['F1 Score']
                early_stop_count = 0
                torch.save(
                    {'model_state_dict': self.gnn_model.state_dict(), 'epoch': epoch + 1, 'val_f1': best_val_f1},
                    f"{save_dir}/best_model_no_distill_gnn.pt")
            else:
                early_stop_count += 1
                if early_stop_count >= early_stop_patience:
                    print(f"触发早停！最佳验证集 F1: {best_val_f1:.4f}")
                    break

        # 训练结束，统计耗时并保存最终模型
        total_training_time = time.time() - start_time
        final_model_path = f"{save_dir}/final_gnn_model_no_distill_gnn.pt"
        torch.save({
            'model_state_dict': self.gnn_model.state_dict(),
            'train_metrics': self.train_metrics,
            'val_metrics': self.val_metrics
        }, final_model_path)

        # 计算模型文件体积
        if os.path.exists(final_model_path):
            self.gnn_size_mb = os.path.getsize(final_model_path) / (1024 * 1024)

        print(f"\nGNN模型训练完成！总耗时: {total_training_time:.2f}秒")
        print(f"最佳GNN模型保存路径: {save_dir}/best_model_no_distill_gnn.pt")
        return self.gnn_model, total_training_time

    def evaluate_model_performance(self, sequences, labels, ids, batch_size=64, dataset_name="validation"):
        """
        评估模型性能：同时评估ESM模型和GNN模型，计算所有指标和推理速度
        :param sequences: 评估序列列表
        :param labels: 评估标签列表
        :param ids: 评估id列表
        :param batch_size: 推理批次大小
        :param dataset_name: 数据集名称（用于标注）
        :return: 性能指标字典
        """
        if self.gnn_model is None: raise ValueError("GNN模型未初始化")
        self.gnn_model.eval()

        # ESM模型预测
        print(f"\n=== 评估ESM特征提取模型性能（{dataset_name}集）===")
        esm_preds, esm_probs, esm_inference_time = self.feature_extractor.predict(sequences,
                                                                                  batch_size=batch_size)
        true_labels = np.array(labels)

        # 提取CLS特征
        print(f"=== 提取{dataset_name}集CLS特征 ===")
        features = self.feature_extractor.get_cls_features(sequences, batch_size=batch_size)
        features = torch.FloatTensor(features).to(self.device)

        # GNN模型预测
        print(f"=== 评估GNN模型性能（{dataset_name}集）===")
        gnn_preds, gnn_probs = [], []
        gnn_start_time = time.time()
        with torch.no_grad():
            for i in tqdm(range(0, len(features), batch_size), desc=f"Evaluating GNN ({dataset_name})"):
                batch_features = features[i:i + batch_size]
                batch_logits = self.gnn_model(batch_features)
                _, batch_preds = batch_logits.max(1)
                gnn_preds.extend(batch_preds.cpu().numpy())
                gnn_probs.extend(F.softmax(batch_logits, dim=1)[:, 1].cpu().numpy())
        gnn_inference_time = time.time() - gnn_start_time

        # 计算指标
        esm_metrics = self._compute_metrics(true_labels, np.array(esm_preds), np.array(esm_probs))
        gnn_metrics = self._compute_metrics(true_labels, np.array(gnn_preds), np.array(gnn_probs))

        # 补充模型参数量、体积、推理时间
        esm_metrics['Inference Time (s)'] = esm_inference_time
        esm_metrics['Param Count (k)'] = self.esm_params_k
        esm_metrics['Model Size (MB)'] = self.esm_size_mb

        gnn_metrics['Inference Time (s)'] = gnn_inference_time
        gnn_metrics['Param Count (k)'] = self.gnn_params_k
        gnn_metrics['Model Size (MB)'] = self.gnn_size_mb

        # 返回完整性能字典
        return {
            'dataset_name': dataset_name,
            'ids': ids,
            'sequences': sequences,
            'true_labels': true_labels,
            'esm_preds': np.array(esm_preds),
            'esm_probs': np.array(esm_probs),
            'gnn_preds': np.array(gnn_preds),
            'gnn_probs': np.array(gnn_probs),
            'esm_metrics': esm_metrics,
            'gnn_metrics': gnn_metrics,
            'speedup_ratio': esm_inference_time / gnn_inference_time  # GNN模型相对ESM模型的提速比
        }

    def get_model_comparison(self, gnn_model_path):
        """
        计算ESM/GNN模型的参数量和体积对比
        :param gnn_model_path: GNN模型权重路径
        :return: 对比指标字典
        """
        return {
            'esm_size_mb': self.esm_size_mb, 'gnn_size_mb': self.gnn_size_mb,
            'size_reduction': (self.esm_size_mb - self.gnn_size_mb) / self.esm_size_mb * 100,
            'esm_params': self.feature_extractor_total_params, 'gnn_params': self.gnn_params_k * 1000,
            'params_reduction': (
                                        self.feature_extractor_total_params - self.gnn_params_k * 1000) / self.feature_extractor_total_params * 100
        }


# ===================== 结果保存工具函数 =====================
def save_prediction_results(performance_dict, output_file):
    """
    将预测结果保存为csv文件，包含id、序列、真实标签、ESM/GNN模型预测结果和概率
    :param performance_dict: 性能指标字典
    :param output_file: 输出文件路径
    """
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    results = pd.DataFrame({
        "id": performance_dict['ids'],
        "sequence": performance_dict['sequences'],
        "sequence_length": [len(seq) for seq in performance_dict['sequences']],
        "true_label": performance_dict['true_labels'],
        "esm_predicted_label": performance_dict['esm_preds'],
        "esm_predicted_label_name": ["阳性（1）" if p == 1 else "阴性（0）" for p in performance_dict['esm_preds']],
        "esm_positive_probability": performance_dict['esm_probs'].round(4),
        "gnn_predicted_label": performance_dict['gnn_preds'],
        "gnn_predicted_label_name": ["阳性（1）" if p == 1 else "阴性（0）" for p in performance_dict['gnn_preds']],
        "gnn_positive_probability": performance_dict['gnn_probs'].round(4)
    })
    results.to_csv(output_file, index=False, encoding="utf-8-sig")
    print(f"\n{performance_dict['dataset_name']}集预测结果已保存至: {output_file}")


def save_metrics_summary(performance_list, output_file):
    """
    将所有数据集的ESM/GNN模型指标汇总保存为csv文件
    :param performance_list: 性能指标列表
    :param output_file: 输出文件路径
    """
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    summary_data = []

    # 整理指标数据
    for perf in performance_list:
        dataset_name = perf['dataset_name']
        esm_row = {
            'Dataset': dataset_name,
            'Model Type': 'Feature Extractor (ESM)',
            **perf['esm_metrics']
        }
        gnn_row = {
            'Dataset': dataset_name,
            'Model Type': 'GNN (No Distill)',
            **perf['gnn_metrics']
        }
        summary_data.append(esm_row)
        summary_data.append(gnn_row)

    summary_df = pd.DataFrame(summary_data)

    # 指标列排序，保证格式统一
    column_order = [
        'Dataset', 'Model Type', 'Accuracy', 'F1 Score', 'Precision',
        'Recall', 'MCC', 'ROC-AUC', 'PR-AUC', 'Inference Time (s)',
        'Param Count (k)', 'Model Size (MB)'
    ]
    for col in column_order:
        if col not in summary_df.columns:
            summary_df[col] = 0.0
    summary_df = summary_df[column_order]

    # 保留四位小数并保存
    summary_df = summary_df.round(4)
    summary_df.to_csv(output_file, index=False, encoding="utf-8-sig")
    print(f"\n全量指标汇总已保存至: {output_file}")


def print_detailed_metrics(performance_list):
    """
    在控制台打印详细的模型指标，方便实时查看
    :param performance_list: 性能指标列表
    """
    print("\n" + "=" * 120)
    print("【详细指标汇总】训练集/验证集/测试集 - ESM特征提取器&GNN模型（无蒸馏）")
    print("=" * 120)

    # 需要打印的指标列表
    metrics_to_print = [
        'Accuracy', 'F1 Score', 'Precision', 'Recall', 'MCC',
        'ROC-AUC', 'PR-AUC', 'Inference Time (s)', 'Param Count (k)', 'Model Size (MB)'
    ]

    # 逐数据集打印
    for perf in performance_list:
        dataset_name = perf['dataset_name']
        print(f"\n📊 数据集: {dataset_name}")
        print("-" * 100)

        print("🔬 ESM特征提取模型:")
        for metric in metrics_to_print:
            value = perf['esm_metrics'].get(metric, 0.0)
            if metric in ['Inference Time (s)']:
                print(f"   - {metric:<20}: {value:.4f}")
            elif metric in ['Param Count (k)', 'Model Size (MB)']:
                print(f"   - {metric:<20}: {value:.2f}")
            else:
                print(f"   - {metric:<20}: {value:.4f}")

        print("🧠 GNN模型 (无蒸馏):")
        for metric in metrics_to_print:
            value = perf['gnn_metrics'].get(metric, 0.0)
            if metric in ['Inference Time (s)']:
                print(f"   - {metric:<20}: {value:.4f}")
            elif metric in ['Param Count (k)', 'Model Size (MB)']:
                print(f"   - {metric:<20}: {value:.2f}")
            else:
                print(f"   - {metric:<20}: {value:.4f}")

        print(f"   - {'Speedup Ratio (GNN/ESM)':<20}: {perf['speedup_ratio']:.2f}x")
        print("-" * 100)


# ===================== 主训练流程函数 =====================
def train_gnn_no_distillation():
    """
    主函数：无知识蒸馏的GNN模型训练全流程
    步骤：加载数据 -> 加载特征提取器 -> 初始化模型 -> 准备数据 -> 训练 -> 评估 -> 保存结果
    """
    print("=== 开始训练GNN模型（无知识蒸馏）===")
    print("=== 适配指定数据集 | val_test | t0-t3为测试集 ===")
    print(f"使用设备: {device}")

    try:
        # 1. 加载所有数据集
        print("\n===================== 加载数据集 =====================")
        train_ids, train_sequences, train_labels = load_csv_data(train_csv)
        val0_ids, val0_sequences, val0_labels = load_csv_data(val0_csv)
        t0_ids, t0_sequences, t0_labels = load_csv_data(t0_csv)
        t1_ids, t1_sequences, t1_labels = load_csv_data(t1_csv)
        t2_ids, t2_sequences, t2_labels = load_csv_data(t2_csv)
        t3_ids, t3_sequences, t3_labels = load_csv_data(t3_csv)

        # 打印数据集统计
        print(f"\n数据集汇总: ")
        print(f"   - 训练集: {len(train_sequences)}条")
        print(f"   - 验证集(val_test): {len(val0_sequences)}条")
        print(f"   - 测试集t0(rawdata_val): {len(t0_sequences)}条")
        print(f"   - 测试集t1(bagel4_1_val): {len(t1_sequences)}条")
        print(f"   - 测试集t2(bagel4_2_val): {len(t2_sequences)}条")
        print(f"   - 测试集t3(bagel4_3_val): {len(t3_sequences)}条")

        # 2. 加载ESM2特征提取器
        print("\n===================== 加载特征提取模型 =====================")
        feature_extractor = PeptideFeatureExtractor(esm2_model_name_or_path=esm2_backbone_path, device=device)
        feature_extractor.load_pretrained_model(finetuned_model_path=finetuned_esm_path)

        # 3. 初始化GNN模型
        print("\n===================== 初始化GNN模型 =====================")
        trainer = StandardGNNTrainer(
            feature_extractor=feature_extractor,
            feature_extractor_total_params=feature_extractor.total_params,
            device=device
        )
        trainer.init_gnn_model(input_dim=feature_extractor.hidden_size, hidden_dim=128)

        # 4. 准备训练和验证数据
        print("\n===================== 准备训练数据 =====================")
        training_data = trainer.prepare_training_data(
            sequences=train_sequences, labels=train_labels,
            batch_size=32, save_dir="results_ultra_light_gnn"
        )
        val0_data = trainer.prepare_validation_data(
            val_sequences=val0_sequences, val_labels=val0_labels,
            batch_size=32, save_dir="results_ultra_light_gnn"
        )

        # 5. 训练GNN模型
        print("\n===================== 训练GNN模型 =====================")
        gnn_model, training_time = trainer.train_gnn(
            training_data=training_data, val0_data=val0_data,
            save_dir="results_ultra_light_gnn",
            batch_size=512, epochs=50, lr=1e-3, early_stop_patience=10
        )

        # 6. 评估所有数据集性能
        print("\n===================== 评估所有数据集 =====================")
        train_perf = trainer.evaluate_model_performance(
            sequences=train_sequences, labels=train_labels, ids=train_ids,
            batch_size=64, dataset_name="train_no_distill_gnn"
        )

        val0_perf = trainer.evaluate_model_performance(
            sequences=val0_sequences, labels=val0_labels, ids=val0_ids,
            batch_size=64, dataset_name="val0_val_test_early_stop_gnn"
        )

        t0_perf = trainer.evaluate_model_performance(
            sequences=t0_sequences, labels=t0_labels, ids=t0_ids,
            batch_size=64, dataset_name="t0_rawdata_val_gnn"
        )

        t1_perf = trainer.evaluate_model_performance(
            sequences=t1_sequences, labels=t1_labels, ids=t1_ids,
            batch_size=64, dataset_name="t1_bagel4_1_val_gnn"
        )

        t2_perf = trainer.evaluate_model_performance(
            sequences=t2_sequences, labels=t2_labels, ids=t2_ids,
            batch_size=64, dataset_name="t2_bagel4_2_val_gnn"
        )

        t3_perf = trainer.evaluate_model_performance(
            sequences=t3_sequences, labels=t3_labels, ids=t3_ids,
            batch_size=64, dataset_name="t3_bagel4_3_val_gnn"
        )

        performance_list = [train_perf, val0_perf, t0_perf, t1_perf, t2_perf, t3_perf]

        # 打印核心指标
        print(f"\n各数据集（GNN模型）核心指标:")
        for perf in performance_list:
            ds_name = perf['dataset_name']
            g_metrics = perf['gnn_metrics']
            print(
                f"   - {ds_name}: F1={g_metrics['F1 Score']:.4f} | ROC-AUC={g_metrics['ROC-AUC']:.4f} | PR-AUC={g_metrics['PR-AUC']:.4f} | 推理提速={perf['speedup_ratio']:.1f}x")

        # 7. 保存预测结果和指标汇总
        print("\n===================== 保存预测结果和指标汇总 =====================")
        save_prediction_results(val0_perf, val0_output)
        save_prediction_results(t0_perf, t0_output)
        save_prediction_results(t1_perf, t1_output)
        save_prediction_results(t2_perf, t2_output)
        save_prediction_results(t3_perf, t3_output)
        save_metrics_summary(performance_list, metrics_summary_output)

        # 打印详细指标
        print_detailed_metrics(performance_list)

        # 计算模型对比指标
        model_comparison = trainer.get_model_comparison("results_ultra_light_gnn/best_model_no_distill_gnn.pt")

        # 打印完成信息
        print("\n" + "=" * 100)
        print("=== GNN模型（无蒸馏）训练全流程成功完成！===")
        print(f"GNN模型保存目录: results_ultra_light_gnn")
        print(f"验证集结果: {val0_output}")
        print(f"测试集t0结果: {t0_output}")
        print(f"测试集t1结果: {t1_output}")
        print(f"测试集t2结果: {t2_output}")
        print(f"测试集t3结果: {t3_output}")
        print(f"全量指标汇总: {metrics_summary_output}")
        print("=" * 100)

        return gnn_model, trainer, performance_list, model_comparison

    except Exception as e:
        # 异常捕获并打印堆栈信息
        print(f"\n程序执行失败: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None, None


if __name__ == "__main__":
    gnn_model, trainer, performance_results, model_comparison = train_gnn_no_distillation()

    if gnn_model is None:
        print("\n训练流程执行失败，请检查错误信息！")
        exit(1)
    else:
        print("\n所有流程执行完毕，结果已保存！")
        exit(0)