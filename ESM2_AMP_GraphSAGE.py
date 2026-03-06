# ===================================== 导入核心依赖库 =====================================
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
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModel
import torch.serialization
from sklearn.preprocessing import LabelEncoder
import time
# PyTorch Geometric图神经网络库，核心GraphSAGE卷积层和全局池化层
from torch_geometric.nn import SAGEConv, global_mean_pool
from torch_geometric.data import Batch, Data

torch.serialization.add_safe_globals([LabelEncoder])


# ===================================== 自定义GraphSAGE图编码器 =====================================
class SAGEEncoder(nn.Module):
    """
    图神经网络GraphSAGE编码器核心类
    堆叠多层SAGEConv卷积层，实现节点特征的聚合与编码
    """
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, dropout=0.0):
        super().__init__()
        self.num_layers = num_layers  # 卷积层数
        self.dropout = dropout        # Dropout失活概率

        # 构建多层SAGEConv卷积层列表
        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))  # 输入层
        for _ in range(num_layers - 2):                            # 隐藏层
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels)) # 输出层

    def forward(self, x, edge_index):
        """
        前向传播函数
        :param x: 节点特征矩阵 [num_nodes, in_channels]
        :param edge_index: 图的边索引矩阵 [2, num_edges]
        :return: 编码后的节点特征
        """
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            # 最后一层不做激活和Dropout
            if i != self.num_layers - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x

# ===================================== 全局路径配置 =====================================
ROOT_DIR = "./"
# 训练集/验证集/测试集数据路径
train_csv = os.path.join(ROOT_DIR, "mydata/newdata/rawdata_train.csv")
val0_csv = os.path.join(ROOT_DIR, "mydata/newdata/rawdata_val.csv")
t0_csv = os.path.join(ROOT_DIR, "mydata/newdata/rawdata_test.csv")
t1_csv = os.path.join(ROOT_DIR, "test_data/bagel4_1_val.csv")
t2_csv = os.path.join(ROOT_DIR, "test_data/bagel4_2_val.csv")
t3_csv = os.path.join(ROOT_DIR, "test_data/bagel4_3_val.csv")

# 预训练模型路径
esm2_backbone_path = os.path.join(ROOT_DIR, "model/esm2_t6_8M_UR50D")
finetuned_esm_candidates = [
    os.path.join(ROOT_DIR, "model/model_AMP_ESM2_8M_256_v3/best_model_amp/esm2_8m_binary_best.pt"),
    os.path.join(ROOT_DIR, "model_AMP_ESM2_8M_256_v3/best_model_amp/esm2_8m_binary_best.pt"),
]

# 预测结果输出路径
val0_output = os.path.join(ROOT_DIR, "results_ultra_light_graphsage/val0_val_test_early_stop_predictions.csv")
t0_output = os.path.join(ROOT_DIR, "results_ultra_light_graphsage/t0_rawdata_test_predictions.csv")
t1_output = os.path.join(ROOT_DIR, "results_ultra_light_graphsage/t1_bagel4_1_val_predictions.csv")
t2_output = os.path.join(ROOT_DIR, "results_ultra_light_graphsage/t2_bagel4_2_val_predictions.csv")
t3_output = os.path.join(ROOT_DIR, "results_ultra_light_graphsage/t3_bagel4_3_val_predictions.csv")
metrics_summary_output = os.path.join(ROOT_DIR, "results_ultra_light_graphsage/metrics_summary.csv")
cumulative_curve_dir = os.path.join(ROOT_DIR, "results_ultra_light_graphsage/cumulative_curves")

# 自动选择设备：优先使用GPU(CUDA)，无则使用CPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ===================================== 通用工具函数 =====================================
def resolve_existing_path(path_candidates, file_desc):
    """按顺序返回第一个存在路径，不存在则抛出异常。"""
    for candidate in path_candidates:
        if os.path.exists(candidate):
            return candidate
    joined = "\n".join(path_candidates)
    raise FileNotFoundError(f"{file_desc}不存在，已尝试以下路径:\n{joined}")


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


def extract_esm_backbone_state_dict(checkpoint_obj):
    """从checkpoint中提取ESM backbone权重，兼容不同前缀。"""
    raw_state = extract_state_dict(checkpoint_obj)
    backbone_state = {}
    for k, v in raw_state.items():
        if k.startswith("esm2_backbone."):
            backbone_state[k] = v
        elif k.startswith("backbone."):
            mapped_key = k.replace("backbone.", "esm2_backbone.", 1)
            backbone_state[mapped_key] = v
    if not backbone_state:
        raise ValueError("checkpoint中未找到ESM2 backbone权重")
    return backbone_state


# ===================================== 累积概率曲线绘制函数 =====================================
def plot_cumulative_curves(graphsage_probs, dataset_name, save_dir):
    """绘制GraphSAGE模型阳性概率的累积分布曲线并保存。"""
    os.makedirs(save_dir, exist_ok=True)
    probs = np.asarray(graphsage_probs, dtype=float)
    probs = probs[~np.isnan(probs)]
    if probs.size == 0:
        print(f"数据集 {dataset_name} 无有效概率值，跳过累积曲线绘制。")
        return

    probs_sorted = np.sort(probs)
    cum_percent = np.arange(1, len(probs_sorted) + 1) / len(probs_sorted) * 100

    plt.figure(figsize=(8, 6))
    plt.plot(probs_sorted, cum_percent, lw=2, label=f"{dataset_name} (n={len(probs_sorted)})")
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 100.0)
    plt.xlabel("Predicted Positive Probability")
    plt.ylabel("Cumulative Percentage (%)")
    plt.title(f"GraphSAGE Cumulative Probability Curve - {dataset_name}")
    plt.grid(alpha=0.3)
    plt.legend(loc="lower right")
    plt.tight_layout()

    safe_name = str(dataset_name).replace("/", "_").replace(" ", "_")
    output_path = os.path.join(save_dir, f"{safe_name}_cumulative_curve.pdf")
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"累积曲线已保存: {output_path}")

# ===================================== 自定义肽段数据集类 =====================================
class PeptideDataset(Dataset):
    """
    自定义Dataset子类，封装肽段序列数据集
    适配PyTorch的DataLoader批量加载数据
    """
    def __init__(self, sequences, labels):
        self.sequences = sequences  # 肽段序列列表
        self.labels = labels        # 肽段标签列表（0/1）
        # 校验：序列数量必须与标签数量一致
        if len(self.sequences) != len(self.labels):
            raise ValueError("序列数量与标签数量不匹配")

    def __len__(self):
        """返回数据集总长度"""
        return len(self.sequences)

    def __getitem__(self, idx):
        """根据索引返回单条数据：序列、标签、序列长度"""
        return self.sequences[idx], self.labels[idx], len(self.sequences[idx])

# ===================================== 肽段特征提取器类（ESM模型） =====================================
class PeptideFeatureExtractor:
    """
    基于预训练ESM2模型的肽段特征提取器核心类
    作用：加载微调后的ESM2模型，提取肽段序列的CLS特征向量，作为后续GraphSAGE模型的输入
    """
    def __init__(self, esm2_model_name_or_path, device=device):
        self.device = device                      # 运行设备
        self.esm2_model_name_or_path = esm2_model_name_or_path  # ESM2模型路径
        self.model = None                         # 特征提取模型实例
        self.tokenizer = None                     # ESM2分词器实例
        self.hidden_size = None                   # ESM2模型的隐藏层维度
        self.total_params = 0                     # 模型总参数量

    def load_pretrained_model(self, finetuned_model_path=None):
        """
        加载预训练ESM2模型+微调权重，冻结所有层仅做特征提取
        :param finetuned_model_path: 微调后的ESM2模型权重路径
        """
        if finetuned_model_path is None:
            finetuned_model_path = resolve_existing_path(finetuned_esm_candidates, "ESM2微调权重")
        print(f"\n正在加载特征提取模型: {finetuned_model_path}")
        try:
            # 加载ESM2分词器
            self.tokenizer = AutoTokenizer.from_pretrained(self.esm2_model_name_or_path)
            # 加载ESM2预训练主干模型
            esm2_backbone = AutoModel.from_pretrained(self.esm2_model_name_or_path)
            # 获取模型隐藏层维度
            self.hidden_size = esm2_backbone.config.hidden_size
            print(f"ESM2 8M隐藏维度: {self.hidden_size}")

            # 定义内部ESM2特征提取器，仅返回CLS特征向量，无分类头
            class ESM2FeatureExtractor(nn.Module):
                def __init__(self, esm2_backbone):
                    super().__init__()
                    self.esm2_backbone = esm2_backbone
                    self.hidden_dim = esm2_backbone.config.hidden_size

                def forward(self, input_ids, attention_mask):
                    # 前向传播获取模型输出
                    outputs = self.esm2_backbone(input_ids=input_ids, attention_mask=attention_mask)
                    # 提取<cls>位置的特征向量 (batch_size, hidden_dim)
                    cls_repr = outputs.last_hidden_state[:, 0, :]
                    # 序列长度归一化，防止长序列特征值过大
                    seq_lengths = attention_mask.sum(dim=1, keepdim=True).float()
                    cls_repr = cls_repr / torch.sqrt(seq_lengths + 1e-8)
                    return cls_repr

            # 初始化特征提取器并迁移到指定设备
            self.model = ESM2FeatureExtractor(esm2_backbone).to(self.device)
            # 加载微调模型权重
            checkpoint = torch.load(finetuned_model_path, map_location=self.device, weights_only=False)

            # 过滤出ESM2主干模型的权重，加载到特征提取器中
            feature_extractor_state_dict = extract_esm_backbone_state_dict(checkpoint)
            self.model.load_state_dict(feature_extractor_state_dict, strict=False)

            # 冻结所有参数，不参与梯度更新，仅做特征提取
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
        批量提取肽段序列的CLS特征向量
        :param sequences: 肽段序列列表
        :param batch_size: 批量大小
        :return: 特征向量数组 (num_samples, hidden_dim)
        """
        if not self.model: raise ValueError("请先加载特征提取模型")
        self.model.eval()  # 模型切换到评估模式
        features = []
        # 构建临时数据集（标签无意义，仅占位）
        dataset = PeptideDataset(sequences, [0] * len(sequences))

        # 自定义数据拼接函数：处理变长序列的动态padding
        def collate_fn(batch):
            sequences, _, _ = zip(*batch)
            # 分词：不自动padding，截断最大长度256
            encoded = self.tokenizer(
                sequences, return_tensors=None, padding="do_not_pad", truncation=True, max_length=256
            )
            input_ids_list, attention_mask_list = encoded["input_ids"], encoded["attention_mask"]
            # 计算当前批次的最大序列长度
            batch_max_len = max(len(ids) for ids in input_ids_list)
            pad_token_id = self.tokenizer.pad_token_id
            # 对序列进行padding到批次最大长度
            padded_input_ids = [ids + [pad_token_id] * (batch_max_len - len(ids)) for ids in input_ids_list]
            padded_attention_mask = [mask + [0] * (batch_max_len - len(mask)) for mask in attention_mask_list]
            return torch.tensor(padded_input_ids, dtype=torch.long), torch.tensor(padded_attention_mask, dtype=torch.long)

        # 构建数据加载器
        dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)
        print(f"提取CLS特征...")
        # 无梯度计算，加速推理
        with torch.no_grad():
            for input_ids, attention_mask in tqdm(dataloader, total=len(dataloader)):
                input_ids, attention_mask = input_ids.to(self.device), attention_mask.to(self.device)
                cls_repr = self.model(input_ids=input_ids, attention_mask=attention_mask)
                features.extend(cls_repr.cpu().numpy())
        return np.array(features)

    def predict(self, sequences, batch_size=16):
        """
        调用ESM模型进行预测，返回预测标签、概率、推理耗时
        :param sequences: 肽段序列列表
        :param batch_size: 批量大小
        :return: predictions, probabilities, inference_time
        """
        if not self.model: raise ValueError("请先加载特征提取模型")
        self.model.eval()
        # 初始化临时预测器，加载带分类头的完整ESM模型
        temp_predictor = PeptidePredictor(self.esm2_model_name_or_path, self.device)
        finetuned_esm_path = resolve_existing_path(finetuned_esm_candidates, "ESM2微调权重")
        temp_predictor.load_pretrained_model(freeze_layers=True, finetuned_model_path=finetuned_esm_path)
        predictions, probabilities, inference_time = temp_predictor.predict(sequences, batch_size)
        return predictions, probabilities, inference_time

# ===================================== 肽段预测器类（带分类头的ESM模型） =====================================
class PeptidePredictor:
    """
    基于ESM2的肽段二分类预测器，包含完整的分类头
    作用：作为ESM模型，输出最终的预测标签和概率值，用于和GraphSAGE模型做性能对比
    """
    def __init__(self, esm2_model_name_or_path, device=device):
        self.device = device
        self.esm2_model_name_or_path = esm2_model_name_or_path
        self.model = None
        self.tokenizer = None
        self.hidden_size = None

    def load_pretrained_model(self, freeze_layers=True, finetuned_model_path=None):
        """加载带分类头的微调ESM2模型，冻结层用于推理"""
        if finetuned_model_path is None:
            finetuned_model_path = resolve_existing_path(finetuned_esm_candidates, "ESM2微调权重")
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
                # 分类头：三层全连接+ReLU+Dropout，输出2分类logits
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

        # 冻结部分层：仅训练后2层，冻结embedding层
        if freeze_layers:
            for i, layer in enumerate(esm2_backbone.encoder.layer):
                for param in layer.parameters():
                    param.requires_grad = (i >= 4)
            for param in esm2_backbone.embeddings.parameters():
                param.requires_grad = False

        # 加载模型和权重，冻结所有参数用于推理
        self.model = ESM2Classifier(esm2_backbone).to(self.device)
        checkpoint = torch.load(finetuned_model_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(extract_state_dict(checkpoint))
        for param in self.model.parameters():
            param.requires_grad = False

    def predict(self, sequences, batch_size=16):
        """批量预测肽段标签和概率，返回预测结果、概率值、推理耗时"""
        self.model.eval()
        predictions = []
        probabilities = []
        dataset = PeptideDataset(sequences, [0] * len(sequences))

        # 同特征提取器的自定义collate_fn，处理变长序列
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
            return torch.tensor(padded_input_ids, dtype=torch.long), torch.tensor(padded_attention_mask, dtype=torch.long)

        dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)
        start_time = time.time()
        with torch.no_grad():
            for input_ids, attention_mask in tqdm(dataloader, total=len(dataloader)):
                input_ids, attention_mask = input_ids.to(self.device), attention_mask.to(self.device)
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                # softmax计算概率，argmax获取预测标签
                probs = torch.softmax(outputs, dim=1).cpu().numpy()
                predictions.extend(np.argmax(probs, axis=1))
                probabilities.extend(probs[:, 1])
        inference_time = time.time() - start_time
        return predictions, probabilities, inference_time

# ===================================== 通用工具函数：加载CSV数据集 =====================================
def load_csv_data(csv_path):
    """
    加载肽段数据集CSV文件，数据清洗与校验
    :param csv_path: csv文件路径
    :return: ids列表, sequences序列列表, labels标签列表
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
    # 遍历数据行，清洗无效数据
    for _, row in df.iterrows():
        id_val, seq = row["id"], str(row[seq_col]).strip().upper()
        # 过滤空序列
        if len(seq) == 0:
            print(f"过滤空序列（id: {id_val}）")
            continue
        # 过滤无效标签（必须是0/1）
        try:
            label = int(row["label"])
            if label not in (0, 1):
                raise ValueError("标签必须为0或1")
        except Exception as e:
            print(f"过滤无效标签（{e}）: {row['label']}（id: {id_val}）")
            continue
        ids.append(id_val)
        sequences.append(seq)
        labels.append(label)
    # 统计数据集信息
    total, positive = len(labels), sum(labels)
    if total == 0:
        raise ValueError(f"数据清洗后无有效样本: {csv_path}")
    ratio = positive / total * 100
    print(f"加载完成 {os.path.basename(csv_path)}：共{total}条有效序列（阳性{positive}条，阴性{total - positive}条，阳性占比{ratio:.2f}%）")
    return ids, sequences, labels

# ===================================== 核心GraphSAGE模型：增强版GraphSAGE轻量化分类器 =====================================
class EnhancedGraphSAGEClassifier(nn.Module):
    """
    无知识蒸馏的增强版GraphSAGE模型，轻量化二分类器
    输入：ESM2提取的CLS特征向量
    输出：肽段0/1分类的logits
    核心特点：超轻量级、无蒸馏、仅用图卷积做特征聚合，参数量极少
    """
    def __init__(self, input_dim=320, hidden_dim=128, latent_dim=64, num_classes=2, num_layers=2):
        super().__init__()
        self.input_dim = input_dim    # 输入特征维度（ESM2的hidden_size）
        self.hidden_dim = hidden_dim  # GraphSAGE隐藏层维度
        self.latent_dim = latent_dim  # GraphSAGE输出维度
        self.num_layers = num_layers  # GraphSAGE层数

        # 输入投影层：将原始特征映射到GraphSAGE的输入维度
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # 初始化GraphSAGE编码器
        self.sage_encoder = SAGEEncoder(
            in_channels=hidden_dim,
            hidden_channels=hidden_dim,
            out_channels=latent_dim,
            num_layers=num_layers,
            dropout=0.2
        )

        # 层归一化+Dropout：防止过拟合，稳定训练
        self.layer_norm = nn.LayerNorm(latent_dim)
        self.dropout = nn.Dropout(0.2)

        # 最终分类头：将图特征映射到2分类
        self.classifier = nn.Linear(latent_dim, num_classes)

        # 权重初始化
        self._init_weights()

    def _init_weights(self):
        """自定义权重初始化策略，提升训练收敛速度"""
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.zeros_(self.input_proj.bias)
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)
        nn.init.ones_(self.layer_norm.weight)
        nn.init.zeros_(self.layer_norm.bias)

    def forward(self, x):
        """
        前向传播函数，核心逻辑：特征投影 -> 图卷积编码 -> 全局池化 -> 分类
        :param x: 输入特征 (batch_size, input_dim)
        :return: 分类logits (batch_size, num_classes)
        """
        batch_size = x.shape[0]

        # 特征投影
        x_proj = self.input_proj(x)

        # 构建图数据：单节点无连接（edge_index为空），每个样本是一个独立的图
        node_feat = x_proj
        edge_index = torch.empty((2, 0), dtype=torch.long, device=x.device)
        batch_idx = torch.arange(batch_size, device=x.device, dtype=torch.long)

        # GraphSAGE编码节点特征
        z = self.sage_encoder(node_feat, edge_index)

        # 全局均值池化，得到图级特征
        global_feat = global_mean_pool(z, batch_idx)

        # 归一化+Dropout+分类
        out = self.layer_norm(global_feat)
        out = self.dropout(out)
        logits = self.classifier(out)
        return logits

# ===================================== GraphSAGE模型训练器核心类 =====================================
class GraphSAGETrainer:
    """
    无蒸馏的GraphSAGE模型训练器，封装完整的训练、验证、评估、预测流程
    核心功能：初始化模型、数据预处理、模型训练、早停策略、性能评估、轻量化对比
    """
    def __init__(self, feature_extractor, feature_extractor_total_params, device=device):
        self.device = device  # 运行设备
        self.feature_extractor = feature_extractor  # 特征提取器实例
        self.feature_extractor_total_params = feature_extractor_total_params  # ESM模型参数量
        self.graphsage_model = None  # GraphSAGE模型实例
        self.optimizer = None      # 优化器实例

        # 训练/验证指标记录字典
        self.train_metrics = {"total_loss": [], "accuracy": [], "f1": [], "recall": [], "mcc": []}
        self.val_metrics = {"accuracy": [], "f1": [], "recall": [], "mcc": [], "auc": []}

        # 模型轻量化相关统计
        self.esm_params_k = self.feature_extractor_total_params / 1000
        self.esm_size_mb = (self.feature_extractor_total_params * 4) / (1024 * 1024)
        self.graphsage_params_k = 0
        self.graphsage_size_mb = 0

    def init_graphsage_model(self, input_dim=320, hidden_dim=128, latent_dim=64):
        """初始化GraphSAGE模型，并统计参数量"""
        self.graphsage_model = EnhancedGraphSAGEClassifier(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            num_layers=2
        ).to(self.device)
        total_params = sum(p.numel() for p in self.graphsage_model.parameters())
        self.graphsage_params_k = total_params / 1000
        print(f"\nGraphSAGE模型初始化完成:")
        print(f"   - 输入维度: {input_dim} | GraphSAGE隐藏层维度: {hidden_dim} | 输出维度: {latent_dim}")
        print(f"   - GraphSAGE编码器层数: 2")
        print(f"   - GraphSAGE模型总参数量: {total_params:,} ({self.graphsage_params_k:.2f}k)")
        return self.graphsage_model

    def prepare_training_data(self, sequences, labels, batch_size=32, save_dir="results_ultra_light_graphsage"):
        """预处理训练数据：提取特征并缓存到npz文件，避免重复提取"""
        os.makedirs(save_dir, exist_ok=True)
        cache_path = os.path.join(save_dir, "training_data_no_distill.npz")

        if os.path.exists(cache_path):
            print(f"加载缓存的训练数据: {cache_path}")
            data = np.load(cache_path)
            return {
                'features': data['features'],
                'labels': data['labels'],
                'sequences': sequences
            }

        # 提取特征并保存缓存
        features = self.feature_extractor.get_cls_features(sequences, batch_size=batch_size)
        labels_np = np.array(labels)
        np.savez_compressed(cache_path, features=features, labels=labels_np)
        print(f"训练特征已保存，维度: {features.shape}")
        return {'features': features, 'labels': labels_np, 'sequences': sequences}

    def prepare_validation_data(self, val_sequences, val_labels, batch_size=32, save_dir="results_ultra_light_graphsage"):
        """预处理验证数据：同训练数据，提取特征并缓存"""
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
        计算二分类任务全量评估指标，包含异常处理
        :param y_true: 真实标签
        :param y_pred: 预测标签
        :param y_prob: 预测概率
        :return: 指标字典
        """
        # ROC-AUC指标：处理单类数据异常
        try:
            roc_auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.0
        except:
            roc_auc = 0.0

        # PR-AUC指标：处理单类数据异常
        try:
            precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_prob)
            pr_auc = auc(recall_curve, precision_curve)
        except:
            pr_auc = 0.0

        return {
            'Accuracy': accuracy_score(y_true, y_pred),
            'F1 Score': f1_score(y_true, y_pred, zero_division=0),
            'Precision': precision_score(y_true, y_pred, zero_division=0),
            'Recall': recall_score(y_true, y_pred, zero_division=0),
            'MCC': matthews_corrcoef(y_true, y_pred),
            'ROC-AUC': roc_auc,
            'PR-AUC': pr_auc
        }

    def train_graphsage(self, training_data, val0_data, save_dir="results_ultra_light_graphsage",
                      batch_size=512, epochs=50, lr=1e-3, early_stop_patience=10):
        """
        核心训练函数：训练GraphSAGE模型，带早停策略、梯度裁剪、模型保存
        :param training_data: 训练数据字典
        :param val0_data: 早停验证集数据字典
        :param save_dir: 模型保存路径
        :param batch_size: 批量大小
        :param epochs: 训练轮次
        :param lr: 学习率
        :param early_stop_patience: 早停耐心值
        :return: 训练好的GraphSAGE模型、总训练耗时
        """
        os.makedirs(save_dir, exist_ok=True)

        # 转换为张量并迁移到指定设备
        train_features = torch.FloatTensor(training_data['features']).to(self.device)
        train_labels = torch.LongTensor(training_data['labels']).to(self.device)
        val0_features = torch.FloatTensor(val0_data['val_features']).to(self.device)
        val0_labels = torch.LongTensor(val0_data['val_labels']).to(self.device)

        # 构建数据加载器
        train_dataset = torch.utils.data.TensorDataset(train_features, train_labels)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val0_dataset = torch.utils.data.TensorDataset(val0_features, val0_labels)
        val0_loader = DataLoader(val0_dataset, batch_size=batch_size)

        # 初始化优化器和损失函数
        self.optimizer = optim.AdamW(self.graphsage_model.parameters(), lr=lr, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()
        best_val_f1, early_stop_count = -1.0, 0  # 早停相关初始化

        print(f"\n开始训练GraphSAGE模型（无蒸馏）...")
        print(f"超参数配置: 批大小={batch_size}, 轮次={epochs}, 学习率={lr}")
        print(f"早停验证集: val_test.csv (val0)")
        start_time = time.time()

        # 训练主循环
        for epoch in range(epochs):
            self.graphsage_model.train()  # 训练模式
            total_loss, correct, total = 0, 0, 0
            train_preds, train_true = [], []

            # 批量训练
            for batch_idx, (features, labels) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs} - Training")):
                self.optimizer.zero_grad()  # 梯度清零
                logits = self.graphsage_model(features)  # 前向传播
                loss = criterion(logits, labels)       # 计算损失

                loss.backward()  # 反向传播
                torch.nn.utils.clip_grad_norm_(self.graphsage_model.parameters(), max_norm=1.0)  # 梯度裁剪，防止梯度爆炸
                self.optimizer.step()  # 更新参数

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
            self.graphsage_model.eval()
            val0_preds, val0_true, val0_probs = [], [], []
            with torch.no_grad():
                for features, labels in val0_loader:
                    logits = self.graphsage_model(features)
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

            # 打印本轮训练结果
            print(f'Epoch {epoch + 1}: 训练准确率={train_acc:.2f}% | 训练F1={train_metrics["F1 Score"]:.4f} | '
                  f'早停验证集 F1={val0_metrics["F1 Score"]:.4f} | 早停验证集 AUC={val0_metrics["ROC-AUC"]:.4f}')

            # 早停策略：验证集F1提升则保存最佳模型，否则计数+1
            if val0_metrics['F1 Score'] > best_val_f1 + 1e-4:
                best_val_f1 = val0_metrics['F1 Score']
                early_stop_count = 0
                torch.save(
                    {'model_state_dict': self.graphsage_model.state_dict(), 'epoch': epoch + 1, 'val_f1': best_val_f1},
                    f"{save_dir}/best_graphsage_model_no_distill.pt")
            else:
                early_stop_count += 1
                if early_stop_count >= early_stop_patience:
                    print(f"触发早停！最佳早停验证集 F1: {best_val_f1:.4f}")
                    break

        # 训练结束，统计耗时并保存最终模型
        total_training_time = time.time() - start_time
        final_model_path = f"{save_dir}/final_graphsage_model_no_distill.pt"
        torch.save({
            'model_state_dict': self.graphsage_model.state_dict(),
            'train_metrics': self.train_metrics,
            'val_metrics': self.val_metrics
        }, final_model_path)

        # 计算GraphSAGE模型文件大小
        if os.path.exists(final_model_path):
            self.graphsage_size_mb = os.path.getsize(final_model_path) / (1024 * 1024)

        print(f"\nGraphSAGE模型训练完成！总耗时: {total_training_time:.2f}秒")
        print(f"最佳GraphSAGE模型保存路径: {save_dir}/best_graphsage_model_no_distill.pt")
        return self.graphsage_model, total_training_time

    def evaluate_model_performance(self, sequences, labels, ids, batch_size=64, dataset_name="validation"):
        """
        核心评估函数：同时评估ESM模型和GraphSAGE模型的性能，计算推理耗时、提速比、全量指标
        :param sequences: 肽段序列列表
        :param labels: 真实标签列表
        :param ids: 肽段ID列表
        :param batch_size: 批量大小
        :param dataset_name: 数据集名称
        :return: 完整性能评估字典
        """
        if self.graphsage_model is None: raise ValueError("GraphSAGE模型未初始化")
        self.graphsage_model.eval()

        # ESM模型预测
        print(f"\n=== 评估ESM模型性能（{dataset_name}集）===")
        esm_preds, esm_probs, esm_inference_time = self.feature_extractor.predict(sequences, batch_size=batch_size)
        true_labels = np.array(labels)

        # 提取特征
        print(f"=== 提取{dataset_name}集CLS特征 ===")
        features = self.feature_extractor.get_cls_features(sequences, batch_size=batch_size)
        features = torch.FloatTensor(features).to(self.device)

        # GraphSAGE模型预测
        print(f"=== 评估GraphSAGE模型性能（{dataset_name}集）===")
        graphsage_preds, graphsage_probs = [], []
        graphsage_start_time = time.time()
        with torch.no_grad():
            for i in tqdm(range(0, len(features), batch_size), desc=f"Evaluating GraphSAGE ({dataset_name})"):
                batch_features = features[i:i + batch_size]
                batch_logits = self.graphsage_model(batch_features)
                _, batch_preds = batch_logits.max(1)
                graphsage_preds.extend(batch_preds.cpu().numpy())
                graphsage_probs.extend(F.softmax(batch_logits, dim=1)[:, 1].cpu().numpy())
        graphsage_inference_time = time.time() - graphsage_start_time

        # 计算评估指标
        esm_metrics = self._compute_metrics(true_labels, np.array(esm_preds), np.array(esm_probs))
        graphsage_metrics = self._compute_metrics(true_labels, np.array(graphsage_preds), np.array(graphsage_probs))

        # 补充模型轻量化和耗时指标
        esm_metrics['Inference Time (s)'] = esm_inference_time
        esm_metrics['Param Count (k)'] = self.esm_params_k
        esm_metrics['Model Size (MB)'] = self.esm_size_mb

        graphsage_metrics['Inference Time (s)'] = graphsage_inference_time
        graphsage_metrics['Param Count (k)'] = self.graphsage_params_k
        graphsage_metrics['Model Size (MB)'] = self.graphsage_size_mb

        return {
            'dataset_name': dataset_name,
            'ids': ids,
            'sequences': sequences,
            'true_labels': true_labels,
            'esm_preds': np.array(esm_preds),
            'esm_probs': np.array(esm_probs),
            'graphsage_preds': np.array(graphsage_preds),
            'graphsage_probs': np.array(graphsage_probs),
            'esm_metrics': esm_metrics,
            'graphsage_metrics': graphsage_metrics,
            'speedup_ratio': esm_inference_time / graphsage_inference_time
        }

    def get_model_comparison(self, graphsage_model_path):
        """计算ESM模型与GraphSAGE模型的轻量化对比指标：体积缩减率、参数量缩减率"""
        return {
            'esm_size_mb': self.esm_size_mb, 'graphsage_size_mb': self.graphsage_size_mb,
            'size_reduction': (self.esm_size_mb - self.graphsage_size_mb) / self.esm_size_mb * 100,
            'esm_params': self.feature_extractor_total_params, 'graphsage_params': self.graphsage_params_k * 1000,
            'params_reduction': (self.feature_extractor_total_params - self.graphsage_params_k * 1000) / self.feature_extractor_total_params * 100
        }

# ===================================== 结果保存与打印工具函数 =====================================
def save_prediction_results(performance_dict, output_file):
    """将预测结果保存为CSV文件，包含序列、真实标签、ESM/GraphSAGE模型预测结果和概率"""
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    results = pd.DataFrame({
        "id": performance_dict['ids'],
        "sequence": performance_dict['sequences'],
        "sequence_length": [len(seq) for seq in performance_dict['sequences']],
        "true_label": performance_dict['true_labels'],
        "esm_predicted_label": performance_dict['esm_preds'],
        "esm_predicted_label_name": ["阳性（1）" if p == 1 else "阴性（0）" for p in performance_dict['esm_preds']],
        "esm_positive_probability": performance_dict['esm_probs'].round(4),
        "graphsage_predicted_label": performance_dict['graphsage_preds'],
        "graphsage_predicted_label_name": ["阳性（1）" if p == 1 else "阴性（0）" for p in performance_dict['graphsage_preds']],
        "graphsage_positive_probability": performance_dict['graphsage_probs'].round(4)
    })
    results.to_csv(output_file, index=False, encoding="utf-8-sig")
    print(f"\n{performance_dict['dataset_name']}集预测结果已保存至: {output_file}")

def save_metrics_summary(performance_list, output_file):
    """将所有数据集的ESM/GraphSAGE模型评估指标保存为汇总CSV文件"""
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    summary_data = []

    for perf in performance_list:
        dataset_name = perf['dataset_name']
        esm_row = {'Dataset': dataset_name, 'Model Type': 'ESM Model', **perf['esm_metrics']}
        graphsage_row = {'Dataset': dataset_name, 'Model Type': 'GraphSAGE Model (No Distill)', **perf['graphsage_metrics']}
        summary_data.append(esm_row)
        summary_data.append(graphsage_row)

    summary_df = pd.DataFrame(summary_data)
    # 指标列排序
    column_order = [
        'Dataset', 'Model Type', 'Accuracy', 'F1 Score', 'Precision',
        'Recall', 'MCC', 'ROC-AUC', 'PR-AUC', 'Inference Time (s)',
        'Param Count (k)', 'Model Size (MB)'
    ]
    for col in column_order:
        if col not in summary_df.columns:
            summary_df[col] = 0.0
    summary_df = summary_df[column_order]
    summary_df = summary_df.round(4)
    summary_df.to_csv(output_file, index=False, encoding="utf-8-sig")
    print(f"\n全量指标汇总已保存至: {output_file}")

def print_detailed_metrics(performance_list):
    """在控制台打印格式化的详细评估指标，便于查看"""
    print("\n" + "=" * 120)
    print("【详细指标汇总】训练集/验证集/测试集 - ESM模型&GraphSAGE模型（无蒸馏）")
    print("=" * 120)

    metrics_to_print = [
        'Accuracy', 'F1 Score', 'Precision', 'Recall', 'MCC',
        'ROC-AUC', 'PR-AUC', 'Inference Time (s)', 'Param Count (k)', 'Model Size (MB)'
    ]

    for perf in performance_list:
        dataset_name = perf['dataset_name']
        print(f"\n📊 数据集: {dataset_name}")
        print("-" * 100)

        print("🧬 ESM模型:")
        for metric in metrics_to_print:
            value = perf['esm_metrics'].get(metric, 0.0)
            if metric in ['Inference Time (s)']:
                print(f"   - {metric:<20}: {value:.4f}")
            elif metric in ['Param Count (k)', 'Model Size (MB)']:
                print(f"   - {metric:<20}: {value:.2f}")
            else:
                print(f"   - {metric:<20}: {value:.4f}")

        print("📈 GraphSAGE模型 (无蒸馏):")
        for metric in metrics_to_print:
            value = perf['graphsage_metrics'].get(metric, 0.0)
            if metric in ['Inference Time (s)']:
                print(f"   - {metric:<20}: {value:.4f}")
            elif metric in ['Param Count (k)', 'Model Size (MB)']:
                print(f"   - {metric:<20}: {value:.2f}")
            else:
                print(f"   - {metric:<20}: {value:.4f}")

        print(f"   - {'Speedup Ratio (GraphSAGE/ESM)':<20}: {perf['speedup_ratio']:.2f}x")
        print("-" * 100)

# ===================================== 主训练流程函数 =====================================
def train_graphsage_no_distillation():
    """
    无知识蒸馏的GraphSAGE模型完整训练流程主函数
    流程：加载数据集 -> 初始化特征提取器 -> 初始化GraphSAGE模型 -> 预处理数据 -> 训练模型 -> 评估所有数据集 -> 保存结果
    """
    print("=== 开始训练GraphSAGE模型（无知识蒸馏）===")
    print("=== 适配新数据集路径 | val_test.csv早停 | 新增t0/t1/t2/t3验证集 ===")
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

        # 打印数据集汇总信息
        print(f"\n数据集汇总: ")
        print(f"   - 训练集: {len(train_sequences)}条")
        print(f"   - 早停验证集(val_test): {len(val0_sequences)}条")
        print(f"   - 验证集(rawdata_test): {len(t0_sequences)}条")
        print(f"   - bagel4_1_val: {len(t1_sequences)}条")
        print(f"   - bagel4_2_val: {len(t2_sequences)}条")
        print(f"   - bagel4_3_val: {len(t3_sequences)}条")

        # 2. 初始化并加载特征提取器（ESM模型）
        print("\n===================== 加载ESM特征提取模型 =====================")
        feature_extractor = PeptideFeatureExtractor(esm2_model_name_or_path=esm2_backbone_path, device=device)
        finetuned_esm_path = resolve_existing_path(finetuned_esm_candidates, "ESM2微调权重")
        feature_extractor.load_pretrained_model(finetuned_model_path=finetuned_esm_path)

        # 3. 初始化GraphSAGE模型训练器和GraphSAGE模型
        print("\n===================== 初始化GraphSAGE模型 =====================")
        trainer = GraphSAGETrainer(
            feature_extractor=feature_extractor,
            feature_extractor_total_params=feature_extractor.total_params,
            device=device
        )
        trainer.init_graphsage_model(input_dim=feature_extractor.hidden_size, hidden_dim=128, latent_dim=64)

        # 4. 预处理训练和验证数据
        print("\n===================== 准备训练数据 =====================")
        training_data = trainer.prepare_training_data(
            sequences=train_sequences, labels=train_labels,
            batch_size=32, save_dir="results_ultra_light_graphsage"
        )
        val0_data = trainer.prepare_validation_data(
            val_sequences=val0_sequences, val_labels=val0_labels,
            batch_size=32, save_dir="results_ultra_light_graphsage"
        )

        # 5. 训练GraphSAGE模型
        print("\n===================== 训练GraphSAGE模型 =====================")
        graphsage_model, training_time = trainer.train_graphsage(
            training_data=training_data, val0_data=val0_data,
            save_dir="results_ultra_light_graphsage",
            batch_size=512, epochs=50, lr=1e-3, early_stop_patience=10
        )

        # 6. 评估所有数据集的性能
        print("\n===================== 评估所有数据集 =====================")
        train_perf = trainer.evaluate_model_performance(sequences=train_sequences, labels=train_labels, ids=train_ids, batch_size=64, dataset_name="train_no_distill_graphsage")
        plot_cumulative_curves(graphsage_probs=train_perf['graphsage_probs'], dataset_name=train_perf['dataset_name'], save_dir=cumulative_curve_dir)

        val0_perf = trainer.evaluate_model_performance(sequences=val0_sequences, labels=val0_labels, ids=val0_ids, batch_size=64, dataset_name="val0_val_test_early_stop")
        plot_cumulative_curves(graphsage_probs=val0_perf['graphsage_probs'], dataset_name=val0_perf['dataset_name'], save_dir=cumulative_curve_dir)

        t0_perf = trainer.evaluate_model_performance(sequences=t0_sequences, labels=t0_labels, ids=t0_ids, batch_size=64, dataset_name="t0_rawdata_test")
        plot_cumulative_curves(graphsage_probs=t0_perf['graphsage_probs'], dataset_name=t0_perf['dataset_name'], save_dir=cumulative_curve_dir)

        t1_perf = trainer.evaluate_model_performance(sequences=t1_sequences, labels=t1_labels, ids=t1_ids, batch_size=64, dataset_name="t1_bagel4_1_val")
        plot_cumulative_curves(graphsage_probs=t1_perf['graphsage_probs'], dataset_name=t1_perf['dataset_name'], save_dir=cumulative_curve_dir)

        t2_perf = trainer.evaluate_model_performance(sequences=t2_sequences, labels=t2_labels, ids=t2_ids, batch_size=64, dataset_name="t2_bagel4_2_val")
        plot_cumulative_curves(graphsage_probs=t2_perf['graphsage_probs'], dataset_name=t2_perf['dataset_name'], save_dir=cumulative_curve_dir)

        t3_perf = trainer.evaluate_model_performance(sequences=t3_sequences, labels=t3_labels, ids=t3_ids, batch_size=64, dataset_name="t3_bagel4_3_val")
        plot_cumulative_curves(graphsage_probs=t3_perf['graphsage_probs'], dataset_name=t3_perf['dataset_name'], save_dir=cumulative_curve_dir)

        performance_list = [train_perf, val0_perf, t0_perf, t1_perf, t2_perf, t3_perf]

        # 7. 计算模型轻量化对比指标
        print("\n===================== 模型轻量化对比 =====================")
        graphsage_model_path = "results_ultra_light_graphsage/final_graphsage_model_no_distill.pt"
        model_comparison = trainer.get_model_comparison(graphsage_model_path)

        # 8. 打印训练最终结果汇总
        print("\n" + "=" * 100)
        print("【无蒸馏】GraphSAGE模型训练最终结果汇总")
        print("=" * 100)

        print(f"\n1. 模型轻量化效果:")
        print(f"   - ESM模型: {model_comparison['esm_size_mb']:.2f} MB | {model_comparison['esm_params']:,} 参数")
        print(f"   - GraphSAGE模型: {model_comparison['graphsage_size_mb']:.2f} MB | {model_comparison['graphsage_params']:,} 参数")
        print(f"   - 体积缩减: {model_comparison['size_reduction']:.1f}% | 参数量缩减: {model_comparison['params_reduction']:.1f}%")
        print(f"   - 训练耗时: {training_time:.2f}秒")

        print(f"\n2. 各数据集（GraphSAGE模型）核心指标:")
        for perf in performance_list:
            ds_name = perf['dataset_name']
            g_metrics = perf['graphsage_metrics']
            print(f"   - {ds_name}: F1={g_metrics['F1 Score']:.4f} | ROC-AUC={g_metrics['ROC-AUC']:.4f} | PR-AUC={g_metrics['PR-AUC']:.4f} | 推理提速={perf['speedup_ratio']:.1f}x")

        # 9. 保存预测结果和指标汇总
        print("\n===================== 保存预测结果和指标汇总 =====================")
        save_prediction_results(val0_perf, val0_output)
        save_prediction_results(t0_perf, t0_output)
        save_prediction_results(t1_perf, t1_output)
        save_prediction_results(t2_perf, t2_output)
        save_prediction_results(t3_perf, t3_output)
        save_metrics_summary(performance_list, metrics_summary_output)

        # 打印详细指标
        print_detailed_metrics(performance_list)

        # 训练完成提示
        print("\n" + "=" * 100)
        print("=== GraphSAGE模型无蒸馏训练全流程成功完成！===")
        print(f"GraphSAGE模型保存目录: results_ultra_light_graphsage")
        print(f"早停验证集结果: {val0_output}")
        print(f"rawdata_test结果: {t0_output}")
        print(f"bagel4_1_val结果: {t1_output}")
        print(f"bagel4_2_val结果: {t2_output}")
        print(f"bagel4_3_val结果: {t3_output}")
        print(f"全量指标汇总: {metrics_summary_output}")
        print(f"累积曲线保存目录: {cumulative_curve_dir}")
        print("=" * 100)

        return graphsage_model, trainer, performance_list, model_comparison

    except Exception as e:
        # 异常捕获与打印堆栈信息
        print(f"\n程序执行失败: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None, None

if __name__ == "__main__":
    graphsage_model, trainer, performance_results, model_comparison = train_graphsage_no_distillation()

    if graphsage_model is None:
        print("\n训练流程执行失败，请检查错误信息！")
        exit(1)
    else:
        print("\n所有流程执行完毕，结果已保存！")
        exit(0)
