import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from torch_geometric.nn import SAGEConv, global_mean_pool, GCNConv
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

# ===================== 全局配置参数区 =====================
ROOT_DIR = "./"
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 数据集路径（用户指定）
TRAIN_CSV = os.path.join(ROOT_DIR, "mydata/newdata/rawdata_val.csv")  # XGBoost的训练集
TEST_CSVs = {
    "t1": os.path.join(ROOT_DIR, "test_data/bagel4_1_val.csv"),
    "t2": os.path.join(ROOT_DIR, "test_data/bagel4_2_val.csv"),
    "t3": os.path.join(ROOT_DIR, "test_data/bagel4_3_val.csv")
}

# 模型路径候选（兼容不同目录结构）
esm2_backbone_path = os.path.join(ROOT_DIR, "model/esm2_t6_8M_UR50D")
finetuned_esm_candidates = [
    os.path.join(ROOT_DIR, "model/model_AMP_ESM2_8M_256_v3/best_model_amp/esm2_8m_binary_best.pt"),
    os.path.join(ROOT_DIR, "model_AMP_ESM2_8M_256_v3/best_model_amp/esm2_8m_binary_best.pt"),
]
rnn_model_candidates = [
    os.path.join(ROOT_DIR, "model/results_ultra_light_rnn/best_rnn_model_no_distill.pt"),
    os.path.join(ROOT_DIR, "results_ultra_light_rnn/best_rnn_model_no_distill.pt"),
]
gnn_model_candidates = [
    os.path.join(ROOT_DIR, "model/results_ultra_light_gnn/best_model_no_distill_gnn.pt"),
    os.path.join(ROOT_DIR, "results_ultra_light_gnn/best_model_no_distill_gnn.pt"),
]
graphsage_model_candidates = [
    os.path.join(ROOT_DIR, "model/results_ultra_light_graphsage/best_graphsage_model_no_distill.pt"),
    os.path.join(ROOT_DIR, "results_ultra_light_graphsage/best_graphsage_model_no_distill.pt"),
]
comdel_model_candidates = [
    os.path.join(ROOT_DIR, "model/results_ultra_light_scratch/cnn_scratch_best.pt"),
    os.path.join(ROOT_DIR, "results_ultra_light_scratch/cnn_scratch_best.pt"),
]

# 结果保存目录（优化命名，更直观）
OUTPUT_DIR = os.path.join(ROOT_DIR, "xgboost_ensemble_result")
XGB_MODEL_PATH = os.path.join(OUTPUT_DIR, "xgboost_ensemble_model.json")  # XGBoost模型保存路径
os.makedirs(OUTPUT_DIR, exist_ok=True)

# XGBoost训练参数（可根据需求调整）
XGB_PARAMS = {
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'max_depth': 6,
    'learning_rate': 0.1,
    'n_estimators': 100,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'seed': 42,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'  # 支持GPU加速
}


def resolve_existing_path(path_candidates, file_desc):
    """按顺序返回第一个存在的路径，否则抛出包含候选路径的错误。"""
    for candidate in path_candidates:
        if os.path.exists(candidate):
            return candidate
    joined = "\n".join(path_candidates)
    raise FileNotFoundError(f"{file_desc}不存在，已尝试以下路径:\n{joined}")


def extract_state_dict(checkpoint_obj):
    """
    兼容常见checkpoint格式：
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

# ===================== 原有基础类（保持不变，仅注释优化） =====================
class PeptideDataset(Dataset):
    """肽段序列数据集类，适配torch的DataLoader加载"""
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels
        if len(self.sequences) != len(self.labels):
            raise ValueError("序列数量与标签数量不匹配")

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx], len(self.sequences[idx])

class PeptideFeatureExtractor:
    """基于ESM2预训练模型提取肽段CLS特征（供RNN/GNN/GraphSAGE使用）"""
    def __init__(self, esm2_model_path, device=device):
        self.device = device
        self.esm2_model_path = esm2_model_path
        self.model = None
        self.tokenizer = None
        self.hidden_size = None

    def load_pretrained_model(self, finetuned_weight_path):
        """加载微调后的ESM2模型，冻结所有层仅用于特征提取"""
        if finetuned_weight_path is None:
            finetuned_weight_path = resolve_existing_path(finetuned_esm_candidates, "ESM2微调权重")
        print(f"\n加载ESM2特征提取模型: {finetuned_weight_path}")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.esm2_model_path)
            esm2_backbone = AutoModel.from_pretrained(self.esm2_model_path)
            self.hidden_size = esm2_backbone.config.hidden_size
            print(f"ESM2隐藏维度: {self.hidden_size}")

            # 仅保留ESM2主干，提取CLS特征
            class ESM2FeatureExtractor(nn.Module):
                def __init__(self, backbone):
                    super().__init__()
                    self.backbone = backbone

                def forward(self, input_ids, attention_mask):
                    outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
                    cls_feat = outputs.last_hidden_state[:, 0, :]  # CLS特征
                    seq_len = attention_mask.sum(dim=1, keepdim=True).float()
                    cls_feat = cls_feat / torch.sqrt(seq_len + 1e-8)  # 长度归一化
                    return cls_feat

            self.model = ESM2FeatureExtractor(esm2_backbone).to(self.device)
            # 加载微调权重（仅加载ESM2主干部分）
            checkpoint = torch.load(finetuned_weight_path, map_location=device, weights_only=False)
            model_state_dict = extract_state_dict(checkpoint)
            state_dict = {}
            for k, v in model_state_dict.items():
                if k.startswith('esm2_backbone.'):
                    state_dict[k] = v
                elif k.startswith('backbone.'):
                    state_dict[k.replace('backbone.', 'esm2_backbone.', 1)] = v
            if not state_dict:
                raise ValueError("ESM2主干权重为空，请检查checkpoint是否匹配当前模型结构")
            self.model.load_state_dict(state_dict, strict=False)
            
            # 冻结参数，加速推理
            for param in self.model.parameters():
                param.requires_grad = False
            self.model.eval()
            print("ESM2特征提取模型加载完成（已冻结）")
        except Exception as e:
            raise RuntimeError(f"ESM2模型加载失败: {e}")

    def get_cls_features(self, sequences, batch_size=16):
        """批量提取序列的CLS特征"""
        if not self.model:
            raise ValueError("请先调用load_pretrained_model加载模型")
        
        dataset = PeptideDataset(sequences, [0]*len(sequences))
        # 动态padding的collate函数
        def collate_fn(batch):
            seqs, _, _ = zip(*batch)
            encoded = self.tokenizer(seqs, padding="do_not_pad", truncation=True, max_length=256)
            max_len = max(len(ids) for ids in encoded["input_ids"])
            pad_id = self.tokenizer.pad_token_id
            input_ids = [ids + [pad_id]*(max_len-len(ids)) for ids in encoded["input_ids"]]
            attention_mask = [mask + [0]*(max_len-len(mask)) for mask in encoded["attention_mask"]]
            return torch.tensor(input_ids, dtype=torch.long), torch.tensor(attention_mask, dtype=torch.long)
        
        dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)
        features = []
        with torch.no_grad():
            for input_ids, attention_mask in tqdm(dataloader, desc="提取ESM2 CLS特征"):
                input_ids, attention_mask = input_ids.to(self.device), attention_mask.to(self.device)
                cls_feat = self.model(input_ids, attention_mask)
                features.extend(cls_feat.cpu().numpy())
        return np.array(features)

# ===================== 四个下游模型定义（保持不变） =====================
class EnhancedRNNClassifier(nn.Module):
    def __init__(self, input_dim=320, hidden_dim=128, num_classes=2, num_layers=2, bidirectional=True):
        super().__init__()
        self.rnn = nn.RNN(input_dim, hidden_dim, num_layers, batch_first=True, bidirectional=bidirectional, dropout=0.3 if num_layers>1 else 0)
        self.layer_norm = nn.LayerNorm(hidden_dim * (2 if bidirectional else 1))
        self.dropout = nn.Dropout(0.2)
        self.classifier = nn.Linear(hidden_dim * (2 if bidirectional else 1), num_classes)
        self._init_weights()

    def _init_weights(self):
        for name, param in self.rnn.named_parameters():
            if 'weight_ih' in name: nn.init.xavier_uniform_(param)
            elif 'weight_hh' in name: nn.init.orthogonal_(param)
            elif 'bias' in name: param.data.fill_(0.0)
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, x):
        x = x.unsqueeze(1)
        rnn_out, _ = self.rnn(x)
        rnn_out = rnn_out.squeeze(1)
        out = self.layer_norm(rnn_out)
        out = self.dropout(out)
        return self.classifier(out)

class GNN_EnhancedClassifier(nn.Module):
    def __init__(self, input_dim=320, hidden_dim=128, num_classes=2, num_layers=2):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.gcn_layers = nn.ModuleList([GCNConv(hidden_dim, hidden_dim) for _ in range(num_layers)])
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(0.2)
        self.classifier = nn.Linear(hidden_dim, num_classes)
        self._init_weights()

    def _init_weights(self):
        for gcn in self.gcn_layers:
            nn.init.xavier_uniform_(gcn.lin.weight)
            nn.init.zeros_(gcn.bias)
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.zeros_(self.input_proj.bias)
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, x):
        batch_size = x.shape[0]
        x = self.input_proj(x)
        edge_index = torch.empty((2,0), dtype=torch.long, device=x.device)
        batch_idx = torch.arange(batch_size, device=x.device)
        for gcn in self.gcn_layers:
            x = F.relu(gcn(x, edge_index))
        x = global_mean_pool(x, batch_idx)
        x = self.layer_norm(x)
        x = self.dropout(x)
        return self.classifier(x)

class SAGEEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, dropout=0.0):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        for _ in range(num_layers-2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))
        self.dropout = dropout

    def forward(self, x, edge_index):
        for i in range(len(self.convs)):
            x = self.convs[i](x, edge_index)
            if i != len(self.convs)-1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x

class GraphSAGE_Model(nn.Module):
    def __init__(self, input_dim=320, hidden_dim=128, latent_dim=64, num_classes=2, num_layers=2):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.sage_encoder = SAGEEncoder(hidden_dim, hidden_dim, latent_dim, num_layers, 0.2)
        self.layer_norm = nn.LayerNorm(latent_dim)
        self.dropout = nn.Dropout(0.2)
        self.classifier = nn.Linear(latent_dim, num_classes)

    def forward(self, x):
        batch_size = x.shape[0]
        x = self.input_proj(x)
        edge_index = torch.empty((2,0), dtype=torch.long, device=x.device)
        batch_idx = torch.arange(batch_size, device=x.device)
        x = self.sage_encoder(x, edge_index)
        x = global_mean_pool(x, batch_idx)
        x = self.layer_norm(x)
        x = self.dropout(x)
        return self.classifier(x)

class AminoAcidTokenizer:
    def __init__(self, max_seq_len=256):
        self.vocab = {'<pad>':0, 'A':1, '<cls>':2, '<sep>':3, 'R':4, 'N':5, 'D':6, 'C':7, 'Q':8, 'E':9, 'G':10, 'H':11, 'I':12, 'L':13, 'K':14, 'M':15, 'F':16, 'P':17, 'S':18, 'T':19, 'W':20, 'Y':21, 'V':22}
        self.max_seq_len = max_seq_len
        self.pad_token_id = 0

    def batch_encode(self, sequences, padding='max_length', truncation=True, max_length=256):
        batch_ids = []
        for seq in sequences:
            seq = seq.strip().upper()
            ids = [self.vocab.get(c, self.pad_token_id) for c in seq]
            if truncation and len(ids) > max_length:
                ids = ids[:max_length]
            if padding == 'max_length':
                ids += [self.pad_token_id]*(max_length - len(ids))
            batch_ids.append(ids)
        return torch.tensor(batch_ids, dtype=torch.long)

class LightweightCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=128, max_seq_len=256):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.embed_dropout = nn.Dropout(0.1)
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(embed_dim, hidden_dim, 3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(hidden_dim, hidden_dim*2, 3, padding=1),
            nn.BatchNorm1d(hidden_dim*2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(hidden_dim*2, hidden_dim, 3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(1)
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 2)
        )

    def forward(self, input_ids):
        x = self.embedding(input_ids)  # [batch, seq_len, embed_dim]
        x = self.embed_dropout(x)
        x = x.permute(0, 2, 1)         # [batch, embed_dim, seq_len]
        x = self.feature_extractor(x).squeeze(-1)
        return self.classifier(x)

# ===================== 工具函数（优化逻辑表述） =====================
def load_csv_data(csv_path):
    """加载数据集，返回id、序列、标签（过滤无效样本）"""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"文件不存在: {csv_path}")
    df = pd.read_csv(csv_path)

    if "seq" in df.columns:
        seq_col = "seq"
    elif "sequence" in df.columns:
        seq_col = "sequence"
    else:
        raise ValueError("缺少列: ['seq' 或 'sequence']")

    if "label" not in df.columns:
        raise ValueError("缺少列: ['label']")

    if "id" not in df.columns:
        df = df.copy()
        df["id"] = np.arange(1, len(df) + 1)
    
    ids, seqs, labels = [], [], []
    for _, row in df.iterrows():
        seq = str(row[seq_col]).strip().upper()
        if len(seq) == 0:
            print(f"过滤空序列（id: {row['id']}）")
            continue
        try:
            label = int(row['label'])
            if label not in (0,1):
                raise ValueError("标签非0/1")
        except:
            print(f"过滤无效标签（id: {row['id']}，标签: {row['label']}）")
            continue
        ids.append(row['id'])
        seqs.append(seq)
        labels.append(label)
    
    total, pos = len(labels), sum(labels)
    if total == 0:
        raise ValueError(f"数据清洗后无有效样本: {csv_path}")
    ratio = pos / total * 100
    print(f"加载{os.path.basename(csv_path)}: 共{total}条有效序列（阳性{pos}条，占比{ratio:.2f}%）")
    return ids, seqs, labels

def load_four_models(feat_dim, device):
    """加载四个预训练的下游模型（仅用于生成预测概率）"""
    rnn_model_path = resolve_existing_path(rnn_model_candidates, "RNN模型权重")
    gnn_model_path = resolve_existing_path(gnn_model_candidates, "GNN模型权重")
    graphsage_model_path = resolve_existing_path(graphsage_model_candidates, "GraphSAGE模型权重")
    comdel_model_path = resolve_existing_path(comdel_model_candidates, "COMDEL模型权重")

    # 1. 加载RNN模型
    rnn_model = EnhancedRNNClassifier(input_dim=feat_dim).to(device)
    rnn_ckpt = torch.load(rnn_model_path, map_location=device, weights_only=False)
    rnn_model.load_state_dict(extract_state_dict(rnn_ckpt))
    rnn_model.eval()
    # 标准化器（无实际作用，仅兼容）
    rnn_scaler = StandardScaler()
    rnn_scaler.mean_ = np.zeros(feat_dim)
    rnn_scaler.scale_ = np.ones(feat_dim)

    # 2. 加载GNN模型
    gnn_model = GNN_EnhancedClassifier(input_dim=feat_dim).to(device)
    gnn_ckpt = torch.load(gnn_model_path, map_location=device, weights_only=False)
    gnn_model.load_state_dict(extract_state_dict(gnn_ckpt))
    gnn_model.eval()

    # 3. 加载GraphSAGE模型
    graphsage_model = GraphSAGE_Model(input_dim=feat_dim).to(device)
    graphsage_ckpt = torch.load(graphsage_model_path, map_location=device, weights_only=False)
    graphsage_model.load_state_dict(extract_state_dict(graphsage_ckpt))
    graphsage_model.eval()

    # 4. 加载COMDEL(CNN)模型
    comdel_tokenizer = AminoAcidTokenizer()
    comdel_model = LightweightCNN(vocab_size=len(comdel_tokenizer.vocab)).to(device)
    comdel_ckpt = torch.load(comdel_model_path, map_location=device, weights_only=False)
    comdel_model.load_state_dict(extract_state_dict(comdel_ckpt))
    comdel_model.eval()

    return rnn_model, rnn_scaler, gnn_model, graphsage_model, comdel_model, comdel_tokenizer

def generate_xgb_features(ids, seqs, labels, extractor, four_models, device):
    """
    生成XGBoost所需的输入特征（四个模型的阳性预测概率）
    :param ids: 样本ID列表
    :param seqs: 序列列表
    :param labels: 真实标签列表
    :param extractor: ESM2特征提取器
    :param four_models: 加载好的四个下游模型
    :param device: 运行设备
    :return: DataFrame（含ID、序列、真实标签、四个模型概率特征）
    """
    rnn_model, rnn_scaler, gnn_model, graphsage_model, comdel_model, comdel_tokenizer = four_models
    
    # 1. 提取ESM2 CLS特征（供RNN/GNN/GraphSAGE使用）
    esm_feats = extractor.get_cls_features(seqs, batch_size=32)
    esm_feats_tensor = torch.FloatTensor(esm_feats).to(device)

    # 2. 定义函数：获取单个模型的阳性预测概率
    def get_model_pos_prob(model, input_data, is_comdel=False, scaler=None):
        prob_list = []
        batch_size = 64
        model.eval()
        with torch.no_grad():
            if is_comdel:
                # COMDEL模型：输入是编码后的序列
                for i in range(0, len(input_data), batch_size):
                    batch_input = input_data[i:i+64].to(device)
                    logits = model(batch_input)
                    prob = F.softmax(logits, dim=1)[:, 1]  # 阳性概率
                    prob_list.extend(prob.cpu().numpy())
            else:
                # RNN/GNN/GraphSAGE：输入是ESM2特征
                batch_data = input_data
                if scaler is not None:
                    batch_data = torch.FloatTensor(scaler.transform(batch_data.cpu().numpy())).to(device)
                for i in range(0, len(batch_data), batch_size):
                    batch_input = batch_data[i:i+64]
                    logits = model(batch_input)
                    prob = F.softmax(logits, dim=1)[:, 1]  # 阳性概率
                    prob_list.extend(prob.cpu().numpy())
        return np.array(prob_list)

    # 3. 生成四个模型的阳性概率
    print("  - 生成RNN模型概率特征...")
    rnn_prob = get_model_pos_prob(rnn_model, esm_feats_tensor, scaler=rnn_scaler)
    print("  - 生成GNN模型概率特征...")
    gnn_prob = get_model_pos_prob(gnn_model, esm_feats_tensor)
    print("  - 生成GraphSAGE模型概率特征...")
    graphsage_prob = get_model_pos_prob(graphsage_model, esm_feats_tensor)
    print("  - 生成COMDEL模型概率特征...")
    comdel_input = comdel_tokenizer.batch_encode(seqs, max_length=256)
    comdel_prob = get_model_pos_prob(comdel_model, comdel_input, is_comdel=True)

    # 4. 构建特征DataFrame
    feat_df = pd.DataFrame({
        "id": ids,
        "sequence": seqs,
        "true_label": labels,
        "rnn_pos_prob": np.round(rnn_prob, 6),
        "gnn_pos_prob": np.round(gnn_prob, 6),
        "graphsage_pos_prob": np.round(graphsage_prob, 6),
        "comdel_pos_prob": np.round(comdel_prob, 6)
    })
    return feat_df

# ===================== 核心流程（逻辑清晰化） =====================
def main():
    """
    核心流程：
    1. 加载ESM2特征提取器 + 四个下游模型
    2. 对训练集：生成四个模型概率特征 → 训练XGBoost
    3. 对每个测试集：生成四个模型概率特征 → XGBoost预测 → 保存结果 + 评估
    """
    print(f"📌 开始XGBoost集成训练与预测 | 设备: {device}")

    # Step 1: 初始化ESM2特征提取器
    extractor = PeptideFeatureExtractor(esm2_backbone_path, device)
    finetuned_esm_path = resolve_existing_path(finetuned_esm_candidates, "ESM2微调权重")
    extractor.load_pretrained_model(finetuned_esm_path)

    # Step 2: 加载四个下游模型（用于生成概率特征）
    four_models = load_four_models(extractor.hidden_size, device)

    # Step 3: 处理训练集 → 生成特征 → 训练XGBoost
    print("\n========== 处理训练集（生成XGBoost输入特征 + 训练模型） ==========")
    train_ids, train_seqs, train_labels = load_csv_data(TRAIN_CSV)
    train_feat_df = generate_xgb_features(train_ids, train_seqs, train_labels, extractor, four_models, device)
    
    # 准备XGBoost训练数据
    feature_cols = ["rnn_pos_prob", "gnn_pos_prob", "graphsage_pos_prob", "comdel_pos_prob"]
    X_train = train_feat_df[feature_cols].values
    y_train = train_feat_df["true_label"].values
    if len(np.unique(y_train)) < 2:
        raise ValueError("训练集仅包含单一类别，无法训练二分类XGBoost模型")
    
    # 训练XGBoost
    print("\n训练XGBoost模型...")
    xgb_model = xgb.XGBClassifier(**XGB_PARAMS)
    xgb_model.fit(X_train, y_train)
    # 保存XGBoost模型
    xgb_model.save_model(XGB_MODEL_PATH)
    print(f"XGBoost模型已保存至: {XGB_MODEL_PATH}")

    # Step 4: 处理每个测试集 → 生成特征 → XGBoost预测 → 评估 + 保存
    for test_name, test_csv in TEST_CSVs.items():
        print(f"\n========== 处理测试集: {test_name} ==========")
        # 加载测试集数据
        test_ids, test_seqs, test_labels = load_csv_data(test_csv)
        # 生成XGBoost输入特征（四个模型的概率）
        test_feat_df = generate_xgb_features(test_ids, test_seqs, test_labels, extractor, four_models, device)
        # 准备测试集特征
        X_test = test_feat_df[feature_cols].values
        y_test = test_feat_df["true_label"].values
        
        # XGBoost预测
        y_pred_prob = xgb_model.predict_proba(X_test)[:, 1]  # 阳性概率
        y_pred = (y_pred_prob >= 0.5).astype(int)            # 预测标签（0/1）
        
        # 计算评估指标
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        try:
            auc = roc_auc_score(y_test, y_pred_prob)
        except:
            auc = 0.0  # 当标签只有一类时AUC无法计算
        
        # 构建最终结果
        result_df = test_feat_df.copy()
        result_df["xgboost_pos_prob"] = np.round(y_pred_prob, 6)
        result_df["xgboost_pred"] = y_pred
        result_df["final_result"] = ["阳性" if p==1 else "阴性" for p in y_pred]
        
        # 保存结果
        result_path = os.path.join(OUTPUT_DIR, f"{test_name}_xgboost_result.csv")
        result_df.to_csv(result_path, index=False, encoding="utf-8-sig")
        
        # 打印评估指标
        print(f"\n📊 测试集 {test_name} 评估指标：")
        print(f"准确率: {acc:.4f} | F1分数: {f1:.4f} | 精确率: {precision:.4f} | 召回率: {recall:.4f} | AUC: {auc:.4f}")
        print(f"测试集 {test_name} 结果已保存至: {result_path}")

    # 打印训练集上的表现（可选）
    print("\n========== 训练集XGBoost表现 ==========")
    y_train_pred = (xgb_model.predict_proba(X_train)[:, 1] >= 0.5).astype(int)
    train_acc = accuracy_score(y_train, y_train_pred)
    train_f1 = f1_score(y_train, y_train_pred, zero_division=0)
    print(f"训练集 - 准确率: {train_acc:.4f} | F1分数: {train_f1:.4f}")

    print("\n🎉 所有流程完成！结果保存在: xgboost_ensemble_result 目录")

if __name__ == "__main__":
    main()
