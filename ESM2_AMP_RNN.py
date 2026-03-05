import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score, roc_curve, \
    auc, precision_recall_curve, matthews_corrcoef, accuracy_score, roc_auc_score
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, AutoModel
import torch.serialization
from sklearn.preprocessing import LabelEncoder
import time

# 解决序列化时LabelEncoder的识别问题
torch.serialization.add_safe_globals([LabelEncoder])

# ===================== 路径配置区 =====================
ROOT_DIR = "./"
# 训练集与验证/测试集路径配置
train_csv = os.path.join(ROOT_DIR, "mydata/newdata/rawdata_train.csv")
val0_csv = os.path.join(ROOT_DIR, "mydata/newdata/rawdata_val.csv")
t0_csv = os.path.join(ROOT_DIR, "mydata/newdata/rawdata_test.csv")
t1_csv = os.path.join(ROOT_DIR, "test_data/bagel4_1_val.csv")
t2_csv = os.path.join(ROOT_DIR, "test_data/bagel4_2_val.csv")
t3_csv = os.path.join(ROOT_DIR, "test_data/bagel4_3_val.csv")

# ESM2预训练模型和微调ESM模型权重路径
esm2_backbone_path = os.path.join(ROOT_DIR, "model/esm2_t6_8M_UR50D")
finetuned_esm_path = os.path.join(ROOT_DIR, "model/model_AMP_ESM2_8M_256_v3/best_model_amp/esm2_8m_binary_best.pt")

# 预测结果输出路径配置
val0_output = os.path.join(ROOT_DIR, "results_ultra_light_rnn/val0_val_test_early_stop_predictions.csv")
t0_output = os.path.join(ROOT_DIR, "results_ultra_light_rnn/t0_rawdata_val_predictions.csv")
t1_output = os.path.join(ROOT_DIR, "results_ultra_light_rnn/t1_bagel4_1_val_predictions.csv")
t2_output = os.path.join(ROOT_DIR, "results_ultra_light_rnn/t2_bagel4_2_val_predictions.csv")
t3_output = os.path.join(ROOT_DIR, "results_ultra_light_rnn/t3_bagel4_3_val_predictions.csv")
metrics_summary_output = os.path.join(ROOT_DIR, "results_ultra_light_rnn/metrics_summary.csv")

# 设备配置：优先使用GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'


# ===================== 数据加载类 =====================
class PeptideDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels
        if len(self.sequences) != len(self.labels):
            raise ValueError("序列数量与标签数量必须一致")

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx], len(self.sequences[idx])


# ===================== 特征提取器类 =====================
class PeptideFeatureExtractor:
    def __init__(self, esm2_model_name_or_path, device=device):
        self.device = device
        self.esm2_model_name_or_path = esm2_model_name_or_path
        self.model = None
        self.tokenizer = None
        self.hidden_size = None
        self.total_params = 0

    def load_pretrained_model(self, finetuned_model_path=None):
        print(f"\n加载特征提取模型: {finetuned_model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.esm2_model_name_or_path)
        esm2_backbone = AutoModel.from_pretrained(self.esm2_model_name_or_path)
        self.hidden_size = esm2_backbone.config.hidden_size
        print(f"ESM2 8M模型隐藏层维度: {self.hidden_size}")

        # 定义特征提取模型结构，仅返回CLS特征
        class ESM2FeatureExtractor(nn.Module):
            def __init__(self, esm2_backbone):
                super().__init__()
                self.esm2_backbone = esm2_backbone
                self.hidden_dim = esm2_backbone.config.hidden_size

            def forward(self, input_ids, attention_mask):
                outputs = self.esm2_backbone(input_ids=input_ids, attention_mask=attention_mask)
                cls_repr = outputs.last_hidden_state[:, 0, :]
                seq_lengths = attention_mask.sum(dim=1, keepdim=True).float()
                cls_repr = cls_repr / torch.sqrt(seq_lengths + 1e-8)
                return cls_repr

        self.model = ESM2FeatureExtractor(esm2_backbone).to(self.device)
        if not os.path.exists(finetuned_model_path):
            raise FileNotFoundError(f"微调模型文件不存在: {finetuned_model_path}")
        checkpoint = torch.load(finetuned_model_path, map_location=self.device, weights_only=False)

        # 加载模型权重，仅匹配backbone部分
        feature_extractor_state_dict = {}
        for k, v in checkpoint['model_state_dict'].items():
            if k.startswith('esm2_backbone.'):
                feature_extractor_state_dict[k] = v
        self.model.load_state_dict(feature_extractor_state_dict, strict=False)

        # 冻结所有层，仅做特征提取不训练
        for param in self.model.parameters():
            param.requires_grad = False

        # 统计模型参数量
        self.total_params = sum(p.numel() for p in self.model.parameters())
        print(f"特征提取模型总参数量: {self.total_params:,}")
        print("特征提取模型加载完成！")

    def get_cls_features(self, sequences, batch_size=16):
        if not self.model: raise ValueError("请先加载特征提取模型")
        self.model.eval()
        features = []
        dataset = PeptideDataset(sequences, [0] * len(sequences))

        # 数据批处理函数，与训练阶段保持一致
        def collate_fn(batch):
            sequences, _, _ = zip(*batch)
            encoded = self.tokenizer(sequences, return_tensors=None, padding="do_not_pad", truncation=True,
                                     max_length=256)
            input_ids_list, attention_mask_list = encoded["input_ids"], encoded["attention_mask"]
            batch_max_len = max(len(ids) for ids in input_ids_list)
            pad_token_id = self.tokenizer.pad_token_id
            padded_input_ids = [ids + [pad_token_id] * (batch_max_len - len(ids)) for ids in input_ids_list]
            padded_attention_mask = [mask + [0] * (batch_max_len - len(mask)) for mask in attention_mask_list]
            return torch.tensor(padded_input_ids, dtype=torch.long), torch.tensor(padded_attention_mask,
                                                                                  dtype=torch.long)

        dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)
        print(f"开始提取CLS特征...")
        with torch.no_grad():
            for input_ids, attention_mask in tqdm(dataloader, total=len(dataloader)):
                input_ids, attention_mask = input_ids.to(self.device), attention_mask.to(self.device)
                cls_repr = self.model(input_ids=input_ids, attention_mask=attention_mask)
                features.extend(cls_repr.cpu().numpy())
        return np.array(features)

    def predict(self, sequences, batch_size=16):
        if not self.model: raise ValueError("请先加载特征提取模型")
        temp_predictor = PeptidePredictor(self.esm2_model_name_or_path, self.device)
        temp_predictor.load_pretrained_model(freeze_layers=True, finetuned_model_path=finetuned_esm_path)
        predictions, probabilities, inference_time = temp_predictor.predict(sequences, batch_size)
        return predictions, probabilities, inference_time


# ===================== ESM模型预测类（仅用于对比评估） =====================
class PeptidePredictor:
    def __init__(self, esm2_model_name_or_path, device=device):
        self.device = device
        self.esm2_model_name_or_path = esm2_model_name_or_path
        self.model = None
        self.tokenizer = None
        self.hidden_size = None

    def load_pretrained_model(self, freeze_layers=True, finetuned_model_path=None):
        print(f"\n加载ESM预测模型: {finetuned_model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.esm2_model_name_or_path)
        esm2_backbone = AutoModel.from_pretrained(self.esm2_model_name_or_path)
        self.hidden_size = esm2_backbone.config.hidden_size

        # 定义完整的分类模型结构
        class ESM2Classifier(nn.Module):
            def __init__(self, esm2_backbone):
                super().__init__()
                self.esm2_backbone = esm2_backbone
                self.hidden_dim = esm2_backbone.config.hidden_size
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

        if freeze_layers:
            for i, layer in enumerate(esm2_backbone.encoder.layer):
                for param in layer.parameters():
                    param.requires_grad = (i >= 4)
            for param in esm2_backbone.embeddings.parameters():
                param.requires_grad = False

        self.model = ESM2Classifier(esm2_backbone).to(self.device)
        checkpoint = torch.load(finetuned_model_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        for param in self.model.parameters():
            param.requires_grad = False

    def predict(self, sequences, batch_size=16):
        self.model.eval()
        predictions = []
        probabilities = []
        dataset = PeptideDataset(sequences, [0] * len(sequences))

        def collate_fn(batch):
            sequences, _, _ = zip(*batch)
            encoded = self.tokenizer(sequences, return_tensors=None, padding="do_not_pad", truncation=True,
                                     max_length=256)
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
                probs = torch.softmax(outputs, dim=1).cpu().numpy()
                predictions.extend(np.argmax(probs, axis=1))
                probabilities.extend(probs[:, 1])
        inference_time = time.time() - start_time
        return predictions, probabilities, inference_time


# ===================== 数据加载函数 =====================
def load_csv_data(csv_path):
    if not os.path.exists(csv_path): raise FileNotFoundError(f"CSV文件不存在: {csv_path}")
    df = pd.read_csv(csv_path)
    required_cols = ['id', 'seq', 'label']
    if not all(col in df.columns for col in required_cols):
        missing = [col for col in required_cols if col not in df.columns]
        raise ValueError(f"CSV文件缺少必要列: {missing}")

    ids, sequences, labels = [], [], []
    for _, row in df.iterrows():
        id_val, seq = row['id'], str(row['seq']).strip().upper()
        if len(seq) == 0:
            print(f"过滤空序列，ID: {id_val}")
            continue
        try:
            label = int(row['label'])
            if label not in (0, 1): raise ValueError(f"标签值非法")
        except Exception as e:
            print(f"过滤无效标签数据，ID: {id_val}，错误: {e}")
            continue
        ids.append(id_val)
        sequences.append(seq)
        labels.append(label)

    total, positive = len(labels), sum(labels)
    print(
        f"加载完成 {os.path.basename(csv_path)}：共{total}条序列，阳性{positive}条，阴性{total - positive}条，阳性占比{positive / total * 100:.2f}%")
    return ids, sequences, labels


# ===================== RNN模型架构-轻量化分类器 =====================
class EnhancedRNNClassifier(nn.Module):
    def __init__(self, input_dim=320, hidden_dim=128, num_classes=2, num_layers=2, bidirectional=True):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        # 双向RNN核心层
        self.rnn = nn.RNN(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=0.3 if num_layers > 1 else 0
        )

        # 层归一化+Dropout防止过拟合
        self.layer_norm = nn.LayerNorm(hidden_dim * self.num_directions)
        self.dropout = nn.Dropout(0.2)

        # 输出分类层
        self.classifier = nn.Linear(hidden_dim * self.num_directions, num_classes)

        # 权重初始化
        self._init_weights()

    def _init_weights(self):
        for name, param in self.rnn.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0.0)

        nn.init.ones_(self.layer_norm.weight)
        nn.init.zeros_(self.layer_norm.bias)
        nn.init.xavier_uniform_(self.classifier.weight)
        if self.classifier.bias is not None:
            nn.init.zeros_(self.classifier.bias)

    def forward(self, x):
        x = x.unsqueeze(1)
        rnn_out, hn = self.rnn(x)
        rnn_out = rnn_out.squeeze(1)
        out = self.layer_norm(rnn_out)
        out = self.dropout(out)
        logits = self.classifier(out)
        return logits


# ===================== RNN模型训练器 =====================
class StandardRNNTrainer:
    def __init__(self, feature_extractor, feature_extractor_total_params, device=device):
        self.device = device
        self.feature_extractor = feature_extractor
        self.feature_extractor_total_params = feature_extractor_total_params
        self.rnn_model = None
        self.optimizer = None
        self.train_metrics = {"total_loss": [], "accuracy": [], "f1": [], "recall": [], "mcc": []}
        self.val_metrics = {"accuracy": [], "f1": [], "recall": [], "mcc": [], "auc": []}
        self.esm_params_k = self.feature_extractor_total_params / 1000
        self.esm_size_mb = (self.feature_extractor_total_params * 4) / (1024 * 1024)
        self.rnn_params_k = 0
        self.rnn_size_mb = 0

    def init_rnn_model(self, input_dim=320, hidden_dim=128):
        self.rnn_model = EnhancedRNNClassifier(input_dim=input_dim, hidden_dim=hidden_dim, num_layers=2,
                                               bidirectional=True).to(self.device)
        total_params = sum(p.numel() for p in self.rnn_model.parameters())
        self.rnn_params_k = total_params / 1000
        print(f"\nRNN模型初始化完成:")
        print(f"输入维度: {input_dim} | RNN隐藏层维度: {hidden_dim} | 双向RNN: 是 | 层数: 2")
        print(f"RNN模型总参数量: {total_params:,} ({self.rnn_params_k:.2f}k)")
        return self.rnn_model

    def prepare_training_data(self, sequences, labels, batch_size=32, save_dir="results_ultra_light_rnn"):
        os.makedirs(save_dir, exist_ok=True)
        cache_path = os.path.join(save_dir, "training_data_no_distill.npz")
        if os.path.exists(cache_path):
            print(f"加载缓存训练数据: {cache_path}")
            data = np.load(cache_path)
            return {'features': data['features'], 'labels': data['labels'], 'sequences': sequences}

        features = self.feature_extractor.get_cls_features(sequences, batch_size=batch_size)
        labels_np = np.array(labels)
        np.savez_compressed(cache_path, features=features, labels=labels_np)
        print(f"训练特征保存完成，特征维度: {features.shape}")
        return {'features': features, 'labels': labels_np, 'sequences': sequences}

    def prepare_validation_data(self, val_sequences, val_labels, batch_size=32, save_dir="results_ultra_light_rnn"):
        os.makedirs(save_dir, exist_ok=True)
        cache_path = os.path.join(save_dir, "validation_data_no_distill.npz")
        if os.path.exists(cache_path):
            print(f"加载缓存验证数据: {cache_path}")
            data = np.load(cache_path)
            return {'val_features': data['val_features'], 'val_labels': data['val_labels']}

        val_features = self.feature_extractor.get_cls_features(val_sequences, batch_size=batch_size)
        val_labels_np = np.array(val_labels)
        np.savez_compressed(cache_path, val_features=val_features, val_labels=val_labels_np)
        print(f"验证特征保存完成，特征维度: {val_features.shape}")
        return {'val_features': val_features, 'val_labels': val_labels_np}

    def _compute_metrics(self, y_true, y_pred, y_prob):
        try:
            roc_auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.0
        except:
            roc_auc = 0.0
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

    def train_rnn(self, training_data, val1_data, save_dir="results_ultra_light_rnn", batch_size=512, epochs=50,
                  lr=1e-3, early_stop_patience=10):
        os.makedirs(save_dir, exist_ok=True)
        train_features = torch.FloatTensor(training_data['features']).to(self.device)
        train_labels = torch.LongTensor(training_data['labels']).to(self.device)
        val1_features = torch.FloatTensor(val1_data['val_features']).to(self.device)
        val1_labels = torch.LongTensor(val1_data['val_labels']).to(self.device)

        train_dataset = torch.utils.data.TensorDataset(train_features, train_labels)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val1_dataset = torch.utils.data.TensorDataset(val1_features, val1_labels)
        val1_loader = DataLoader(val1_dataset, batch_size=batch_size)

        self.optimizer = optim.AdamW(self.rnn_model.parameters(), lr=lr, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()
        best_val_f1, early_stop_count = 0.0, 0

        print(f"\n开始训练RNN模型（无蒸馏）")
        print(f"训练超参数: 批大小={batch_size}, 训练轮次={epochs}, 学习率={lr}, 早停耐心值={early_stop_patience}")
        start_time = time.time()

        for epoch in range(epochs):
            self.rnn_model.train()
            total_loss, correct, total = 0, 0, 0
            train_preds, train_true = [], []

            for batch_idx, (features, labels) in enumerate(
                    tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs} 训练中")):
                self.optimizer.zero_grad()
                logits = self.rnn_model(features)
                loss = criterion(logits, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.rnn_model.parameters(), max_norm=1.0)
                self.optimizer.step()

                total_loss += loss.item()
                _, predicted = logits.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                train_preds.extend(predicted.cpu().numpy())
                train_true.extend(labels.cpu().numpy())

            train_acc = 100. * correct / total
            avg_loss = total_loss / len(train_loader)
            train_metrics = self._compute_metrics(np.array(train_true), np.array(train_preds), None)
            self.train_metrics["total_loss"].append(avg_loss)
            self.train_metrics["accuracy"].append(train_acc)
            self.train_metrics["f1"].append(train_metrics['F1 Score'])
            self.train_metrics["recall"].append(train_metrics['Recall'])
            self.train_metrics["mcc"].append(train_metrics['MCC'])

            # 验证阶段
            self.rnn_model.eval()
            val1_preds, val1_true, val1_probs = [], [], []
            with torch.no_grad():
                for features, labels in val1_loader:
                    logits = self.rnn_model(features)
                    _, predicted = logits.max(1)
                    val1_preds.extend(predicted.cpu().numpy())
                    val1_true.extend(labels.cpu().numpy())
                    val1_probs.extend(F.softmax(logits, dim=1)[:, 1].cpu().numpy())

            val1_metrics = self._compute_metrics(np.array(val1_true), np.array(val1_preds), np.array(val1_probs))
            self.val_metrics["accuracy"].append(val1_metrics['Accuracy'] * 100)
            self.val_metrics["f1"].append(val1_metrics['F1 Score'])
            self.val_metrics["recall"].append(val1_metrics['Recall'])
            self.val_metrics["mcc"].append(val1_metrics['MCC'])
            self.val_metrics["auc"].append(val1_metrics['ROC-AUC'])

            print(
                f'Epoch {epoch + 1}: 训练准确率={train_acc:.2f}% | 训练F1={train_metrics["F1 Score"]:.4f} | 验证集F1={val1_metrics["F1 Score"]:.4f} | 验证集AUC={val1_metrics["ROC-AUC"]:.4f}')

            # 早停逻辑
            if val1_metrics['F1 Score'] > best_val_f1 + 1e-4:
                best_val_f1 = val1_metrics['F1 Score']
                early_stop_count = 0
                torch.save(
                    {'model_state_dict': self.rnn_model.state_dict(), 'epoch': epoch + 1, 'val_f1': best_val_f1},
                    f"{save_dir}/best_rnn_model_no_distill.pt")
            else:
                early_stop_count += 1
                if early_stop_count >= early_stop_patience:
                    print(f"触发早停，最佳验证集F1: {best_val_f1:.4f}")
                    break

        total_training_time = time.time() - start_time
        final_model_path = f"{save_dir}/final_rnn_model_no_distill.pt"
        torch.save({'model_state_dict': self.rnn_model.state_dict(), 'train_metrics': self.train_metrics,
                    'val_metrics': self.val_metrics}, final_model_path)
        if os.path.exists(final_model_path):
            self.rnn_size_mb = os.path.getsize(final_model_path) / (1024 * 1024)

        print(f"\nRNN模型训练完成，总耗时: {total_training_time:.2f}秒")
        print(f"最佳模型保存路径: {save_dir}/best_rnn_model_no_distill.pt")
        return self.rnn_model, total_training_time

    def evaluate_model_performance(self, sequences, labels, ids, batch_size=64, dataset_name="validation"):
        if self.rnn_model is None: raise ValueError("RNN模型未初始化")
        self.rnn_model.eval()

        print(f"\n=== 评估ESM模型性能 - {dataset_name} ===")
        esm_preds, esm_probs, esm_inference_time = self.feature_extractor.predict(sequences,
                                                                                  batch_size=batch_size)
        true_labels = np.array(labels)

        print(f"=== 提取{dataset_name}数据集CLS特征 ===")
        features = self.feature_extractor.get_cls_features(sequences, batch_size=batch_size)
        features = torch.FloatTensor(features).to(self.device)

        print(f"=== 评估RNN模型性能 - {dataset_name} ===")
        rnn_preds, rnn_probs = [], []
        rnn_start_time = time.time()
        with torch.no_grad():
            for i in tqdm(range(0, len(features), batch_size), desc=f"评估中 {dataset_name}"):
                batch_features = features[i:i + batch_size]
                batch_logits = self.rnn_model(batch_features)
                _, batch_preds = batch_logits.max(1)
                rnn_preds.extend(batch_preds.cpu().numpy())
                rnn_probs.extend(F.softmax(batch_logits, dim=1)[:, 1].cpu().numpy())
        rnn_inference_time = time.time() - rnn_start_time

        esm_metrics = self._compute_metrics(true_labels, np.array(esm_preds), np.array(esm_probs))
        rnn_metrics = self._compute_metrics(true_labels, np.array(rnn_preds), np.array(rnn_probs))

        esm_metrics['Inference Time (s)'] = esm_inference_time
        esm_metrics['Param Count (k)'] = self.esm_params_k
        esm_metrics['Model Size (MB)'] = self.esm_size_mb

        rnn_metrics['Inference Time (s)'] = rnn_inference_time
        rnn_metrics['Param Count (k)'] = self.rnn_params_k
        rnn_metrics['Model Size (MB)'] = self.rnn_size_mb

        return {
            'dataset_name': dataset_name, 'ids': ids, 'sequences': sequences, 'true_labels': true_labels,
            'esm_preds': np.array(esm_preds), 'esm_probs': np.array(esm_probs),
            'rnn_preds': np.array(rnn_preds), 'rnn_probs': np.array(rnn_probs),
            'esm_metrics': esm_metrics, 'rnn_metrics': rnn_metrics,
            'speedup_ratio': esm_inference_time / rnn_inference_time
        }

    def get_model_comparison(self, rnn_model_path):
        return {
            'esm_size_mb': self.esm_size_mb, 'rnn_size_mb': self.rnn_size_mb,
            'size_reduction': (self.esm_size_mb - self.rnn_size_mb) / self.esm_size_mb * 100,
            'esm_params': self.feature_extractor_total_params, 'rnn_params': self.rnn_params_k * 1000,
            'params_reduction': (self.feature_extractor_total_params - self.rnn_params_k * 1000) / self.feature_extractor_total_params * 100
        }


# ===================== 结果保存函数 =====================
def save_prediction_results(performance_dict, output_file):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    results = pd.DataFrame({
        "id": performance_dict['ids'],
        "sequence": performance_dict['sequences'],
        "sequence_length": [len(seq) for seq in performance_dict['sequences']],
        "true_label": performance_dict['true_labels'],
        "ESM_predicted_label": performance_dict['esm_preds'],
        "ESM_predicted_label_name": ["阳性（1）" if p == 1 else "阴性（0）" for p in performance_dict['esm_preds']],
        "ESM_positive_probability": performance_dict['esm_probs'].round(4),
        "RNN_predicted_label": performance_dict['rnn_preds'],
        "RNN_predicted_label_name": ["阳性（1）" if p == 1 else "阴性（0）" for p in performance_dict['rnn_preds']],
        "RNN_positive_probability": performance_dict['rnn_probs'].round(4)
    })
    results.to_csv(output_file, index=False, encoding="utf-8-sig")
    print(f"{performance_dict['dataset_name']} 预测结果保存至: {output_file}")


def save_metrics_summary(performance_list, output_file):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    summary_data = []
    for perf in performance_list:
        dataset_name = perf['dataset_name']
        esm_row = {'Dataset': dataset_name, 'Model Type': 'ESM Model', **perf['esm_metrics']}
        rnn_row = {'Dataset': dataset_name, 'Model Type': 'RNN Model (No Distill)', **perf['rnn_metrics']}
        summary_data.append(esm_row)
        summary_data.append(rnn_row)

    summary_df = pd.DataFrame(summary_data)
    column_order = ['Dataset', 'Model Type', 'Accuracy', 'F1 Score', 'Precision', 'Recall', 'MCC', 'ROC-AUC', 'PR-AUC',
                    'Inference Time (s)', 'Param Count (k)', 'Model Size (MB)']
    for col in column_order:
        if col not in summary_df.columns:
            summary_df[col] = 0.0
    summary_df = summary_df[column_order].round(4)
    summary_df.to_csv(output_file, index=False, encoding="utf-8-sig")
    print(f"指标汇总表保存至: {output_file}")


# ===================== 指标打印函数 =====================
def print_detailed_metrics(performance_list):
    print("\n" + "=" * 120)
    print("【详细指标汇总】ESM模型 & RNN轻量化模型（无蒸馏）")
    print("=" * 120)
    metrics_to_print = ['Accuracy', 'F1 Score', 'Precision', 'Recall', 'MCC', 'ROC-AUC', 'PR-AUC', 'Inference Time (s)',
                        'Param Count (k)', 'Model Size (MB)']

    for perf in performance_list:
        dataset_name = perf['dataset_name']
        print(f"\n📊 数据集: {dataset_name}")
        print("-" * 100)
        print("📌 ESM模型:")
        for metric in metrics_to_print:
            value = perf['esm_metrics'].get(metric, 0.0)
            if metric in ['Inference Time (s)']:
                print(f"   - {metric:<20}: {value:.4f}")
            elif metric in ['Param Count (k)', 'Model Size (MB)']:
                print(f"   - {metric:<20}: {value:.2f}")
            else:
                print(f"   - {metric:<20}: {value:.4f}")

        print("📌 RNN模型(无蒸馏):")
        for metric in metrics_to_print:
            value = perf['rnn_metrics'].get(metric, 0.0)
            if metric in ['Inference Time (s)']:
                print(f"   - {metric:<20}: {value:.4f}")
            elif metric in ['Param Count (k)', 'Model Size (MB)']:
                print(f"   - {metric:<20}: {value:.2f}")
            else:
                print(f"   - {metric:<20}: {value:.4f}")

        print(f"   - 推理提速比          : {perf['speedup_ratio']:.2f}x")
        print("-" * 100)


# ===================== 主训练流程 =====================
def train_rnn_no_distillation():
    print("=== 启动RNN轻量化模型训练流程（无知识蒸馏）===")
    print(f"运行设备: {device}")
    try:
        # 加载所有数据集
        print("\n===================== 加载数据集 =====================")
        train_ids, train_sequences, train_labels = load_csv_data(train_csv)
        val0_ids, val0_sequences, val0_labels = load_csv_data(val0_csv)
        t0_ids, t0_sequences, t0_labels = load_csv_data(t0_csv)
        t1_ids, t1_sequences, t1_labels = load_csv_data(t1_csv)
        t2_ids, t2_sequences, t2_labels = load_csv_data(t2_csv)
        t3_ids, t3_sequences, t3_labels = load_csv_data(t3_csv)

        print(f"\n数据集汇总:")
        print(f"训练集: {len(train_sequences)}条 | val0验证集: {len(val0_sequences)}条")
        print(f"t0测试集: {len(t0_sequences)}条 | t1测试集: {len(t1_sequences)}条")
        print(f"t2测试集: {len(t2_sequences)}条 | t3测试集: {len(t3_sequences)}条")

        # 加载特征提取模型
        print("\n===================== 加载特征提取模型 =====================")
        feature_extractor = PeptideFeatureExtractor(esm2_model_name_or_path=esm2_backbone_path, device=device)
        feature_extractor.load_pretrained_model(finetuned_model_path=finetuned_esm_path)

        # 初始化训练器和RNN模型
        print("\n===================== 初始化训练器 =====================")
        trainer = StandardRNNTrainer(feature_extractor=feature_extractor,
                                     feature_extractor_total_params=feature_extractor.total_params, device=device)
        trainer.init_rnn_model(input_dim=feature_extractor.hidden_size, hidden_dim=128)

        # 准备训练数据
        print("\n===================== 准备训练数据 =====================")
        training_data = trainer.prepare_training_data(sequences=train_sequences, labels=train_labels, batch_size=32,
                                                      save_dir="results_ultra_light_rnn")
        val1_data = trainer.prepare_validation_data(val_sequences=val0_sequences, val_labels=val0_labels, batch_size=32,
                                                    save_dir="results_ultra_light_rnn")

        # 训练RNN模型
        print("\n===================== 训练RNN模型 =====================")
        rnn_model, training_time = trainer.train_rnn(training_data=training_data, val1_data=val1_data,
                                                     save_dir="results_ultra_light_rnn", batch_size=512,
                                                     epochs=50, lr=1e-3, early_stop_patience=10)

        # 评估所有数据集
        print("\n===================== 评估所有数据集 =====================")
        train_perf = trainer.evaluate_model_performance(sequences=train_sequences, labels=train_labels, ids=train_ids,
                                                        batch_size=64, dataset_name="train_set")
        val0_perf = trainer.evaluate_model_performance(sequences=val0_sequences, labels=val0_labels, ids=val0_ids,
                                                       batch_size=64, dataset_name="val0_set")
        t0_perf = trainer.evaluate_model_performance(sequences=t0_sequences, labels=t0_labels, ids=t0_ids,
                                                     batch_size=64, dataset_name="t0_set")
        t1_perf = trainer.evaluate_model_performance(sequences=t1_sequences, labels=t1_labels, ids=t1_ids,
                                                     batch_size=64, dataset_name="t1_set")
        t2_perf = trainer.evaluate_model_performance(sequences=t2_sequences, labels=t2_labels, ids=t2_ids,
                                                     batch_size=64, dataset_name="t2_set")
        t3_perf = trainer.evaluate_model_performance(sequences=t3_sequences, labels=t3_labels, ids=t3_ids,
                                                     batch_size=64, dataset_name="t3_set")

        performance_list = [train_perf, val0_perf, t0_perf, t1_perf, t2_perf, t3_perf]

        # 模型轻量化对比
        print("\n===================== 模型轻量化对比 =====================")
        rnn_model_path = "results_ultra_light_rnn/final_rnn_model_no_distill.pt"
        model_comparison = trainer.get_model_comparison(rnn_model_path)

        # 打印汇总结果
        print("\n" + "=" * 100)
        print("【训练完成】RNN轻量化模型 结果汇总")
        print("=" * 100)
        print(f"\n1. 模型轻量化效果:")
        print(
            f"ESM模型: {model_comparison['esm_size_mb']:.2f}MB | {model_comparison['esm_params']:,} 参数")
        print(f"RNN模型: {model_comparison['rnn_size_mb']:.2f}MB | {model_comparison['rnn_params']:,} 参数")
        print(
            f"体积缩减: {model_comparison['size_reduction']:.1f}% | 参数量缩减: {model_comparison['params_reduction']:.1f}%")
        print(f"训练总耗时: {training_time:.2f}秒")

        print(f"\n2. RNN模型核心指标:")
        for perf in performance_list:
            ds_name = perf['dataset_name']
            r_metrics = perf['rnn_metrics']
            print(
                f"{ds_name}: F1={r_metrics['F1 Score']:.4f} | AUC={r_metrics['ROC-AUC']:.4f} | 推理提速={perf['speedup_ratio']:.1f}x")

        # 保存结果
        print("\n===================== 保存结果文件 =====================")
        save_prediction_results(val0_perf, val0_output)
        save_prediction_results(t0_perf, t0_output)
        save_prediction_results(t1_perf, t1_output)
        save_prediction_results(t2_perf, t2_output)
        save_prediction_results(t3_perf, t3_output)
        save_metrics_summary(performance_list, metrics_summary_output)

        # 打印详细指标
        print_detailed_metrics(performance_list)

        print("\n" + "=" * 100)
        print("=== 所有流程执行完成，结果已全部保存 ===")
        print(f"结果保存目录: results_ultra_light_rnn")
        print("=" * 100)

        return rnn_model, trainer, performance_list, model_comparison

    except Exception as e:
        print(f"\n程序执行异常: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None, None


# ===================== 程序入口 =====================
if __name__ == "__main__":
    rnn_model, trainer, performance_results, model_comparison = train_rnn_no_distillation()
    if rnn_model is None:
        print("\n训练流程执行失败！")
        exit(1)
    else:
        print("\n执行成功！")
        exit(0)