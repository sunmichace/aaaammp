# NOTE: comment translated to English.
import torch 
import torch .nn as nn 
import torch .nn .functional as F 
import torch .optim as optim 
from torch .utils .data import Dataset ,DataLoader 
from sklearn .metrics import (
classification_report ,confusion_matrix ,precision_score ,recall_score ,
f1_score ,roc_curve ,auc ,precision_recall_curve ,matthews_corrcoef ,accuracy_score ,roc_auc_score 
)
import numpy as np 
import pandas as pd 
import os 
from tqdm import tqdm 
import seaborn as sns 
import matplotlib .pyplot as plt 
from transformers import AutoTokenizer ,AutoModel 
import torch .serialization 
from sklearn .preprocessing import LabelEncoder 
import time 
# NOTE: comment translated to English.
from torch_geometric .nn import SAGEConv ,global_mean_pool 
from torch_geometric .data import Batch ,Data 

torch .serialization .add_safe_globals ([LabelEncoder ])


# NOTE: comment translated to English.
class SAGEEncoder (nn .Module ):
    """Docstring translated to English."""
    def __init__ (self ,in_channels ,hidden_channels ,out_channels ,num_layers =2 ,dropout =0.0 ):
        super ().__init__ ()
        self .num_layers =num_layers # NOTE: comment translated to English.
        self .dropout =dropout # NOTE: comment translated to English.

        # NOTE: comment translated to English.
        self .convs =nn .ModuleList ()
        self .convs .append (SAGEConv (in_channels ,hidden_channels ))# NOTE: comment translated to English.
        for _ in range (num_layers -2 ):# NOTE: comment translated to English.
            self .convs .append (SAGEConv (hidden_channels ,hidden_channels ))
        self .convs .append (SAGEConv (hidden_channels ,out_channels ))# NOTE: comment translated to English.

    def forward (self ,x ,edge_index ):
        """Docstring translated to English."""
        for i in range (self .num_layers ):
            x =self .convs [i ](x ,edge_index )
            # NOTE: comment translated to English.
            if i !=self .num_layers -1 :
                x =F .relu (x )
                x =F .dropout (x ,p =self .dropout ,training =self .training )
        return x 

        # NOTE: comment translated to English.
ROOT_DIR ="./"
# NOTE: comment translated to English.
train_csv =os .path .join (ROOT_DIR ,"mydata/newdata/rawdata_train.csv")
val0_csv =os .path .join (ROOT_DIR ,"mydata/newdata/rawdata_val.csv")
t0_csv =os .path .join (ROOT_DIR ,"mydata/newdata/rawdata_test.csv")
t1_csv =os .path .join (ROOT_DIR ,"test_data/bagel4_1_val.csv")
t2_csv =os .path .join (ROOT_DIR ,"test_data/bagel4_2_val.csv")
t3_csv =os .path .join (ROOT_DIR ,"test_data/bagel4_3_val.csv")

# NOTE: comment translated to English.
esm2_backbone_path =os .path .join (ROOT_DIR ,"model/esm2_t6_8M_UR50D")
finetuned_esm_candidates =[
os .path .join (ROOT_DIR ,"model/model_AMP_ESM2_8M_256_v3/best_model_amp/esm2_8m_binary_best.pt"),
os .path .join (ROOT_DIR ,"model_AMP_ESM2_8M_256_v3/best_model_amp/esm2_8m_binary_best.pt"),
]

# NOTE: comment translated to English.
val0_output =os .path .join (ROOT_DIR ,"results_ultra_light_graphsage/val0_val_test_early_stop_predictions.csv")
t0_output =os .path .join (ROOT_DIR ,"results_ultra_light_graphsage/t0_rawdata_test_predictions.csv")
t1_output =os .path .join (ROOT_DIR ,"results_ultra_light_graphsage/t1_bagel4_1_val_predictions.csv")
t2_output =os .path .join (ROOT_DIR ,"results_ultra_light_graphsage/t2_bagel4_2_val_predictions.csv")
t3_output =os .path .join (ROOT_DIR ,"results_ultra_light_graphsage/t3_bagel4_3_val_predictions.csv")
metrics_summary_output =os .path .join (ROOT_DIR ,"results_ultra_light_graphsage/metrics_summary.csv")
cumulative_curve_dir =os .path .join (ROOT_DIR ,"results_ultra_light_graphsage/cumulative_curves")

# NOTE: comment translated to English.
device ='cuda'if torch .cuda .is_available ()else 'cpu'

# NOTE: comment translated to English.
def resolve_existing_path (path_candidates ,file_desc ):
    """Docstring translated to English."""
    for candidate in path_candidates :
        if os .path .exists (candidate ):
            return candidate 
    joined ="\n".join (path_candidates )
    raise FileNotFoundError (f"{file_desc }不存在，已尝试以下路径:\n{joined }")


def extract_state_dict (checkpoint_obj ):
    """Docstring translated to English."""
    if isinstance (checkpoint_obj ,dict )and "model_state_dict"in checkpoint_obj :
        return checkpoint_obj ["model_state_dict"]
    if isinstance (checkpoint_obj ,dict )and "state_dict"in checkpoint_obj :
        return checkpoint_obj ["state_dict"]
    if isinstance (checkpoint_obj ,dict ):
        tensor_keys =[k for k ,v in checkpoint_obj .items ()if isinstance (v ,torch .Tensor )]
        if tensor_keys :
            return {k :checkpoint_obj [k ]for k in tensor_keys }
    raise ValueError ("无法从checkpoint中解析state_dict")


def extract_esm_backbone_state_dict (checkpoint_obj ):
    """Docstring translated to English."""
    raw_state =extract_state_dict (checkpoint_obj )
    backbone_state ={}
    for k ,v in raw_state .items ():
        if k .startswith ("esm2_backbone."):
            backbone_state [k ]=v 
        elif k .startswith ("backbone."):
            mapped_key =k .replace ("backbone.","esm2_backbone.",1 )
            backbone_state [mapped_key ]=v 
    if not backbone_state :
        raise ValueError ("checkpoint中未找到ESM2 backbone权重")
    return backbone_state 


    # NOTE: comment translated to English.
def plot_cumulative_curves (graphsage_probs ,dataset_name ,save_dir ):
    """Docstring translated to English."""
    os .makedirs (save_dir ,exist_ok =True )
    probs =np .asarray (graphsage_probs ,dtype =float )
    probs =probs [~np .isnan (probs )]
    if probs .size ==0 :
        print (f"数据集 {dataset_name } 无有效概率值，跳过累积曲线绘制。")
        return 

    probs_sorted =np .sort (probs )
    cum_percent =np .arange (1 ,len (probs_sorted )+1 )/len (probs_sorted )*100 

    plt .figure (figsize =(8 ,6 ))
    plt .plot (probs_sorted ,cum_percent ,lw =2 ,label =f"{dataset_name } (n={len (probs_sorted )})")
    plt .xlim (0.0 ,1.0 )
    plt .ylim (0.0 ,100.0 )
    plt .xlabel ("Predicted Positive Probability")
    plt .ylabel ("Cumulative Percentage (%)")
    plt .title (f"GraphSAGE Cumulative Probability Curve - {dataset_name }")
    plt .grid (alpha =0.3 )
    plt .legend (loc ="lower right")
    plt .tight_layout ()

    safe_name =str (dataset_name ).replace ("/","_").replace (" ","_")
    output_path =os .path .join (save_dir ,f"{safe_name }_cumulative_curve.pdf")
    plt .savefig (output_path ,dpi =300 )
    plt .close ()
    print (f"累积曲线已保存: {output_path }")

    # NOTE: comment translated to English.
class PeptideDataset (Dataset ):
    """Docstring translated to English."""
    def __init__ (self ,sequences ,labels ):
        self .sequences =sequences # NOTE: comment translated to English.
        self .labels =labels # NOTE: comment translated to English.
        # NOTE: comment translated to English.
        if len (self .sequences )!=len (self .labels ):
            raise ValueError ("序列数量与标签数量不匹配")

    def __len__ (self ):
        """Docstring translated to English."""
        return len (self .sequences )

    def __getitem__ (self ,idx ):
        """Docstring translated to English."""
        return self .sequences [idx ],self .labels [idx ],len (self .sequences [idx ])

        # NOTE: comment translated to English.
class PeptideFeatureExtractor :
    """Docstring translated to English."""
    def __init__ (self ,esm2_model_name_or_path ,device =device ):
        self .device =device # NOTE: comment translated to English.
        self .esm2_model_name_or_path =esm2_model_name_or_path # NOTE: comment translated to English.
        self .model =None # NOTE: comment translated to English.
        self .tokenizer =None # NOTE: comment translated to English.
        self .hidden_size =None # NOTE: comment translated to English.
        self .total_params =0 # NOTE: comment translated to English.

    def load_pretrained_model (self ,finetuned_model_path =None ):
        """Docstring translated to English."""
        if finetuned_model_path is None :
            finetuned_model_path =resolve_existing_path (finetuned_esm_candidates ,"ESM2微调权重")
        print (f"\n正在加载特征提取模型: {finetuned_model_path }")
        try :
        # NOTE: comment translated to English.
            self .tokenizer =AutoTokenizer .from_pretrained (self .esm2_model_name_or_path )
            # NOTE: comment translated to English.
            esm2_backbone =AutoModel .from_pretrained (self .esm2_model_name_or_path )
            # NOTE: comment translated to English.
            self .hidden_size =esm2_backbone .config .hidden_size 
            print (f"ESM2 8M隐藏维度: {self .hidden_size }")

            # NOTE: comment translated to English.
            class ESM2FeatureExtractor (nn .Module ):
                def __init__ (self ,esm2_backbone ):
                    super ().__init__ ()
                    self .esm2_backbone =esm2_backbone 
                    self .hidden_dim =esm2_backbone .config .hidden_size 

                def forward (self ,input_ids ,attention_mask ):
                # NOTE: comment translated to English.
                    outputs =self .esm2_backbone (input_ids =input_ids ,attention_mask =attention_mask )
                    # NOTE: comment translated to English.
                    cls_repr =outputs .last_hidden_state [:,0 ,:]
                    # NOTE: comment translated to English.
                    seq_lengths =attention_mask .sum (dim =1 ,keepdim =True ).float ()
                    cls_repr =cls_repr /torch .sqrt (seq_lengths +1e-8 )
                    return cls_repr 

                    # NOTE: comment translated to English.
            self .model =ESM2FeatureExtractor (esm2_backbone ).to (self .device )
            # NOTE: comment translated to English.
            checkpoint =torch .load (finetuned_model_path ,map_location =self .device ,weights_only =False )

            # NOTE: comment translated to English.
            feature_extractor_state_dict =extract_esm_backbone_state_dict (checkpoint )
            self .model .load_state_dict (feature_extractor_state_dict ,strict =False )

            # NOTE: comment translated to English.
            for param in self .model .parameters ():
                param .requires_grad =False 
            print ("已冻结特征提取模型所有层")

            # NOTE: comment translated to English.
            self .total_params =sum (p .numel ()for p in self .model .parameters ())
            print (f"特征提取模型总参数量: {self .total_params :,}")
            print ("特征提取模型加载完成！")
        except Exception as e :
            print (f"模型加载失败: {e }")
            raise 

    def get_cls_features (self ,sequences ,batch_size =16 ):
        """Docstring translated to English."""
        if not self .model :raise ValueError ("请先加载特征提取模型")
        self .model .eval ()# NOTE: comment translated to English.
        features =[]
        # NOTE: comment translated to English.
        dataset =PeptideDataset (sequences ,[0 ]*len (sequences ))

        # NOTE: comment translated to English.
        def collate_fn (batch ):
            sequences ,_ ,_ =zip (*batch )
            # NOTE: comment translated to English.
            encoded =self .tokenizer (
            sequences ,return_tensors =None ,padding ="do_not_pad",truncation =True ,max_length =256 
            )
            input_ids_list ,attention_mask_list =encoded ["input_ids"],encoded ["attention_mask"]
            # NOTE: comment translated to English.
            batch_max_len =max (len (ids )for ids in input_ids_list )
            pad_token_id =self .tokenizer .pad_token_id 
            # NOTE: comment translated to English.
            padded_input_ids =[ids +[pad_token_id ]*(batch_max_len -len (ids ))for ids in input_ids_list ]
            padded_attention_mask =[mask +[0 ]*(batch_max_len -len (mask ))for mask in attention_mask_list ]
            return torch .tensor (padded_input_ids ,dtype =torch .long ),torch .tensor (padded_attention_mask ,dtype =torch .long )

            # NOTE: comment translated to English.
        dataloader =DataLoader (dataset ,batch_size =batch_size ,collate_fn =collate_fn )
        print (f"提取CLS特征...")
        # NOTE: comment translated to English.
        with torch .no_grad ():
            for input_ids ,attention_mask in tqdm (dataloader ,total =len (dataloader )):
                input_ids ,attention_mask =input_ids .to (self .device ),attention_mask .to (self .device )
                cls_repr =self .model (input_ids =input_ids ,attention_mask =attention_mask )
                features .extend (cls_repr .cpu ().numpy ())
        return np .array (features )

    def predict (self ,sequences ,batch_size =16 ):
        """Docstring translated to English."""
        if not self .model :raise ValueError ("请先加载特征提取模型")
        self .model .eval ()
        # NOTE: comment translated to English.
        temp_predictor =PeptidePredictor (self .esm2_model_name_or_path ,self .device )
        finetuned_esm_path =resolve_existing_path (finetuned_esm_candidates ,"ESM2微调权重")
        temp_predictor .load_pretrained_model (freeze_layers =True ,finetuned_model_path =finetuned_esm_path )
        predictions ,probabilities ,inference_time =temp_predictor .predict (sequences ,batch_size )
        return predictions ,probabilities ,inference_time 

        # NOTE: comment translated to English.
class PeptidePredictor :
    """Docstring translated to English."""
    def __init__ (self ,esm2_model_name_or_path ,device =device ):
        self .device =device 
        self .esm2_model_name_or_path =esm2_model_name_or_path 
        self .model =None 
        self .tokenizer =None 
        self .hidden_size =None 

    def load_pretrained_model (self ,freeze_layers =True ,finetuned_model_path =None ):
        """Docstring translated to English."""
        if finetuned_model_path is None :
            finetuned_model_path =resolve_existing_path (finetuned_esm_candidates ,"ESM2微调权重")
        print (f"\n加载临时预测模型: {finetuned_model_path }")
        self .tokenizer =AutoTokenizer .from_pretrained (self .esm2_model_name_or_path )
        esm2_backbone =AutoModel .from_pretrained (self .esm2_model_name_or_path )
        self .hidden_size =esm2_backbone .config .hidden_size 

        # NOTE: comment translated to English.
        class ESM2Classifier (nn .Module ):
            def __init__ (self ,esm2_backbone ):
                super ().__init__ ()
                self .esm2_backbone =esm2_backbone 
                self .hidden_dim =esm2_backbone .config .hidden_size 
                # NOTE: comment translated to English.
                self .classifier =nn .Sequential (
                nn .Linear (self .hidden_dim ,512 ),
                nn .ReLU (),
                nn .Dropout (0.6 ),
                nn .Linear (512 ,256 ),
                nn .ReLU (),
                nn .Dropout (0.6 ),
                nn .Linear (256 ,2 )
                )

            def forward (self ,input_ids ,attention_mask ):
                outputs =self .esm2_backbone (input_ids =input_ids ,attention_mask =attention_mask )
                cls_repr =outputs .last_hidden_state [:,0 ,:]
                seq_lengths =attention_mask .sum (dim =1 ,keepdim =True ).float ()
                cls_repr =cls_repr /torch .sqrt (seq_lengths +1e-8 )
                return self .classifier (cls_repr )

                # NOTE: comment translated to English.
        if freeze_layers :
            for i ,layer in enumerate (esm2_backbone .encoder .layer ):
                for param in layer .parameters ():
                    param .requires_grad =(i >=4 )
            for param in esm2_backbone .embeddings .parameters ():
                param .requires_grad =False 

                # NOTE: comment translated to English.
        self .model =ESM2Classifier (esm2_backbone ).to (self .device )
        checkpoint =torch .load (finetuned_model_path ,map_location =self .device ,weights_only =False )
        self .model .load_state_dict (extract_state_dict (checkpoint ))
        for param in self .model .parameters ():
            param .requires_grad =False 

    def predict (self ,sequences ,batch_size =16 ):
        """Docstring translated to English."""
        self .model .eval ()
        predictions =[]
        probabilities =[]
        dataset =PeptideDataset (sequences ,[0 ]*len (sequences ))

        # NOTE: comment translated to English.
        def collate_fn (batch ):
            sequences ,_ ,_ =zip (*batch )
            encoded =self .tokenizer (
            sequences ,return_tensors =None ,padding ="do_not_pad",truncation =True ,max_length =256 
            )
            input_ids_list ,attention_mask_list =encoded ["input_ids"],encoded ["attention_mask"]
            batch_max_len =max (len (ids )for ids in input_ids_list )
            pad_token_id =self .tokenizer .pad_token_id 
            padded_input_ids =[ids +[pad_token_id ]*(batch_max_len -len (ids ))for ids in input_ids_list ]
            padded_attention_mask =[mask +[0 ]*(batch_max_len -len (mask ))for mask in attention_mask_list ]
            return torch .tensor (padded_input_ids ,dtype =torch .long ),torch .tensor (padded_attention_mask ,dtype =torch .long )

        dataloader =DataLoader (dataset ,batch_size =batch_size ,collate_fn =collate_fn )
        start_time =time .time ()
        with torch .no_grad ():
            for input_ids ,attention_mask in tqdm (dataloader ,total =len (dataloader )):
                input_ids ,attention_mask =input_ids .to (self .device ),attention_mask .to (self .device )
                outputs =self .model (input_ids =input_ids ,attention_mask =attention_mask )
                # NOTE: comment translated to English.
                probs =torch .softmax (outputs ,dim =1 ).cpu ().numpy ()
                predictions .extend (np .argmax (probs ,axis =1 ))
                probabilities .extend (probs [:,1 ])
        inference_time =time .time ()-start_time 
        return predictions ,probabilities ,inference_time 

        # NOTE: comment translated to English.
def load_csv_data (csv_path ):
    """Docstring translated to English."""
    if not os .path .exists (csv_path ):
        raise FileNotFoundError (f"CSV文件不存在: {csv_path }")
    try :
        df =pd .read_csv (csv_path )
    except Exception as e :
        raise IOError (f"读取CSV失败: {str (e )}")

    if "seq"in df .columns :
        seq_col ="seq"
    elif "sequence"in df .columns :
        seq_col ="sequence"
    else :
        raise ValueError ("CSV缺少必要列: ['seq' 或 'sequence']")
    if "label"not in df .columns :
        raise ValueError ("CSV缺少必要列: ['label']")
    if "id"not in df .columns :
        df =df .copy ()
        df ["id"]=np .arange (1 ,len (df )+1 )

    ids ,sequences ,labels =[],[],[]
    # NOTE: comment translated to English.
    for _ ,row in df .iterrows ():
        id_val ,seq =row ["id"],str (row [seq_col ]).strip ().upper ()
        # NOTE: comment translated to English.
        if len (seq )==0 :
            print (f"过滤空序列（id: {id_val }）")
            continue 
            # NOTE: comment translated to English.
        try :
            label =int (row ["label"])
            if label not in (0 ,1 ):
                raise ValueError ("标签必须为0或1")
        except Exception as e :
            print (f"过滤无效标签（{e }）: {row ['label']}（id: {id_val }）")
            continue 
        ids .append (id_val )
        sequences .append (seq )
        labels .append (label )
        # NOTE: comment translated to English.
    total ,positive =len (labels ),sum (labels )
    if total ==0 :
        raise ValueError (f"数据清洗后无有效样本: {csv_path }")
    ratio =positive /total *100 
    print (f"加载完成 {os .path .basename (csv_path )}：共{total }条有效序列（阳性{positive }条，阴性{total -positive }条，阳性占比{ratio :.2f}%）")
    return ids ,sequences ,labels 

    # NOTE: comment translated to English.
class EnhancedGraphSAGEClassifier (nn .Module ):
    """Docstring translated to English."""
    def __init__ (self ,input_dim =320 ,hidden_dim =128 ,latent_dim =64 ,num_classes =2 ,num_layers =2 ):
        super ().__init__ ()
        self .input_dim =input_dim # NOTE: comment translated to English.
        self .hidden_dim =hidden_dim # NOTE: comment translated to English.
        self .latent_dim =latent_dim # NOTE: comment translated to English.
        self .num_layers =num_layers # NOTE: comment translated to English.

        # NOTE: comment translated to English.
        self .input_proj =nn .Linear (input_dim ,hidden_dim )

        # NOTE: comment translated to English.
        self .sage_encoder =SAGEEncoder (
        in_channels =hidden_dim ,
        hidden_channels =hidden_dim ,
        out_channels =latent_dim ,
        num_layers =num_layers ,
        dropout =0.2 
        )

        # NOTE: comment translated to English.
        self .layer_norm =nn .LayerNorm (latent_dim )
        self .dropout =nn .Dropout (0.2 )

        # NOTE: comment translated to English.
        self .classifier =nn .Linear (latent_dim ,num_classes )

        # NOTE: comment translated to English.
        self ._init_weights ()

    def _init_weights (self ):
        """Docstring translated to English."""
        nn .init .xavier_uniform_ (self .input_proj .weight )
        nn .init .zeros_ (self .input_proj .bias )
        nn .init .xavier_uniform_ (self .classifier .weight )
        nn .init .zeros_ (self .classifier .bias )
        nn .init .ones_ (self .layer_norm .weight )
        nn .init .zeros_ (self .layer_norm .bias )

    def forward (self ,x ):
        """Docstring translated to English."""
        batch_size =x .shape [0 ]

        # NOTE: comment translated to English.
        x_proj =self .input_proj (x )

        # NOTE: comment translated to English.
        node_feat =x_proj 
        edge_index =torch .empty ((2 ,0 ),dtype =torch .long ,device =x .device )
        batch_idx =torch .arange (batch_size ,device =x .device ,dtype =torch .long )

        # NOTE: comment translated to English.
        z =self .sage_encoder (node_feat ,edge_index )

        # NOTE: comment translated to English.
        global_feat =global_mean_pool (z ,batch_idx )

        # NOTE: comment translated to English.
        out =self .layer_norm (global_feat )
        out =self .dropout (out )
        logits =self .classifier (out )
        return logits 

        # NOTE: comment translated to English.
class GraphSAGETrainer :
    """Docstring translated to English."""
    def __init__ (self ,feature_extractor ,feature_extractor_total_params ,device =device ):
        self .device =device # NOTE: comment translated to English.
        self .feature_extractor =feature_extractor # NOTE: comment translated to English.
        self .feature_extractor_total_params =feature_extractor_total_params # NOTE: comment translated to English.
        self .graphsage_model =None # NOTE: comment translated to English.
        self .optimizer =None # NOTE: comment translated to English.

        # NOTE: comment translated to English.
        self .train_metrics ={"total_loss":[],"accuracy":[],"f1":[],"recall":[],"mcc":[]}
        self .val_metrics ={"accuracy":[],"f1":[],"recall":[],"mcc":[],"auc":[]}

        # NOTE: comment translated to English.
        self .esm_params_k =self .feature_extractor_total_params /1000 
        self .esm_size_mb =(self .feature_extractor_total_params *4 )/(1024 *1024 )
        self .graphsage_params_k =0 
        self .graphsage_size_mb =0 

    def init_graphsage_model (self ,input_dim =320 ,hidden_dim =128 ,latent_dim =64 ):
        """Docstring translated to English."""
        self .graphsage_model =EnhancedGraphSAGEClassifier (
        input_dim =input_dim ,
        hidden_dim =hidden_dim ,
        latent_dim =latent_dim ,
        num_layers =2 
        ).to (self .device )
        total_params =sum (p .numel ()for p in self .graphsage_model .parameters ())
        self .graphsage_params_k =total_params /1000 
        print (f"\nGraphSAGE模型初始化完成:")
        print (f"   - 输入维度: {input_dim } | GraphSAGE隐藏层维度: {hidden_dim } | 输出维度: {latent_dim }")
        print (f"   - GraphSAGE编码器层数: 2")
        print (f"   - GraphSAGE模型总参数量: {total_params :,} ({self .graphsage_params_k :.2f}k)")
        return self .graphsage_model 

    def prepare_training_data (self ,sequences ,labels ,batch_size =32 ,save_dir ="results_ultra_light_graphsage"):
        """Docstring translated to English."""
        os .makedirs (save_dir ,exist_ok =True )
        cache_path =os .path .join (save_dir ,"training_data_no_distill.npz")

        if os .path .exists (cache_path ):
            print (f"加载缓存的训练数据: {cache_path }")
            data =np .load (cache_path )
            return {
            'features':data ['features'],
            'labels':data ['labels'],
            'sequences':sequences 
            }

            # NOTE: comment translated to English.
        features =self .feature_extractor .get_cls_features (sequences ,batch_size =batch_size )
        labels_np =np .array (labels )
        np .savez_compressed (cache_path ,features =features ,labels =labels_np )
        print (f"训练特征已保存，维度: {features .shape }")
        return {'features':features ,'labels':labels_np ,'sequences':sequences }

    def prepare_validation_data (self ,val_sequences ,val_labels ,batch_size =32 ,save_dir ="results_ultra_light_graphsage"):
        """Docstring translated to English."""
        os .makedirs (save_dir ,exist_ok =True )
        cache_path =os .path .join (save_dir ,f"validation_data_no_distill.npz")

        if os .path .exists (cache_path ):
            print (f"加载缓存的验证数据: {cache_path }")
            data =np .load (cache_path )
            return {'val_features':data ['val_features'],'val_labels':data ['val_labels']}

        val_features =self .feature_extractor .get_cls_features (val_sequences ,batch_size =batch_size )
        val_labels_np =np .array (val_labels )
        np .savez_compressed (cache_path ,val_features =val_features ,val_labels =val_labels_np )
        print (f"验证集特征已保存，维度: {val_features .shape }")
        return {'val_features':val_features ,'val_labels':val_labels_np }

    def _compute_metrics (self ,y_true ,y_pred ,y_prob ):
        """Docstring translated to English."""
        # NOTE: comment translated to English.
        try :
            roc_auc =roc_auc_score (y_true ,y_prob )if len (np .unique (y_true ))>1 else 0.0 
        except :
            roc_auc =0.0 

            # NOTE: comment translated to English.
        try :
            precision_curve ,recall_curve ,_ =precision_recall_curve (y_true ,y_prob )
            pr_auc =auc (recall_curve ,precision_curve )
        except :
            pr_auc =0.0 

        return {
        'Accuracy':accuracy_score (y_true ,y_pred ),
        'F1 Score':f1_score (y_true ,y_pred ,zero_division =0 ),
        'Precision':precision_score (y_true ,y_pred ,zero_division =0 ),
        'Recall':recall_score (y_true ,y_pred ,zero_division =0 ),
        'MCC':matthews_corrcoef (y_true ,y_pred ),
        'ROC-AUC':roc_auc ,
        'PR-AUC':pr_auc 
        }

    def train_graphsage (self ,training_data ,val0_data ,save_dir ="results_ultra_light_graphsage",
    batch_size =512 ,epochs =50 ,lr =1e-3 ,early_stop_patience =10 ):
        """Docstring translated to English."""
        os .makedirs (save_dir ,exist_ok =True )

        # NOTE: comment translated to English.
        train_features =torch .FloatTensor (training_data ['features']).to (self .device )
        train_labels =torch .LongTensor (training_data ['labels']).to (self .device )
        val0_features =torch .FloatTensor (val0_data ['val_features']).to (self .device )
        val0_labels =torch .LongTensor (val0_data ['val_labels']).to (self .device )

        # NOTE: comment translated to English.
        train_dataset =torch .utils .data .TensorDataset (train_features ,train_labels )
        train_loader =DataLoader (train_dataset ,batch_size =batch_size ,shuffle =True )
        val0_dataset =torch .utils .data .TensorDataset (val0_features ,val0_labels )
        val0_loader =DataLoader (val0_dataset ,batch_size =batch_size )

        # NOTE: comment translated to English.
        self .optimizer =optim .AdamW (self .graphsage_model .parameters (),lr =lr ,weight_decay =1e-4 )
        criterion =nn .CrossEntropyLoss ()
        best_val_f1 ,early_stop_count =-1.0 ,0 # NOTE: comment translated to English.

        print (f"\n开始训练GraphSAGE模型（无蒸馏）...")
        print (f"超参数配置: 批大小={batch_size }, 轮次={epochs }, 学习率={lr }")
        print (f"早停验证集: val_test.csv (val0)")
        start_time =time .time ()

        # NOTE: comment translated to English.
        for epoch in range (epochs ):
            self .graphsage_model .train ()# NOTE: comment translated to English.
            total_loss ,correct ,total =0 ,0 ,0 
            train_preds ,train_true =[],[]

            # NOTE: comment translated to English.
            for batch_idx ,(features ,labels )in enumerate (tqdm (train_loader ,desc =f"Epoch {epoch +1 }/{epochs } - Training")):
                self .optimizer .zero_grad ()# NOTE: comment translated to English.
                logits =self .graphsage_model (features )# NOTE: comment translated to English.
                loss =criterion (logits ,labels )# NOTE: comment translated to English.

                loss .backward ()# NOTE: comment translated to English.
                torch .nn .utils .clip_grad_norm_ (self .graphsage_model .parameters (),max_norm =1.0 )# NOTE: comment translated to English.
                self .optimizer .step ()# NOTE: comment translated to English.

                # NOTE: comment translated to English.
                total_loss +=loss .item ()
                _ ,predicted =logits .max (1 )
                total +=labels .size (0 )
                correct +=predicted .eq (labels ).sum ().item ()
                train_preds .extend (predicted .cpu ().numpy ())
                train_true .extend (labels .cpu ().numpy ())

                # NOTE: comment translated to English.
            train_acc =100. *correct /total 
            avg_loss =total_loss /len (train_loader )
            train_metrics =self ._compute_metrics (np .array (train_true ),np .array (train_preds ),None )
            self .train_metrics ["total_loss"].append (avg_loss )
            self .train_metrics ["accuracy"].append (train_acc )
            self .train_metrics ["f1"].append (train_metrics ['F1 Score'])
            self .train_metrics ["recall"].append (train_metrics ['Recall'])
            self .train_metrics ["mcc"].append (train_metrics ['MCC'])

            # NOTE: comment translated to English.
            self .graphsage_model .eval ()
            val0_preds ,val0_true ,val0_probs =[],[],[]
            with torch .no_grad ():
                for features ,labels in val0_loader :
                    logits =self .graphsage_model (features )
                    _ ,predicted =logits .max (1 )
                    val0_preds .extend (predicted .cpu ().numpy ())
                    val0_true .extend (labels .cpu ().numpy ())
                    val0_probs .extend (F .softmax (logits ,dim =1 )[:,1 ].cpu ().numpy ())

                    # NOTE: comment translated to English.
            val0_metrics =self ._compute_metrics (np .array (val0_true ),np .array (val0_preds ),np .array (val0_probs ))
            self .val_metrics ["accuracy"].append (val0_metrics ['Accuracy']*100 )
            self .val_metrics ["f1"].append (val0_metrics ['F1 Score'])
            self .val_metrics ["recall"].append (val0_metrics ['Recall'])
            self .val_metrics ["mcc"].append (val0_metrics ['MCC'])
            self .val_metrics ["auc"].append (val0_metrics ['ROC-AUC'])

            # NOTE: comment translated to English.
            print (f'Epoch {epoch +1 }: 训练准确率={train_acc :.2f}% | 训练F1={train_metrics ["F1 Score"]:.4f} | '
            f'早停验证集 F1={val0_metrics ["F1 Score"]:.4f} | 早停验证集 AUC={val0_metrics ["ROC-AUC"]:.4f}')

            # NOTE: comment translated to English.
            if val0_metrics ['F1 Score']>best_val_f1 +1e-4 :
                best_val_f1 =val0_metrics ['F1 Score']
                early_stop_count =0 
                torch .save (
                {'model_state_dict':self .graphsage_model .state_dict (),'epoch':epoch +1 ,'val_f1':best_val_f1 },
                f"{save_dir }/best_graphsage_model_no_distill.pt")
            else :
                early_stop_count +=1 
                if early_stop_count >=early_stop_patience :
                    print (f"触发早停！最佳早停验证集 F1: {best_val_f1 :.4f}")
                    break 

                    # NOTE: comment translated to English.
        total_training_time =time .time ()-start_time 
        final_model_path =f"{save_dir }/final_graphsage_model_no_distill.pt"
        torch .save ({
        'model_state_dict':self .graphsage_model .state_dict (),
        'train_metrics':self .train_metrics ,
        'val_metrics':self .val_metrics 
        },final_model_path )

        # NOTE: comment translated to English.
        if os .path .exists (final_model_path ):
            self .graphsage_size_mb =os .path .getsize (final_model_path )/(1024 *1024 )

        print (f"\nGraphSAGE模型训练完成！总耗时: {total_training_time :.2f}秒")
        print (f"最佳GraphSAGE模型保存路径: {save_dir }/best_graphsage_model_no_distill.pt")
        return self .graphsage_model ,total_training_time 

    def evaluate_model_performance (self ,sequences ,labels ,ids ,batch_size =64 ,dataset_name ="validation"):
        """Docstring translated to English."""
        if self .graphsage_model is None :raise ValueError ("GraphSAGE模型未初始化")
        self .graphsage_model .eval ()

        # NOTE: comment translated to English.
        print (f"\n=== 评估ESM模型性能（{dataset_name }集）===")
        esm_preds ,esm_probs ,esm_inference_time =self .feature_extractor .predict (sequences ,batch_size =batch_size )
        true_labels =np .array (labels )

        # NOTE: comment translated to English.
        print (f"=== 提取{dataset_name }集CLS特征 ===")
        features =self .feature_extractor .get_cls_features (sequences ,batch_size =batch_size )
        features =torch .FloatTensor (features ).to (self .device )

        # NOTE: comment translated to English.
        print (f"=== 评估GraphSAGE模型性能（{dataset_name }集）===")
        graphsage_preds ,graphsage_probs =[],[]
        graphsage_start_time =time .time ()
        with torch .no_grad ():
            for i in tqdm (range (0 ,len (features ),batch_size ),desc =f"Evaluating GraphSAGE ({dataset_name })"):
                batch_features =features [i :i +batch_size ]
                batch_logits =self .graphsage_model (batch_features )
                _ ,batch_preds =batch_logits .max (1 )
                graphsage_preds .extend (batch_preds .cpu ().numpy ())
                graphsage_probs .extend (F .softmax (batch_logits ,dim =1 )[:,1 ].cpu ().numpy ())
        graphsage_inference_time =time .time ()-graphsage_start_time 

        # NOTE: comment translated to English.
        esm_metrics =self ._compute_metrics (true_labels ,np .array (esm_preds ),np .array (esm_probs ))
        graphsage_metrics =self ._compute_metrics (true_labels ,np .array (graphsage_preds ),np .array (graphsage_probs ))

        # NOTE: comment translated to English.
        esm_metrics ['Inference Time (s)']=esm_inference_time 
        esm_metrics ['Param Count (k)']=self .esm_params_k 
        esm_metrics ['Model Size (MB)']=self .esm_size_mb 

        graphsage_metrics ['Inference Time (s)']=graphsage_inference_time 
        graphsage_metrics ['Param Count (k)']=self .graphsage_params_k 
        graphsage_metrics ['Model Size (MB)']=self .graphsage_size_mb 

        return {
        'dataset_name':dataset_name ,
        'ids':ids ,
        'sequences':sequences ,
        'true_labels':true_labels ,
        'esm_preds':np .array (esm_preds ),
        'esm_probs':np .array (esm_probs ),
        'graphsage_preds':np .array (graphsage_preds ),
        'graphsage_probs':np .array (graphsage_probs ),
        'esm_metrics':esm_metrics ,
        'graphsage_metrics':graphsage_metrics ,
        'speedup_ratio':esm_inference_time /graphsage_inference_time 
        }

    def get_model_comparison (self ,graphsage_model_path ):
        """Docstring translated to English."""
        return {
        'esm_size_mb':self .esm_size_mb ,'graphsage_size_mb':self .graphsage_size_mb ,
        'size_reduction':(self .esm_size_mb -self .graphsage_size_mb )/self .esm_size_mb *100 ,
        'esm_params':self .feature_extractor_total_params ,'graphsage_params':self .graphsage_params_k *1000 ,
        'params_reduction':(self .feature_extractor_total_params -self .graphsage_params_k *1000 )/self .feature_extractor_total_params *100 
        }

        # NOTE: comment translated to English.
def save_prediction_results (performance_dict ,output_file ):
    """Docstring translated to English."""
    os .makedirs (os .path .dirname (output_file ),exist_ok =True )
    results =pd .DataFrame ({
    "id":performance_dict ['ids'],
    "sequence":performance_dict ['sequences'],
    "sequence_length":[len (seq )for seq in performance_dict ['sequences']],
    "true_label":performance_dict ['true_labels'],
    "esm_predicted_label":performance_dict ['esm_preds'],
    "esm_predicted_label_name":["阳性（1）"if p ==1 else "阴性（0）"for p in performance_dict ['esm_preds']],
    "esm_positive_probability":performance_dict ['esm_probs'].round (4 ),
    "graphsage_predicted_label":performance_dict ['graphsage_preds'],
    "graphsage_predicted_label_name":["阳性（1）"if p ==1 else "阴性（0）"for p in performance_dict ['graphsage_preds']],
    "graphsage_positive_probability":performance_dict ['graphsage_probs'].round (4 )
    })
    results .to_csv (output_file ,index =False ,encoding ="utf-8-sig")
    print (f"\n{performance_dict ['dataset_name']}集预测结果已保存至: {output_file }")

def save_metrics_summary (performance_list ,output_file ):
    """Docstring translated to English."""
    os .makedirs (os .path .dirname (output_file ),exist_ok =True )
    summary_data =[]

    for perf in performance_list :
        dataset_name =perf ['dataset_name']
        esm_row ={'Dataset':dataset_name ,'Model Type':'ESM Model',**perf ['esm_metrics']}
        graphsage_row ={'Dataset':dataset_name ,'Model Type':'GraphSAGE Model (No Distill)',**perf ['graphsage_metrics']}
        summary_data .append (esm_row )
        summary_data .append (graphsage_row )

    summary_df =pd .DataFrame (summary_data )
    # NOTE: comment translated to English.
    column_order =[
    'Dataset','Model Type','Accuracy','F1 Score','Precision',
    'Recall','MCC','ROC-AUC','PR-AUC','Inference Time (s)',
    'Param Count (k)','Model Size (MB)'
    ]
    for col in column_order :
        if col not in summary_df .columns :
            summary_df [col ]=0.0 
    summary_df =summary_df [column_order ]
    summary_df =summary_df .round (4 )
    summary_df .to_csv (output_file ,index =False ,encoding ="utf-8-sig")
    print (f"\n全量指标汇总已保存至: {output_file }")

def print_detailed_metrics (performance_list ):
    """Docstring translated to English."""
    print ("\n"+"="*120 )
    print ("【详细指标汇总】训练集/验证集/测试集 - ESM模型&GraphSAGE模型（无蒸馏）")
    print ("="*120 )

    metrics_to_print =[
    'Accuracy','F1 Score','Precision','Recall','MCC',
    'ROC-AUC','PR-AUC','Inference Time (s)','Param Count (k)','Model Size (MB)'
    ]

    for perf in performance_list :
        dataset_name =perf ['dataset_name']
        print (f"\n📊 数据集: {dataset_name }")
        print ("-"*100 )

        print ("🧬 ESM模型:")
        for metric in metrics_to_print :
            value =perf ['esm_metrics'].get (metric ,0.0 )
            if metric in ['Inference Time (s)']:
                print (f"   - {metric :<20}: {value :.4f}")
            elif metric in ['Param Count (k)','Model Size (MB)']:
                print (f"   - {metric :<20}: {value :.2f}")
            else :
                print (f"   - {metric :<20}: {value :.4f}")

        print ("📈 GraphSAGE模型 (无蒸馏):")
        for metric in metrics_to_print :
            value =perf ['graphsage_metrics'].get (metric ,0.0 )
            if metric in ['Inference Time (s)']:
                print (f"   - {metric :<20}: {value :.4f}")
            elif metric in ['Param Count (k)','Model Size (MB)']:
                print (f"   - {metric :<20}: {value :.2f}")
            else :
                print (f"   - {metric :<20}: {value :.4f}")

        print (f"   - {'Speedup Ratio (GraphSAGE/ESM)':<20}: {perf ['speedup_ratio']:.2f}x")
        print ("-"*100 )

        # NOTE: comment translated to English.
def train_graphsage_no_distillation ():
    """Docstring translated to English."""
    print ("=== 开始训练GraphSAGE模型（无知识蒸馏）===")
    print ("=== 适配新数据集路径 | val_test.csv早停 | 新增t0/t1/t2/t3验证集 ===")
    print (f"使用设备: {device }")

    try :
    # NOTE: comment translated to English.
        print ("\n===================== 加载数据集 =====================")
        train_ids ,train_sequences ,train_labels =load_csv_data (train_csv )
        val0_ids ,val0_sequences ,val0_labels =load_csv_data (val0_csv )
        t0_ids ,t0_sequences ,t0_labels =load_csv_data (t0_csv )
        t1_ids ,t1_sequences ,t1_labels =load_csv_data (t1_csv )
        t2_ids ,t2_sequences ,t2_labels =load_csv_data (t2_csv )
        t3_ids ,t3_sequences ,t3_labels =load_csv_data (t3_csv )

        # NOTE: comment translated to English.
        print (f"\n数据集汇总: ")
        print (f"   - 训练集: {len (train_sequences )}条")
        print (f"   - 早停验证集(val_test): {len (val0_sequences )}条")
        print (f"   - 验证集(rawdata_test): {len (t0_sequences )}条")
        print (f"   - bagel4_1_val: {len (t1_sequences )}条")
        print (f"   - bagel4_2_val: {len (t2_sequences )}条")
        print (f"   - bagel4_3_val: {len (t3_sequences )}条")

        # NOTE: comment translated to English.
        print ("\n===================== 加载ESM特征提取模型 =====================")
        feature_extractor =PeptideFeatureExtractor (esm2_model_name_or_path =esm2_backbone_path ,device =device )
        finetuned_esm_path =resolve_existing_path (finetuned_esm_candidates ,"ESM2微调权重")
        feature_extractor .load_pretrained_model (finetuned_model_path =finetuned_esm_path )

        # NOTE: comment translated to English.
        print ("\n===================== 初始化GraphSAGE模型 =====================")
        trainer =GraphSAGETrainer (
        feature_extractor =feature_extractor ,
        feature_extractor_total_params =feature_extractor .total_params ,
        device =device 
        )
        trainer .init_graphsage_model (input_dim =feature_extractor .hidden_size ,hidden_dim =128 ,latent_dim =64 )

        # NOTE: comment translated to English.
        print ("\n===================== 准备训练数据 =====================")
        training_data =trainer .prepare_training_data (
        sequences =train_sequences ,labels =train_labels ,
        batch_size =32 ,save_dir ="results_ultra_light_graphsage"
        )
        val0_data =trainer .prepare_validation_data (
        val_sequences =val0_sequences ,val_labels =val0_labels ,
        batch_size =32 ,save_dir ="results_ultra_light_graphsage"
        )

        # NOTE: comment translated to English.
        print ("\n===================== 训练GraphSAGE模型 =====================")
        graphsage_model ,training_time =trainer .train_graphsage (
        training_data =training_data ,val0_data =val0_data ,
        save_dir ="results_ultra_light_graphsage",
        batch_size =512 ,epochs =50 ,lr =1e-3 ,early_stop_patience =10 
        )

        # NOTE: comment translated to English.
        print ("\n===================== 评估所有数据集 =====================")
        train_perf =trainer .evaluate_model_performance (sequences =train_sequences ,labels =train_labels ,ids =train_ids ,batch_size =64 ,dataset_name ="train_no_distill_graphsage")
        plot_cumulative_curves (graphsage_probs =train_perf ['graphsage_probs'],dataset_name =train_perf ['dataset_name'],save_dir =cumulative_curve_dir )

        val0_perf =trainer .evaluate_model_performance (sequences =val0_sequences ,labels =val0_labels ,ids =val0_ids ,batch_size =64 ,dataset_name ="val0_val_test_early_stop")
        plot_cumulative_curves (graphsage_probs =val0_perf ['graphsage_probs'],dataset_name =val0_perf ['dataset_name'],save_dir =cumulative_curve_dir )

        t0_perf =trainer .evaluate_model_performance (sequences =t0_sequences ,labels =t0_labels ,ids =t0_ids ,batch_size =64 ,dataset_name ="t0_rawdata_test")
        plot_cumulative_curves (graphsage_probs =t0_perf ['graphsage_probs'],dataset_name =t0_perf ['dataset_name'],save_dir =cumulative_curve_dir )

        t1_perf =trainer .evaluate_model_performance (sequences =t1_sequences ,labels =t1_labels ,ids =t1_ids ,batch_size =64 ,dataset_name ="t1_bagel4_1_val")
        plot_cumulative_curves (graphsage_probs =t1_perf ['graphsage_probs'],dataset_name =t1_perf ['dataset_name'],save_dir =cumulative_curve_dir )

        t2_perf =trainer .evaluate_model_performance (sequences =t2_sequences ,labels =t2_labels ,ids =t2_ids ,batch_size =64 ,dataset_name ="t2_bagel4_2_val")
        plot_cumulative_curves (graphsage_probs =t2_perf ['graphsage_probs'],dataset_name =t2_perf ['dataset_name'],save_dir =cumulative_curve_dir )

        t3_perf =trainer .evaluate_model_performance (sequences =t3_sequences ,labels =t3_labels ,ids =t3_ids ,batch_size =64 ,dataset_name ="t3_bagel4_3_val")
        plot_cumulative_curves (graphsage_probs =t3_perf ['graphsage_probs'],dataset_name =t3_perf ['dataset_name'],save_dir =cumulative_curve_dir )

        performance_list =[train_perf ,val0_perf ,t0_perf ,t1_perf ,t2_perf ,t3_perf ]

        # NOTE: comment translated to English.
        print ("\n===================== 模型轻量化对比 =====================")
        graphsage_model_path ="results_ultra_light_graphsage/final_graphsage_model_no_distill.pt"
        model_comparison =trainer .get_model_comparison (graphsage_model_path )

        # NOTE: comment translated to English.
        print ("\n"+"="*100 )
        print ("【无蒸馏】GraphSAGE模型训练最终结果汇总")
        print ("="*100 )

        print (f"\n1. 模型轻量化效果:")
        print (f"   - ESM模型: {model_comparison ['esm_size_mb']:.2f} MB | {model_comparison ['esm_params']:,} 参数")
        print (f"   - GraphSAGE模型: {model_comparison ['graphsage_size_mb']:.2f} MB | {model_comparison ['graphsage_params']:,} 参数")
        print (f"   - 体积缩减: {model_comparison ['size_reduction']:.1f}% | 参数量缩减: {model_comparison ['params_reduction']:.1f}%")
        print (f"   - 训练耗时: {training_time :.2f}秒")

        print (f"\n2. 各数据集（GraphSAGE模型）核心指标:")
        for perf in performance_list :
            ds_name =perf ['dataset_name']
            g_metrics =perf ['graphsage_metrics']
            print (f"   - {ds_name }: F1={g_metrics ['F1 Score']:.4f} | ROC-AUC={g_metrics ['ROC-AUC']:.4f} | PR-AUC={g_metrics ['PR-AUC']:.4f} | 推理提速={perf ['speedup_ratio']:.1f}x")

            # NOTE: comment translated to English.
        print ("\n===================== 保存预测结果和指标汇总 =====================")
        save_prediction_results (val0_perf ,val0_output )
        save_prediction_results (t0_perf ,t0_output )
        save_prediction_results (t1_perf ,t1_output )
        save_prediction_results (t2_perf ,t2_output )
        save_prediction_results (t3_perf ,t3_output )
        save_metrics_summary (performance_list ,metrics_summary_output )

        # NOTE: comment translated to English.
        print_detailed_metrics (performance_list )

        # NOTE: comment translated to English.
        print ("\n"+"="*100 )
        print ("=== GraphSAGE模型无蒸馏训练全流程成功完成！===")
        print (f"GraphSAGE模型保存目录: results_ultra_light_graphsage")
        print (f"早停验证集结果: {val0_output }")
        print (f"rawdata_test结果: {t0_output }")
        print (f"bagel4_1_val结果: {t1_output }")
        print (f"bagel4_2_val结果: {t2_output }")
        print (f"bagel4_3_val结果: {t3_output }")
        print (f"全量指标汇总: {metrics_summary_output }")
        print (f"累积曲线保存目录: {cumulative_curve_dir }")
        print ("="*100 )

        return graphsage_model ,trainer ,performance_list ,model_comparison 

    except Exception as e :
    # NOTE: comment translated to English.
        print (f"\n程序执行失败: {e }")
        import traceback 
        traceback .print_exc ()
        return None ,None ,None ,None 

if __name__ =="__main__":
    graphsage_model ,trainer ,performance_results ,model_comparison =train_graphsage_no_distillation ()

    if graphsage_model is None :
        print ("\n训练流程执行失败，请检查错误信息！")
        exit (1 )
    else :
        print ("\n所有流程执行完毕，结果已保存！")
        exit (0 )
