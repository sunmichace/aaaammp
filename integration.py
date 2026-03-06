import torch 
import torch .nn as nn 
import torch .nn .functional as F 
from torch .utils .data import Dataset ,DataLoader 
from sklearn .metrics import accuracy_score ,f1_score ,precision_score ,recall_score 
from sklearn .preprocessing import StandardScaler 
import numpy as np 
import pandas as pd 
import os 
from tqdm import tqdm 
from transformers import AutoTokenizer ,AutoModel 
# NOTE: comment translated to English.
from torch_geometric .nn import SAGEConv ,global_mean_pool ,GCNConv 

# NOTE: comment translated to English.
# NOTE: comment translated to English.
ROOT_DIR ="./"
# NOTE: comment translated to English.
device ='cuda'if torch .cuda .is_available ()else 'cpu'

# NOTE: comment translated to English.
INFERENCE_CSV_PATH =os .path .join (ROOT_DIR ,"mydata/newdata/rawdata_test.csv")
# NOTE: comment translated to English.
DATASET_NAME =None 

# NOTE: comment translated to English.
esm2_backbone_path =os .path .join (ROOT_DIR ,"model/esm2_t6_8M_UR50D")
# NOTE: comment translated to English.
finetuned_esm_candidates =[
os .path .join (ROOT_DIR ,"model/model_AMP_ESM2_8M_256_v3/best_model_amp/esm2_8m_binary_best.pt"),
os .path .join (ROOT_DIR ,"model_AMP_ESM2_8M_256_v3/best_model_amp/esm2_8m_binary_best.pt"),
]

# NOTE: comment translated to English.
rnn_model_candidates =[
os .path .join (ROOT_DIR ,"model/results_ultra_light_rnn/best_rnn_model_no_distill.pt"),
os .path .join (ROOT_DIR ,"results_ultra_light_rnn/best_rnn_model_no_distill.pt"),
]
gnn_model_candidates =[
os .path .join (ROOT_DIR ,"model/results_ultra_light_gnn/best_model_no_distill_gnn.pt"),
os .path .join (ROOT_DIR ,"results_ultra_light_gnn/best_model_no_distill_gnn.pt"),
]
graphsage_model_candidates =[
os .path .join (ROOT_DIR ,"model/results_ultra_light_graphsage/best_graphsage_model_no_distill.pt"),
os .path .join (ROOT_DIR ,"results_ultra_light_graphsage/best_graphsage_model_no_distill.pt"),
]
comdel_model_candidates =[
os .path .join (ROOT_DIR ,"model/results_ultra_light_scratch/cnn_scratch_best.pt"),
os .path .join (ROOT_DIR ,"results_ultra_light_scratch/cnn_scratch_best.pt"),
]

# NOTE: comment translated to English.
OUTPUT_DIR =os .path .join (ROOT_DIR ,"single_inference_result_4models")
os .makedirs (OUTPUT_DIR ,exist_ok =True )


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


def infer_dataset_name (csv_path ,manual_name =None ):
    """Docstring translated to English."""
    if manual_name and str (manual_name ).strip ():
        return str (manual_name ).strip ()
    return os .path .splitext (os .path .basename (csv_path ))[0 ]

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
            esm2_backbone =AutoModel .from_pretrained (self .esm2_model_name_or_path )
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
            checkpoint =torch .load (finetuned_model_path ,map_location =device ,weights_only =False )

            # NOTE: comment translated to English.
            model_state_dict =extract_state_dict (checkpoint )
            feature_extractor_state_dict ={}
            for k ,v in model_state_dict .items ():
                if k .startswith ('esm2_backbone.'):
                    feature_extractor_state_dict [k ]=v 
                elif k .startswith ('backbone.'):
                    mapped_key =k .replace ('backbone.','esm2_backbone.',1 )
                    feature_extractor_state_dict [mapped_key ]=v 
            if not feature_extractor_state_dict :
                raise ValueError ("ESM2主干权重为空，请检查checkpoint是否匹配当前模型结构")
                # NOTE: comment translated to English.
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
                # NOTE: comment translated to English.
                cls_repr =self .model (input_ids =input_ids ,attention_mask =attention_mask )
                features .extend (cls_repr .cpu ().numpy ())
                # NOTE: comment translated to English.
        return np .array (features )

        # NOTE: comment translated to English.
class EnhancedRNNClassifier (nn .Module ):
    def __init__ (self ,input_dim =320 ,hidden_dim =128 ,num_classes =2 ,num_layers =2 ,bidirectional =True ):
        super ().__init__ ()
        self .input_dim =input_dim 
        self .hidden_dim =hidden_dim 
        self .num_layers =num_layers 
        self .bidirectional =bidirectional 
        self .num_directions =2 if bidirectional else 1 

        # NOTE: comment translated to English.
        self .rnn =nn .RNN (
        input_size =input_dim ,
        hidden_size =hidden_dim ,
        num_layers =num_layers ,
        batch_first =True ,
        bidirectional =bidirectional ,
        dropout =0.3 if num_layers >1 else 0 
        )

        # NOTE: comment translated to English.
        self .layer_norm =nn .LayerNorm (hidden_dim *self .num_directions )
        self .dropout =nn .Dropout (0.2 )

        # NOTE: comment translated to English.
        self .classifier =nn .Linear (hidden_dim *self .num_directions ,num_classes )

        # NOTE: comment translated to English.
        self ._init_weights ()

    def _init_weights (self ):
        for name ,param in self .rnn .named_parameters ():
            if 'weight_ih'in name :
                nn .init .xavier_uniform_ (param .data )
            elif 'weight_hh'in name :
                nn .init .orthogonal_ (param .data )
            elif 'bias'in name :
                param .data .fill_ (0.0 )

        nn .init .ones_ (self .layer_norm .weight )
        nn .init .zeros_ (self .layer_norm .bias )
        nn .init .xavier_uniform_ (self .classifier .weight )
        if self .classifier .bias is not None :
            nn .init .zeros_ (self .classifier .bias )

    def forward (self ,x ):
        x =x .unsqueeze (1 )
        rnn_out ,hn =self .rnn (x )
        rnn_out =rnn_out .squeeze (1 )
        out =self .layer_norm (rnn_out )
        out =self .dropout (out )
        logits =self .classifier (out )
        return logits 

        # NOTE: comment translated to English.
class GNN_EnhancedClassifier (nn .Module ):
    """Docstring translated to English."""
    def __init__ (self ,input_dim =320 ,hidden_dim =128 ,num_classes =2 ,num_layers =2 ):
        super ().__init__ ()
        self .input_dim =input_dim 
        self .hidden_dim =hidden_dim 
        self .num_layers =num_layers 

        # NOTE: comment translated to English.
        self .input_proj =nn .Linear (input_dim ,hidden_dim )

        # NOTE: comment translated to English.
        self .gcn_layers =nn .ModuleList ()
        self .gcn_layers .append (GCNConv (hidden_dim ,hidden_dim ))
        for _ in range (num_layers -1 ):
            self .gcn_layers .append (GCNConv (hidden_dim ,hidden_dim ))

        self .layer_norm =nn .LayerNorm (hidden_dim )
        self .dropout =nn .Dropout (0.2 )
        self .classifier =nn .Linear (hidden_dim ,num_classes )

        self ._init_weights ()

    def _init_weights (self ):
        """Docstring translated to English."""
        for gcn in self .gcn_layers :
            nn .init .xavier_uniform_ (gcn .lin .weight )
            if gcn .bias is not None :
                nn .init .zeros_ (gcn .bias )
        nn .init .xavier_uniform_ (self .input_proj .weight )
        nn .init .zeros_ (self .input_proj .bias )
        nn .init .xavier_uniform_ (self .classifier .weight )
        nn .init .zeros_ (self .classifier .bias )
        nn .init .ones_ (self .layer_norm .weight )
        nn .init .zeros_ (self .layer_norm .bias )

    def forward (self ,x ):
        """Docstring translated to English."""
        batch_size =x .shape [0 ]
        x_proj =self .input_proj (x )
        node_feat =x_proj 
        # NOTE: comment translated to English.
        edge_index =torch .empty ((2 ,0 ),dtype =torch .long ,device =x .device )
        batch_idx =torch .arange (batch_size ,device =x .device )

        # NOTE: comment translated to English.
        gcn_out =node_feat 
        for gcn in self .gcn_layers :
            gcn_out =gcn (gcn_out ,edge_index )
            gcn_out =F .relu (gcn_out )

            # NOTE: comment translated to English.
        global_feat =global_mean_pool (gcn_out ,batch_idx )
        out =self .layer_norm (global_feat )
        out =self .dropout (out )
        logits =self .classifier (out )
        return logits 

        # NOTE: comment translated to English.
class SAGEEncoder (nn .Module ):
    """Docstring translated to English."""
    def __init__ (self ,in_channels ,hidden_channels ,out_channels ,num_layers =2 ,dropout =0.0 ):
        super ().__init__ ()
        self .num_layers =num_layers 
        self .dropout =dropout 

        self .convs =nn .ModuleList ()
        self .convs .append (SAGEConv (in_channels ,hidden_channels ))
        for _ in range (num_layers -2 ):
            self .convs .append (SAGEConv (hidden_channels ,hidden_channels ))
        self .convs .append (SAGEConv (hidden_channels ,out_channels ))

    def forward (self ,x ,edge_index ):
        """Docstring translated to English."""
        for i in range (self .num_layers ):
            x =self .convs [i ](x ,edge_index )
            if i !=self .num_layers -1 :
                x =F .relu (x )
                x =F .dropout (x ,p =self .dropout ,training =self .training )
        return x 

class GraphSAGE_Model (nn .Module ):
    """Docstring translated to English."""
    def __init__ (self ,input_dim =320 ,hidden_dim =128 ,latent_dim =64 ,num_classes =2 ,num_layers =2 ):
        super ().__init__ ()
        self .input_dim =input_dim 
        self .hidden_dim =hidden_dim 
        self .latent_dim =latent_dim 
        self .num_layers =num_layers 

        self .input_proj =nn .Linear (input_dim ,hidden_dim )
        self .sage_encoder =SAGEEncoder (
        in_channels =hidden_dim ,
        hidden_channels =hidden_dim ,
        out_channels =latent_dim ,
        num_layers =num_layers ,
        dropout =0.2 
        )
        self .layer_norm =nn .LayerNorm (latent_dim )
        self .dropout =nn .Dropout (0.2 )
        self .classifier =nn .Linear (latent_dim ,num_classes )

    def forward (self ,x ):
        """Docstring translated to English."""
        batch_size =x .shape [0 ]
        x_proj =self .input_proj (x )
        node_feat =x_proj 
        edge_index =torch .empty ((2 ,0 ),dtype =torch .long ,device =x .device )
        batch_idx =torch .arange (batch_size ,device =x .device ,dtype =torch .long )

        z =self .sage_encoder (node_feat ,edge_index )
        global_feat =global_mean_pool (z ,batch_idx )
        out =self .layer_norm (global_feat )
        out =self .dropout (out )
        logits =self .classifier (out )
        return logits 

        # NOTE: comment translated to English.
class AminoAcidTokenizer :
    """Docstring translated to English."""
    def __init__ (self ,max_seq_len =256 ):
    # NOTE: comment translated to English.
        self .vocab ={
        '<pad>':0 ,'A':1 ,'<cls>':2 ,'<sep>':3 ,'R':4 ,'N':5 ,'D':6 ,
        'C':7 ,'Q':8 ,'E':9 ,'G':10 ,'H':11 ,'I':12 ,'L':13 ,'K':14 ,
        'M':15 ,'F':16 ,'P':17 ,'S':18 ,'T':19 ,'W':20 ,'Y':21 ,'V':22 
        }
        self .id2token ={v :k for k ,v in self .vocab .items ()}
        self .max_seq_len =max_seq_len # NOTE: comment translated to English.
        self .pad_token_id =0 # NOTE: comment translated to English.
        self .vocab_size =len (self .vocab )# NOTE: comment translated to English.

    def batch_encode (self ,sequences ,padding ='max_length',truncation =True ,max_length =256 ):
        """Docstring translated to English."""
        batch_input_ids =[]
        for seq in sequences :
            seq =seq .strip ().upper ()
            # NOTE: comment translated to English.
            input_ids =[self .vocab .get (char ,self .pad_token_id )for char in seq ]
            # NOTE: comment translated to English.
            if truncation and len (input_ids )>max_length :
                input_ids =input_ids [:max_length ]
                # NOTE: comment translated to English.
            if padding =='max_length':
                input_ids +=[self .pad_token_id ]*(max_length -len (input_ids ))
            batch_input_ids .append (input_ids )
        return torch .tensor (batch_input_ids ,dtype =torch .long )

class LightweightCNN (nn .Module ):
    """Docstring translated to English."""
    def __init__ (self ,vocab_size ,embed_dim =128 ,hidden_dim =128 ,max_seq_len =256 ):
        super ().__init__ ()
        self .max_seq_len =max_seq_len 
        # NOTE: comment translated to English.
        self .embedding =nn .Embedding (vocab_size ,embed_dim ,padding_idx =0 )
        self .embed_dropout =nn .Dropout (0.1 )
        # NOTE: comment translated to English.
        self .feature_extractor =nn .Sequential (
        nn .Conv1d (embed_dim ,hidden_dim ,kernel_size =3 ,padding =1 ),
        nn .BatchNorm1d (hidden_dim ),
        nn .ReLU (),
        nn .MaxPool1d (2 ),
        nn .Conv1d (hidden_dim ,hidden_dim *2 ,kernel_size =3 ,padding =1 ),
        nn .BatchNorm1d (hidden_dim *2 ),
        nn .ReLU (),
        nn .MaxPool1d (2 ),
        nn .Conv1d (hidden_dim *2 ,hidden_dim ,kernel_size =3 ,padding =1 ),
        nn .BatchNorm1d (hidden_dim ),
        nn .ReLU (),
        nn .AdaptiveMaxPool1d (1 )
        )
        # NOTE: comment translated to English.
        self .classifier =nn .Sequential (
        nn .Linear (hidden_dim ,64 ),
        nn .ReLU (),
        nn .Dropout (0.3 ),
        nn .Linear (64 ,2 )
        )

    def forward (self ,input_ids ):
        """Docstring translated to English."""
        x =self .embedding (input_ids )# [batch, seq_len, embed_dim]
        x =self .embed_dropout (x )
        x =x .permute (0 ,2 ,1 )# NOTE: comment translated to English.
        x =self .feature_extractor (x )# [batch, hidden_dim, 1]
        x =x .squeeze (-1 )# [batch, hidden_dim]
        logits =self .classifier (x )# [batch, 2]
        return logits 

        # NOTE: comment translated to English.
def load_csv_data (csv_path ):
    """Docstring translated to English."""
    if not os .path .exists (csv_path ):
        raise FileNotFoundError (f"CSV文件不存在: {csv_path }")
    try :
        df =pd .read_csv (csv_path )
    except Exception as e :
        raise IOError (f"读取CSV失败: {str (e )}")

        # NOTE: comment translated to English.
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

def load_trained_models (feat_dim ,device ):
    """Docstring translated to English."""
    rnn_model_path =resolve_existing_path (rnn_model_candidates ,"RNN模型权重")
    gnn_model_path =resolve_existing_path (gnn_model_candidates ,"GNN模型权重")
    graphsage_model_path =resolve_existing_path (graphsage_model_candidates ,"GraphSAGE模型权重")
    comdel_model_path =resolve_existing_path (comdel_model_candidates ,"COMDEL模型权重")

    print ("\n🔧 加载RNN模型（含标准化器）...")
    rnn_model =EnhancedRNNClassifier (input_dim =feat_dim ,hidden_dim =128 ).to (device )
    rnn_ckpt =torch .load (rnn_model_path ,map_location =device ,weights_only =False )
    rnn_model .load_state_dict (extract_state_dict (rnn_ckpt ))
    # NOTE: comment translated to English.
    rnn_scaler =StandardScaler ()
    rnn_scaler .mean_ =np .zeros (feat_dim )
    rnn_scaler .scale_ =np .ones (feat_dim )
    rnn_model .eval ()

    print ("\n🔧 加载GNN模型...")
    gnn_model =GNN_EnhancedClassifier (input_dim =feat_dim ,hidden_dim =128 ,num_classes =2 ,num_layers =2 ).to (device )
    gnn_ckpt =torch .load (gnn_model_path ,map_location =device ,weights_only =False )
    gnn_model .load_state_dict (extract_state_dict (gnn_ckpt ))
    gnn_model .eval ()

    print ("\n🔧 加载GraphSAGE模型...")
    graphsage_model =GraphSAGE_Model (input_dim =feat_dim ,hidden_dim =128 ,latent_dim =64 ).to (device )
    graphsage_ckpt =torch .load (graphsage_model_path ,map_location =device ,weights_only =False )
    graphsage_model .load_state_dict (extract_state_dict (graphsage_ckpt ))
    graphsage_model .eval ()

    print ("\n🔧 加载COMDEL模型（CNN）...")
    comdel_tokenizer =AminoAcidTokenizer (max_seq_len =256 )
    comdel_model =LightweightCNN (vocab_size =comdel_tokenizer .vocab_size ,embed_dim =128 ,hidden_dim =128 ).to (device )
    comdel_ckpt =torch .load (comdel_model_path ,map_location =device ,weights_only =False )
    comdel_model .load_state_dict (extract_state_dict (comdel_ckpt ))
    comdel_model .eval ()

    return rnn_model ,rnn_scaler ,gnn_model ,graphsage_model ,comdel_model ,comdel_tokenizer 

    # NOTE: comment translated to English.
def single_inference ():
    """Docstring translated to English."""
    dataset_name =infer_dataset_name (INFERENCE_CSV_PATH ,DATASET_NAME )

    # NOTE: comment translated to English.
    ids ,seqs ,labels =load_csv_data (INFERENCE_CSV_PATH )
    true_labels =np .array (labels )

    # NOTE: comment translated to English.
    extractor =PeptideFeatureExtractor (esm2_backbone_path ,device )
    finetuned_esm_path =resolve_existing_path (finetuned_esm_candidates ,"ESM2微调权重")
    extractor .load_pretrained_model (finetuned_model_path =finetuned_esm_path )

    # NOTE: comment translated to English.
    features =extractor .get_cls_features (seqs ,batch_size =32 )
    features_tensor =torch .FloatTensor (features ).to (device )

    # NOTE: comment translated to English.
    rnn_model ,rnn_scaler ,gnn_model ,graphsage_model ,comdel_model ,comdel_tokenizer =load_trained_models (
    extractor .hidden_size ,device 
    )

    # NOTE: comment translated to English.
    def infer_model (model ,input_data ,is_comdel =False ,scaler =None ):
        """Docstring translated to English."""
        preds ,probs =[],[]
        batch_size =64 
        model .eval ()
        # NOTE: comment translated to English.
        with torch .no_grad ():
            if is_comdel :
            # NOTE: comment translated to English.
                for i in tqdm (range (0 ,len (input_data ),batch_size ),desc ="COMDEL(CNN)推理"):
                    batch_input =input_data [i :i +64 ].to (device )
                    logits =model (batch_input )
                    prob =F .softmax (logits ,dim =1 )# NOTE: comment translated to English.
                    preds .extend (torch .argmax (prob ,dim =1 ).cpu ().numpy ())# NOTE: comment translated to English.
                    probs .extend (prob [:,1 ].cpu ().numpy ())# NOTE: comment translated to English.
            else :
            # NOTE: comment translated to English.
                batch_data =input_data 
                # NOTE: comment translated to English.
                if scaler is not None :
                    batch_data_np =batch_data .cpu ().numpy ()
                    batch_data_np =scaler .transform (batch_data_np )
                    batch_data =torch .FloatTensor (batch_data_np ).to (device )

                for i in tqdm (range (0 ,len (batch_data ),batch_size ),desc =f"{model .__class__ .__name__ }推理"):
                    batch_input =batch_data [i :i +64 ]
                    logits =model (batch_input )
                    prob =F .softmax (logits ,dim =1 )
                    preds .extend (torch .argmax (prob ,dim =1 ).cpu ().numpy ())
                    probs .extend (prob [:,1 ].cpu ().numpy ())
        return np .array (preds ),np .array (probs )

        # NOTE: comment translated to English.
    rnn_preds ,rnn_probs =infer_model (rnn_model ,features_tensor ,scaler =rnn_scaler )
    gnn_preds ,gnn_probs =infer_model (gnn_model ,features_tensor )
    graphsage_preds ,graphsage_probs =infer_model (graphsage_model ,features_tensor )
    # NOTE: comment translated to English.
    comdel_input_ids =comdel_tokenizer .batch_encode (seqs ,padding ='max_length',truncation =True ,max_length =256 )
    comdel_preds ,comdel_probs =infer_model (comdel_model ,comdel_input_ids ,is_comdel =True )

    # NOTE: comment translated to English.
    all_preds =np .stack ([rnn_preds ,gnn_preds ,graphsage_preds ,comdel_preds ],axis =1 )
    # NOTE: comment translated to English.
    joint_preds =(np .sum (all_preds ,axis =1 )==4 ).astype (int )
    # NOTE: comment translated to English.
    joint_probs =np .mean ([rnn_probs ,gnn_probs ,graphsage_probs ,comdel_probs ],axis =0 )

    # NOTE: comment translated to English.
    result_df =pd .DataFrame ({
    "id":ids ,
    "sequence":seqs ,
    "true_label":true_labels ,
    "rnn_pred":rnn_preds ,
    "rnn_pos_prob":np .round (rnn_probs ,4 ),
    "gnn_pred":gnn_preds ,
    "gnn_pos_prob":np .round (gnn_probs ,4 ),
    "graphsage_pred":graphsage_preds ,
    "graphsage_pos_prob":np .round (graphsage_probs ,4 ),
    "comdel_cnn_pred":comdel_preds ,
    "comdel_cnn_pos_prob":np .round (comdel_probs ,4 ),
    "joint_pred":joint_preds ,
    "joint_pos_prob":np .round (joint_probs ,4 ),
    "final_result":["阳性"if p ==1 else "阴性"for p in joint_preds ]
    })
    result_path =os .path .join (OUTPUT_DIR ,f"{dataset_name }_4models_inference_result.csv")
    result_df .to_csv (result_path ,index =False ,encoding ="utf-8-sig")
    print (f"\n📄 推理结果已保存至: {result_path }")

    # NOTE: comment translated to English.
    def calculate_metrics (y_true ,y_pred ):
        acc =accuracy_score (y_true ,y_pred )# NOTE: comment translated to English.
        f1 =f1_score (y_true ,y_pred ,zero_division =0 )# NOTE: comment translated to English.
        precision =precision_score (y_true ,y_pred ,zero_division =0 )# NOTE: comment translated to English.
        recall =recall_score (y_true ,y_pred ,zero_division =0 )# NOTE: comment translated to English.
        return acc ,f1 ,precision ,recall 

        # NOTE: comment translated to English.
    rnn_acc ,rnn_f1 ,rnn_precision ,rnn_recall =calculate_metrics (true_labels ,rnn_preds )
    gnn_acc ,gnn_f1 ,gnn_precision ,gnn_recall =calculate_metrics (true_labels ,gnn_preds )
    graphsage_acc ,graphsage_f1 ,graphsage_precision ,graphsage_recall =calculate_metrics (true_labels ,graphsage_preds )
    comdel_acc ,comdel_f1 ,comdel_precision ,comdel_recall =calculate_metrics (true_labels ,comdel_preds )
    joint_acc ,joint_f1 ,joint_precision ,joint_recall =calculate_metrics (true_labels ,joint_preds )

    # NOTE: comment translated to English.
    print ("\n📊 推理结果指标汇总：")
    print (f"RNN模型       - 准确率: {rnn_acc :.4f} | F1: {rnn_f1 :.4f} | 精确率: {rnn_precision :.4f} | 召回率: {rnn_recall :.4f}")
    print (f"GNN模型       - 准确率: {gnn_acc :.4f} | F1: {gnn_f1 :.4f} | 精确率: {gnn_precision :.4f} | 召回率: {gnn_recall :.4f}")
    print (f"GraphSAGE模型 - 准确率: {graphsage_acc :.4f} | F1: {graphsage_f1 :.4f} | 精确率: {graphsage_precision :.4f} | 召回率: {graphsage_recall :.4f}")
    print (f"COMDEL(CNN)模型 - 准确率: {comdel_acc :.4f} | F1: {comdel_f1 :.4f} | 精确率: {comdel_precision :.4f} | 召回率: {comdel_recall :.4f}")
    print (f"四模型联合     - 准确率: {joint_acc :.4f} | F1: {joint_f1 :.4f} | 精确率: {joint_precision :.4f} | 召回率: {joint_recall :.4f}")
    print (f"四模型联合     - 正样本预测数: {sum (joint_preds )} | 实际正样本数: {sum (true_labels )}")

if __name__ =="__main__":
    print (f"📌 开始单个数据集四模型推理 | 使用设备: {device }")
    single_inference ()
    print ("\n🎉 四模型推理完成！所有结果已保存至 single_inference_result_4models 目录")
