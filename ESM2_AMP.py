# -*- coding: utf-8 -*-
import torch 
import torch .nn as nn 
import torch .optim as optim 
from torch .utils .data import Dataset ,DataLoader 
from sklearn .metrics import (
classification_report ,confusion_matrix ,
precision_score ,recall_score ,f1_score ,
roc_curve ,auc ,precision_recall_curve ,matthews_corrcoef ,accuracy_score 
)
import numpy as np 
import pandas as pd 
import os 
import shutil 
from tqdm import tqdm 
import matplotlib .pyplot as plt 
import seaborn as sns 
from transformers import AutoTokenizer ,AutoModel 
import torch .serialization 
from sklearn .preprocessing import LabelEncoder 


torch .serialization .add_safe_globals ([LabelEncoder ])

plt .rcParams ["axes.unicode_minus"]=False 


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


    # NOTE: comment translated to English.
class PeptideDataset (Dataset ):
    """Docstring translated to English."""
    def __init__ (self ,sequences ,labels ):
        self .sequences =sequences # NOTE: comment translated to English.
        self .labels =labels # NOTE: comment translated to English.

    def __len__ (self ):
        """Docstring translated to English."""
        return len (self .sequences )

    def __getitem__ (self ,idx ):
        """Docstring translated to English."""
        return self .sequences [idx ],self .labels [idx ],len (self .sequences [idx ])


        # NOTE: comment translated to English.
class PeptidePredictor :
    """Docstring translated to English."""
    def __init__ (self ,esm2_model_name_or_path ,device ='cuda'if torch .cuda .is_available ()else 'cpu'):
        """Docstring translated to English."""
        self .device =device # NOTE: comment translated to English.
        self .esm2_model_name_or_path =esm2_model_name_or_path # NOTE: comment translated to English.
        self .model =None # NOTE: comment translated to English.
        self .tokenizer =None # NOTE: comment translated to English.

        self .all_folds_metrics =[]# NOTE: comment translated to English.

        # NOTE: comment translated to English.
        self .train_metrics ={
        "loss":[],"accuracy":[],"precision":[],"recall":[],"f1":[],"mcc":[]
        }
        self .val_metrics ={
        "loss":[],"accuracy":[],"precision":[],"recall":[],"f1":[],"mcc":[]
        }
        self .test_metrics ={
        "loss":[],"accuracy":[],"precision":[],"recall":[],"f1":[],"mcc":[]
        }

    def load_pretrained_model (self ,freeze_layers =True ):
        """Docstring translated to English."""
        print (f"正在加载ESM2 8M模型: {self .esm2_model_name_or_path }")
        try :
        # NOTE: comment translated to English.
            self .tokenizer =AutoTokenizer .from_pretrained (self .esm2_model_name_or_path )
            # NOTE: comment translated to English.
            esm2_backbone =AutoModel .from_pretrained (self .esm2_model_name_or_path )
            print (f"ESM2 8M隐藏维度: {esm2_backbone .config .hidden_size }")

            # NOTE: comment translated to English.
            class ESM2Classifier (nn .Module ):
                def __init__ (self ,esm2_backbone ):
                    super ().__init__ ()
                    self .esm2_backbone =esm2_backbone # NOTE: comment translated to English.
                    self .hidden_dim =esm2_backbone .config .hidden_size # NOTE: comment translated to English.
                    # NOTE: comment translated to English.
                    self .classifier =nn .Sequential (
                    nn .Linear (self .hidden_dim ,512 ),
                    nn .ReLU (),
                    nn .Dropout (0.6 ),
                    nn .Linear (512 ,256 ),
                    nn .ReLU (),
                    nn .Dropout (0.6 ),
                    nn .Linear (256 ,2 )# NOTE: comment translated to English.
                    )

                def forward (self ,input_ids ,attention_mask ):
                    """Docstring translated to English."""
                    # NOTE: comment translated to English.
                    outputs =self .esm2_backbone (input_ids =input_ids ,attention_mask =attention_mask )
                    # NOTE: comment translated to English.
                    cls_repr =outputs .last_hidden_state [:,0 ,:]
                    # NOTE: comment translated to English.
                    seq_lengths =attention_mask .sum (dim =1 ,keepdim =True ).float ()
                    cls_repr =cls_repr /torch .sqrt (seq_lengths +1e-8 )# NOTE: comment translated to English.
                    # NOTE: comment translated to English.
                    return self .classifier (cls_repr )

                    # NOTE: comment translated to English.
            if freeze_layers :
            # NOTE: comment translated to English.
                for i ,layer in enumerate (esm2_backbone .encoder .layer ):
                    for param in layer .parameters ():
                        param .requires_grad =(i >=4 )
                        # NOTE: comment translated to English.
                for param in esm2_backbone .embeddings .parameters ():
                    param .requires_grad =False 

                    # NOTE: comment translated to English.
            self .model =ESM2Classifier (esm2_backbone ).to (self .device )
            print ("ESM2 8M二分类模型（CLS归一化+增强建模）加载完成！")
        except Exception as e :
            print (f"模型加载失败: {e }")
            raise 

    def _compute_binary_metrics (self ,y_true ,y_pred ):
        """Docstring translated to English."""
        precision =precision_score (y_true ,y_pred ,zero_division =0 )# NOTE: comment translated to English.
        recall =recall_score (y_true ,y_pred ,zero_division =0 )# NOTE: comment translated to English.
        f1 =f1_score (y_true ,y_pred ,zero_division =0 )# NOTE: comment translated to English.
        mcc =matthews_corrcoef (y_true ,y_pred )# NOTE: comment translated to English.
        accuracy =accuracy_score (y_true ,y_pred )# NOTE: comment translated to English.
        return accuracy ,precision ,recall ,f1 ,mcc 

    def _plot_pr_curve (self ,y_true ,y_probs ,save_dir ):
        """Docstring translated to English."""
        precision ,recall ,_ =precision_recall_curve (y_true ,y_probs )
        pr_auc =auc (recall ,precision )
        plt .figure (figsize =(8 ,6 ))
        plt .plot (recall ,precision ,lw =2 ,label =f'PR Curve (PR-AUC = {pr_auc :.3f})')
        plt .axhline (y =np .mean (y_true ),color ='r',linestyle ='--',label =f'Random Guess (PR-AUC = {np .mean (y_true ):.3f})')
        plt .xlim ([0.0 ,1.0 ])
        plt .ylim ([0.0 ,1.05 ])
        plt .xlabel ('Recall')
        plt .ylabel ('Precision')
        plt .title ('Binary Classification PR Curve (More Reliable for Imbalanced Data)')
        plt .legend (loc ="lower right")
        plt .tight_layout ()
        plt .savefig (f"{save_dir }/pr_curve.pdf",dpi =300 )
        plt .close ()
        print (f"PR曲线已保存至: {save_dir }/pr_curve.pdf")

    def _plot_length_sensitivity (self ,sequences ,y_true ,y_pred ,save_dir ):
        """Docstring translated to English."""
        length_groups ={
        "Short sequences (≤30aa)":[],
        "Medium sequences (30-100aa)":[],
        "Long sequences (>100aa)":[]
        }
        # NOTE: comment translated to English.
        for seq ,true ,pred in zip (sequences ,y_true ,y_pred ):
            l =len (seq )
            if l <=30 :
                length_groups ["Short sequences (≤30aa)"].append ((true ,pred ))
            elif 30 <l <=100 :
                length_groups ["Medium sequences (30-100aa)"].append ((true ,pred ))
            else :
                length_groups ["Long sequences (>100aa)"].append ((true ,pred ))

        metrics ={"f1":[],"accuracy":[]}
        labels =[]
        for group_name ,data in length_groups .items ():
            if not data :
                continue 
            labels .append (group_name )
            yt ,yp =zip (*data )
            metrics ["f1"].append (f1_score (yt ,yp ,zero_division =0 ))
            metrics ["accuracy"].append (accuracy_score (yt ,yp ))

        x =np .arange (len (labels ))
        width =0.35 
        plt .figure (figsize =(10 ,6 ))
        plt .bar (x -width /2 ,metrics ["accuracy"],width ,label ='Accuracy')
        plt .bar (x +width /2 ,metrics ["f1"],width ,label ='F1 Score')
        plt .ylim (0 ,1.05 )
        plt .xlabel ('Sequence Length Groups')
        plt .ylabel ('Score')
        plt .title ('Model Performance on Different Sequence Lengths (Length Sensitivity)')
        plt .xticks (x ,labels ,rotation =30 )
        plt .legend ()
        plt .tight_layout ()
        plt .savefig (f"{save_dir }/length_sensitivity.pdf",dpi =300 )
        plt .close ()
        print (f"长度敏感性分析图已保存至: {save_dir }/length_sensitivity.pdf")

    def _plot_loss_curve (self ,save_dir ):
        """Docstring translated to English."""
        epochs =range (1 ,len (self .train_metrics ["loss"])+1 )
        plt .figure (figsize =(8 ,5 ))
        plt .plot (epochs ,self .train_metrics ["loss"],'b-',label ='Training Loss')
        plt .plot (epochs ,self .val_metrics ["loss"],'g-',label ='Validation Loss')
        plt .title ('Training/Validation Loss Curve')
        plt .xlabel ('Epoch')
        plt .ylabel ('Loss Value')
        plt .legend ()
        plt .tight_layout ()
        plt .savefig (f"{save_dir }/loss_curve.pdf",dpi =300 )
        plt .close ()

    def _plot_accuracy_curve (self ,save_dir ):
        """Docstring translated to English."""
        epochs =range (1 ,len (self .train_metrics ["accuracy"])+1 )
        plt .figure (figsize =(8 ,5 ))
        plt .plot (epochs ,self .train_metrics ["accuracy"],'b-',label ='Training Accuracy')
        plt .plot (epochs ,self .val_metrics ["accuracy"],'g-',label ='Validation Accuracy')
        plt .title ('Training/Validation Accuracy Curve')
        plt .xlabel ('Epoch')
        plt .ylabel ('Accuracy (%)')
        plt .ylim (0 ,100 )
        plt .legend ()
        plt .tight_layout ()
        plt .savefig (f"{save_dir }/accuracy_curve.pdf",dpi =300 )
        plt .close ()

    def _plot_mcc_curve (self ,save_dir ):
        """Docstring translated to English."""
        epochs =range (1 ,len (self .train_metrics ["mcc"])+1 )
        plt .figure (figsize =(8 ,5 ))
        plt .plot (epochs ,self .train_metrics ["mcc"],'b-',label ='Training MCC')
        plt .plot (epochs ,self .val_metrics ["mcc"],'g-',label ='Validation MCC')
        plt .ylim (-1 ,1 )
        plt .title ('Training/Validation MCC Curve (Matthews Correlation Coefficient)')
        plt .xlabel ('Epoch')
        plt .ylabel ('MCC Value')
        plt .legend ()
        plt .tight_layout ()
        plt .savefig (f"{save_dir }/mcc_curve.pdf",dpi =300 )
        plt .close ()
        print (f"MCC曲线已保存至: {save_dir }/mcc_curve.pdf")

    def _plot_roc_curve (self ,y_true ,y_probs ,save_dir ):
        """Docstring translated to English."""
        fpr ,tpr ,_ =roc_curve (y_true ,y_probs )
        roc_auc =auc (fpr ,tpr )
        plt .figure (figsize =(8 ,6 ))
        plt .plot (fpr ,tpr ,lw =2 ,label =f'ROC Curve (AUC = {roc_auc :.3f})')
        plt .plot ([0 ,1 ],[0 ,1 ],'k--',lw =2 )# NOTE: comment translated to English.
        plt .xlim ([0.0 ,1.0 ])
        plt .ylim ([0.0 ,1.05 ])
        plt .xlabel ('False Positive Rate (FPR)')
        plt .ylabel ('True Positive Rate (TPR)')
        plt .title ('Binary Classification ROC Curve')
        plt .legend (loc ="lower right")
        plt .tight_layout ()
        plt .savefig (f"{save_dir }/roc_curve.pdf",dpi =300 )
        plt .close ()

    def _plot_confusion_matrix (self ,y_true ,y_pred ,save_dir ):
        """Docstring translated to English."""
        cm =confusion_matrix (y_true ,y_pred )
        plt .figure (figsize =(8 ,6 ))
        sns .heatmap (cm ,annot =True ,fmt ='d',cmap ='Blues',
        xticklabels =['0 (Negative)','1 (Positive)'],
        yticklabels =['0 (Negative)','1 (Positive)'])
        plt .xlabel ('Predicted Label')
        plt .ylabel ('True Label')
        plt .title ('Binary Classification Confusion Matrix')
        plt .tight_layout ()
        plt .savefig (f"{save_dir }/confusion_matrix.pdf",dpi =300 )
        plt .close ()

    def _reset_metrics (self ):
        """Docstring translated to English."""
        self .train_metrics ={key :[]for key in self .train_metrics }
        self .val_metrics ={key :[]for key in self .val_metrics }

    def train_model (self ,train_sequences ,train_labels ,val_sequences ,val_labels ,
    test_sequences ,test_labels ,fold_num ,
    save_metrics_dir ="model_metrics",
    batch_size =16 ,epochs =10 ,lr =1e-4 ,
    early_stop_patience =3 ,min_delta =0.001 ):
        """Docstring translated to English."""
        # NOTE: comment translated to English.
        self ._reset_metrics ()

        # NOTE: comment translated to English.
        fold_save_dir =os .path .join (save_metrics_dir ,f"fold_{fold_num }")
        os .makedirs (fold_save_dir ,exist_ok =True )

        # NOTE: comment translated to English.
        X_train ,y_train =train_sequences ,train_labels 
        X_val ,y_val =val_sequences ,val_labels 
        X_test ,y_test =test_sequences ,test_labels 
        print (f"\n--- Fold {fold_num } ---")
        print (f"训练集: {len (X_train )}条（阳性{sum (y_train )}条）")
        print (f"验证集: {len (X_val )}条（阳性{sum (y_val )}条）")

        # NOTE: comment translated to English.
        train_dataset =PeptideDataset (X_train ,y_train )
        val_dataset =PeptideDataset (X_val ,y_val )
        test_dataset =PeptideDataset (X_test ,y_test )

        # NOTE: comment translated to English.
        def collate_fn (batch ):
            sequences ,labels ,seq_lengths =zip (*batch )
            # NOTE: comment translated to English.
            encoded =self .tokenizer (
            sequences ,
            return_tensors =None ,
            padding ="do_not_pad",
            truncation =True ,
            max_length =256 
            )
            input_ids_list =encoded ["input_ids"]
            attention_mask_list =encoded ["attention_mask"]
            # NOTE: comment translated to English.
            batch_max_len =max (len (ids )for ids in input_ids_list )
            pad_token_id =self .tokenizer .pad_token_id 
            padded_input_ids =[]
            padded_attention_mask =[]
            # NOTE: comment translated to English.
            for ids ,mask in zip (input_ids_list ,attention_mask_list ):
                pad_len =batch_max_len -len (ids )
                padded_ids =ids +[pad_token_id ]*pad_len 
                padded_mask =mask +[0 ]*pad_len 
                padded_input_ids .append (padded_ids )
                padded_attention_mask .append (padded_mask )
                # NOTE: comment translated to English.
            return (torch .tensor (padded_input_ids ,dtype =torch .long ),
            torch .tensor (padded_attention_mask ,dtype =torch .long ),
            torch .tensor (labels ,dtype =torch .long ))

            # NOTE: comment translated to English.
        train_loader =DataLoader (train_dataset ,batch_size =batch_size ,shuffle =True ,collate_fn =collate_fn )
        val_loader =DataLoader (val_dataset ,batch_size =batch_size ,collate_fn =collate_fn )
        test_loader =DataLoader (test_dataset ,batch_size =batch_size ,collate_fn =collate_fn )

        # NOTE: comment translated to English.
        criterion =nn .CrossEntropyLoss ()# NOTE: comment translated to English.
        optimizer =optim .Adam (
        [p for p in self .model .parameters ()if p .requires_grad ],# NOTE: comment translated to English.
        lr =lr ,
        weight_decay =1e-4 # NOTE: comment translated to English.
        )

        # NOTE: comment translated to English.
        best_val_f1 =-1.0 
        early_stop_count =0 

        # NOTE: comment translated to English.
        for epoch in range (epochs ):
            self .model .train ()# NOTE: comment translated to English.
            running_loss =0.0 
            correct =0 
            total =0 
            train_y_true =[]
            train_y_pred =[]

            # NOTE: comment translated to English.
            progress_bar =tqdm (enumerate (train_loader ),total =len (train_loader ),leave =False )
            for batch_idx ,(input_ids ,attention_mask ,targets )in progress_bar :
            # NOTE: comment translated to English.
                input_ids =input_ids .to (self .device )
                attention_mask =attention_mask .to (self .device )
                targets =targets .to (self .device )

                # NOTE: comment translated to English.
                outputs =self .model (input_ids =input_ids ,attention_mask =attention_mask )
                loss =criterion (outputs ,targets )# NOTE: comment translated to English.

                # NOTE: comment translated to English.
                optimizer .zero_grad ()# NOTE: comment translated to English.
                loss .backward ()# NOTE: comment translated to English.
                optimizer .step ()# NOTE: comment translated to English.

                # NOTE: comment translated to English.
                running_loss +=loss .item ()
                _ ,predicted =outputs .max (1 )# NOTE: comment translated to English.
                total +=targets .size (0 )
                correct +=predicted .eq (targets ).sum ().item ()
                train_y_true .extend (targets .cpu ().numpy ())
                train_y_pred .extend (predicted .cpu ().numpy ())
                # NOTE: comment translated to English.
                progress_bar .set_description (
                f"Fold {fold_num } Epoch {epoch +1 } - Loss: {running_loss /(batch_idx +1 ):.4f}, "
                f"Acc: {100. *correct /total :.2f}%"
                )

                # NOTE: comment translated to English.
            train_loss =running_loss /len (train_loader )
            train_acc ,train_precision ,train_recall ,train_f1 ,train_mcc =self ._compute_binary_metrics (train_y_true ,train_y_pred )
            self .train_metrics ["loss"].append (train_loss )
            self .train_metrics ["accuracy"].append (train_acc *100 )
            self .train_metrics ["precision"].append (train_precision )
            self .train_metrics ["recall"].append (train_recall )
            self .train_metrics ["f1"].append (train_f1 )
            self .train_metrics ["mcc"].append (train_mcc )

            # NOTE: comment translated to English.
            val_loss ,val_acc ,val_y_true ,val_y_pred =self .evaluate_model (val_loader ,criterion )
            val_acc ,val_precision ,val_recall ,val_f1 ,val_mcc =self ._compute_binary_metrics (val_y_true ,val_y_pred )
            self .val_metrics ["loss"].append (val_loss )
            self .val_metrics ["accuracy"].append (val_acc *100 )
            self .val_metrics ["precision"].append (val_precision )
            self .val_metrics ["recall"].append (val_recall )
            self .val_metrics ["f1"].append (val_f1 )
            self .val_metrics ["mcc"].append (val_mcc )

            print (f"Fold {fold_num } Epoch {epoch +1 } - Val Loss: {val_loss :.4f}, Val Acc: {val_acc *100 :.2f}%, Val F1: {val_f1 :.4f}")

            # NOTE: comment translated to English.
            if val_f1 >best_val_f1 +min_delta :
                best_val_f1 =val_f1 
                early_stop_count =0 
                # NOTE: comment translated to English.
                model_filename =os .path .join (fold_save_dir ,"esm2_8m_binary_best.pt")
                torch .save ({
                'model_state_dict':self .model .state_dict (),
                'epoch':epoch +1 ,
                'val_f1':val_f1 ,
                'fold_num':fold_num ,
                },model_filename )
            else :
                early_stop_count +=1 
                if early_stop_count >=early_stop_patience :
                    print (f"Fold {fold_num } - Early stopping triggered after {epoch +1 } epochs.")
                    break 

                    # NOTE: comment translated to English.
        print (f"--- Evaluating Fold {fold_num } best model on Test Set ---")
        best_model_path =os .path .join (fold_save_dir ,"esm2_8m_binary_best.pt")
        checkpoint =torch .load (best_model_path ,map_location =self .device ,weights_only =False )
        self .model .load_state_dict (extract_state_dict (checkpoint ))

        # NOTE: comment translated to English.
        self ._plot_loss_curve (fold_save_dir )
        self ._plot_accuracy_curve (fold_save_dir )
        self ._plot_mcc_curve (fold_save_dir )

        # NOTE: comment translated to English.
        test_loss ,test_acc ,test_y_true ,test_y_pred =self .evaluate_model (test_loader ,criterion )
        test_acc ,test_precision ,test_recall ,test_f1 ,test_mcc =self ._compute_binary_metrics (test_y_true ,test_y_pred )

        # NOTE: comment translated to English.
        self .all_folds_metrics .append ({
        'fold':fold_num ,
        'accuracy':test_acc ,
        'precision':test_precision ,
        'recall':test_recall ,
        'f1':test_f1 ,
        'mcc':test_mcc ,
        'model_path':best_model_path 
        })

        # NOTE: comment translated to English.
        self ._plot_confusion_matrix (test_y_true ,test_y_pred ,fold_save_dir )
        test_probs =self ._get_test_probabilities (test_loader )
        self ._plot_roc_curve (test_y_true ,test_probs ,fold_save_dir )
        self ._plot_pr_curve (test_y_true ,test_probs ,fold_save_dir )
        self ._plot_length_sensitivity (X_test ,test_y_true ,test_y_pred ,fold_save_dir )

        # NOTE: comment translated to English.
        report =classification_report (
        test_y_true ,test_y_pred ,
        target_names =['阴性（0）','阳性（1）'],
        zero_division =0 ,
        output_dict =True 
        )
        report_str =classification_report (
        test_y_true ,test_y_pred ,
        target_names =['阴性（0）','阳性（1）'],
        zero_division =0 
        )
        report_str +=f"\n马修斯相关系数（MCC）: {test_mcc :.4f}"
        print (f"Fold {fold_num } Test Set Classification Report:\n",report_str )

        with open (os .path .join (fold_save_dir ,"classification_report.txt"),"w",encoding ="utf-8")as f :
            f .write (report_str )

        return self .model 

    def _get_test_probabilities (self ,dataloader ):
        """Docstring translated to English."""
        self .model .eval ()
        all_probs =[]
        with torch .no_grad ():
            for input_ids ,attention_mask ,_ in dataloader :
                input_ids =input_ids .to (self .device )
                attention_mask =attention_mask .to (self .device )
                outputs =self .model (input_ids =input_ids ,attention_mask =attention_mask )
                probs =torch .softmax (outputs ,dim =1 )[:,1 ].cpu ().numpy ()# NOTE: comment translated to English.
                all_probs .extend (probs )
        return np .array (all_probs )

    def evaluate_model (self ,dataloader ,criterion ):
        """Docstring translated to English."""
        self .model .eval ()# NOTE: comment translated to English.
        test_loss =0 
        correct =0 
        total =0 
        y_true =[]
        y_pred =[]
        with torch .no_grad ():# NOTE: comment translated to English.
            for input_ids ,attention_mask ,targets in dataloader :
                input_ids =input_ids .to (self .device )
                attention_mask =attention_mask .to (self .device )
                targets =targets .to (self .device )
                outputs =self .model (input_ids =input_ids ,attention_mask =attention_mask )
                loss =criterion (outputs ,targets )
                test_loss +=loss .item ()
                _ ,predicted =outputs .max (1 )
                total +=targets .size (0 )
                correct +=predicted .eq (targets ).sum ().item ()
                y_true .extend (targets .cpu ().numpy ())
                y_pred .extend (predicted .cpu ().numpy ())
        test_loss /=len (dataloader )
        test_acc =correct /total 
        return test_loss ,test_acc ,y_true ,y_pred 

    def predict (self ,sequences ,batch_size =16 ,model_path =None ):
        """Docstring translated to English."""
        if not self .model and not model_path :
            raise ValueError ("请加载模型或指定模型路径")
        if model_path :
            print (f"加载ESM2 8M模型: {model_path }")
            checkpoint =torch .load (model_path ,map_location =self .device ,weights_only =False )
            self .model .load_state_dict (extract_state_dict (checkpoint ))
        self .model .eval ()
        predictions =[]
        probabilities =[]
        # NOTE: comment translated to English.
        dataset =PeptideDataset (sequences ,[0 ]*len (sequences ))

        # NOTE: comment translated to English.
        def collate_fn (batch ):
            sequences ,_ ,_ =zip (*batch )
            encoded =self .tokenizer (
            sequences ,return_tensors =None ,padding ="do_not_pad",truncation =True ,max_length =256 
            )
            input_ids_list =encoded ["input_ids"]
            attention_mask_list =encoded ["attention_mask"]
            batch_max_len =max (len (ids )for ids in input_ids_list )
            pad_token_id =self .tokenizer .pad_token_id 
            padded_input_ids =[]
            padded_attention_mask =[]
            for ids ,mask in zip (input_ids_list ,attention_mask_list ):
                pad_len =batch_max_len -len (ids )
                padded_ids =ids +[pad_token_id ]*pad_len 
                padded_mask =mask +[0 ]*pad_len 
                padded_input_ids .append (padded_ids )
                padded_attention_mask .append (padded_mask )
            return torch .tensor (padded_input_ids ,dtype =torch .long ),torch .tensor (padded_attention_mask ,dtype =torch .long )

        dataloader =DataLoader (dataset ,batch_size =batch_size ,collate_fn =collate_fn )
        print (f"开始预测 {len (sequences )} 条序列...")
        with torch .no_grad ():
            for input_ids ,attention_mask in tqdm (dataloader ,total =len (dataloader )):
                input_ids =input_ids .to (self .device )
                attention_mask =attention_mask .to (self .device )
                outputs =self .model (input_ids =input_ids ,attention_mask =attention_mask )
                probs =torch .softmax (outputs ,dim =1 ).cpu ().numpy ()
                predictions .extend (np .argmax (probs ,axis =1 ))# NOTE: comment translated to English.
                probabilities .extend (probs [:,1 ])# NOTE: comment translated to English.
        return predictions ,probabilities 

    def save_predictions (self ,ids ,sequences ,predictions ,probabilities ,output_file ,true_labels =None ):
        """Docstring translated to English."""
        output_dir =os .path .dirname (output_file )
        if output_dir and not os .path .exists (output_dir ):
            os .makedirs (output_dir ,exist_ok =True )
        results =[]
        for i ,(id_val ,seq ,pred ,prob )in enumerate (zip (ids ,sequences ,predictions ,probabilities )):
            true_label =true_labels [i ]if (true_labels and i <len (true_labels ))else "未知"
            results .append ({
            "id":id_val ,"sequence":seq ,"sequence_length":len (seq ),
            "true_label":true_label ,"predicted_label":pred ,
            "predicted_label_name":"阳性（1）"if pred ==1 else "阴性（0）",
            "positive_probability":round (float (prob ),4 )
            })
        pd .DataFrame (results ).to_csv (output_file ,index =False ,encoding ="utf-8-sig")
        print (f"预测结果已保存至: {output_file }")

    def generate_cross_validation_summary (self ,save_dir ):
        """Docstring translated to English."""
        if not self .all_folds_metrics :
            print ("没有可用的交叉验证指标来生成报告。")
            return 

        df_metrics =pd .DataFrame (self .all_folds_metrics )

        summary_report ="五折交叉验证测试集性能综合报告\n"
        summary_report +="="*50 +"\n\n"

        summary_report +=df_metrics [['fold','accuracy','precision','recall','f1','mcc']].to_string (index =False )+"\n\n"

        # NOTE: comment translated to English.
        mean_metrics =df_metrics [['accuracy','precision','recall','f1','mcc']].mean ()
        std_metrics =df_metrics [['accuracy','precision','recall','f1','mcc']].std ()

        summary_report +="各指标平均值 ± 标准差:\n"
        summary_report +=f"  Accuracy: {mean_metrics ['accuracy']:.4f} ± {std_metrics ['accuracy']:.4f}\n"
        summary_report +=f"  Precision: {mean_metrics ['precision']:.4f} ± {std_metrics ['precision']:.4f}\n"
        summary_report +=f"  Recall: {mean_metrics ['recall']:.4f} ± {std_metrics ['recall']:.4f}\n"
        summary_report +=f"  F1-Score: {mean_metrics ['f1']:.4f} ± {std_metrics ['f1']:.4f}\n"
        summary_report +=f"  MCC: {mean_metrics ['mcc']:.4f} ± {std_metrics ['mcc']:.4f}\n"

        print ("\n"+"="*50 )
        print (summary_report )

        # NOTE: comment translated to English.
        with open (os .path .join (save_dir ,"cross_validation_summary.txt"),"w",encoding ="utf-8")as f :
            f .write (summary_report )

            # NOTE: comment translated to English.
        plt .figure (figsize =(12 ,8 ))
        metrics_to_plot =['accuracy','precision','recall','f1','mcc']
        df_metrics [metrics_to_plot ].boxplot ()
        plt .title ('五折交叉验证各指标分布')
        plt .ylabel ('分数')
        plt .xticks (rotation =45 )
        plt .grid (False )
        plt .tight_layout ()
        plt .savefig (os .path .join (save_dir ,"cross_validation_boxplot.pdf"),dpi =300 )
        plt .close ()
        print (f"\n交叉验证综合报告已保存至: {os .path .join (save_dir ,'cross_validation_summary.txt')}")
        print (f"交叉验证箱线图已保存至: {os .path .join (save_dir ,'cross_validation_boxplot.pdf')}")

    def select_best_model (self ,save_dir ):
        """Docstring translated to English."""
        if not self .all_folds_metrics :
            print ("没有可用的模型指标，无法选择最佳模型。")
            return None 

        df_metrics =pd .DataFrame (self .all_folds_metrics )
        # NOTE: comment translated to English.
        df_metrics_sorted =df_metrics .sort_values (by =['f1','mcc'],ascending =[False ,False ])
        best_model_info =df_metrics_sorted .iloc [0 ]

        best_fold =int (best_model_info ['fold'])
        best_f1 =best_model_info ['f1']
        best_mcc =best_model_info ['mcc']
        best_accuracy =best_model_info ['accuracy']
        best_model_path =best_model_info ['model_path']
        if not os .path .exists (best_model_path ):
            raise FileNotFoundError (f"最佳模型文件不存在: {best_model_path }")

            # NOTE: comment translated to English.
        best_model_dir =os .path .join (save_dir ,"best_model_amp")
        os .makedirs (best_model_dir ,exist_ok =True )

        # NOTE: comment translated to English.
        dest_model_path =os .path .join (best_model_dir ,"esm2_8m_binary_best.pt")
        shutil .copy2 (best_model_path ,dest_model_path )

        # NOTE: comment translated to English.
        best_model_info_str =f"最佳模型选择报告\n"
        best_model_info_str +="="*50 +"\n\n"
        best_model_info_str +=f"选择标准：测试集F1分数最高（F1相同则MCC最高）\n\n"
        best_model_info_str +=f"最佳模型所属折数：Fold {best_fold }\n"
        best_model_info_str +=f"模型原始路径：{best_model_path }\n"
        best_model_info_str +=f"模型保存路径：{dest_model_path }\n\n"
        best_model_info_str +=f"测试集性能指标：\n"
        best_model_info_str +=f"  Accuracy: {best_accuracy :.4f}\n"
        best_model_info_str +=f"  Precision: {best_model_info ['precision']:.4f}\n"
        best_model_info_str +=f"  Recall: {best_model_info ['recall']:.4f}\n"
        best_model_info_str +=f"  F1-Score: {best_f1 :.4f}\n"
        best_model_info_str +=f"  MCC: {best_mcc :.4f}\n\n"
        best_model_info_str +=f"五折交叉验证平均指标（参考）：\n"
        mean_metrics =df_metrics [['accuracy','precision','recall','f1','mcc']].mean ()
        best_model_info_str +=f"  平均Accuracy: {mean_metrics ['accuracy']:.4f}\n"
        best_model_info_str +=f"  平均Precision: {mean_metrics ['precision']:.4f}\n"
        best_model_info_str +=f"  平均Recall: {mean_metrics ['recall']:.4f}\n"
        best_model_info_str +=f"  平均F1-Score: {mean_metrics ['f1']:.4f}\n"
        best_model_info_str +=f"  平均MCC: {mean_metrics ['mcc']:.4f}\n"

        # NOTE: comment translated to English.
        info_file_path =os .path .join (best_model_dir ,"best_model_info.txt")
        with open (info_file_path ,"w",encoding ="utf-8")as f :
            f .write (best_model_info_str )

        print ("\n"+"="*60 )
        print ("最佳模型选择完成！")
        print (best_model_info_str )

        return dest_model_path 


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
        id_val =row ['id']
        seq =str (row [seq_col ]).strip ().upper ()# NOTE: comment translated to English.
        if len (seq )==0 :
            print (f"过滤空序列（id: {id_val }）")
            continue 
            # NOTE: comment translated to English.
        try :
            label =int (row ['label'])
            if label not in (0 ,1 ):
                raise ValueError (f"标签必须为0或1，当前为{label }")
        except Exception as e :
            print (f"过滤无效标签（{e }）: {row ['label']}（id: {id_val }）")
            continue 
        ids .append (id_val )
        sequences .append (seq )
        labels .append (label )
    if len (sequences )==0 :
        raise ValueError (f"数据清洗后无有效样本: {csv_path }")
    print (f"从 {os .path .basename (csv_path )} 加载完成：共{len (sequences )}条有效序列（阳性{sum (labels )}条）")
    return ids ,sequences ,labels 


    # NOTE: comment translated to English.
if __name__ =="__main__":
# NOTE: comment translated to English.
    ROOT_DIR ="./"
    base_fold_path =os .path .join (ROOT_DIR ,"mydata/newdata/fold_")# NOTE: comment translated to English.
    test_csv =os .path .join (ROOT_DIR ,"mydata/newdata/rawdata_test.csv")# NOTE: comment translated to English.
    esm2_model_path =os .path .join (ROOT_DIR ,"model/esm2_t6_8M_UR50D")# NOTE: comment translated to English.
    save_metrics_dir =os .path .join (ROOT_DIR ,"model/model_AMP_ESM2_8M_256_v3")# NOTE: comment translated to English.

    # NOTE: comment translated to English.
    batch_size =8 
    epochs =10 
    learning_rate =1e-4 
    early_stop_patience =3 
    device ='cuda'if torch .cuda .is_available ()else 'cpu'
    print (f"使用设备: {device }")

    # NOTE: comment translated to English.
    try :
        print ("\n加载最终测试集...")
        test_ids ,test_sequences ,test_labels =load_csv_data (test_csv )
        if len (test_sequences )==0 :
            print ("测试集无有效数据，无法训练")
            exit (1 )
    except Exception as e :
        print (f"数据加载失败: {e }")
        exit (1 )

        # NOTE: comment translated to English.
    try :
        predictor =PeptidePredictor (esm2_model_name_or_path =esm2_model_path ,device =device )
    except Exception as e :
        print (f"模型初始化失败: {e }")
        exit (1 )

        # NOTE: comment translated to English.
    num_folds =5 
    for fold in range (1 ,num_folds +1 ):
        train_csv_path =f"{base_fold_path }{fold }_train.csv"
        val_csv_path =f"{base_fold_path }{fold }_val.csv"

        print (f"\n{'='*60 }")
        print (f"加载 Fold {fold } 的训练集和验证集...")
        try :
            train_ids ,train_sequences ,train_labels =load_csv_data (train_csv_path )
            val_ids ,val_sequences ,val_labels =load_csv_data (val_csv_path )
            if len (train_sequences )==0 or len (val_sequences )==0 :
                print (f"Fold {fold } 训练集/验证集无有效数据，跳过此折。")
                continue 
        except Exception as e :
            print (f"Fold {fold } 数据加载失败: {e }，跳过此折。")
            continue 

            # NOTE: comment translated to English.
        predictor .load_pretrained_model (freeze_layers =True )

        # NOTE: comment translated to English.
        predictor .train_model (
        train_sequences =train_sequences ,train_labels =train_labels ,
        val_sequences =val_sequences ,val_labels =val_labels ,
        test_sequences =test_sequences ,test_labels =test_labels ,
        fold_num =fold ,
        save_metrics_dir =save_metrics_dir ,
        batch_size =batch_size ,epochs =epochs ,lr =learning_rate ,
        early_stop_patience =early_stop_patience ,min_delta =0.001 
        )

        # NOTE: comment translated to English.
        if device =='cuda':
            torch .cuda .empty_cache ()

            # NOTE: comment translated to English.
    print ("\n"+"="*60 )
    print ("五折交叉验证全部完成！")
    predictor .generate_cross_validation_summary (save_metrics_dir )

    # NOTE: comment translated to English.
    best_model_path =predictor .select_best_model (save_metrics_dir )

    # NOTE: comment translated to English.
    print (f"\n所有模型和指标文件已保存至: {save_metrics_dir }")
    if best_model_path is None :
        print ("未选出最佳模型：请检查各折训练是否成功生成有效指标。")
    else :
        print (f"最佳模型已保存至: {best_model_path }")
        print (f"最佳模型说明文件: {os .path .join (os .path .dirname (best_model_path ),'best_model_info.txt')}")
