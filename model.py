#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from simpletransformers.classification import ClassificationModel

import torch 
import numpy as np
import random

import json
import os
import argparse


# In[2]:
#class for correct dump of json data 
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


class LM_RE_classification_model():
    def __init__(self, 
                 model_type, 
                 model_path,
                 save_model_path=None,
                 model_args=None,
                 spec_tokens=None,
                 labels_nb=5):
        """Loads the language model into the ClassificationModel simpletransformers for future use.
           model_type - str - supported model type (e.g. xlmroberta);
           model_path - str -path to the model;
           save_model_path - str - path where to save the trained model;
           model_args - dict - model parameters;
           spec_tokens - set of str - special tokens to be added to the model dictionary;
           labels_nb is the number of classes that the model should define."""
        
        self.model_type = model_type
        self.model_path = model_path
        self.model_args = model_args
        self.labels_nb = labels_nb
        
        self.save_model_path = save_model_path
        
        self.model = ClassificationModel(model_type, 
                                         model_path, 
                                         args=model_args, 
                                         num_labels=labels_nb)
        
        self._add_spec_tokens_to_vocab(spec_tokens)
        
                    
    
    def _add_spec_tokens_to_vocab(self, spec_tokens):
        """Adds special tokens to the model dictionary if they are not already there."""
        if spec_tokens is not None:
            special_token_set = set()
            for cur_spec_token in spec_tokens:
                special_token_set.add(cur_spec_token)
                    
                    
            special_tokens_dict = {'additional_special_tokens': list(special_token_set)}
            num_added_toks = self.model.tokenizer.add_special_tokens(special_tokens_dict)
            self.model.model.resize_token_embeddings(len(self.model.tokenizer))
    
    def fit(self, 
            train_df, 
            eval_df, 
            fit_args=None,
            save_report = True,
            save_report_path = None):
        """Trains the model using the specified parameters.
         train_df - pandas dataframe with text and labels columns, data on which the model is trained;
         eval_df - pandas dataframe similar, used for model validation;
         fit_args - Dictionary, ClassificationModel training parameters in SimpleTransformers;
         save_report - whether to save model training data;
         save_report_path - the path to save the model training data, if not specified, is saved to save_model_path. """
        
        if fit_args is None:
            raise RuntimeError("There are no model args.")
            
        training_report = self.model.train_model(train_df, 
                                            eval_df = eval_df, 
                                            output_dir=self.save_model_path,
                                            args=fit_args)
        
        report_dict = dict()

        report_dict["training_report"] = training_report
        report_dict["model_args"] = self.model_args 
        report_dict["training_args"] = fit_args
        
        # Сохранение отчёта об обучении
        if save_report:
            if save_report_path is None:
                cur_save_report_path = self.save_model_path + "/training_report.json"
            else:
                cur_save_report_path = save_report_path
            
            with open(cur_save_report_path, "w") as training_report_file:
                json.dump(report_dict, training_report_file, cls = NpEncoder)
        
        return report_dict
    
    
    def predict(self, test_data, return_df=True, return_both=False, save_path=None):
        """Predicts classes.
         test_data - as a df with a text field.
         return_df - if True, will return df with predictions, otherwise - tuple output of the Classification Model;
         return_both - if True, ignore the previous one, return df and preds.
         save_path - path where to save pred_df.
        
         Returns:
         tuple - (predicted classes; output network activities) """ 
        preds = self.model.predict(test_data["text"].to_list())
        
        res_df = test_data
        res_df["pred_labels"] = preds[0]
        
        res_df = self._add_pred_to_df(res_df, preds)
        
        
        if save_path:
            res_df.to_csv(save_path, sep="\t", index=False)
        
        if return_both:
            return res_df, preds
        
        if return_df:
            return res_df
        else:
            return preds
    
    
    def _add_pred_to_df(self, df, pred):
        
        df_preds = df.copy()
        
        classes_nb = len(pred[1][0])
        classes_dict = {i:list() for i in range(classes_nb)}

        for cur_pred in pred[1]:
            for i in range(classes_nb):
                classes_dict[i].append(cur_pred[i])

        for cur_key in classes_dict:
            df_preds[f"act_cls_{cur_key}"] = classes_dict[cur_key]
        
        return df_preds

# In[3]:


if __name__=="__main__":
    model_type = "xlmroberta"
    model_path = "/home/aselivanov/language_models/roberta_sag/sag_xlm_roberta_kfu_2ep/"

    # basic
    train_data_path = "../data/dev_DDI/tr.tsv"
    valid_data_path = "../data/dev_DDI/tr.tsv"

    random_seed = 42
    n_gpu = 1
    os.environ["CUDA_VISIBLE_DEVICES"]="0"

    # training settings
    use_lr_decay = True
    batch_size = 8
    learning_rate = 1e-5
    nb_epochs  = 2
    max_seq_length = 512
    warmup_ratio = 0.1

    #early stopping
    early_stopping = True
    early_stopping_patience = 4
    early_stopping_metric = "eval_loss"
    early_stopping_minimize = True

    # saving model
    save_every_epoch = False
    save_best_model = True
    save_model_dir = "../model/roberta_test_pipeline/"

    wandb_project = ""
    


# In[4]:


if __name__=="__main__":
    model_args = dict()

    if use_lr_decay:
        model_args["scheduler"] = "polynomial_decay_schedule_with_warmup"

    # Saving model
    model_args["save_model_every_epoch"] = save_every_epoch
    model_args["save_best_model"] = save_best_model
    model_args["output_dir"] = save_model_dir
    model_args["best_model_dir"] = save_model_dir + "/best/"
    model_args["save_eval_checkpoints"] = False

    # Early Stopping
    model_args["use_early_stopping"] = early_stopping
    model_args["early_stopping_metric"] = early_stopping_metric
    model_args["early_stopping_patience"] = early_stopping_patience
    model_args["early_stopping_minimize"] = early_stopping_minimize
    model_args["early_stopping_consider_epochs"] = False

    # Other
    model_args["evaluate_during_training"] = True
    model_args["manual_seed"] = random_seed
    model_args["wandb_project"] = wandb_project

    # Model parametrs
    model_args["n_gpu"] = n_gpu
    model_args["num_train_epochs"] = nb_epochs
    model_args["learning_rate"] = learning_rate
    model_args["train_batch_size"] = batch_size
    model_args["warmup_ratio"] = warmup_ratio
    model_args["overwrite_output_dir"] = True
    model_args["max_seq_length"] = max_seq_length


# In[5]:


if __name__ == "__main__":
    random_seed = 42
    train_data_path = "../data/dev_DDI/tr.tsv"
    valid_data_path = "../data/dev_DDI/tr.tsv"
    
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    
    train_df = pd.read_csv(train_data_path, sep="\t")

    if valid_data_path:
        valid_df = pd.read_csv(valid_data_path, sep="\t")
    
    
    model = LM_RE_classification_model("xlmroberta", 
                                       "/home/aselivanov/language_models/roberta_sag/sag_xlm_roberta_kfu_2ep/",
                                       save_model_path="../model/roberta_test_pipeline/",
                                       model_args=model_args,
                                       labels_nb=4)
    
    
    report = model.fit(train_df, valid_df, fit_args=model_args)
    
    print(report)
    
    del model
    
    print("prediction")
    pred_model = LM_RE_classification_model("xlmroberta",
                                            "../model/roberta_test_pipeline/",
                                            save_model_path=None,
                                            model_args=None,
                                            labels_nb=4)
    
    test_data_path = "../data/dev_DDI/ts.tsv"
    test_df = pd.read_csv(test_data_path, sep="\t")
    
    preds = pred_model.predict(test_df["text"].values)
    
    print("Succes.")

