from model import LM_RE_classification_model
import pandas as pd
import argparse
import json
import os

import random
from torch import manual_seed 
import numpy as np

RANDOM_SEED = 42

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
manual_seed(RANDOM_SEED)


# Аргументы командной строки
parser = argparse.ArgumentParser()

parser.add_argument("train_data_path", type=str, help="Path to prepared train data in .tsv format ")
parser.add_argument("eval_data_path", type=str, help="Path to prepared eval data in .tsv format ")

parser.add_argument("--model_path", 
                    type=str, 
                    help="Path to the model in torch huggingface transformers format, which is compatible with Classification Model SimpleTransformers.", 
                    required=True)

#TODO: определять тип модели по данным модели
parser.add_argument("--model_type", 
                    type=str, 
                    help="Model type",
                    required=True)

parser.add_argument("--model_out_path", 
                    type=str, 
                    help="Path to save the model", 
                    default="../outputs/")

parser.add_argument("--args_path",
                    type=str,
                    help="Path to model config in a .json format.")

parser.add_argument("--n_gpu",
                    type=int,
                    help="the number of gpu to use",
                    default=1)

parser.add_argument("--cuda_visible_devices",
                   type=str,
                   help="Environment variable which video card to use.",
                   default="0")

#Сделать как-то иначе, уучше
parser.add_argument("--special_tokens",
                   type=str,
                   help="string, comma-separated list of tokens that need to be added to the model dictionary",
                   default=" [SEP] , [TXTSEP] ")

parser.add_argument("--label_nb",
                   type=int,
                   help="The number of model classes",
                   default=0)


args = parser.parse_args()

train_path = args.train_data_path
eval_path = args.eval_data_path

model_path = args.model_path
model_type = args.model_type

model_out_path = args.model_out_path

args_path = args.args_path

special_tokens = args.special_tokens.split(",")
label_nb = args.label_nb

n_gpu = args.n_gpu
os.environ["CUDA_VISIBLE_DEVICES"]=args.cuda_visible_devices

# Read the data
train_df = pd.read_csv(train_path, sep="\t")
eval_df = pd.read_csv(eval_path, sep="\t")

#if there are nan values
train_df.fillna(0, inplace=True)
eval_df.fillna(0, inplace=True)

# Read model config
with open(args_path, "r") as model_args_file:
    model_args=json.load(model_args_file)
    
model_args["output_dir"] = model_out_path
model_args["best_model_dir"] = model_out_path + "/best/"
model_args["manual_seed"] = RANDOM_SEED
model_args["n_gpu"] = n_gpu
model_args["labels_list"]=[i for i in range(train_df.labels.nunique())]

if label_nb==0:
    if "label_nb" in model_args:
        label_nb=model_args["label_nb"]
    
    elif "labels_list" in model_args:
        label_nb=len(model_args["labels_list"])

    else:
        label_nb=train_df.labels.nunique()
        #TODO: warning

#Creating model and fit
print('class number: %s'%label_nb)

fit_model = LM_RE_classification_model(model_type, 
                                       model_path,
                                       save_model_path=model_out_path,
                                       model_args=model_args,
                                       spec_tokens=special_tokens,
                                       labels_nb=label_nb)

fit_report = fit_model.fit(train_df, 
                           eval_df, 
                           fit_args=model_args)


print("Model saved in: {}. Training is completed.".format(model_out_path))
#TODO: time
