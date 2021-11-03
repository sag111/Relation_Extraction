from model import LM_RE_classification_model
import pandas as pd
import argparse
import json
import os

parser = argparse.ArgumentParser()

parser.add_argument("data_path", type=str, help="Path to prepared train data in .tsv format")

parser.add_argument("--model_path", 
                    type=str, 
                    help="Path to the model in torch huggingface transformers format, which is compatible with Classification Model SimpleTransformers.", 
                    required=True)

parser.add_argument("--res_path",
                    type=str,
                    help="Path for saving predictions",
                    default="/results/")

parser.add_argument("--args_path",
                    type=str,
                    help="The path to the model parameters in .json format. If not specified, the file model_args.json from model_path is taken. In the same place, you should indicate the type of model under the key model_type ")

parser.add_argument("--cuda_visible_devices",
                   type=str,
                   help="Environment variable which video card to use.",
                   default="0")


args = parser.parse_args()

data_path = args.data_path
model_path = args.model_path
res_path = args.res_path
args_path = args.args_path

os.environ["CUDA_VISIBLE_DEVICES"]=args.cuda_visible_devices


if args_path:
    model_args_path = args_path
else:
    model_args_path = model_path+"/model_args.json"

with open(model_args_path, "r") as model_args_file:
    model_args=json.load(model_args_file)

model_type=model_args["model_type"]

#Read data
data_to_pred = pd.read_csv(data_path, sep="\t")

if "label_nb" in model_args:
    label_nb=model_args["label_nb"]
    
elif "labels_list" in model_args:
    label_nb=len(model_args["labels_list"])
    
else:
    label_nb=data_to_pred.labels.nunique()
    #TODO: warning

print("Class number: {}".format(label_nb))
    

pred_model = LM_RE_classification_model(model_type, 
                                       model_path,
                                       model_args=model_args,
                                       labels_nb=label_nb)

preds_df = pred_model.predict(data_to_pred,
                              save_path=res_path)

print("The work is done. Result is saved in {}".format(res_path))
