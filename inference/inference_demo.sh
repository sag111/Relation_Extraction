#! /bin/sh

#path to the json file with "relations" field, that contains pairs of entities to be classified
export INP_JSON_FILE="./inference_data.json"
#path to trained model
export RE_MODEL_DIR="/s/ls4/users/romanrybka/pharm_er/true_RE/Anton_RE/model/RDRS_multicontext_fold1/best"
#path to a file that contains class indices of the model and their names
export LABEL_MAP="/s/ls4/users/romanrybka/pharm_er/true_RE/Anton_RE/model/RDRS_multicontext_fold1/best/label_map.json"
#the path where the temporal files with predictions will be saved, as well as all intermediate files that are obtained during the inference
export TEMP_DIR="./temp_data/"
#path to the json file with predicted relations in the "relations" field
export RES_JSON_FILE="./RE/inference_results.json"


mkdir -p $TEMP_DIR
echo prepare dataframe for the model evaluation, here is the statistics of the data:
python ../prepare_ds.py $INP_JSON_FILE $TEMP_DIR/inference_data.tsv --mode NER_fixed --label_map_path $LABEL_MAP
echo start model prediction
python ../predict.py $TEMP_DIR/inference_data.tsv --model_path $RE_MODEL_DIR --res_path $TEMP_DIR/inference_data_pred.tsv

echo evaluating results
python ../evaluate.py $TEMP_DIR/inference_data_pred.tsv

echo saving file with predictions in json format
python ../save_json_pred.py --data_path $TEMP_DIR/inference_data_pred.tsv --origin_json_file_path ./inference_data.json --label_map_path $LABEL_MAP --res_path $RES_JSON_FILE

echo deleting temporal files
rm -rf $TEMP_DIR
echo successful completion
