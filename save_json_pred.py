import json
import pandas as pd

if __name__ == "__main__":
    if not hasattr(__builtins__, '__IPYTHON__'):
        import argparse

        parser = argparse.ArgumentParser()

        parser.add_argument("--data_path", type=str, help="Data path with predicted relations in .tsv format")
        parser.add_argument("--res_path", type=str, help="The path to the output data in .json format, where the field with predicted relations is filled")
        parser.add_argument("--origin_json_file_path", type=str, help="The path to the original json file, on the basis of which the prediction was made ")
        parser.add_argument("--label_map_path", type=str,
                            help="Path before matching label indices to label names")

        args = parser.parse_args()

        data_path = args.data_path
        res_path = args.res_path
        origin_json_file = args.origin_json_file_path
        label_map = args.label_map_path

    else:
        data_path = "./results/predictions_all_RE_roberta_large_exp1.tsv"
        res_path = "./data/pred_all_RE.json"
        origin_json_file = "./data/test_RE_all_types_RDRS.json"
        label_map = "./data/prepared/label_map.json"
    with open(label_map, 'r') as f:
        label_map = json.load(f)
    reversed_label_map = {v:k for k, v in label_map.items()}
    with open(origin_json_file, 'r') as f:
        origin_json_data = json.load(f)
    pred_df = pd.read_csv(data_path, sep = '\t')
    dict_flag = True if type(origin_json_data)==dict else False
    a = set(pred_df['text_id'].map(str))
    b = set([rev['text_id'] for rev in origin_json_data])
    assert set.intersection(a, b) == a
    for _, cur_row in pred_df.iterrows():
        if dict_flag:
            cur_sample = origin_json_data[str(cur_row['text_id'])]
        else:
            for cur_sample in origin_json_data:
                if "text_id" in cur_sample.keys():
                    if str(cur_sample['text_id'])==str(cur_row['text_id']):
                        break
                else:
                    if cur_sample['meta']['xmi']['meta']['fileName'].replace('.xmi', '')==str(cur_row["text_id"]):
                        break
            else:
                raise Exception("There is no text with id %s in origin json file"%cur_row['text_id'])
        for r, rel in enumerate(cur_sample['relations']):
            if 'rel_id' in rel.keys():
                if int(rel['rel_id'])==int(cur_row['rel_id']):
                    break
            elif 'relation_id' in rel.keys():
                if int(rel['relation_id'])==int(cur_row['rel_id']):
                    break
            else:
                if r==int(cur_row['rel_id']):
                    break
            #else:
            #    raise Exception("Relations in original json file don't have ids")
        else:
            raise Exception("There is no relation with relation id %s in text with id %s in original json file"%(cur_row['rel_id'], cur_row['text_id']))
        rel['relation_type'] = reversed_label_map[cur_row['pred_labels']]
    with open(res_path, 'w') as f:
        json.dump(origin_json_data, f)
