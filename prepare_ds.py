#!/usr/bin/env python
# coding: utf-8

# In[1]:


import json
import pandas as pd
from numpy import mean, std
from os.path import dirname
import warnings


# In[2]:


class Dataset_processor():
    def __init__(self, data_path, 
                 entity_separator=" [SEP] ",
                 text_separator=" [TXTSEP] ",
                 mode = "NER_fixed",
                 map_dict=None):
        """Loads data in .json format, converts it to .csv for reading by language models, changes the type of text to the desired one.
         Input data:
         data_path - str - path to json file with source data. See also data_path
         entity_separator - str is a special token for separating a pair of entities.
         map_dict - dict - dictionary-map with correlation of classes and indices (key - class name, value - index)
         """ 
        assert mode in ['NER_fixed', 'Joint']
        self.data_path = data_path
        self.raw_data = self.load_data(data_path)
        if mode=="Joint":
            self.raw_data = self.convert2RE_format(self.raw_data)
            with open('RE_filled_file.json', 'w') as f:
                json.dump(self.raw_data, f)
        self.entity_separator = entity_separator
        self.text_separator = text_separator
        self.special_tokens = [entity_separator, text_separator]

            

        
        self.df_data_cols = ["rel_id","first_entity_text","second_entity_text","relation_type", "text", "text_id"]
        
        self.rel_df = self.get_data_rel_df(self.raw_data)
        
        self.set_label_map(self.rel_df["relation_type"].values, map_dict)
        #self.re_class_map = self.get_label_map()
        
        
    def load_data(self, data_path):
        with open(data_path, "r") as inp_file:
            inp_data = json.load(inp_file)

        if type(inp_data) is dict:
            inp_data = list(inp_data.values())
        
        return inp_data
    
    
    def _get_text_rel_df(self, test_text_data):

        cols = self.df_data_cols
        relation_list = list()

        for i, cur_relation in enumerate(test_text_data["relations"]):
            temp_dict = dict()
            
            if "relation_id" in cur_relation.keys():
                temp_dict['rel_id'] = cur_relation["relation_id"]
            else:
                temp_dict["rel_id"] = i
            temp_dict["first_entity_text"] = cur_relation["first_entity"]["text"]
            temp_dict["second_entity_text"] = cur_relation["second_entity"]["text"]
            temp_dict["relation_type"] = cur_relation["relation_type"]

            relation_list.append(temp_dict)

        if relation_list:
            cur_data_rels_df = pd.DataFrame(relation_list)
            cur_data_rels_df["text"] = test_text_data["text"]
            if "text_id" in test_text_data.keys():
                cur_data_rels_df["text_id"] = test_text_data["text_id"]
            else:
                cur_data_rels_df["text_id"] = test_text_data['meta']['xmi']['meta']['fileName'].replace('.xmi', '')#["text_id"]
        else:
            cur_data_rels_df = pd.DataFrame(columns=cols)
        return cur_data_rels_df
    
    
    def get_data_rel_df(self, raw_data, verbose=True):

        cols = self.df_data_cols
        
        res_rel_df = pd.DataFrame(columns = cols)
        
        rel_nb = 0

        for cur_text_data in self.raw_data:
            cur_rel_df = self._get_text_rel_df(cur_text_data)
            rel_nb+=len(cur_rel_df)
            res_rel_df = res_rel_df.append(cur_rel_df)
        
        if verbose:
            print("Число связей в dataframe: ", rel_nb)
        return res_rel_df
    

    def sorting(self, s):
        if s is None:
            return 0
        else:
            return len(s) 
    
    def set_label_map(self, raw_labels, map_dict=None):
        """assignment of classes to indices"""
        pass
        if map_dict:        
            self.re_class_map = map_dict
            self.reverse_re_class_map = {v:k for k,v in map_dict.items()}

        else:
            raw_labels_set = list(set(raw_labels))
            raw_labels_set.sort(key=self.sorting)
            self.re_class_map = {k:num for num, k in enumerate(raw_labels_set)}
            self.reverse_re_class_map = {v:k for k,v in self.re_class_map.items()}
    
    
    def get_label_map(self, reverse=False):
        """get assignment of classes to indices"""
        if reverse:
            return self.reverse_re_class_map
        else:
            return self.re_class_map
    
    
    def _get_txt_len_list(self,re_df):
        return [len(cur_txt) for cur_txt in re_df.text.values]
    
    def data_stats(self):
        """Corpus statistics."""
        print("Corpus statistics.")
        print()
        print("Number of texts: ", len(self.raw_data))
        print("Number of relations: ", len(self.rel_df))
        print()
        print("Number of texts by relation type: ")
        print(self.rel_df["relation_type"].value_counts())
        print()
        print()
        txt_len = self._get_txt_len_list(self.rel_df)
        
        print("Average length of text in characters in the corpus: ", mean(txt_len), "+-", std(txt_len))
        print("Min: ", min(txt_len))
        print("Max: ", max(txt_len))
        
        
    def process_data(self, res_format="pair_and_text", verbose=True):
        """res_form - str - the form in which the texts should be presented.
                Valid values:
                - pair_and_text - the text of the pair of entities and the text;
                - pair - text of a pair of entities;
                - text - the original text.
                
            verbose - display statistics on the length of texts after conversion.
            
            Returns:
            pandas dataframe:
                the labels column contains class indices;
                column text - converted text;
                column inp_text - original texts. """
        
        res_form = list()
        res_txt_length = list()
        
        available_res_forms = {"pair_and_text", "pair", "text"}
        
        if res_format not in available_res_forms:
            raise NotImplementedError(f"Выходная форма {res_form} не реализована. Возможные варианты: {available_res_forms}.")
        
        
        for cur_row_id, cur_row in self.rel_df.iterrows():
            
            if res_format == "pair_and_text":
                cur_res_form = cur_row["first_entity_text"] + self.entity_separator + cur_row["second_entity_text"] + self.text_separator + cur_row["text"]
            
            if res_format == "pair":
                cur_res_form = cur_row["first_entity_text"] + self.entity_separator + cur_row["second_entity_text"]
            
            if res_format == "text":
                cur_res_form = cur_row["text"]
            
            
            res_txt_length.append(len(cur_res_form))
            res_form.append(cur_res_form)
            
        if verbose and res_txt_length:
            print("The minimum length of the final form of the text in characters", 
                  min(res_txt_length), 
                  "\nmax: ", 
                  max(res_txt_length), 
                  "\nmean: ", 
                  sum(res_txt_length)/len(res_txt_length))
            print()
        
        res_rel_df = self.rel_df.copy()
        res_rel_df["inp_text"] = res_rel_df["text"]
        res_rel_df["text"] = res_form
        
        res_rel_df["labels"] = res_rel_df["relation_type"].map(self.re_class_map)
        return res_rel_df

    def get_tag_from_priority(self, cur_entity):
        '''
        since we have a multiclass classification, then for relations we select the highest priority tag among the presented
        cur_entity - the current entity with a list of tags
        '''
        priority_entities = (
            'Medication:MedTypeDrugname',
            'ADR',
            'Medication:MedTypeDrugclass',
            'Medication:MedTypeDrugform',
            'Medication:MedTypeDrugBrand',
            'Disease:DisTypeDiseasename',
            'Disease',
            'Disease:DisTypeADE-Neg',
            'Disease:DisTypeBNE-Pos',
            'Disease:DisTypeWorse',
            'Disease:DisTypeNegatedADE',
            'Disease:DisTypeIndication',
            'Medication:MedTypeSourceInfodrug')
        if cur_entity["tag"]:
            temp_tag = cur_entity["tag"][0]
        else:
            temp_tag = ""
        cur_tag = None
        for temp_cur_tag in cur_entity["tag"]:
            for comparison_cur_tag in priority_entities:
                if temp_cur_tag == comparison_cur_tag:
                    cur_tag = temp_cur_tag
                    break
        if not cur_tag:
            cur_tag = temp_tag
        return cur_tag

    def rel_data_generator(self, entities_d):
        '''
         generator on the list of entities, returns at once all relations with 4 types of relationships that we are working on,
         except for None links. None is not yet available, since debugging is in progress
         entities_li - a list of entities from which links will be formed
         ''' 
        entities_li = list(entities_d.values())
        rel_id=0

        for rel_type in ['ADR_Drugname', 'Diseasename_Indication', 'Drugname_Diseasename', 'Drugname_SourceInfodrug']: #
            tag_1, tag_2 = rel_type.split('_')
            for head_i, cur_head_ent in enumerate(entities_li):
                cur_head_tag = cur_head_ent["tag"][0] 

                if cur_head_tag.find(tag_1)<0 and cur_head_tag.find(tag_2)<0:
                    continue
                for cur_tail_ent in entities_li[head_i+1:]:
                    cur_tail_tag = self.get_tag_from_priority(cur_tail_ent)
                    if cur_tail_tag.find(tag_1) < 0 and cur_tail_tag.find(tag_2)<0 or cur_tail_tag==cur_head_tag:
                        continue
                    new_rel = dict.fromkeys(['first_entity', 'second_entity', 'relation_type'])
                    new_rel['first_entity'] = cur_head_ent
                    new_rel['second_entity'] = cur_tail_ent
                    new_rel['relation_type'] = 'potential_' + rel_type
                    new_rel['relation_id'] = rel_id
                    rel_id+=1
                    yield new_rel

    def convert2RE_format(self, raw_data):
        if type(raw_data)==dict:
            for s_id in raw_data:
                raw_data[s_id]["text_id"] = s_id
                raw_data[s_id]['relations'] = [rel for rel in  self.rel_data_generator(raw_data[s_id]['entities'])]
        elif type(raw_data)==list:
            for s_id, s in enumerate(raw_data):
                try:
                    s["text_id"] = s['meta']['text_id']
                except:
                    pass
                s['relations'] = [rel for rel in self.rel_data_generator(s['entities'])]
        return raw_data


# In[3]:


if __name__=="__main__":
    if not hasattr(__builtins__, '__IPYTHON__'):
        import argparse
        parser = argparse.ArgumentParser()
    
        parser.add_argument("data_path", type=str, help="Path to origin .json file")
        parser.add_argument("res_path", type=str, help="Path to save in a .tsv format")
        parser.add_argument("--mode", type=str, help="The mode in which the script is run. If NER_fixed, then everything is as usual, at the input of .json, if Joint, then at the json input, in which you need to manually form relations field")
        parser.add_argument("--entity_sep", type=str, default=" [SEP] ")
        parser.add_argument("--txt_sep", type=str, default=" [TXTSEP] ")
        parser.add_argument("--label_map_path",type=str, help="json, in which the linkage classes are mapped to their identifiers (to avoid confusion) ", default="")
        parser.add_argument("--save_label_map", help="Flag whether to save the map of labels and indices (to the same place where the result is saved)", action='store_true')
        parser.add_argument("--res_format", type=str, 
                            help="Output text form, available: pair_and_text, text, pair", 
                            default='pair_and_text')
        
        args = parser.parse_args()

        data_path = args.data_path
        entity_separator = args.entity_sep
        text_separator = args.txt_sep
        res_path = args.res_path
        label_map_path = args.label_map_path
        save_label_map = args.save_label_map
        res_format = args.res_format
        mode = args.mode
        
    else:
        data_path = "../data/dev_DDI/tr.json"
        res_path = "../data/dev_DDI/tr.tsv"
        entity_separator = " [SEP] "
        text_separator = " [TXTSEP] "
        label_map_path = None
        save_label_map = True
        res_format = "pair_and_text"
        mode = "NER_fixed"
        
    if label_map_path:
        with open(label_map_path, "r") as label_map_file:
            map_dict = json.load(label_map_file)
    else:
        map_dict = None
    
    data_processor = Dataset_processor(data_path, 
                                       entity_separator=entity_separator,
                                       text_separator=text_separator,
                                       map_dict=map_dict,
                                       mode = mode)
    
    if (not label_map_path) and (save_label_map):
        label_map = data_processor.get_label_map()
        
        saving_path = dirname(res_path) + "/" + "label_map.json"
        
        with open(saving_path, "w") as map_file:
            json.dump(label_map, map_file)
    
    res_df = data_processor.process_data(res_format=res_format)
    res_df.to_csv(res_path, sep="\t", index=False)

  


