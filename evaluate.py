#!/usr/bin/env python
# coding: utf-8

# In[10]:


from sklearn.metrics import classification_report
import pandas as pd
import argparse


# In[11]:


class Evaluator():
    def __init__(self, data_df):
        
        self.res_df = data_df
    
    def classification_report(self, return_dict = False, label_names=None):
        if return_dict:
            return classification_report(self.res_df["labels"], self.res_df["pred_labels"], digits=3, output_dict=True)
        else:
            return classification_report(self.res_df["labels"], self.res_df["pred_labels"], digits=3, target_names=label_names)
        
        
    def examples(self, n, example_type="correct"):
        res_data = self.res_df
        if example_type=="correct":
            
            typed_pred_df = res_data[res_data["labels"] == res_data["pred_labels"]]
        
        if example_type=="incorrect":
            typed_pred_df = res_data[res_data["labels"] != res_data["pred_labels"]]
            
        else:
            typed_pred_df = res_data
        
        return typed_pred_df.sample(n)


# In[16]:


if __name__=="__main__":
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("data_path", type=str, help="Path to the prediction file in .tsv format ")
    parser.add_argument("-n", 
                        type=str, 
                        help="Number of displayed text examples",
                        default=5)
    #TODO: путь к label_map
    
    args = parser.parse_args()
    
    data_path = args.data_path
    n = args.n
    
    test_df = pd.read_csv(data_path, sep="\t")
    test_df.fillna(value=0, inplace=True)
    
    # evaluation
    evaluator = Evaluator(test_df)
    
    classification_rep = evaluator.classification_report(return_dict=True)
    
    print(evaluator.classification_report(return_dict=False))
    
    correct_examples = evaluator.examples(n)
    incorrect_examples = evaluator.examples(n, example_type="incorrect")
    
    print("examples of correctly predicted texts: ")
    print()
    for cur_row_nb, cur_row in correct_examples.iterrows():
        print("Text №{}".format(cur_row_nb))
        print(cur_row.text)
        print()
        print("Entity 1:", cur_row.first_entity_text)
        print("Entity 2:", cur_row.second_entity_text)
        print("Predicted label: ", cur_row.pred_labels)
        print("Real label: ", cur_row.labels, cur_row.relation_type)
        print()
        print()
    print()
    
    print("примеры неправильно предсказанных текстов: ")
    for cur_row_nb, cur_row in incorrect_examples.iterrows():
        print("Text №{}".format(cur_row_nb))
        print(cur_row.text)
        print()
        print("Entity 1:", cur_row.first_entity_text)
        print("Entity 2:", cur_row.second_entity_text)
        print("Predicted label: ", cur_row.pred_labels)
        print("Real label: ", cur_row.labels, cur_row.relation_type)
        print()
        print()


# In[ ]:
