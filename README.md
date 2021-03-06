# Relation_Extraction
This repository provides code and additional materials of the paper: "Extraction of the Relations between Significant Pharmacological Entities in Russian-Language Reviews of Internet Users on Medications".

In this work, we trained a model to recognize 4 types of relationships between entities in drug review texts: ADR–Drugname, Drugname–Diseasename, Drugname–SourceInfoDrug, Diseasename–Indication. The input of the model is a review text and a pair of entities, between which it is required to determine the fact of a relationship and one of the 4 types of relationship, listed above.

Data
---
Proposed model is trained on a subset of 908 reviews of the [Russian Drug Review Corpus (RDRS)](https://arxiv.org/pdf/2105.00059.pdf). The subset contains the pairs of entities marked with the 4 listed types of relationships:
- ADR-Drugname — the relationship between the drug and its side effects
- Drugname-SourceInfodrug — the relationship between the medication and the source of information about it (e.g., “was advised at the pharmacy”, e.g., “was advised at the pharmacy”, “the doctor recommended it”); 
- Drugname-Diseasname — the relationship between the drug and the disease
- Diseasename-Indication — the connection between the illness and its symptoms (e.g., “cough”, “fever 39 degrees”)
Also, this subset contains pairs of the same entity types between which there is no relationship: for example, a drug and an unrelated side effect that appeared after taking another drug; in other words, this side effect is related to another drug.

Model
--- 
Weights of our model is available on [hugging face repository](https://huggingface.co/sagteam/pharm-relation-extraction/tree/main). Proposed model is based on the  [XLM-RoBERTA-large](https://arxiv.org/abs/1911.02116) topology. After the additional training as a langauge model on corpus of unmarked drug reviews, this model was trained as a classification model on 80% of the texts from subset of the corps described above. This model showed the best accuracy on one of the folds of the cross-validation. For additional details see original paper.

Results
---
Here are the accuracy, estimated by the f1 score metric for the recognition of relationships on the best fold.

| ADR–Drugname  | Drugname–Diseasename | Drugname–SourceInfoDrug | Diseasename–Indication |
| ------------- | -------------------- | ----------------------- | ---------------------- |
| 0.955         | 0.892                | 0.922                   | 0.891                  |

Requirements
---
Create anaconda env via `conda env create -f relation_extraction_env.yml`. Then activate it via `conda activate relation_extraction`

How to use
---
Go to the inference folder and see instructions. This folder contains a demonstration of evaluating the proposed model on 5 reviews.

Citing & Authors
---
If you have found our results helpful in your work, feel free to cite our publication and this repository as
```
@article{*coming soon*,
  title={Extraction of the Relations between Significant PharmacologicalEntities in Russian-Language Reviews of Internet Users on Medications},
  author={Alexander Sboev, Anton Selivanov, Ivan Moloshnikov, Roman Rybka, Artem Gryaznov, Sanna Sboeva, Gleb Rylkov},
  year={2021}
}
```
