Description
---
This folder contains a demonstration of evaluating the proposed model on 5 reviews.

Requirements
---
Go to the root folder. Create anaconda env via `conda env create -f relation_extraction_env.yml`. Then activate it via `conda activate relation_extraction`.

Start instructions
---
1. Download trained Relation Extraction model from [hugging face repository](https://huggingface.co/sagteam/pharm-relation-extraction/tree/main). You can do it with these commands: `git lfs install` and `git clone https://huggingface.co/sagteam/pharm-relation-extraction` or by downloading each file of the model using the web interface
2. Replace the value of the RE_MODEL_PATH variable with the path to the Relation Extraction model
3. Run `. inference.sh`
