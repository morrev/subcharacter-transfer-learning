# subcharacter-transfer-learning

## Summary
Models and training scripts for using pretrained Chinese subcharacter (glyph, radical, subcomponent) together with pretrained Japanese BERT, to assess performance improvement on Japanese NLP tasks (title classification on Livedoor and Wikipedia, and embedding similarity on JWSAN), relative to a pretrained Japanese BERT baseline.

## Motivation
Incorporating subcharacter embeddings has yielded improved performance in Chinese and Japanese NLP tasks separately, compared to only using character embeddings. Recent papers make use of the fact that the graphical subcharacters of Chinese characters contain both semantic and phonetic information. However, transferring subcharacter embeddings from Chinese to Japanese remains largely unexplored. We assess the effect of including Chinese subcharacter embeddings in Japanese pretrained models for Japanese NLP tasks. 

## Requirements
Dependencies are stored in the Pipenv and Pipenv.lock files, which can be installed via:
`pipenv install`

## Data preparation
* Run `./prepare_data.sh` 
  * this git clones:
    * GDCE-SSA: repo for livedoor train/test split and processing
    * JWE: radical and subcomponent mappings
    * ChineseBERT: glyph embeddings (pretrained model in `/ChineseBERT-large` and models in `/ChineseBert/` by default) 
  * this curls:
    * livedoor: 9 label title classification dataset
    * JWSAN: word similarity pair scores
    * wikipedia title dataset: 12 label title classification dataset
  * by default, downloaded files saved to ./data directory

## Model training

The main file to train and evaluate our code can be found in `train.ipynb`. Available configuration includes:

### Model Parameters (for both pooled and unpooled models)
```
N_EPOCHS = 1
LR = 1e-5
PATIENCE = 2
BATCH_SIZE = 2
```

### Model Type
```
pooled = 0  # if 1, pooled; if 0, unpooled 
subcomponent = 2 # if 0, radical; if 1, subcomponent; if 2, glyph; if 3 or other number, baseline
frozen = 0 # if 1, frozen weights; if 0, unfrozen
livedoor = 1 # if 1, load livedoor data; if 0, load wikipedia data
````

Trained models will be saved to ./data/models

Currently, `train.ipynb` contains a small example run. For training the full dataset, comment out the following lines: 
```
iend = 20
X_train = X_train[:iend]
X_val = X_val[:iend]
X_test = X_test[:iend]
y_train = y_train[:iend]
y_val = y_val[:iend]
y_test = y_test[:iend]
```

### Data
Our trained model can be found on Google Drive: https://drive.google.com/drive/folders/1M6CpAuvvZqvgUBcIKwmZvESLj1b7aGCT?usp=sharing

