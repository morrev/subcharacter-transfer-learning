# subcharacter-transfer-learning

## Requirements
Dependencies are stored in the Pipenv and Pipenv.lock files, which can be installed via:
```pipenv install``

## Data preparation
* Run `./prepare_data.sh` 
  * this git clones:
    * GDCE-SSA: repo for livedoor train/test split and processing
    * JWE: radical and subcomponent mappings
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
