# subcharacter-transfer-learning

Data preparation:
* Run `./prepare_data.sh` 
  * this git clones:
    * GDCE-SSA: repo for livedoor train/test split and processing
    * JWE: radical and subcomponent mappings
  * this curls:
    * livedoor: 9 label title classification dataset
    * JWSAN: word similarity pair scores
    * wikipedia title dataset: 12 label title classification dataset
  * by default, downloaded files saved to ./data directory
