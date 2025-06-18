### Train DLM on molecular SELFIES
To train the DLM with the multitask descriptor regression loss on our curated SELFIES data, please use the following command:
```
python main.py \
  model=small \
  data=SELFIES \
  wandb.name=dlm-mtr \
  parameterization=subs \
  model.length=1024 \
  sampling.steps=1000
```


### Acknowledgements
This repository was built off of [MDLM](https://github.com/kuleshov-group/mdlm).
