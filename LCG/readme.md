# NADO

## Training
Usage: First run 
```shell
    python pretrain.py
```
to train the base distribution.

Then run 
```shell
    python finetune.py 
``` 
to generate the data for training NADO layers. After the data has been generated and dumped, the script will terminate itself. Run:
```shell
    python finetune.py
``` 
again to actually train the model.

## Evaluation/Inference/Parameter tuning.

See the argument descriptions of each script argument for details