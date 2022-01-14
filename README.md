<div align="center">    
 
# Uncertainty estimation for deep learning-based automated analysis of 12-lead electrocardiograms      

<!--
[![Paper](http://img.shields.io/badge/paper-arxiv.1001.2234-B31B1B.svg)](https://www.nature.com/articles/nature14539)
-->
 
</div>
 
## Description   
Repository accompanying the paper "Uncertainty estimation for deep learning-based automated analysis of 12-lead electrocardiograms" by Vranken and Van de Leur et al. It contains all the uncertainty estimation methods described in the paper. 

## How to run   
First, install dependencies   
```bash
# clone project   
git clone https://github.com/rutgervandeleur/uncertainty

# install project   
cd uncertainty
pip install -e .   
pip install -r requirements.txt
 ```
Next, make sure the [CPSC2018 dataset](http://2018.icbeb.org/Challenge.html) is downloaded and the correct paths are set in the CPSC2018 config file
```bash
# edit config file
vim project/configs/CPSC2018.json
```

The original CPSC2018 `REFERENCES.csv` file does not contain an iterable primary key. However, this is required by the PyTorch dataloader and therefore has to be added. In the current implementation, the `REFERENCES.csv` file was altered to contain a column named `id`, with an incrementing integer. This file (for the CPSC2018 validation set) can be found [here](https://github.com/rutgervandeleur/uncertainty/blob/master/project/data/CPSC2018/validation_set/CPSC2018_REFERENCES.csv). For the other sets, the file can easily be modified using pandas:
```python
import pandas as pd

df = pd.read_csv('REFERENCES.csv')
df['id'] = df.index + 1

pd.to_csv('CPSC2018_REFERENCES.csv')
```


*Alternatively, you can use a dataset of your choice by extending the ```torch.utils.data.Dataset``` class in the [project/utils/dataloader.py](project/utils/dataloader.py) file, and initializing it in the [project/main.py](project/main.py) file.*


After linking the dataset with iterable `id` column, you can now train a model with uncertainty estimation by passing the desired methods to the program. 
```bash
# go to project folder
cd project

# train model (example: Variational Inference with Bayesian Decomposition)   
python main.py --epistemic_method varinf --aleatoric_method bayesdecomp
```
For ensembling (ensemble, ssensemble) methods, the models are run on the GPU by default when ```torch.cuda.is_available()```. If you want to run on the cpu instead, pass:```gpus 0``` to main call.

The following methods are available:
#### Epistemic uncertainty estimation methods:
- Monte Carlo Dropout [**mcdropout**]
- Ensemble [**ensemble**]
- Snapshot Ensemble [**ssensemble**]
- Variational Inference [**varinf**]
- None [**none**]

#### Aleatoric uncertainty estimation methods:
- Auxiliary Output [**auxout**]
- Bayesian Decomposition [**bayesdecomp**] (Requires the Variational Inference epistemic method)
- None [**none**]


### Citation   
```
@article{VrankenUncertainty2021,
  title={Uncertainty estimation for deep learning-based automated analysis of 12-lead electrocardiograms},
  author={Jeroen F. Vranken, Rutger R. van de Leur, Deepak K. Gupta, Luis E. Juarez Orozco, Rutger J. Hassink, Pim van der Harst, Pieter A. Doevendans, Sadaf Gulshad, René van Es},
  journal={European Heart Journal: Digital Health},
  year={2021}
}
```   
