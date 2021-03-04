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
*Alternatively, you can use a dataset of your choice by extending the ```torch.utils.data.Dataset``` class in the [project/utils/dataloader.py](project/utils/dataloader.py) file, and initializing it in the [project/main.py](project/main.py) file.*


After linking the dataset, you can now train a model with uncertainty estimation by passing the desired methods to the program. 
```bash
# go to project folder
cd project

# train model (example: Variational Inference with Bayesian Decomposition)   
python main.py --epistemic_method varinf --aleatoric_method bayesdecomp
```

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
  journal={submitted},
  year={2021}
}
```   
