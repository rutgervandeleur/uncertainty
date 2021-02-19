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
 Next, navigate to any file and run it.   
 ```bash
# module folder
cd project

# run module (example: mnist as your main contribution)   
python lit_classifier_main.py    
```

### Citation   
```
@article{VrankenUncertainty2021,
  title={Uncertainty estimation for deep learning-based automated analysis of 12-lead electrocardiograms},
  author={Jeroen F. Vranken, Rutger R. van de Leur, Deepak K. Gupta, Luis E. Juarez Orozco, Rutger J. Hassink, Pim van der Harst, Pieter A. Doevendans, Sadaf Gulshad, René van Es},
  journal={submitted},
  year={2021}
}
```   
