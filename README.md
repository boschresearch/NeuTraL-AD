This repository provides a PyTorch implementation of **Neural Transformation Learning for Deep Anomaly Detection Beyond Images** published in ICML 2021 by Chen Qiu et al.
The paper can be found here https://arxiv.org/abs/2103.16440. Please cite the above paper when reporting, reproducing or extending the results.

# NTL Experiment

To run the experiment with NTL on time series data and tabular data, please run the command and replace \$# with available options (see below): 

```
python Launch_Exps.py --config-file $1 --dataset-name $2 
```

config-file: config_thyroid.yml; config_arrhy.yml; config_kdd.yml; config_kddrev.yml; config_arabic.yml; config_characters.yml; config_natops.yml; config_epilepsy.yml;

dataset-name: thyroid; arrhythmia; kdd; kddrev; arabic_digits; characters; natops; epilepsy;  

