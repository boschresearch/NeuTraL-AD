# Neural Transformation Learning for Anomaly Detection (NeuTraLAD)

This is the companion code for a PyTorch implementation of Neural Transformation Learning reported in the paper
**Neural Transformation Learning for Deep Anomaly Detection Beyond Images** by Chen Qiu et al. 
The paper is published in ICML 2021 and can be found here https://arxiv.org/abs/2103.16440. 
The code allows the users to reproduce and extend the results reported in the study. Please cite the
above paper when reporting, reproducing or extending the results.

## Purpose of the project

This software is a research prototype, solely developed for and published as
part of the publication cited above. It will neither be maintained nor monitored in any way.

## How to use

To run the experiment with NeuTraLAD on time series data and tabular data, please run the command and replace \$# with available options (see below): 

```
python Launch_Exps.py --config-file $1 --dataset-name $2 
```

config-file: 

config_thyroid.yml; config_arrhy.yml; config_kdd.yml; config_kddrev.yml; config_arabic.yml; config_characters.yml; config_natops.yml; config_epilepsy.yml; config_rs.yml

dataset-name: 

thyroid; arrhythmia; kdd; kddrev; arabic_digits; characters; natops; epilepsy; racket_sports

## Datasets

Tabular datasets are provided in the folder DATA. Time series datasets are modified on the UEA datasets from https://www.timeseriesclassification.com/

## License

Neural Transformation Learning for Anomaly Detection (NeuTraLAD) is open-sourced under the AGPL-3.0 license. See the
[LICENSE](LICENSE) file for details.

For a list of other open source components included in Neural Transformation Learning for Anomaly Detection (NeuTraLAD), see the
file [3rd-party-licenses.txt](3rd-party-licenses.txt).