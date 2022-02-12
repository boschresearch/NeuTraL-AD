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

This repo contains the code of experiments with NeuTraLAD on various data types including time series data (one-vs-rest setting), tabular data, image data (one-vs-rest setting), text data (one-vs-rest setting), and graph data (one-vs-rest setting). 
Please run the command and replace \$# with available options (see below): 

```
python Launch_Exps.py --config-file $1 --dataset-name $2 
```

**config-file:** 

> Tabular Data: config_thyroid.yml; config_arrhy.yml; config_kdd.yml; config_kddrev.yml;

> Time Series: config_arabic.yml; config_characters.yml; config_natops.yml; config_epilepsy.yml; config_rs.yml;

> Image Data: config_fmnist.yml; config_cifar10_feat.yml; 

> Text Data: config_reuters.yml;

> Graph Data: config_bio.yml; config_molecule.yml; config_social.yml;

**dataset-name:** 

> Tabular Data: thyroid; arrhythmia; kdd; kddrev;

> Time Series: arabic_digits; characters; natops; epilepsy; racket_sports;

> Image Data: fmnist; cifar10_feat; 

> Text Data: reuters;

> Graph Data: dd; thyroid; nci1; mutag; imdb; reddit;

## Datasets

Time series datasets are modified on the UEA datasets from https://www.timeseriesclassification.com/

Arrhythmia and Thyroid datasets are taken from https://github.com/lironber/GOAD 

KDD and KDDrev datasets can be downloaded from https://kdd.ics.uci.edu/databases/kddcup99/

Graph Data are modified on the TUDataset from https://chrsmrrs.github.io/datasets/

Cifar10_feat is the last-layer features of Cifar 10 extracted by a ResNet152 pretrained on ImageNet.
## License

Neural Transformation Learning for Anomaly Detection (NeuTraLAD) is open-sourced under the AGPL-3.0 license. See the
[LICENSE](LICENSE) file for details.

For a list of other open source components included in Neural Transformation Learning for Anomaly Detection (NeuTraLAD), see the
file [3rd-party-licenses.txt](3rd-party-licenses.txt).