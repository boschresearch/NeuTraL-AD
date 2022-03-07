# Neural Transformation Learning for Anomaly Detection (NeuTraLAD)

This is the companion code for a PyTorch implementation of Neural Transformation Learning reported in the paper
**Neural Transformation Learning for Deep Anomaly Detection Beyond Images** by Chen Qiu et al. 
The paper is published in ICML 2021 and can be found here https://arxiv.org/abs/2103.16440. 
The code allows the users to reproduce and extend the results reported in the study. Please cite the
above paper when reporting, reproducing or extending the results.

## Purpose of the project

This software is a research prototype, solely developed for and published as
part of the publication cited above. It will neither be maintained nor monitored in any way.

## Reproduce the Results

This repo contains the code of experiments with NeuTraLAD on various data types including time series data (one-vs-rest setting), tabular data, image data (one-vs-rest setting), text data (one-vs-rest setting), and graph data (one-vs-rest setting). 

Please run the command and replace \$# with available options (see below): 

```
python Launch_Exps.py --config-file $1 --dataset-name $2 
```

**config-file:** 

* Tabular Data: config_thyroid.yml; config_arrhy.yml; config_kdd.yml; config_kddrev.yml;

* Time Series: config_arabic.yml; config_characters.yml; config_natops.yml; config_epilepsy.yml; config_rs.yml;

* Image Data: config_fmnist.yml; config_cifar10_feat.yml; 

* Text Data: config_reuters.yml;

* Graph Data: config_bio.yml; config_molecule.yml; config_social.yml;


**dataset-name:** 

* Tabular Data: thyroid; arrhythmia; kdd; kddrev;

* Time Series: arabic_digits; characters; natops; epilepsy; racket_sports;

* Image Data: fmnist; cifar10_feat; 

* Text Data: reuters;

* Graph Data: dd; thyroid; nci1; aids; imdb; reddit;

## How to Use
1. When using your own data, please put your data files under [DATA](DATA).

2. Create a config file which contains your hyper-parameters under [config_files](config_files).  

3. Add your data loader to the function ''load_data'' in the [loader/LoadData.py](loader/LoadData.py).
* For time series data, the shape is (batch size, #channels, sequence length).
* For image data, the shape is (batch size, #channels, height, width).
* For tabular data/features from pre-trained model, the shape is (batch size, feature dim).

## Datasets

* Time series datasets are downloaded from the UEA datasets from https://www.timeseriesclassification.com/.
The processed data of NATOPS, Epilepsy, and Racket_Sports are available under [DATA](DATA). 

* Arrhythmia and Thyroid datasets are downloaded from https://github.com/lironber/GOAD. Please put the data under [DATA](DATA).  

* KDD and KDDrev datasets are downloaded from https://kdd.ics.uci.edu/databases/kddcup99/. Please put the data under [DATA](DATA).  

* Graph Data are downloaded from TUDataset https://chrsmrrs.github.io/datasets/. Please put the data under [DATA](DATA).  

* Cifar10_feat is the last-layer features of Cifar 10 extracted by a ResNet152 pretrained on ImageNet. [Extract_img_features.py](Extract_img_features.py) is used to extract features.
## License

Neural Transformation Learning for Anomaly Detection (NeuTraLAD) is open-sourced under the AGPL-3.0 license. See the
[LICENSE](LICENSE) file for details.

For a list of other open source components included in Neural Transformation Learning for Anomaly Detection (NeuTraLAD), see the
file [3rd-party-licenses.txt](3rd-party-licenses.txt).