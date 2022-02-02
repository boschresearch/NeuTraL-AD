import argparse
from config.base import Grid, Config
from evaluation.Experiments import runExperiment
from evaluation.Kvariants_Eval import KVariantEval
from torch.backends import cudnn
cudnn.benchmark = True

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config-file', dest='config_file', default='config_kdd.yml')
    parser.add_argument('--dataset-name', dest='dataset_name', default='kdd')
    return parser.parse_args()

def EndtoEnd_Experiments(config_file, dataset_name):

    model_configurations = Grid(config_file, dataset_name)
    model_configuration = Config(**model_configurations[0])
    dataset =model_configuration.dataset
    result_folder = model_configuration.result_folder+model_configuration.exp_name

    risk_assesser = KVariantEval(dataset, result_folder, model_configurations)

    risk_assesser.risk_assessment(runExperiment)

if __name__ == "__main__":
    args = get_args()
    config_file = 'config_files/'+args.config_file

    EndtoEnd_Experiments(config_file, args.dataset_name)
