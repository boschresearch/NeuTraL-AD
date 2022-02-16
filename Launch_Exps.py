# Neural Transformation Learning for Anomaly Detection (NeuTraLAD) - a self-supervised method for anomaly detection
# Copyright (c) 2022 Robert Bosch GmbH
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
#

import argparse
from config.base import Grid, Config
from evaluation.Experiments import runExperiment,runTextExperiment,runGraphExperiment
from evaluation.Kvariants_Eval import KVariantEval
from evaluation.Kfolds_Eval import KFoldEval

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config-file', dest='config_file', default='config_thyroid.yml')
    parser.add_argument('--dataset-name', dest='dataset_name', default='thyroid')
    return parser.parse_args()

def EndtoEnd_Experiments(config_file, dataset_name):

    model_configurations = Grid(config_file, dataset_name)
    model_configuration = Config(**model_configurations[0])
    dataset =model_configuration.dataset
    result_folder = model_configuration.result_folder+model_configuration.exp_name

    if dataset_name in ['dd','proteins','nci1','mutag','imdb','reddit']:
        exp_class = runGraphExperiment
        risk_assesser = KFoldEval(dataset,result_folder,model_configurations)
    else:
        if dataset_name == 'reuters':
            exp_class = runTextExperiment
        else:
            exp_class = runExperiment
        risk_assesser = KVariantEval(dataset, result_folder, model_configurations)

    risk_assesser.risk_assessment(exp_class)

if __name__ == "__main__":
    args = get_args()
    config_file = 'config_files/'+args.config_file

    EndtoEnd_Experiments(config_file, args.dataset_name)
