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

import os
import json
import torch
import numpy as np
import random
from utils import Logger

class KFoldEval:
    """
    Class implementing a sufficiently general framework to do model ASSESSMENT
    """

    def __init__(self, dataset, exp_path, model_configs):
        self.num_folds = 10
        self.num_cls = dataset.num_cls
        self.data_name = dataset.data_name
        self.model_configs = model_configs
        self.dataset_class = dataset
        self._NESTED_FOLDER = exp_path
        self._FOLD_BASE = 'FOLD_'
        self._RESULTS_FILENAME = '_results.json'
        self._ASSESSMENT_FILENAME = 'assessment_results.json'

    def process_results(self):

        VAL_aucs,TS_aucs,TS_f1s, = [],[], []
        results = {}

        for normal_cls in range(self.num_cls):
            Fold_VAL_aucs,Fold_VAL_f1s,Fold_TS_aucs,Fold_TS_f1s= [],[],[],[]

            for i in range(self.num_folds):
                try:
                    config_filename = os.path.join(self._NESTED_FOLDER, self._FOLD_BASE + str(i),
                                                   str(normal_cls)+self._RESULTS_FILENAME)
                    with open(config_filename, 'r') as fp:
                        variant_scores = json.load(fp)

                        Fold_VAL_aucs.append(variant_scores['VAL_auc_'+str(normal_cls)])
                        Fold_TS_aucs.append(variant_scores['TS_auc_'+str(normal_cls)])
                        Fold_TS_f1s.append(variant_scores['TS_f1_' + str(normal_cls)])

                except Exception as e:
                    print(e)
            Fold_VAL_aucs = np.array(Fold_VAL_aucs)
            Fold_TS_aucs = np.array(Fold_TS_aucs)
            Fold_TS_f1s = np.array(Fold_TS_f1s)

            results['avg_VAL_auc_' + str(normal_cls)] = Fold_VAL_aucs.mean()
            results['std_VAL_auc_' + str(normal_cls)] = Fold_VAL_aucs.std()
            results['avg_TS_auc_' + str(normal_cls)] = Fold_TS_aucs.mean()
            results['std_TS_auc_' + str(normal_cls)] = Fold_TS_aucs.std()
            results['avg_TS_f1_' + str(normal_cls)] = Fold_TS_f1s.mean()
            results['std_TS_f1_' + str(normal_cls)] = Fold_TS_f1s.std()

            VAL_aucs.append(Fold_VAL_aucs)
            TS_aucs.append(Fold_TS_aucs)
            TS_f1s.append(Fold_TS_f1s)

        VAL_aucs = np.array(VAL_aucs)
        TS_aucs = np.array(TS_aucs)
        TS_f1s = np.array(TS_f1s)

        avg_VAL_auc = np.mean(VAL_aucs,0)
        avg_TS_auc = np.mean(TS_aucs,0)
        avg_TS_f1 = np.mean(TS_f1s,0)

        results['avg_VAL_auc'] = avg_VAL_auc.mean()
        results['std_VAL_auc'] = avg_VAL_auc.std()
        results['avg_TS_auc'] = avg_TS_auc.mean()
        results['std_TS_auc'] = avg_TS_auc.std()
        results['avg_TS_f1'] = avg_TS_f1.mean()
        results['std_TS_f1'] = avg_TS_f1.std()

        with open(os.path.join(self._NESTED_FOLDER, self._ASSESSMENT_FILENAME), 'w') as fp:
            json.dump(results, fp)

    def risk_assessment(self, experiment_class):

        if not os.path.exists(self._NESTED_FOLDER):
            os.makedirs(self._NESTED_FOLDER)

        for fold_k in range(self.num_folds):

            kfold_folder = os.path.join(self._NESTED_FOLDER, self._FOLD_BASE + str(fold_k))
            if not os.path.exists(kfold_folder):
                os.makedirs(kfold_folder)

            self._risk_assessment_helper(fold_k,experiment_class, kfold_folder)

        self.process_results()

    def _risk_assessment_helper(self, fold_k, experiment_class, exp_path):

        best_config = self.model_configs[0]
        experiment = experiment_class(best_config, exp_path)

        for cls in range(self.num_cls):

            json_outer_results = os.path.join(exp_path, str(cls)+self._RESULTS_FILENAME)
            if not os.path.exists(json_outer_results):

                logger = Logger(str(os.path.join(exp_path, str(cls)+'_experiment.log')), mode='a')

                val_auc_list,test_auc_list,test_f1_list = [], [],[]
                num_repeat = self.model_configs[0]['num_repeat']
                saved_results = {}
                # Mitigate bad random initializations
                for i in range(num_repeat):
                    torch.backends.cudnn.deterministic = True
                    torch.backends.cudnn.benchmark = False
                    np.random.seed(i + 40)
                    random.seed(i + 40)
                    torch.manual_seed(i + 40)
                    torch.cuda.manual_seed(i + 40)
                    torch.cuda.manual_seed_all(i + 40)

                    dataset = self.dataset_class()
                    train_data, val_data = dataset.get_model_selection_fold(fold_k,cls)
                    test_data = dataset.get_test_fold(fold_k)

                    val_auc, test_auc, test_ap,test_f1,scores,labels= experiment.run_test([train_data,val_data,test_data],cls, logger)
                    print(f'Final training run {i + 1}, val auc:{val_auc},test auc:{test_auc}, test f1:{test_f1}')
                    saved_results['scores_' + str(i)] = scores.tolist()
                    saved_results['labels_' + str(i)] = labels.tolist()

                    val_auc_list.append(val_auc)
                    test_auc_list.append(test_auc)
                    test_f1_list.append(test_f1)

                val_auc = sum(val_auc_list) / num_repeat
                test_auc = sum(test_auc_list) / num_repeat
                test_f1 = sum(test_f1_list) / num_repeat
                if best_config['save_scores']:
                    save_path = os.path.join(self._NESTED_FOLDER, self._FOLD_BASE + str(fold_k),
                                                   str(cls)+'scores_labels.json')
                    json.dump(saved_results, open(save_path, 'w'))
                logger.log('End of Outer fold. Normal cls:'+str(cls)+' VAL auc: ' + str(val_auc)
                           +' TS auc: ' + str(test_auc) + ' TS f1: ' + str(test_f1))

                with open(os.path.join(exp_path, str(cls)+self._RESULTS_FILENAME), 'w') as fp:
                    json.dump({'best_config': best_config, 'VAL_auc_'+str(cls): val_auc,
                               'TS_auc_'+str(cls): test_auc,'TS_f1_'+str(cls): test_f1}, fp)

            else:
                    # Do not recompute experiments for this outer fold.
                print(
                    f"File {json_outer_results} already present! Shutting down to prevent loss of previous experiments")
                continue
