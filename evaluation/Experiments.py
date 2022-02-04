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
# This source code is derived from A Fair Comparison of Graph Neural Networks for Graph Classification (ICLR 2020)
#   (https://github.com/diningphil/gnn-comparison)
# Copyright (C)  2020  University of Pisa,
# licensed under GNU GENERAL PUBLIC LICENSE Version 3, 29 June 2007,
# cf. 3rd-party-licenses.txt file in the root directory of this source tree.

from config.base import Config
from torch.utils.data import DataLoader

class runExperiment():

    def __init__(self, model_configuration, exp_path):
        self.model_config = Config.from_dict(model_configuration)
        self.exp_path = exp_path
    def run_valid(self, train_data,val_data, logger):

        model_class = self.model_config.model

        loss_class = self.model_config.loss
        optim_class = self.model_config.optimizer
        sched_class = self.model_config.scheduler
        stopper_class = self.model_config.early_stopper
        network = self.model_config.network
        trainer_class = self.model_config.trainer
        shuffle = self.model_config['shuffle'] if 'shuffle' in self.model_config else True

        try:
            batch_size = self.model_config['batch_size']
        except:
            batch_size = int(len(train_data)/4)
        if len(train_data) >batch_size:
            drop = True
        else:
            drop = False
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=shuffle,
                                  drop_last=drop)
        if len(val_data) == 0:
            val_loader = None
        else:
            val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False,
                                    drop_last=False)

        try:
            x_dim = self.model_config['x_dim']
        except:
            x_dim = train_data.dim_features
        model = model_class(network(),x_dim, config=self.model_config)
        trainer = trainer_class(model, loss_function=loss_class(self.model_config['loss_temp']),
                         device=self.model_config['device'])

        optimizer = optim_class(model.parameters(),
                                lr=self.model_config['learning_rate'], weight_decay=self.model_config['l2'])

        if sched_class is not None:
            scheduler = sched_class(optimizer)
        else:
            scheduler = None

        val_loss,val_auc, _,_,_ = trainer.train(train_loader=train_loader,
                                                              max_epochs=self.model_config['training_epochs'],
                                                              optimizer=optimizer, scheduler=scheduler,
                                                              validation_loader=val_loader,
                                                              early_stopping=stopper_class,
                                                              logger=logger)
        return val_loss, val_auc

    def run_test(self, train_data,val_data,test_data, logger):

        model_class = self.model_config.model


        loss_class = self.model_config.loss
        optim_class = self.model_config.optimizer
        sched_class = self.model_config.scheduler
        stopper_class = self.model_config.early_stopper
        network = self.model_config.network
        trainer_class = self.model_config.trainer
        shuffle = self.model_config['shuffle'] if 'shuffle' in self.model_config else True


        try:
            x_dim = self.model_config['x_dim']
        except:
            x_dim = train_data.dim_features

        try:
            batch_size = self.model_config['batch_size']
        except:
            batch_size = int(len(train_data)/4)
        if len(train_data) >batch_size:
            drop = True
        else:
            drop = False
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=shuffle,
                                  drop_last=drop)


        if len(val_data) == 0:
            val_loader = None
        else:
            val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False,
                                    drop_last=False)

        if len(test_data) == 0:
            test_loader = None
        else:
            test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False,
                                     drop_last=False)

        model = model_class(network(),x_dim, config=self.model_config)
        optimizer = optim_class(model.parameters(),
                                lr=self.model_config['learning_rate'], weight_decay=self.model_config['l2'])

        if sched_class is not None:
            scheduler = sched_class(optimizer)
        else:
            scheduler = None

        trainer = trainer_class(model, loss_function=loss_class(self.model_config['loss_temp']),
                         device=self.model_config['device'])

        val_loss,val_auc,test_auc,test_ap,test_f1 = \
            trainer.train(train_loader=train_loader,
                      max_epochs=self.model_config['training_epochs'],
                      optimizer=optimizer, scheduler=scheduler,
                      validation_loader=val_loader, test_loader=test_loader, early_stopping=stopper_class,
                      logger=logger)

        return val_auc, test_auc, test_ap,test_f1
