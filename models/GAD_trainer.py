# Neural Transformation Learning for Anomaly Detection (NeuTraLAD) - a self-supervised method for anomaly detection
# Copyright (c) 2022 Robert Bosch GmbH
#
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

import time
import torch
from sklearn.metrics import roc_auc_score,average_precision_score
import numpy as np
from utils import compute_pre_recall_f1,format_time

class GAD_trainer:

    def __init__(self, model, loss_function,device='cuda'):

        self.loss_fun = loss_function
        self.device = torch.device(device)
        self.model = model.to(self.device)

    def _train(self, train_loader, optimizer):


        self.model.train()

        loss_all = 0
        for data in train_loader:

            data = data.to(self.device)
            optimizer.zero_grad()
            z = self.model(data)

            loss = self.loss_fun(z)
            loss_mean = loss.mean()
            loss_mean.backward()
            optimizer.step()

            loss_all += loss.sum()


        return loss_all.item() / len(train_loader.dataset)

    def detect_outliers(self, loader,cls):
        model = self.model
        model.eval()

        loss_in = 0
        loss_out = 0
        target_all = []
        score_all = []
        for data in loader:
            with torch.no_grad():
                data = data.to(self.device)
                label = data.y!=cls
                z= model.detect(data)

                score = self.loss_fun(z,eval=True)
                target_all.append(label)
                score_all.append(score)
                loss_in += score[label==0].sum()
                loss_out += score[label==1].sum()

        try:
            score_all = np.concatenate(score_all)
        except:
            score_all = torch.cat(score_all).cpu().numpy()
        target_all = torch.cat(target_all).cpu().numpy()
        auc = roc_auc_score(target_all, score_all)
        f1 = compute_pre_recall_f1(target_all,score_all)
        ap = average_precision_score(target_all, score_all)

        return auc, ap,f1,loss_in.item() / (target_all == 0).sum(), loss_out.item() / (target_all == 1).sum()


    def train(self, train_loader,cls,max_epochs=100, optimizer=None, scheduler=None,
              validation_loader=None, test_loader=None, early_stopping=None, logger=None, log_every=10):

        early_stopper = early_stopping() if early_stopping is not None else None

        val_auc, val_f1, = -1, -1
        test_auc, test_f1, test_score = None, None,None,

        time_per_epoch = []
        torch.cuda.empty_cache()



        for epoch in range(1, max_epochs+1):

            start = time.time()
            train_loss = self._train(train_loader, optimizer)
            end = time.time() - start
            time_per_epoch.append(end)

            if scheduler is not None:
                scheduler.step()


            if test_loader is not None:
                test_auc, test_f1, test_pre, test_recall, test_score, test_loss = self.detect_outliers(test_loader, cls)
            if validation_loader is not None:
                val_auc, val_f1, val_pre, val_recall, _,val_loss = self.detect_outliers(validation_loader,cls)



                if early_stopper is not None and early_stopper.stop(epoch, val_loss, val_auc,
                                                                  val_f1, test_loss, test_auc,test_f1,test_score,
                                                                        train_loss):
                    break

            if epoch % log_every == 0 or epoch == 1:
                msg = f'Epoch: {epoch}, TR loss: {train_loss,reg_term}, VL loss: {val_loss} VL auc: {val_auc} TS loss: {test_loss} TS auc: {test_auc}'

                if logger is not None:
                    logger.log(msg)
                    print(msg)
                else:
                    print(msg)

        time_per_epoch = torch.tensor(time_per_epoch)
        avg_time_per_epoch = float(time_per_epoch.mean())

        elapsed = format_time(avg_time_per_epoch)

        if early_stopper is not None:
            train_loss, val_loss,val_auc,val_f1,test_loss,test_auc,test_f1,test_score, best_epoch\
                = early_stopper.get_best_vl_metrics()
            msg = f'Stopping at epoch {best_epoch}, TR loss: {train_loss}, VAL loss: {val_loss}, VAL auc: {val_auc} VAL f1: {val_f1},' \
                f'TS loss: {test_loss}, TS auc: {test_auc} TS f1: {test_f1}'
            if logger is not None:
                logger.log(msg)
                print(msg)
            else:
                print(msg)

        # np.savez('PLOTS/NTLOCC_P1',train_loss = np.array(train_loss_all),val_loss = np.array(val_loss_all), val_auc = np.array(val_auc_all),test_auc = np.array(test_auc_all))
        return val_loss,val_auc, val_f1,test_auc,test_f1,test_score