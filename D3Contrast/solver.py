import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import time
from utils.utils import *
from model.D3Contrast import D3Contrast
from data_factory.data_loader import get_loader_segment
from einops import rearrange
from metrics.metrics import *
import warnings
warnings.filterwarnings('ignore')


def my_best_f1(score, label):
    best_f1 = (0,0,0)
    best_thre = 0
    best_pred = None
    # score = minmax_scale(score)
    for q in np.arange(0.01, 0.901, 0.01):
    # for q in np.arange(0.01, 0.901, 0.01):
        thre = np.quantile(score, 1-q)
        pred = score > thre
        pred = pred.astype(int)
        label = label.astype(int)
        p,r,f1,_ = precision_recall_fscore_support(label, pred, average='binary')
        print(f'q: {q}, p: {p}, r: {r}, f1: {f1}')
        if f1 > best_f1[2]:
            best_f1 = (p, r, f1)
            best_thre = thre
            best_pred = pred

    return (best_f1, best_thre, best_pred)


def my_kl_loss(p, q):
    # B N D
    res = p * (torch.log(p + 0.0000001) - torch.log(q + 0.0000001))
    # B N
    return torch.sum(res, dim=-1)


def inter_intra_dist(p,q,w_de=True,train=1,temp=1):
    # B N D
    if train:
        if w_de:
            p_loss = torch.mean(my_kl_loss(p,q.detach()*temp)) + torch.mean(my_kl_loss(q.detach(),p*temp))
            q_loss = torch.mean(my_kl_loss(p.detach(),q*temp)) + torch.mean(my_kl_loss(q,p.detach()*temp))
        else:
            p_loss = torch.mean(my_kl_loss(p,q.detach())) 
            q_loss = torch.mean(my_kl_loss(q,p.detach())) 
    else:
        if w_de:
            p_loss = my_kl_loss(p,q.detach()) + my_kl_loss(q.detach(),p)
            q_loss = my_kl_loss(p.detach(),q) + my_kl_loss(q,p.detach())

        else:
            p_loss = my_kl_loss(p,q.detach())
            q_loss = my_kl_loss(q,p.detach())

    return p_loss,q_loss


def normalize_tensor(tensor):
    # tensor: B N D
    sum_tensor = torch.sum(tensor,dim=-1,keepdim=True)
    normalized_tensor = tensor / sum_tensor
    return normalized_tensor


def anomaly_score(time_dist, fre_dist, win_size, train=1, temp=1, w_de=True):
    time_dist = torch.softmax(time_dist, dim=-1)
    fre_dist = torch.softmax(fre_dist, dim=-1)
                            
    time_dist = normalize_tensor(time_dist)
    fre_dist = normalize_tensor(fre_dist)

    time_loss, fre_loss = inter_intra_dist(time_dist,fre_dist,w_de,train=train,temp=temp)

    return time_loss, fre_loss


def adjust_learning_rate(optimizer, epoch, lr_):
    lr_adjust = {epoch: lr_ * (0.5 ** ((epoch - 1) // 1))}
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, dataset_name='', delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.best_score2 = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.val_loss2_min = np.Inf
        self.delta = delta
        self.dataset = dataset_name

    def __call__(self, val_loss, val_loss2, model, path):
        score = -val_loss
        score2 = -val_loss2
        if self.best_score is None:
            self.best_score = score
            self.best_score2 = score2
            self.save_checkpoint(val_loss, val_loss2, model, path)
        elif score < self.best_score + self.delta or score2 < self.best_score2 + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_score2 = score2
            self.save_checkpoint(val_loss, val_loss2, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, val_loss2, model, path):
        torch.save(model.state_dict(), os.path.join(path, str(self.dataset) + '_checkpoint.pth'))
        print('Save model')
        self.val_loss_min = val_loss
        self.val_loss2_min = val_loss2

        
class Solver(object):
    DEFAULTS = {}

    def __init__(self, config):

        self.__dict__.update(Solver.DEFAULTS, **config)

        self.train_loader = get_loader_segment(self.index, 'dataset/'+self.data_path, batch_size=self.batch_size, win_size=self.win_size, mode='train', dataset=self.dataset, )
        self.vali_loader = get_loader_segment(self.index, 'dataset/'+self.data_path, batch_size=self.batch_size, win_size=self.win_size, mode='val', dataset=self.dataset)
        self.test_loader = get_loader_segment(self.index, 'dataset/'+self.data_path, batch_size=self.batch_size, win_size=self.win_size, mode='test', dataset=self.dataset)
        self.thre_loader = get_loader_segment(self.index, 'dataset/'+self.data_path, batch_size=self.batch_size, win_size=self.win_size, mode='thre', dataset=self.dataset)

        self.build_model()
        
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        if self.loss_fuc == 'MAE':
            self.criterion = nn.L1Loss()
        elif self.loss_fuc == 'MSE':
            self.criterion = nn.MSELoss()
        

    def build_model(self):
        self.model = DCdetector(win_size=self.win_size, enc_in=self.input_c, c_out=self.output_c, n_heads=self.n_heads, d_model=self.d_model, e_layers=self.e_layers, channel=self.input_c)
        
        if torch.cuda.is_available():
            self.model.cuda()
            
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        
        
    def vali(self, vali_loader):
        self.model.eval()
        loss_1 = []
        loss_2 = []
        win_size = self.win_size
        for i, (input_data, _) in enumerate(vali_loader):
            input = input_data.float().to(self.device)
            temporal, fre = self.model(input)
            time_loss, fre_loss = anomaly_score(temporal, fre, win_size=win_size, train=1)
            

            loss_1.append((time_loss).item())
            loss_2.append((fre_loss).item())

        return np.average(loss_1), np.average(loss_2)


    def train(self):

        time_now = time.time()
        path = self.model_save_path
        if not os.path.exists(path):
            os.makedirs(path)
        win_size = self.win_size
        early_stopping = EarlyStopping(patience=5, verbose=True, dataset_name=self.data_path)
        train_steps = len(self.train_loader)

        for epoch in range(self.num_epochs):
            iter_count = 0

            epoch_time = time.time()
            self.model.train()
            for i, (input_data, labels) in enumerate(self.train_loader):

                self.optimizer.zero_grad()
                iter_count += 1
                input = input_data.float().to(self.device)
                temporal, fre = self.model(input)

                loss = 0.
                time_loss, fre_loss = anomaly_score(temporal, fre, win_size=win_size, train=1, temp=1)

                loss1 = time_loss + fre_loss
                loss2 = self.criterion(temporal, input)
                loss = 0.5 * loss2 + 0.5 * loss1

                if (i + 1) % 100 == 0:
                    print("DIS: {:.15f}".format(loss1))
                    print(f'MSE {loss2.item()} Loss {loss.item()}')
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.num_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()
 
                loss1.backward()
                self.optimizer.step()

            vali_loss1, vali_loss2 = self.vali(self.test_loader)
            print('Vali',vali_loss1, vali_loss2)

            print(
                "Epoch: {0}, Cost time: {1:.3f}s ".format(
                    epoch + 1, time.time() - epoch_time))
            early_stopping(vali_loss1, vali_loss2, self.model, path)
            if early_stopping.early_stop:
                break
            adjust_learning_rate(self.optimizer, epoch + 1, self.lr)

            
    def test(self):
        self.model.load_state_dict(
            torch.load(
                os.path.join(str(self.model_save_path), str(self.data_path) + '_checkpoint.pth')))
        self.model.eval()
        temperature = 50
        win_size=self.win_size
        mse_loss = nn.MSELoss(reduction='none')
        cont_beta = 1.0
        # (1) stastic on the train set
        attens_energy = []
        for i, (input_data, labels) in enumerate(self.train_loader):
            input = input_data.float().to(self.device)
            temporal, fre = self.model(input)
            time_loss, fre_loss = anomaly_score(temporal, fre, win_size=win_size, train=0)
            
            loss1 = time_loss + fre_loss
            mse_loss_ = mse_loss(temporal, input)

            metric1 = torch.softmax((loss1), dim = -1)
            metric2 = mse_loss_.mean(-1)
            metric = metric1 * (cont_beta) + metric2 * (1-cont_beta)
            cri = metric.detach().cpu().numpy()
            attens_energy.append(cri)

        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        train_energy = np.array(attens_energy)

        # (2) find the threshold
        attens_energy = []
        for i, (input_data, labels) in enumerate(self.thre_loader):
            input = input_data.float().to(self.device)
            temporal, fre = self.model(input)
            time_loss, fre_loss = anomaly_score(temporal, fre, win_size=win_size, train=0)
            
            loss1 = time_loss + fre_loss
            mse_loss_ = mse_loss(temporal, input)

            metric1 = torch.softmax((loss1), dim = -1)
            metric2 = mse_loss_.mean(-1)
            metric = metric1 * (cont_beta) + metric2 * (1-cont_beta)
            cri = metric.detach().cpu().numpy()
            attens_energy.append(cri)

        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        test_energy = np.array(attens_energy)
        combined_energy = np.concatenate([train_energy, test_energy], axis=0)
        thresh = np.percentile(combined_energy, 100 - self.anormly_ratio)
        print("Threshold :", thresh)

        # (3) evaluation on the test set
        test_labels = []
        attens_energy = []
        test_data = []
        for i, (input_data, labels) in enumerate(self.thre_loader):
            input = input_data.float().to(self.device)
            temporal, fre = self.model(input)
            time_loss, fre_loss = anomaly_score(temporal, fre, win_size=win_size, train=0)
            
            loss1 = time_loss + fre_loss
            mse_loss_ = mse_loss(temporal, input)

            metric1 = torch.softmax((loss1), dim = -1)
            metric2 = mse_loss_.mean(-1)
            metric = metric1 * (cont_beta) + metric2 * (1-cont_beta)
            cri = metric.detach().cpu().numpy()
            attens_energy.append(cri)
            test_labels.append(labels)
            test_data.append(input_data.cpu().numpy().reshape(-1,input_data.shape[-1]))
            
        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        test_labels = np.concatenate(test_labels, axis=0).reshape(-1)
        test_energy = np.array(attens_energy)
        test_labels = np.array(test_labels)
        test_data = np.concatenate(test_data,axis=0)

        pred = (test_energy > thresh).astype(int)
        gt = test_labels.astype(int)
        
        matrix = [self.index]
        scores_simple = combine_all_evaluation_scores(pred, gt, test_energy)
        for key, value in scores_simple.items():
            matrix.append(value)
            print('{0:21} : {1:0.4f}'.format(key, value))

        anomaly_state = False
        for i in range(len(gt)):
            if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
                anomaly_state = True
                for j in range(i, 0, -1):
                    if gt[j] == 0:
                        break
                    else:
                        if pred[j] == 0:
                            pred[j] = 1
                for j in range(i, len(gt)):
                    if gt[j] == 0:
                        break
                    else:
                        if pred[j] == 0:
                            pred[j] = 1
            elif gt[i] == 0:
                anomaly_state = False
            if anomaly_state:
                pred[i] = 1

        pred = np.array(pred)
        gt = np.array(gt)

        from sklearn.metrics import precision_recall_fscore_support
        from sklearn.metrics import accuracy_score

        accuracy = accuracy_score(gt, pred)
        precision, recall, f_score, support = precision_recall_fscore_support(gt, pred, average='binary')
        print("Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f} ".format(accuracy, precision, recall, f_score))
        
        if self.data_path == 'UCR' or 'UCR_AUG':
            import csv
            with open('result/'+self.data_path+'.csv', 'a+') as f:
                writer = csv.writer(f)
                writer.writerow(matrix)

        return accuracy, precision, recall, f_score
