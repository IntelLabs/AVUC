# Copyright (C) 2020 Intel Corporation
#
# BSD-3-Clause License
#
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
# 1. Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
# 3. Neither the name of the copyright holder nor the names of its contributors
#    may be used to endorse or promote products derived from this software
#    without specific prior written permission.
#  
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
# THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS
# BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY,
# OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT
# OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
# OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
# OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
# EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#                       
# AvULoss   -> compute accuracy versus uncertainty calibration loss
# AUAvULoss -> compute accuracy versus uncertainty calibration loss
#              without uncertainty threshold
# accuracy_versus_uncertainty -> compute AvU metric
# eval_AvU  -> get AvU scores at differemt uncertainty thresholds
# predictive_entropy -> compute predictive uncertainty of the model
# mutual_information -> compute model uncertainty of the model
#
# @authors: Ranganath Krishnan
#
# ===============================================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch.nn.functional as F
import torch
from torch import nn
import numpy as np
from sklearn.metrics import auc


class AUAvULoss(nn.Module):
    """
    Calculates Accuracy vs Uncertainty Loss of a model.
    The input to this loss is logits from Monte_carlo sampling of the model, true labels,
    and the type of uncertainty to be used [0: predictive uncertainty (default); 
    1: model uncertainty]
    """
    def __init__(self, beta=1):
        super(AUAvULoss, self).__init__()
        self.beta = beta
        self.eps = 1e-10

    def entropy(self, prob):
        return -1 * torch.sum(prob * torch.log(prob + self.eps), dim=-1)

    def expected_entropy(self, mc_preds):
        return torch.mean(self.entropy(mc_preds), dim=0)

    def predictive_uncertainty(self, mc_preds):
        """
        Compute the entropy of the mean of the predictive distribution
        obtained from Monte Carlo sampling.
        """
        return self.entropy(torch.mean(mc_preds, dim=0))

    def model_uncertainty(self, mc_preds):
        """
        Compute the difference between the entropy of the mean of the
        predictive distribution and the mean of the entropy.
        """
        return self.entropy(torch.mean(
            mc_preds, dim=0)) - self.expected_entropy(mc_preds)

    def auc_avu(self, logits, labels, unc):
        """ returns AvU at various uncertainty thresholds"""
        th_list = np.linspace(0, 1, 21)
        umin = torch.min(unc)
        umax = torch.max(unc)
        avu_list = []
        unc_list = []

        probs = F.softmax(logits, dim=1)
        confidences, predictions = torch.max(probs, 1)

        auc_avu = torch.ones(1, device=labels.device)
        auc_avu.requires_grad_(True)

        for t in th_list:
            unc_th = umin + (torch.tensor(t) * (umax - umin))
            n_ac = torch.zeros(
                1,
                device=labels.device)  # number of samples accurate and certain
            n_ic = torch.zeros(1, device=labels.device
                               )  # number of samples inaccurate and certain
            n_au = torch.zeros(1, device=labels.device
                               )  # number of samples accurate and uncertain
            n_iu = torch.zeros(1, device=labels.device
                               )  # number of samples inaccurate and uncertain

            for i in range(len(labels)):
                if ((labels[i].item() == predictions[i].item())
                        and unc[i].item() <= unc_th.item()):
                    """ accurate and certain """
                    n_ac += confidences[i] * (1 - torch.tanh(unc[i]))
                elif ((labels[i].item() == predictions[i].item())
                      and unc[i].item() > unc_th.item()):
                    """ accurate and uncertain """
                    n_au += confidences[i] * torch.tanh(unc[i])
                elif ((labels[i].item() != predictions[i].item())
                      and unc[i].item() <= unc_th.item()):
                    """ inaccurate and certain """
                    n_ic += (1 - confidences[i]) * (1 - torch.tanh(unc[i]))
                elif ((labels[i].item() != predictions[i].item())
                      and unc[i].item() > unc_th.item()):
                    """ inaccurate and uncertain """
                    n_iu += (1 - confidences[i]) * torch.tanh(unc[i])

            AvU = (n_ac + n_iu) / (n_ac + n_au + n_ic + n_iu + 1e-10)
            avu_list.append(AvU.data.cpu().numpy())
            unc_list.append(unc_th)

        auc_avu = auc(th_list, avu_list)
        return auc_avu

    def accuracy_vs_uncertainty(self, prediction, true_label, uncertainty,
                                optimal_threshold):
        n_ac = torch.zeros(
            1,
            device=true_label.device)  # number of samples accurate and certain
        n_ic = torch.zeros(1, device=true_label.device
                           )  # number of samples inaccurate and certain
        n_au = torch.zeros(1, device=true_label.device
                           )  # number of samples accurate and uncertain
        n_iu = torch.zeros(1, device=true_label.device
                           )  # number of samples inaccurate and uncertain

        avu = torch.ones(1, device=true_label.device)
        avu.requires_grad_(True)

        for i in range(len(true_label)):
            if ((true_label[i].item() == prediction[i].item())
                    and uncertainty[i].item() <= optimal_threshold):
                """ accurate and certain """
                n_ac += 1
            elif ((true_label[i].item() == prediction[i].item())
                  and uncertainty[i].item() > optimal_threshold):
                """ accurate and uncertain """
                n_au += 1
            elif ((true_label[i].item() != prediction[i].item())
                  and uncertainty[i].item() <= optimal_threshold):
                """ inaccurate and certain """
                n_ic += 1
            elif ((true_label[i].item() != prediction[i].item())
                  and uncertainty[i].item() > optimal_threshold):
                """ inaccurate and uncertain """
                n_iu += 1

        print('n_ac: ', n_ac, ' ; n_au: ', n_au, ' ; n_ic: ', n_ic, ' ;n_iu: ',
              n_iu)
        avu = (n_ac + n_iu) / (n_ac + n_au + n_ic + n_iu)

        return avu

    def forward(self, logits, labels, type=0):

        probs = F.softmax(logits, dim=1)
        confidences, predictions = torch.max(probs, 1)

        if type == 0:
            unc = self.entropy(probs)
        else:
            unc = self.model_uncertainty(probs)

        th_list = np.linspace(0, 1, 21)
        umin = torch.min(unc)
        umax = torch.max(unc)
        avu_list = []
        unc_list = []

        probs = F.softmax(logits, dim=1)
        confidences, predictions = torch.max(probs, 1)

        auc_avu = torch.ones(1, device=labels.device)
        auc_avu.requires_grad_(True)

        for t in th_list:
            unc_th = umin + (torch.tensor(t, device=labels.device) *
                             (umax - umin))
            n_ac = torch.zeros(
                1,
                device=labels.device)  # number of samples accurate and certain
            n_ic = torch.zeros(1, device=labels.device
                               )  # number of samples inaccurate and certain
            n_au = torch.zeros(1, device=labels.device
                               )  # number of samples accurate and uncertain
            n_iu = torch.zeros(1, device=labels.device
                               )  # number of samples inaccurate and uncertain

            for i in range(len(labels)):
                if ((labels[i].item() == predictions[i].item())
                        and unc[i].item() <= unc_th.item()):
                    """ accurate and certain """
                    n_ac += confidences[i] * (1 - torch.tanh(unc[i]))
                elif ((labels[i].item() == predictions[i].item())
                      and unc[i].item() > unc_th.item()):
                    """ accurate and uncertain """
                    n_au += confidences[i] * torch.tanh(unc[i])
                elif ((labels[i].item() != predictions[i].item())
                      and unc[i].item() <= unc_th.item()):
                    """ inaccurate and certain """
                    n_ic += (1 - confidences[i]) * (1 - torch.tanh(unc[i]))
                elif ((labels[i].item() != predictions[i].item())
                      and unc[i].item() > unc_th.item()):
                    """ inaccurate and uncertain """
                    n_iu += (1 - confidences[i]) * torch.tanh(unc[i])

            AvU = (n_ac + n_iu) / (n_ac + n_au + n_ic + n_iu + self.eps)
            avu_list.append(AvU)
            unc_list.append(unc_th)

        auc_avu = auc(th_list, avu_list)
        avu_loss = -1 * self.beta * torch.log(auc_avu + self.eps)
        return avu_loss, auc_avu


class AvULoss(nn.Module):
    """
    Calculates Accuracy vs Uncertainty Loss of a model.
    The input to this loss is logits from Monte_carlo sampling of the model, true labels,
    and the type of uncertainty to be used [0: predictive uncertainty (default); 
    1: model uncertainty]
    """
    def __init__(self, beta=1):
        super(AvULoss, self).__init__()
        self.beta = beta
        self.eps = 1e-10

    def entropy(self, prob):
        return -1 * torch.sum(prob * torch.log(prob + self.eps), dim=-1)

    def expected_entropy(self, mc_preds):
        return torch.mean(self.entropy(mc_preds), dim=0)

    def predictive_uncertainty(self, mc_preds):
        """
        Compute the entropy of the mean of the predictive distribution
        obtained from Monte Carlo sampling.
        """
        return self.entropy(torch.mean(mc_preds, dim=0))

    def model_uncertainty(self, mc_preds):
        """
        Compute the difference between the entropy of the mean of the
        predictive distribution and the mean of the entropy.
        """
        return self.entropy(torch.mean(
            mc_preds, dim=0)) - self.expected_entropy(mc_preds)

    def accuracy_vs_uncertainty(self, prediction, true_label, uncertainty,
                                optimal_threshold):
        # number of samples accurate and certain
        n_ac = torch.zeros(1, device=true_label.device)
        # number of samples inaccurate and certain
        n_ic = torch.zeros(1, device=true_label.device)
        # number of samples accurate and uncertain
        n_au = torch.zeros(1, device=true_label.device)
        # number of samples inaccurate and uncertain
        n_iu = torch.zeros(1, device=true_label.device)

        avu = torch.ones(1, device=true_label.device)
        avu.requires_grad_(True)

        for i in range(len(true_label)):
            if ((true_label[i].item() == prediction[i].item())
                    and uncertainty[i].item() <= optimal_threshold):
                """ accurate and certain """
                n_ac += 1
            elif ((true_label[i].item() == prediction[i].item())
                  and uncertainty[i].item() > optimal_threshold):
                """ accurate and uncertain """
                n_au += 1
            elif ((true_label[i].item() != prediction[i].item())
                  and uncertainty[i].item() <= optimal_threshold):
                """ inaccurate and certain """
                n_ic += 1
            elif ((true_label[i].item() != prediction[i].item())
                  and uncertainty[i].item() > optimal_threshold):
                """ inaccurate and uncertain """
                n_iu += 1

        print('n_ac: ', n_ac, ' ; n_au: ', n_au, ' ; n_ic: ', n_ic, ' ;n_iu: ',
              n_iu)
        avu = (n_ac + n_iu) / (n_ac + n_au + n_ic + n_iu)

        return avu

    def forward(self, logits, labels, optimal_uncertainty_threshold, type=0):

        probs = F.softmax(logits, dim=1)
        confidences, predictions = torch.max(probs, 1)

        if type == 0:
            unc = self.entropy(probs)
        else:
            unc = self.model_uncertainty(probs)

        unc_th = torch.tensor(optimal_uncertainty_threshold,
                              device=logits.device)

        n_ac = torch.zeros(
            1, device=logits.device)  # number of samples accurate and certain
        n_ic = torch.zeros(
            1,
            device=logits.device)  # number of samples inaccurate and certain
        n_au = torch.zeros(
            1,
            device=logits.device)  # number of samples accurate and uncertain
        n_iu = torch.zeros(
            1,
            device=logits.device)  # number of samples inaccurate and uncertain

        avu = torch.ones(1, device=logits.device)
        avu_loss = torch.zeros(1, device=logits.device)

        for i in range(len(labels)):
            if ((labels[i].item() == predictions[i].item())
                    and unc[i].item() <= unc_th.item()):
                """ accurate and certain """
                n_ac += confidences[i] * (1 - torch.tanh(unc[i]))
            elif ((labels[i].item() == predictions[i].item())
                  and unc[i].item() > unc_th.item()):
                """ accurate and uncertain """
                n_au += confidences[i] * torch.tanh(unc[i])
            elif ((labels[i].item() != predictions[i].item())
                  and unc[i].item() <= unc_th.item()):
                """ inaccurate and certain """
                n_ic += (1 - confidences[i]) * (1 - torch.tanh(unc[i]))
            elif ((labels[i].item() != predictions[i].item())
                  and unc[i].item() > unc_th.item()):
                """ inaccurate and uncertain """
                n_iu += (1 - confidences[i]) * torch.tanh(unc[i])

        avu = (n_ac + n_iu) / (n_ac + n_au + n_ic + n_iu + self.eps)
        p_ac = (n_ac) / (n_ac + n_ic)
        p_ui = (n_iu) / (n_iu + n_ic)
        #print('Actual AvU: ', self.accuracy_vs_uncertainty(predictions, labels, uncertainty, optimal_threshold))
        avu_loss = -1 * self.beta * torch.log(avu + self.eps)
        return avu_loss


def entropy(prob):
    return -1 * np.sum(prob * np.log(prob + 1e-15), axis=-1)


def predictive_entropy(mc_preds):
    """
    Compute the entropy of the mean of the predictive distribution
    obtained from Monte Carlo sampling during prediction phase.
    """
    return entropy(np.mean(mc_preds, axis=0))


def mutual_information(mc_preds):
    """
    Compute the difference between the entropy of the mean of the
    predictive distribution and the mean of the entropy.
    """
    MI = entropy(np.mean(mc_preds, axis=0)) - np.mean(entropy(mc_preds),
                                                      axis=0)
    return MI


def eval_avu(pred_label, true_label, uncertainty):
    """ returns AvU at various uncertainty thresholds"""
    t_list = np.linspace(0, 1, 21)
    umin = np.amin(uncertainty, axis=0)
    umax = np.amax(uncertainty, axis=0)
    avu_list = []
    unc_list = []
    for t in t_list:
        u_th = umin + (t * (umax - umin))
        n_ac = 0
        n_ic = 0
        n_au = 0
        n_iu = 0
        for i in range(len(true_label)):
            if ((true_label[i] == pred_label[i]) and uncertainty[i] <= u_th):
                n_ac += 1
            elif ((true_label[i] == pred_label[i]) and uncertainty[i] > u_th):
                n_au += 1
            elif ((true_label[i] != pred_label[i]) and uncertainty[i] <= u_th):
                n_ic += 1
            elif ((true_label[i] != pred_label[i]) and uncertainty[i] > u_th):
                n_iu += 1

        AvU = (n_ac + n_iu) / (n_ac + n_au + n_ic + n_iu + 1e-15)
        avu_list.append(AvU)
        unc_list.append(u_th)
    return np.asarray(avu_list), np.asarray(unc_list)


def accuracy_vs_uncertainty(pred_label, true_label, uncertainty,
                            optimal_threshold):

    n_ac = 0
    n_ic = 0
    n_au = 0
    n_iu = 0
    for i in range(len(true_label)):
        if ((true_label[i] == pred_label[i])
                and uncertainty[i] <= optimal_threshold):
            n_ac += 1
        elif ((true_label[i] == pred_label[i])
              and uncertainty[i] > optimal_threshold):
            n_au += 1
        elif ((true_label[i] != pred_label[i])
              and uncertainty[i] <= optimal_threshold):
            n_ic += 1
        elif ((true_label[i] != pred_label[i])
              and uncertainty[i] > optimal_threshold):
            n_iu += 1

    AvU = (n_ac + n_iu) / (n_ac + n_au + n_ic + n_iu)
    return AvU
