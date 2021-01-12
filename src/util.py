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
# Utily functions for variational inference in Bayesian deep neural networks
# ELBO_loss -> to compute evidence lower bound loss
# get_rho -> variance (sigma) is represented by softplus function  'sigma = log(1 + exp(rho))' 
#            to make sure it remains always positive and non-transformed 'rho' gets 
#            updated during training.
# MOPED   ->   set the priors and initialize approximate variational posteriors of Bayesian NN
#              with Empirical Bayes
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


def ELBO_loss(out, y, kl_loss, num_data_samples, batch_size):
    nll_loss = F.cross_entropy(out, y)
    return nll_loss + ((1.0 / num_data_samples) * kl_loss)


def get_rho(sigma, delta):
    rho = torch.log(torch.expm1(delta * torch.abs(sigma)) + 1e-20)
    return rho


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


def MOPED(model, det_model, det_checkpoint, delta):
    """
    Set the priors and initialize surrogate posteriors of Bayesian NN with Empirical Bayes
    MOPED (Model Priors with Empirical Bayes using Deterministic DNN)
    Ref: https://arxiv.org/abs/1906.05323
         'Specifying Weight Priors in Bayesian Deep Neural Networks with Empirical Bayes'. AAAI 2020.
    """
    det_model.load_state_dict(torch.load(det_checkpoint))
    for (idx, layer), (det_idx,
                       det_layer) in zip(enumerate(model.modules()),
                                         enumerate(det_model.modules())):
        if (str(layer) == 'Conv1dVariational()'
                or str(layer) == 'Conv2dVariational()'
                or str(layer) == 'Conv3dVariational()'
                or str(layer) == 'ConvTranspose1dVariational()'
                or str(layer) == 'ConvTranspose2dVariational()'
                or str(layer) == 'ConvTranspose3dVariational()'):
            #set the priors
            layer.prior_weight_mu.data = det_layer.weight
            layer.prior_bias_mu.data = det_layer.bias

            #initialize surrogate posteriors
            layer.mu_kernel.data = det_layer.weight
            layer.rho_kernel.data = get_rho(det_layer.weight.data, delta)
            layer.mu_bias.data = det_layer.bias
            layer.rho_bias.data = get_rho(det_layer.bias.data, delta)
        elif (str(layer) == 'LinearVariational()'):
            #set the priors
            layer.prior_weight_mu.data = det_layer.weight
            layer.prior_bias_mu.data = det_layer.bias

            #initialize the surrogate posteriors
            layer.mu_weight.data = det_layer.weight
            layer.rho_weight.data = get_rho(det_layer.weight.data, delta)
            layer.mu_bias.data = det_layer.bias
            layer.rho_bias.data = get_rho(det_layer.bias.data, delta)
        elif str(layer).startswith('Batch'):
            #initialize parameters
            layer.weight.data = det_layer.weight
            layer.bias.data = det_layer.bias
            layer.running_mean.data = det_layer.running_mean
            layer.running_var.data = det_layer.running_var
            layer.num_batches_tracked.data = det_layer.num_batches_tracked

    model.state_dict()
    return model


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
