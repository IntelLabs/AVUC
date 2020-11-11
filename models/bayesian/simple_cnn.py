from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F

from variational_layers.conv_variational import Conv2dVariational
from variational_layers.linear_variational import LinearVariational

prior_mu = 0.0
prior_sigma = 1.0
posterior_mu_init = 0.0
posterior_rho_init = -3.0


class SCNN(nn.Module):
    def __init__(self):
        super(SCNN, self).__init__()
        self.conv1 = Conv2dVariational(prior_mu, prior_sigma,
                                       posterior_mu_init, posterior_rho_init,
                                       1, 32, 3, 1)
        self.conv2 = Conv2dVariational(prior_mu, prior_sigma,
                                       posterior_mu_init, posterior_rho_init,
                                       32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.1)
        self.dropout2 = nn.Dropout2d(0.1)
        self.fc1 = LinearVariational(prior_mu, prior_sigma, posterior_mu_init,
                                     posterior_rho_init, 9216, 128)
        self.fc2 = LinearVariational(prior_mu, prior_sigma, posterior_mu_init,
                                     posterior_rho_init, 128, 10)

    def forward(self, x):
        kl_sum = 0
        x, kl = self.conv1(x)
        kl_sum += kl
        x = F.relu(x)
        x, kl = self.conv2(x)
        kl_sum += kl
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x, kl = self.fc1(x)
        kl_sum += kl
        x = F.relu(x)
        x = self.dropout2(x)
        x, kl = self.fc2(x)
        kl_sum += kl
        output = F.log_softmax(x, dim=1)
        return output, kl
