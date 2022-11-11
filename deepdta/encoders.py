"""Module for encoding protein sequences and ligands."""

import torch
from torch.autograd import Variable
import torch.nn.functional as F
from torch import nn

import numpy as np


torch.manual_seed(2)
np.random.seed(3)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CNN(nn.Sequential):
    """Convolutional neural network for encoding protein sequences and ligands."""

    def __init__(self, encoding, **config):
        super().__init__()
        if encoding == "drug":
            in_ch = [63] + config["cnn_drug_filters"]
            kernels = config["cnn_drug_kernels"]
            layer_size = len(config["cnn_drug_filters"])
            self.conv = nn.ModuleList(
                [
                    nn.Conv1d(
                        in_channels=in_ch[i],
                        out_channels=in_ch[i + 1],
                        kernel_size=kernels[i],
                    )
                    for i in range(layer_size)
                ]
            )
            self.conv = self.conv.double()
            n_size_d = self._get_conv_output((63, 100))
            # n_size_d = 1000
            self.fc1 = nn.Linear(n_size_d, config["hidden_dim_drug"])

        if encoding == "protein":
            in_ch = [26] + config["cnn_target_filters"]
            kernels = config["cnn_target_kernels"]
            layer_size = len(config["cnn_target_filters"])
            self.conv = nn.ModuleList(
                [
                    nn.Conv1d(
                        in_channels=in_ch[i],
                        out_channels=in_ch[i + 1],
                        kernel_size=kernels[i],
                    )
                    for i in range(layer_size)
                ]
            )
            self.conv = self.conv.double()
            n_size_p = self._get_conv_output((26, 1000))

            self.fc1 = nn.Linear(n_size_p, config["hidden_dim_protein"])

    def _get_conv_output(self, shape):
        var_bs = 1
        inp = Variable(torch.rand(var_bs, *shape))
        output_feat = self._forward_features(inp.double())
        n_size = output_feat.data.view(var_bs, -1).size(1)
        return n_size

    def _forward_features(self, var_x):
        for layer in self.conv:
            var_x = F.relu(layer(var_x))
        var_x = F.adaptive_max_pool1d(var_x, output_size=1)
        return var_x

    def forward(self, v):
        v = self._forward_features(v.double())
        v = v.view(v.size(0), -1)
        v = self.fc1(v.float())
        return v
