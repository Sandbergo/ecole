import gzip
import pickle
import numpy as np
import ecole
import torch
import torch.nn.functional as F
import torch_geometric
import os
from pathlib import Path
from utilities import Logger


class MLPPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        emb_size = 64
        var_nfeats = 19

        self.output_module = torch.nn.Sequential(
            torch.nn.Linear(var_nfeats, emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, 1, bias=False),
        )

    def forward(self, constraint_features, edge_indices, edge_features, variable_features):

        # A final MLP on the variable features
        output = self.output_module(variable_features).squeeze(-1)
        return output
