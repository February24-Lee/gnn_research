import torch.nn as nn
import torch.nn.functional as F
import torch

from torch_geometric.nn import GATConv

from typing import List

class GAT(nn.Module):
    def __init__(self,
                input_dim   : int       = None,
                num_layers  : int       = 2,
                hidden_dim  : List[int] = [32],
                num_heads   : List[int] = [1, 1],
                output_dim  : int       = None,
                drop_out    : float     = 0.6):
        super().__init__()
        self.input_dim          = input_dim
        self.num_layers         = num_layers
        self.hidden_dim         = hidden_dim
        self.num_heads          = num_heads
        self.output_dim         = output_dim
        self.drop_out           = drop_out

        GAT_module_list = nn.ModuleList()
        x_dim = input_dim
        for idx in range(self.num_layers-1):
            GAT_module_list.append(GATConv(x_dim,
                                        hidden_dim[idx],
                                        num_heads[idx]))
            x_dim = hidden_dim[idx] * num_heads[idx]
        GAT_module_list.append(GATConv(x_dim,
                                    output_dim))
        self.GAT_module_list = GAT_module_list
        
    def forward(self, x, edge_idx):
        for gat_layer in self.GAT_module_list[:-1]:
            x = F.dropout(x, p = self.drop_out, training = self.training)
            x = F.elu(gat_layer(x, edge_idx))
        x = F.dropout(x, p = self.drop_out, training = self.training)
        x = self.GAT_module_list[-1](x,edge_idx)
        out = F.log_softmax(x)
        return out