import torch
import torch.nn     as nn
import torch.nn.functional as F
from torch.tensor   import Tensor
from torch.optim    import Optimizer

from torch_geometric.nn import GATConv
from torch_geometric.data import Batch

from torchmetrics import Accuracy

import pytorch_lightning as pl

from typing     import Callable, List, Tuple

from .base      import Base_pl
from ..typing   import lr_schedule



class GAT(nn.Module):
    def __init__(self,
                input_dim   : int           = None,
                num_layers  : int           = 2,
                hidden_dim  : List[int]     = [32],
                num_heads   : List[int]     = [1, 1],
                output_dim  : int           = None,
                drop_out    : float         = 0.6):
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

class GAT_pl(Base_pl):
    def __init__(self,
                input_dim   : int       = None,
                num_layers  : int       = 2,
                hidden_dim  : List[int] = [32],
                num_heads   : List[int] = [1, 1],
                output_dim  : int       = None,
                drop_out    : float     = 0.6,
                criterion   : Callable  = F.nll_loss,
                train_mask  : torch.Tensor  = None,
                test_mask   : torch.Tensor  = None,
                val_mask    : torch.Tensor  = None,
                optimizer   : Optimizer     = None,
                lr_sche     : lr_schedule   = None) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.model = GAT(input_dim  = input_dim,
                        num_layers  = num_layers,
                        hidden_dim  = hidden_dim,
                        num_heads   = num_heads,
                        output_dim  = output_dim,
                        drop_out    = drop_out)
        
    def training_step(self, batch : Batch, batch_dix : int) -> torch.Tensor:
        train_acc  = Accuracy().cuda()
        y_hat = self(batch.x, batch.edge_index)
        criterion = self.hparams['criterion']
        mask = self.hparams['train_mask']
        loss = criterion(y_hat[mask], batch.y[mask])
        #self.logger.experiment.log('train_acc', train_acc(y_hat[mask].argmax(-1), batch.y[mask]), on_epoch=True, logger=True)
        self.log('train_acc', train_acc(y_hat[mask].argmax(-1), batch.y[mask]))
        self.logger.experiment.log({'train_acc': train_acc(y_hat[mask].argmax(-1), batch.y[mask])})
        return loss

    def validation_step(self, batch : Batch, batch_dix : int) -> torch.Tensor:
        val_acc  = Accuracy().cuda()
        mask = self.hparams['val_mask']
        y_hat = self(batch.x, batch.edge_index)
        self.log('val_acc', val_acc(y_hat[mask].argmax(-1), batch.y[mask]))
        self.logger.experiment.log({'val_acc': val_acc(y_hat[mask].argmax(-1), batch.y[mask])})

    def test_step(self, batch : Batch, batch_dix : int) -> torch.Tensor:
        test_acc  = Accuracy().cuda()
        mask = self.hparams['test_mask']
        y_hat = self(batch.x, batch.edge_index)
        self.log('test_acc', test_acc(y_hat[mask].argmax(-1), batch.y[mask]))
        self.logger.experiment.log({'test_acc': test_acc(y_hat[mask].argmax(-1), batch.y[mask])})
    
        

