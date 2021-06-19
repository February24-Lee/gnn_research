import torch
from pytorch_lightning      import LightningModule
from torch.nn               import Module
from torch.optim            import Optimizer
from torch_geometric.data   import Batch

from ..typing           import lr_schedule
from typing             import List, Tuple


class Base_pl(LightningModule):
    def __init__(self) -> None:
        super().__init__()
        self.model : Module = None


    # --- fix 
    def setup(self, stage) -> None:
        self.logger.log_hyperparams(self.hparams)
        

    def forward(self, x : torch.Tensor, edge_idx : torch.Tensor) -> torch.Tensor:
        return self.model(x, edge_idx)

    def training_step(self, batch : Batch, batch_dix : int) -> torch.Tensor:
        NotImplemented

    def validation_step(self, batch : Batch, batch_dix : int) -> torch.Tensor:
        NotImplemented

    def configure_optimizers(self) : #-> Tuple[List[Optimizer], List[lr_schedule]]:
        optim = self.hparams['optimizer']
        lr_sche = self.hparams['lr_sche']
        optimizer = optim(self.model.parameters())
        scheduler = lr_sche(optimizer)
        return {'optimizer' : optimizer, 'lr_scheduler' : {'scheduler' : scheduler, 'monitor' : 'val_acc'}}
    

    