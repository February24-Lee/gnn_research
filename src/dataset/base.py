from pytorch_lightning          import LightningDataModule
from torch_geometric.data       import Batch, DataLoader
import torch


class Base_pl_data(LightningDataModule):
    def __init__(self, dataset) -> None:
        super().__init__()
        self.dataset            : Batch     = dataset
        self.train_batch_size   : int       = None
        self.test_batch_size    : int       = None
        self.val_batch_size     : int       = None
        self.is_shuffle         : bool      = True

        self.train_mask     : torch.Tensor = None
        self.test_mask      : torch.Tensor = None
        self.val_mask       : torch.Tensor = None

        self.train_dataset  : Batch        = None
        self.test_dataset   : Batch        = None
        self.val_dataset    : Batch        = None


    @property
    def num_features(self) -> int:
        return self.dataset.num_features

    @property
    def num_classes(self) -> int:
        return self.dataset.num_classes
    
    def train_dataloader(self) :
        return DataLoader(self.train_dataset, batch_size=self.train_batch_size, shuffle=self.is_shuffle)

    def val_dataloader(self) :
        return DataLoader(self.val_dataset, batch_size=self.val_batch_size)

    def test_dataloader(self) :
        return DataLoader(self.test_dataset, batch_size=self.test_batch_size)