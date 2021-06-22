from pytorch_lightning          import LightningDataModule
from torch_geometric.data       import Batch, DataLoader
import torch
import numpy as np


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
    
    
def split_train_test_val(label_dataset : torch.Tensor,
                        classes_num: int, 
                        each_class_num : int = 20,
                        val_num : int = 500):
    class_list = np.arange(classes_num)
    train_idx, test_idx, val_idx = torch.zeros(len(label_dataset)), torch.ones(len(label_dataset)), torch.zeros(len(label_dataset))
    find_idx_fn = lambda x : torch.nonzero(label_dataset == x, as_tuple=True)[0][:each_class_num]
    for class_idx in class_list : 
        idx = find_idx_fn(class_idx)
        train_idx[idx] = 1
        test_idx[idx] = 0
        
    _val_idx = torch.nonzero(test_idx, as_tuple=True)[0][:val_num]
    test_idx[_val_idx] = 0
    val_idx[_val_idx] = 1
    return train_idx.bool(), test_idx.bool(), val_idx.bool()
    