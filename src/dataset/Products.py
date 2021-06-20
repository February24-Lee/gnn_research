import torch
from ogb.nodeproppred import PygNodePropPredDataset
from.base import Base_pl_data

def products(data_path : str = 'data'):
    return PygNodePropPredDataset(root          = data_path,
                                name            = 'ogbn-products')


class products_pl(Base_pl_data):
        def __init__(self,
                data_path       = 'data') -> None:
                dataset = products(data_path=data_path)
                dataset[0].y = dataset[0].y.squeeze()
                super().__init__(dataset)
                
                self.test_dataset       = self.dataset 
                self.train_dataset      = self.dataset
                self.val_dataset        = self.dataset
                
                total_num = len(dataset[0].y)
                _train_idx, _test_idx, _val_test = torch.zeros(total_num), torch.zeros(total_num), torch.zeros(total_num)
                split_idx = dataset.get_idx_split()
                train_idx, valid_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]

                _train_idx[train_idx]   = 1
                _test_idx[test_idx]     = 1
                _val_test[valid_idx]    = 1
                
                self.train_mask          = _train_idx.bool()
                self.test_mask           = _test_idx.bool()
                self.val_mask            = _val_test.bool()

                self.val_batch_size     = 1
                self.train_batch_size   = 1
                self.test_bathc_dataset = 1