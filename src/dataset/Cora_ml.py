from torch_geometric.datasets import CitationFull
import torch_geometric.transforms as T
from .base import split_train_test_val, Base_pl_data

def cora_ml(data_path : str = 'data', transfrom = None):
    return CitationFull(root        = data_path,
                        name        = 'Cora_ML',
                        transform   = transfrom) 


class cora_ml_pl(Base_pl_data):
    def __init__(self,
                transform               = T.Compose([T.NormalizeFeatures()]),
                data_path               = 'data',
                each_train_idx_count    = 20,
                val_num                 = 500) -> None:
        dataset = cora_ml(data_path = data_path, transfrom=transform)
        super().__init__(dataset)
        
        self.test_dataset       = self.dataset 
        self.train_dataset      = self.dataset
        self.val_dataset        = self.dataset
        
        self.train_mask, self.test_mask, self.val_mask = split_train_test_val(dataset[0].y,
                                                                            dataset.num_classes,
                                                                            each_train_idx_count,
                                                                            val_num)
        
        self.val_batch_size     = 1
        self.train_batch_size   = 1
        self.test_batch_size    = 1