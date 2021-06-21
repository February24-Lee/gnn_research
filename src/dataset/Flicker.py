from    torch_geometric.datasets   import Flickr
import  torch_geometric.transforms as T

from.base import Base_pl_data

def flickr(data_path : str = 'data', transform = None):
    return Flickr(root       = data_path,
                  transform   = transform)


class flicker_pl(Base_pl_data):
        def __init__(self,
                transform       = T.Compose([T.NormalizeFeatures()]),
                data_path       = 'data') -> None:
                dataset = flickr(data_path=data_path,
                                transform=transform)
                super().__init__(dataset)
                
                self.test_dataset       = self.dataset 
                self.train_dataset      = self.dataset
                self.val_dataset        = self.dataset

                self.train_mask          = self.dataset[0].train_mask
                self.test_mask           = self.dataset[0].test_mask
                self.val_mask            = self.dataset[0].val_mask

                self.val_batch_size     = 1
                self.train_batch_size   = 1
                self.test_batch_size    = 1