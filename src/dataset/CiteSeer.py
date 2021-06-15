from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T

def citeSeer(data_path : str = 'data',
            ):
    return Planetoid(root       = data_path,
                    name        = 'CiteSeer',
                    transform   = T.NormalizeFeatures())