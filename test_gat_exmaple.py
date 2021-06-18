from typing import List
from torch  import optim
from torch.optim.optimizer      import Optimizer
from torch_geometric.data.data  import Data
from src.dataset    import citeSeer
from src.model      import GAT

import torch
import torch.nn.functional as F

from torch_geometric.data import Dataset

EPOCH = 200


# --- dataloader 
'''
전체 graph를 사용하기 때문에 dataloader 불필요.
'''
citeSeer_ds = citeSeer('data')
citeseer = citeSeer_ds[0]

# --- Model
gat_model = GAT(citeSeer_ds.num_features,
        num_layers = 2,
        hidden_dim = [64],
        num_heads = [8],
        output_dim = citeSeer_ds.num_classes)

# --- Setting
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
gat_model.to(device)
citeseer = citeseer.to(device)
optimizer = torch.optim.Adam(gat_model.parameters(), lr = 0.005, weight_decay = 5e-4)
criterion = F.nll_loss

# --- training


def train(data : Dataset) -> None:
    gat_model.train()
    optimizer.zero_grad()
    out = gat_model(data.x, data.edge_index)
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    
@torch.no_grad()
def test(data : Dataset) -> dict:
    gat_model.eval()
    out= gat_model(data.x, data.edge_index)
    acc_dic = {}
    for name, mask in data('train_mask', 'test_mask', 'val_mask'):
        acc = float((out[mask].argmax(-1) == data.y[mask]).sum() / mask.sum())
        acc_dic[name[:-5]] = acc
    return acc_dic


for epoch in range(EPOCH):
    train(citeseer)
    acc_dic = test(citeseer)
    print(f"Epoch : {epoch+1:03d}, Train : {acc_dic['train']:.4f}, Test : {acc_dic['test']:.4f}")