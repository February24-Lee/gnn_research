from pytorch_lightning              import Trainer, seed_everything
from pytorch_lightning.callbacks    import ModelCheckpoint
from pytorch_lightning.loggers      import TestTubeLogger, WandbLogger

from src.dataset                    import *
from src.model                      import GAT_pl

from torch.random                   import seed
from torch.optim                    import Adam
from torch.optim.lr_scheduler       import ReduceLROnPlateau

import torch
import numpy as np
import random

RANDOM_SEED = 224

# -- For Reproduction
seed_everything(RANDOM_SEED)
'''
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed_all(RANDOM_SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
'''
DATA_LIST = {'citeseer' : citeSeer_pl,
            'core_ml'   : cora_ml_pl,
            'pumbed'    : pubmed_pl,
            'arxiv'     : arxiv_pl,
            'flicker'   : flicker_pl,
            'products'  : products_pl}

if __name__ == "__main__":
    
    # --- dataloader
    for dataset_name, dataset in DATA_LIST.items():
        dataloader = dataset(data_path = 'data')
        
        # --- logger
        #logger = TestTubeLogger('log', 'test1')
        logger = WandbLogger(f'test_gat_{dataset_name}', save_dir='log', project='gnn_research', log_model='all')

        # --- Model
        model = GAT_pl( input_dim   = dataloader.num_features,
                        output_dim  = dataloader.num_classes,
                        train_mask  = dataloader.train_mask,
                        test_mask   = dataloader.test_mask,
                        val_mask    = dataloader.val_mask,
                        optimizer   = Adam,
                        lr_sche     = ReduceLROnPlateau,
                        num_layers= 4,
                        hidden_dim = [64, 64, 64],
                        num_heads = [8, 8, 8, 8])

        # --- callback
        checkpoint_callback = ModelCheckpoint(monitor='val_acc', save_last=5)
        trainer = Trainer(gpus=1, max_epochs=100, callbacks=[checkpoint_callback], logger=logger)
        
        trainer.fit(model, dataloader)
        # --- test 
        trainer.test()