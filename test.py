from pytorch_lightning              import Trainer
from pytorch_lightning              import callbacks
from pytorch_lightning.callbacks    import ModelCheckpoint
from pytorch_lightning.loggers      import TestTubeLogger
from src.dataset import citeSeer_pl
from src.model import GAT_pl
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

import torch
import numpy as np
import random

RANDOM_SEED = 224

# -- For Reproduction
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed_all(RANDOM_SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

if __name__ == "__main__":
    # --- logger
    logger = TestTubeLogger('log', 'test1')

    # --- dataloader
    dataloader = citeSeer_pl()

    # --- Model
    model = GAT_pl( input_dim   = dataloader.num_features,
                    output_dim  = dataloader.num_classes,
                    train_mask  = dataloader.train_mask,
                    test_mask   = dataloader.test_mask,
                    val_mask    = dataloader.val_mask,
                    optimizer   = Adam,
                    lr_sche     = ReduceLROnPlateau)

    # --- callback
    checkpoint_callback = ModelCheckpoint(monitor='val_acc', save_last=5)

    # --- trainer
    trainer = Trainer(gpus=1, max_epochs=20, callbacks=[checkpoint_callback], logger=logger)
    trainer.fit(model, dataloader)