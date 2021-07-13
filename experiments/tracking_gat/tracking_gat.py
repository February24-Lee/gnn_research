import sys, os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
sys.path.insert(0, parentdir)

from pytorch_lightning              import Trainer, seed_everything
from pytorch_lightning.callbacks    import ModelCheckpoint, LearningRateMonitor, lr_monitor, EarlyStopping
from pytorch_lightning.loggers      import WandbLogger

from src.dataset                    import DATASET_LIST
from src.model                      import MODEL_LIST

from torch.optim                    import Adam
from torch.optim.lr_scheduler       import ReduceLROnPlateau

import torch
import yaml
import wandb
from functools import partial

RANDOM_SEED = 224
OPTIM_DIC = {'adam': Adam}

# -- For Reproduction
seed_everything(RANDOM_SEED)


def experinment(configs, logger):
    # --- test
    dataloader = DATASET_LIST[configs['dataset_name']](**configs['dataset_params'])

    # --- Optimizer
    # TODO 하드 코딩 고치기
    partial_optim   = partial(OPTIM_DIC[configs['optim_name']],  lr =configs['optim_params']['lr'])
    #lr_sche         = ReduceLROnPlateau
    lr_sche         = None

    # --- Model
    model = MODEL_LIST[configs['model_name']]( input_dim   = dataloader.num_features,
                                            output_dim  = dataloader.num_classes,
                                            train_mask  = dataloader.train_mask,
                                            test_mask   = dataloader.test_mask,
                                            val_mask    = dataloader.val_mask,
                                            optimizer   = partial_optim,
                                            lr_sche     = lr_sche,
                                            **configs['model_params'])

    # --- callback
    checkpoint_callback = ModelCheckpoint(dirpath   = 'experiments/tracking_gat/log/', 
                                        filename    = '{epoch}-{val_loss:.2f}-{val_acc:.2f}-{train_acc:.2f}',
                                        monitor     = 'val_loss',
                                        mode        = 'min',
                                        save_last   = True,
                                        save_top_k  = -1)
    lr_monitor          = LearningRateMonitor(logging_interval="epoch")
    #early_stopping      = EarlyStopping(monitor     = 'val_loss', 
    #                                    patience    = 10,
    #                                    mode        = 'min')
    trainer = Trainer(gpus                  = [1],
                    max_epochs              = 300,
                    check_val_every_n_epoch = 10,
                    callbacks               = [checkpoint_callback, lr_monitor],
                    logger                  = logger)
    
    trainer.fit(model, dataloader)
    # --- test 
    trainer.test()

if __name__ == "__main__":
    with open('experiments/tracking_gat/gat_citeseer.yaml', 'r') as f:
        configs = yaml.safe_load(f)

    # --- logger
    logger_name = f"{configs['model_name']}_{configs['dataset_name']}"
    logger = WandbLogger(f"{logger_name}", save_dir='experiments/tracking_gat/log', project='gnn_research', log_model='all')

    experinment(configs         = configs,
                logger          = logger)    