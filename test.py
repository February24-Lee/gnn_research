from logging import log
from pytorch_lightning              import Trainer, seed_everything
from pytorch_lightning.callbacks    import ModelCheckpoint
from pytorch_lightning.loggers      import  WandbLogger
from wandb.sweeps import config
from wandb.sweeps.config.cfg import SweepConfig

from src.dataset                    import DATASET_LIST
from src.model                      import MODEL_LIST

from torch.optim                    import Adam
from torch.optim.lr_scheduler       import ReduceLROnPlateau

import torch
import yaml
import argparse
import wandb
from functools import partial

RANDOM_SEED = 224
OPTIM_DIC = {'adam': Adam}

# -- For Reproduction
seed_everything(RANDOM_SEED)

#sweep_default = dict(
#       lr          = 1e-3,
#        num_layers  = 2,
#        hidden_dim  = 32,
#        drop_out    = 0.0,
#        num_heads   = 4
#    )

wandb.init()
sweep_config = wandb.config

def experinment(sweep_config, configs, logger):
    # --- for sweep config
    configs['optim_params']['lr'] = sweep_config['lr']
    configs['model_params']['num_layers'] = sweep_config['num_layers']
    configs['model_params']['hidden_dim'] = sweep_config['hidden_dim']
    configs['model_params']['drop_out'] = sweep_config['drop_out']
    configs['model_params']['num_heads'] = sweep_config['num_heads']


    # --- test
    dataloader = DATASET_LIST[configs['dataset_name']](**configs['dataset_params'])

    # --- Optimizer
    # TODO 하드 코딩 고치기
    partial_optim = partial(OPTIM_DIC[configs['optim_name']],  lr =configs['optim_params']['lr'])

    # --- Model
    model = MODEL_LIST[configs['model_name']]( input_dim   = dataloader.num_features,
                                            output_dim  = dataloader.num_classes,
                                            train_mask  = dataloader.train_mask,
                                            test_mask   = dataloader.test_mask,
                                            val_mask    = dataloader.val_mask,
                                            optimizer   = partial_optim,
                                            lr_sche     = ReduceLROnPlateau,
                                            **configs['model_params'])

    # --- callback
    checkpoint_callback = ModelCheckpoint(monitor='val_acc', save_last=5)
    trainer = Trainer(gpus=1, max_epochs=100, callbacks=[checkpoint_callback], logger=logger)
    
    trainer.fit(model, dataloader)
    # --- test 
    trainer.test()

if __name__ == "__main__":
    #parser = argparse.ArgumentParser()
    #parser.add_argument('--config', '-c', type = str, default='gat_arxiv.yaml')
    #args = parser.parse_args()
    #with open('configs/'+args.config, 'r') as f:
    #    configs = yaml.safe_load(f)

    with open('configs/gat_arxiv.yaml', 'r') as f:
        configs = yaml.safe_load(f)

    # --- logger
    logger_name = f"{configs['model_name']}_{configs['dataset_name']}"
    logger = WandbLogger(f"{logger_name}", save_dir='log', project='gnn_research', log_model='all')

    print(sweep_config)
    experinment(sweep_config    = sweep_config,
                configs         = configs,
                logger          = logger)    