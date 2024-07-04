
from mulooc.models.equimulooc import LightningEquiMuLOOC
from mulooc.dataloading.datamodule import PrecomputedDataModule
from pytorch_lightning.cli import LightningCLI
from pytorch_lightning.cli import SaveConfigCallback
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import yaml
import os
import torch
from hashlib import sha256


class LoggerSaveConfigCallback(SaveConfigCallback):
    def save_config(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
        if trainer.logger is not None:
            experiment_name = trainer.logger.experiment.name
            # Required for proper reproducibility
            config = self.parser.dump(self.config, skip_none=False)
            with open(self.config_filename, "r") as config_file:
                config = yaml.load(config_file, Loader=yaml.FullLoader)
                trainer.logger.experiment.config.update(config, allow_val_change=True)
            
            
            
            new_experiment_name = experiment_name+f'_{config["data"]["task"]}'
                
            if not os.path.exists(os.path.join(self.config['ckpt_path'], new_experiment_name)) and  (not config['test'] or config['resume_id'] is None):
                os.makedirs(os.path.join(self.config['ckpt_path'], new_experiment_name))
                
            with open(os.path.join(os.path.join(self.config['ckpt_path'], new_experiment_name), "config.yaml"), 'w') as outfile:
                yaml.dump(config, outfile, default_flow_style=False)

            
            trainer.logger.experiment.name = new_experiment_name
            
            #add a checkpoint callback that saves the model every epoch
            ## and that saves the best model based on validation loss
            
            


class MyLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.add_argument("--log", default=False)
        parser.add_argument("--log_model", default=False)
        parser.add_argument("--ckpt_path", default="checkpoints-finetuning")
        parser.add_argument("--resume_from_checkpoint", default=None)
        parser.add_argument("--resume_id", default=None)
        parser.add_argument('--test', default=False)
        parser.add_argument('--early_stopping_patience', default=5)
        parser.add_argument('--project', default='MuLOOC-equivariance-sandbox')
        

if __name__ == "__main__":
    

    cli = MyLightningCLI(model_class=LightningEquiMuLOOC, datamodule_class=PrecomputedDataModule, seed_everything_default=123,
                         run=False, save_config_callback=LoggerSaveConfigCallback, save_config_kwargs={"overwrite": True})

    # cli.instantiate_classes()
    

    if cli.config.log:
        logger = WandbLogger(project=cli.config.project,id = cli.config.resume_id)
        experiment_name = logger.experiment.name+f"_{cli.config['data']['task']}"
        ckpt_path = cli.config.ckpt_path
    else:
        logger = None
        # random experiment name
        experiment_name = 'nowandb' + sha256(str(cli.config).encode()).hexdigest()

    cli.trainer.logger = logger

    try:
        if not os.path.exists(os.path.join(ckpt_path, experiment_name)) and not cli.config.test:
            os.makedirs(os.path.join(ckpt_path, experiment_name))
    except:
        pass
    
    best_val_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath=os.path.join(cli.config.ckpt_path, experiment_name),
        filename='best-val-{step}',
        save_top_k=1,
        mode='min',
        every_n_epochs=1
    )
    
    # early_stopping_callback = EarlyStopping(
    #     monitor='val_loss',
    #     patience=cli.config.early_stopping_patience,
    #     mode='min'
    # )
    
    cli.trainer.callbacks = cli.trainer.callbacks[:-1]+[best_val_callback]
    
    if not cli.config.test:    
        cli.trainer.fit(model=cli.model, datamodule=cli.datamodule)
        # load the best val path checkpoint
        ckpt = torch.load(best_val_callback.best_model_path)['state_dict']
        cli.model.load_state_dict(ckpt)
            
        
    cli.trainer.test(model=cli.model, datamodule=cli.datamodule)
    parent_folder = os.path.dirname(best_val_callback.best_model_path)
    for file in os.listdir(parent_folder):
            os.remove(os.path.join(parent_folder, file))
    os.rmdir(parent_folder)
    
