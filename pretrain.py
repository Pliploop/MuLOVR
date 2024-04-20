
from mulooc.models.mulooc import LightningMuLOOC
from mulooc.dataloading.datamodule import AudioDataModule
from pytorch_lightning.cli import SaveConfigCallback, LightningCLI
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import yaml
import os
from jsonargparse import lazy_instance
from pytorch_lightning.strategies import DDPStrategy
import wandb



class LoggerSaveConfigCallback(SaveConfigCallback):
    def save_config(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
        if trainer.logger is not None:
            experiment_name = trainer.logger.experiment.name
            # Required for proper reproducibility
            config = self.parser.dump(self.config, skip_none=False)
            with open(self.config_filename, "r") as config_file:
                config = yaml.load(config_file, Loader=yaml.FullLoader)
                if trainer.global_rank == 0:
                    trainer.logger.experiment.config.update(config, allow_val_change=True)
            with open(os.path.join(os.path.join(self.config['ckpt_path'], experiment_name), "config.yaml"), 'w') as outfile:
                yaml.dump(config, outfile, default_flow_style=False)
                
            #instanciate a ModelCheckpoint saving the model every epoch`
            
            
            

class MyLightningCLI(LightningCLI):
    
    trainer_defaults = {
        "strategy": lazy_instance(DDPStrategy, find_unused_parameters=False),
    }
    def add_arguments_to_parser(self, parser):
        parser.add_argument("--log", default=False)
        parser.add_argument("--log_model", default=False)
        parser.add_argument("--ckpt_path", default="checkpoints")
        parser.add_argument("--resume_id", default=None)
        parser.add_argument("--resume_from_checkpoint", default=None)
        parser.add_argument("--project", default="MuLOOC-sandbox")


if __name__ == "__main__":

    cli = MyLightningCLI(model_class=LightningMuLOOC, datamodule_class=AudioDataModule, seed_everything_default=123,
                         run=False, save_config_callback=LoggerSaveConfigCallback, save_config_kwargs={"overwrite": True},trainer_defaults=MyLightningCLI.trainer_defaults)
    
    # cli.instantiate_classes()

    if cli.config.log:
        logger = WandbLogger(project=cli.config.project, id=cli.config.resume_id)
        if cli.trainer.global_rank == 0:
            experiment_name = logger.experiment.name
        else:
            api = wandb.Api()
            runs = api.runs(cli.config.project)
            latest_run = runs[0]
            experiment_name = latest_run.name
        ckpt_path = cli.config.ckpt_path
    else:
        logger = None

    # if cli.trainer.global_rank == 0:
    #     print('trainer rank 0')
    #     cli.trainer.logger = logger
    # else:
    #     cli.trainer.logger = None

    cli.trainer.logger = logger
    
    print("checkpoint path",ckpt_path)
    print("experiment name",experiment_name)    
    recent_callback_step_latest = ModelCheckpoint(
                dirpath=os.path.join(cli.config.ckpt_path, experiment_name),
                filename='checkpoint-{step}-recent',  # This means all checkpoints are saved, not just the top k
                every_n_train_steps = 1000,  # Replace with your desired value
                save_top_k = 1
            )
            
    recent_callback_step = ModelCheckpoint(
        dirpath=os.path.join(cli.config.ckpt_path, experiment_name),
        filename='checkpoint-{step}',  # This means all checkpoints are saved, not just the top k
        every_n_train_steps = 50000,  # Replace with your desired value
        save_top_k = -1
    )
    callbacks = [recent_callback_step_latest,recent_callback_step ]
    
    if cli.config.log:
        cli.trainer.callbacks = cli.trainer.callbacks[:-1]+callbacks
        print("logging")

    try:
        if not os.path.exists(os.path.join(ckpt_path, experiment_name)):
            os.makedirs(os.path.join(ckpt_path, experiment_name))
    except:
        pass
    
    print("let's fit this mf")

    cli.trainer.fit(model=cli.model, datamodule=cli.datamodule, ckpt_path=cli.config.resume_from_checkpoint)
