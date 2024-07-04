from torch import nn
import pytorch_lightning as pl
import torch
from mulooc.models.utils.task_metrics import *
from mulooc.models.encoders import *
from mulooc.models.mulooc import MuLOOC
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt


class Probe(nn.Module):
    
    def __init__(self, encoder ,layer_dims = [512], num_classes = 50, dropout = 0, activation = 'relu', freeze_encoder = True, encoder_checkpoint = None, checkpoint_head = None, **kwargs):
        super(Probe, self).__init__()
        self.encoder = encoder
        self.layer_dims = layer_dims
        self.num_classes = num_classes
        self.dropout = dropout
        self.activation = activation
        self.freeze_encoder = freeze_encoder
        self.encoder_checkpoint = encoder_checkpoint
        self.checkpoint_head = checkpoint_head
        
        # print(self.encoder)
        
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
            self.encoder.eval()
            print('Encoder frozen') 
            
        self.head = self.build_head()
        
        head_device = next(self.head.parameters()).device
        
        if encoder_checkpoint is not None:
            ckpt = torch.load(encoder_checkpoint, map_location=head_device)
            self.encoder.load_state_dict(ckpt['state_dict'], strict = False)
            print(f'Encoder loaded from {encoder_checkpoint}')
            
        if checkpoint_head is not None:
            self.load_head_weights_from_checkpoint(checkpoint_head)
    
    def build_head(self):
        
        layers = []
        in_features = self.encoder.embed_dim
        previous_dim = in_features
        
        if len(self.layer_dims) > 0:
            for dim in self.layer_dims:
                layers.append(nn.Linear(previous_dim, dim))
                layers.append(nn.ReLU())
                if self.dropout > 0:
                    layers.append(nn.Dropout(self.dropout))
                previous_dim = dim
            
        layers.append(nn.Linear(previous_dim, self.num_classes))
        
        return nn.Sequential(*layers)
    
    
    def forward(self, x):
        encoded = self.encoder.extract_features(x)['encoded']
        
        if encoded.dim() == 3: ## sequence model
            logits = self.head(encoded[:,1:,:].mean(1))
        else:
            logits = self.head(encoded)
        return {
            'logits': logits,
            'encoded': encoded
        }
    
    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
        self.eval()
        print('Probe frozen')
        
    def load_head_weights_from_checkpoint(self, checkpoint):
        checkpoint = torch.load(checkpoint)['state_dict']
        self.head.load_state_dict(checkpoint, strict = True)
        print(f'Head loaded from {checkpoint}')
    
        
                
class LightningProbe(Probe, pl.LightningModule):
    
    def __init__(self, encoder, task = None,loss_fn =None,optimizer = None, layer_dims = [512], num_classes = 50, dropout = 0, activation = 'relu', freeze_encoder = True, checkpoint = None, checkpoint_head = None, **kwargs):
        super().__init__(encoder, layer_dims, num_classes, dropout, activation, freeze_encoder, checkpoint, checkpoint_head, **kwargs)
        
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.task = task
        
        
        self.test_agg = {
            'logits': [],
            'labels': [],
            'activations': []
        }
        
        self.get_metrics = eval(f'{self.task}_metrics')
    
    def log_metrics(self,metrics, stage = 'train'):
        # metrics is a dictionary containing the metric name and the value
        for k,v in metrics.items():
            if stage == 'train' or stage == 'val':
                self.log(f'{stage}_{k}',v, on_step = True, on_epoch = True, prog_bar = True, sync_dist = True)
            else:
                self.log(f'{stage}_{k}',v, on_step = False, on_epoch = True, prog_bar = True, sync_dist = True)
    
        
    def training_step(self, batch, batch_idx):
        audio, labels = batch['audio'], batch['labels']
        logits = self(audio)['logits']
        loss = self.loss_fn(logits, labels)
        
        metrics = self.get_metrics(logits, labels, self.num_classes)
        self.log_metrics(metrics, stage = 'train')
        self.log('train_loss', loss, on_step = False, on_epoch = True, prog_bar = True, sync_dist = True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        audio, labels = batch['audio'], batch['labels']
        logits = self(audio)['logits']
        
        loss = self.loss_fn(logits, labels)
        
        
        
        metrics = self.get_metrics(logits, labels, self.num_classes)
        self.log('val_loss', loss, on_step = False, on_epoch = True, prog_bar = True, sync_dist = True)
        self.log_metrics(metrics, stage = 'val')
        
        return loss
    
    def test_step(self, batch, batch_idx):
        audio, labels = batch['audio'], batch['labels']
        audio = audio.squeeze(0)# remove batch dimension for test
        if audio.shape[0] > 64:
            audio = audio[:64]
        encoded = self(audio)['encoded']
        if encoded.dim() == 3:
            logits = self.head(encoded[:,1:,:].mean(1).mean(0,keepdim = True))
        else:
            logits = self.head(encoded).mean(0,keepdim = True)
            
        #if linear probing, record the activations of each part of the embedding representation
        if len(self.layer_dims) == 0:
            # output activated logits * weight of classifier
            if isinstance(self.loss_fn,nn.CrossEntropyLoss):
                # softmax is applied to get the activated logits,
                activated_logits = torch.softmax(logits, dim = -1)
            elif isinstance(self.loss_fn,nn.BCEWithLogitsLoss):
                activated_logits = torch.sigmoid(logits)
            activations = torch.matmul(activated_logits, self.head[0].weight.T)
            
        
        self.test_agg['logits'].append(logits.detach().cpu())
        self.test_agg['labels'].append(labels.detach().cpu())
        self.test_agg['activations'].append(activations.detach().cpu())
    
    def on_test_epoch_end(self):
        self.test_agg['logits'] = torch.cat(self.test_agg['logits'], dim=0)
        self.test_agg['labels'] = torch.cat(self.test_agg['labels'], dim=0)
        self.test_agg['activations'] = torch.cat(self.test_agg['activations'], dim=0)
        
        metrics = self.get_metrics(self.test_agg['logits'], self.test_agg['labels'], self.num_classes)
        self.log_metrics(metrics, stage = 'test')
        
        print(self.test_agg['activations'].shape)
        

        if len(self.layer_dims) == 0 and self.logger is not None:
            

            fig, ax = plt.subplots(1, 1, figsize=(12, 6))
            ax.set_xlabel('Dimension')
            ax.set_ylabel('Activation')
            ax.set_title('Activations of Embedding Dimensions')

            bins = 50

            window = 50
            stride = self.test_agg['activations'].shape[-1]//bins
            dummy_activations_binned = torch.nn.functional.avg_pool1d(self.test_agg['activations'].unsqueeze(0), kernel_size=window, stride=stride).squeeze(0)

            x = np.arange(dummy_activations_binned.shape[1]).repeat(dummy_activations_binned.shape[0])
            y = dummy_activations_binned.T.numpy().flatten()

            sns.barplot(x=x, y=y, ax=ax,errorbar=("pi", 50), capsize=.1,
                err_kws={"color": "0", "linewidth": 2},
                linewidth=1, edgecolor="0",facecolor='0.8')
            
            self.logger.log({"activations": fig})
        
        
        
        print('Test metrics:', metrics)
        
        
        
        
        #clear test aggregation
        self.test_agg = {
            'logits': [],
            'labels': []
        }
    
    def configure_optimizers(self):
        if self.optimizer is None:
            return torch.optim.Adam(self.parameters(), lr=1e-4)
        else:
            return self.optimizer(self.parameters())
    
    # when saving a checkpoint, only save the head
    def on_save_checkpoint(self, checkpoint):
        checkpoint['state_dict'] = self.head.state_dict()