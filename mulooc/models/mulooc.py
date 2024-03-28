import pytorch_lightning as pl
from mulooc.models.losses import NTXent
import torch
from torch import nn
import matplotlib.pyplot as plt
import wandb
from torch import optim
from pytorch_lightning.cli import OptimizerCallable

class MuLOOC(nn.Module):
    
    def __init__(self,
                 encoder,
                 head_dims =[128],
                 temperature = 0.1,):
        super().__init__()
        self.encoder = encoder
        
        self.head_dims = head_dims
        self.encoder_dim = self.encoder.embed_dim
        
        self.heads = []
        
        for dim in head_dims:
            head = nn.Sequential(
                nn.Linear(self.encoder_dim, self.encoder_dim, bias=False),
                nn.ReLU(),
                nn.Linear(self.encoder_dim, dim, bias=False),
            )
            
            self.heads.append(head)
            
        self.heads = nn.ModuleList(self.heads)
        self.temperature = temperature
        self.loss = NTXent(temperature = temperature)
        
    
        
        
    def forward(self,x):
        
        wav = x['audio']
    
        wav = wav.contiguous().view(-1,1,wav.shape[-1]) ## [B*N_augmentations,T]
                
        encoded = self.encoder(wav)
        projected = [self.head(encoded) for head in self.heads]
        
        return {
            'projected':projected,
            'encoded':encoded,
            "wav":wav,
        }
        
    def forward_losses(self,x):
        out_ = self(x)
        
        labeled = x['labeled']
        labeled = labeled.contiguous().view(-1)
        # invert the semi-supervised contrastive matrix
        
        B, N_augmentations,_, T = x['audio'].shape
        matrices = self.get_contrastive_matrices(B,N_augmentations,T,x['augs'])
        
        negative_mask = torch.ones_like(matrices['invariant'])
        
        assert len(out_['projected']) == len(matrices), "Number of heads and number of loss matrices do not match"
        
        
        loss = []
        losses = {}
        sims = {}
        for i  in enumerate(matrices.items()):
            head = i[1][0]
            loss_head = self.loss(out_['projected'][i], matrices[head], negative_mask)
            sims[head] = self.loss.get_similarities(out_['projected'][i])
            losses[head] = loss_head
            loss.append(loss_head)

        loss = torch.stack(loss)
        
        return  {
            'loss':loss,
            'losses':losses,
            'sims': sims,
            'matrices':matrices
        }
    
    
    def extract_features(self,x,head=-1):
        
        # head -1 means the superspace above all heads
        # head -2 means the concatenated space of all heads
        # head n means the nth head
        
        with torch.no_grad():
            out_ = self(x)
            
            if head == -1:
                return out_['encoded']
            
            if head == -2:
                return torch.cat(out_['projected'],dim=-1)
            
            return out_['projected'][head]
        
    def get_contrastive_matrices(self,B,N,T,augs):
        
        ## returns a matrix of shape [B*N_augmentations,B*N_augmentations] with 1s where the labels are the same
        ## and 0s where the labels are different
        
        all_invariant_matrix = self.get_ssl_contrastive_matrix(B,N,device = labels.device)
        var_matrices = {}
        # sl_contrastive_matrix = self.get_sl_contrastive_matrix(B,N,labels, device = labels.device)
        for aug in augs:
            labels = augs[aug] #shape [B,N_aug]
            labels = labels.contiguous().view(-1)
            var_mat = self.get_sl_contrastive_matrix(B,N,labels, device = labels.device)
            var_mat = (var_mat == 0).int()
            var_mat = var_mat * all_invariant_matrix
            var_matrices[aug] = var_mat    
        
        
        matrices = {"invariant" : all_invariant_matrix, **var_matrices}
        
        return matrices        
        
    def get_ssl_contrastive_matrix(self,B,N,device):
        
        contrastive_matrix = torch.zeros(B*N,B*N,device = device)
        indices = torch.arange(0, B * N, 1, device=device)

        i_indices, j_indices = torch.meshgrid(indices, indices)
        mask = (i_indices // N) == (j_indices // N)
        contrastive_matrix[i_indices[mask], j_indices[mask]] = 1
        contrastive_matrix[j_indices[mask], i_indices[mask]] = 1

        return contrastive_matrix
    
    def get_sl_contrastive_matrix(self,B,N,labels, device):
        
        ## labels is of shape [B,N_augmentations,n_classes]
        ## labels is a one_hot encoding of the labels
        
        ## returns a matrix of shape [B*N_augmentations,B*N_augmentations] with 1s where the labels are the same
        ## and 0s where the labels are different
        
        indices = torch.arange(0, B * N, 1, device=device)
        
        i_indices, j_indices = torch.meshgrid(indices, indices)
        
        # if the label is -1 then there is no corresponding class in the batch
        
        x = (labels[i_indices] == labels[j_indices])*(labels[i_indices]==1)
        # contrastive_matrix = x.any(dim=-1).int()
        
        contrastive_matrix = (x.sum(-1) >= 1).int()
        
        return contrastive_matrix
    
    
    


class LightningMuLOOC(MuLOOC,pl.LightningModule):
    
    def __init__(self, encoder, head_dims=[128], temperature=0.1, optimizer = None):
        super().__init__(encoder, head_dims,temperature=temperature)
        
        self.optimizer = optimizer
        
        
    def configure_optimizers(self):
        if self.optimizer is None:
            optimizer = optim.Adam(
                self.parameters(), lr=1e-4, betas=(0.9, 0.999), eps=1e-8)
        else:
            optimizer = self.optimizer(self.parameters())
            
        return optimizer
    
    def training_step(self, batch, batch_idx):
            
        out_ = self.forward_losses(batch)
        loss = out_['loss']
        
        self.logging(out_)
        
        loss = loss.sum()
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True,sync_dist=True)
        
        return loss
    
    
    def validation_step(self,batch,batch_idx):
        
        out_ = self.forward_losses(batch)
        loss = out_['loss']
        
        # self.logging(out_)
        
        loss = loss.sum()
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True,sync_dist=True)
        
        return loss
    
    def logging(self,out_): 
        
        losses = out_['losses']
        sims = out_['sims']
        matrices = out_['matrices']
        
        for head in losses:
            self.log(f'{head}_loss',losses[head],on_step=True, on_epoch=True, prog_bar=True, logger=True,sync_dist=True)
        
        if self.logger:
            
            if self.global_step % 2000 == 0:
                for head in sims:
                    
                    
                    fig, ax = plt.subplots(2, 1)
                    
                    self.log_similarity(sims[head],f"{head}_similarity",ax = ax[0])
                    
                    
                    ax[1].imshow(matrices[head].detach(
                    ).cpu().numpy(), cmap="plasma")
                    self.logger.log_image(
                        'target_contrastive_matrix', [wandb.Image(fig)])
                    plt.close(fig)
                
            
    def log_similarity(self,similarity,name,ax = None):
        if ax is None:
            fig, ax = plt.subplots(1, 1)
        # remove diagonal
        similarity[torch.eye(similarity.shape[0],device = similarity.device).bool()] = 0
        ax.imshow(similarity.detach(
        ).cpu().numpy(), cmap="plasma")
        self.logger.log_image(
            name, [wandb.Image(fig)])
        plt.close(fig)
    