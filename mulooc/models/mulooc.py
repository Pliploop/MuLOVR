import pytorch_lightning as pl
from mulooc.models.losses import NTXent
import torch
from torch import nn
import matplotlib.pyplot as plt
import wandb
from torch import optim
from pytorch_lightning.cli import OptimizerCallable
from mulooc.models.encoders import *
import torch.distributed as dist


class MuLOOC(nn.Module):
    
    def __init__(self,
                 encoder,
                 head_dims = [[512,128]],
                 temperature = 0.1,
                 feat_extract_head = -2,
                 **kwargs):
        super(MuLOOC,self).__init__()
        
        self.encoder = encoder
        self.head_dims = head_dims
        self.encoder_dim = self.encoder.embed_dim
        self.heads = []
        
        for dim in head_dims:
            head = []
            last_dim = self.encoder_dim
            for d in dim:
                head.append(nn.Linear(last_dim,d,bias = False))
                head.append(nn.ReLU())
                last_dim = d
            self.heads.append(nn.Sequential(*head))
            
        self.heads = nn.ModuleList(self.heads)
        self.temperature = temperature
        self.loss = NTXent(temperature = temperature)
        self.feat_extract_head = feat_extract_head
        
        if self.feat_extract_head == -2:
            self.embed_dim = sum([dim[-1] for dim in self.head_dims])
        elif self.feat_extract_head == -1:
            self.embed_dim = self.encoder_dim
        elif self.feat_extract_head >= 0:
            self.embed_dim = self.head_dims[self.feat_extract_head]
        
        
    def forward(self,x):
        wav = x['audio']
        if wav.dim() == 4:
            wav = wav.contiguous().view(-1,1,wav.shape[-1]) ## [B*N_augmentations,T]
        elif wav.dim() == 5: # spectrogram : [B*N_augmentations,1,F,T]
            wav = wav.contiguous().view(-1,1,wav.shape[-2],wav.shape[-1])
                
        encoded = self.encoder(wav)
        projected = [head(encoded) for head in self.heads]
        
        return {
            'projected':projected,
            'encoded':encoded,
            "wav":wav,
        }
        
    def forward_losses(self,x):
        out_ = self(x)
        
        B, N_augmentations = x['audio'].shape[:2]
        device = x['audio'].device
        matrices = self.get_contrastive_matrices(B,N_augmentations,x['augs'],device)
        
        negative_mask = torch.ones_like(matrices['invariant'])
        
        assert len(out_['projected']) <= len(matrices), "Number of heads and number of loss matrices do not match"
        
        
        loss = []
        losses = {}
        sims = {}
        for i in enumerate(matrices.items()):
            if i[0] < len(out_['projected']): # if there are more matrices that heads, we ignore the extra matrices
                head = i[1][0]
                loss_head = self.loss(out_['projected'][i[0]], matrices[head], negative_mask)
                if isinstance(loss_head,dict): # if distributed training
                    
                    matrices[head] = loss_head['positive_mask']
                    sims[head] = loss_head['similarity']
                    loss_head = loss_head['loss']
                else:
                    sims[head] = self.loss.get_similarities(out_['projected'][i[0]])
                
                # if dist.is_available() and dist.is_initialized():
                #     dist.all_reduce(loss_head)
                #     loss_head = loss_head / dist.get_world_size()
                    
                losses[head] = loss_head
                loss.append(loss_head)

        loss = torch.stack(loss)
        
        return  {
            'loss':loss,
            'losses':losses,
            'sims': sims,
            'matrices':matrices
        }
    
    
    def extract_features(self,x,head=None):
        
        # head -1 : the superspace above all heads
        # head -2 : the concatenated space of all heads
        # head n : the nth head
        if head is None:
            head = self.feat_extract_head

        with torch.no_grad():
            out_ = self({
                'audio':x,
            })
            
            if head == -1:
                return {'encoded': out_['encoded']}
            
            if head == -2:
                return {"encoded" : torch.cat(out_['projected'],dim=-1)}
            
            return {"encoded": out_['projected'][head]}
        
    def get_contrastive_matrices(self,B,N,augs,device):
        
        ## returns a matrix of shape [B*N_augmentations,B*N_augmentations] with 1s where the labels are the same
        ## and 0s where the labels are different
        
        
        all_invariant_matrix = self.get_ssl_contrastive_matrix(B,N,device = device)
        var_matrices = {}
        # sl_contrastive_matrix = self.get_sl_contrastive_matrix(B,N,labels, device = labels.device)

        for aug in augs:
            if len(var_matrices) < len(self.heads):
         
                labels = augs[aug] #shape [B,N_aug]
                # print(labels)
                labels = labels.contiguous().view(-1)
                # print(labels)
                var_mat = self.get_sl_contrastive_matrix(B,N,(labels == 0).int(), device = device)
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
        
        
        indices = torch.arange(0, B * N, 1, device=device)
        i_indices, j_indices = torch.meshgrid(indices, indices)
        
        
        if labels.dim() == 3:
            x = (labels[i_indices] == labels[j_indices])*(labels[i_indices]==1)
            contrastive_matrix = (x.sum(-1) >= 1).int()
        
        else:
            contrastive_matrix = torch.mm(labels.unsqueeze(-1).float(),labels.unsqueeze(-1).t().float()).int()
            contrastive_matrix[torch.eye(contrastive_matrix.shape[0],device = contrastive_matrix.device).bool()] = 0
        
        return contrastive_matrix
    
    
    


class LightningMuLOOC(MuLOOC,pl.LightningModule):
    
    def __init__(self,
                 encoder,
                 head_dims = [[512,128]],
                 temperature=0.1,
                 feat_extract_head = -2,
                 optimizer=None,**kwargs):
        super().__init__(encoder, head_dims,temperature=temperature,feat_extract_head=feat_extract_head)
        
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
        
        if isinstance(loss,dict):
            loss = loss['loss']
        
        loss = loss.mean()
        
        self.logging(out_)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True,sync_dist=True)
        
        return loss
    
    
    def validation_step(self,batch,batch_idx):
        
        out_ = self.forward_losses(batch)
        loss = out_['loss']
        
        loss = loss.mean()
        # self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True,sync_dist=True)
        
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
                    
                    self.log_similarity(sims[head].clone(),f"{head}_similarity",ax = ax[0])
                    
                    
                    ax[1].imshow(matrices[head].detach(
                    ).cpu().numpy(), cmap="plasma")
                    self.logger.log_image(
                        f'{head}_target_contrastive_matrix', [wandb.Image(fig)])
                    plt.close(fig)
                
            
    def log_similarity(self,similarity,name,ax = None):
        if ax is None:
            fig, ax = plt.subplots(1, 1)
        # remove diagonal
        similarity[torch.eye(similarity.shape[0],device = similarity.device).bool()] = 0
        ax.imshow(similarity.detach(
        ).cpu().numpy(), cmap="plasma")
        if ax is None:
            self.logger.log_image(
                name, [wandb.Image(fig)])
            plt.close(fig)
            
            