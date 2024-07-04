# two lightning variants of muLooc
# one takes the embedding and the parameters and predicts the transformed embedding
# the other takes the embedding and the transformed embedding and predicts the parameters
# in all cases, features are extracted using the self extract_features method
# what changed is the way the features are used

import torch
from mulooc.models.mulooc import MuLOOC
from mulooc.models.utils.mlp import MLP
from pytorch_lightning import LightningModule
import wandb
from torchmetrics.functional import r2_score


# create an R2 metric between predicted and true values

    

class EmbeddingEquiMuLOOC(torch.nn.Module):
    
    def __init__(self,
                 transformation_embedding_layers = [128],
                 predictor_layers=  [2048,2048],
                 chunk_level = True,
                 embed_dim = None,
                 conditioned = True,
                 probing_space = [0],
                 input_size = 1,
                 **kwargs):
        
        
        super(EmbeddingEquiMuLOOC,self).__init__()
        
        self.embed_dim = embed_dim
            
        
        
        # loss function is a linear combination of cosine distance and MSE
        self.cosine_loss_weight = 0.5
        self.mse_loss_weight = 0.5
        
        # create self.range such that we can extract the relevant indices.
        # relevant indices are given by probing_space. if probing space is [0], relevant indices are [0...embed_dim]
        # if [1], [embed_dim...2*embed_dim] etc.
        # the last dimension MUST be taken into account
        
        self.range = torch.arange(0,self.embed_dim)
        self.range = torch.cat([self.range + i * self.embed_dim for i in probing_space])
        
        self.embed_dim = self.embed_dim * len(probing_space)
        self.input_size = input_size
        
        
        self.predictor_layers = predictor_layers + [self.embed_dim]
        self.transformation_embedding_size = transformation_embedding_layers[-1]
        
        self.transform_embed = MLP(input_size=input_size,hidden_sizes=transformation_embedding_layers,conditional=False)
        self.embedding_predictor = MLP(input_size=self.embed_dim,hidden_sizes=self.predictor_layers,conditional=conditioned,conditioning_size=self.transformation_embedding_size)
        self.chunk_level = chunk_level
        
        print(f'Probing space: {probing_space}',f'Range: {self.range}. {self.range.shape}',f'Embed dim: {self.embed_dim}',sep='\n')

        # this is the version that predicts the embedding from the transformation parameters
        
    def forward(self,clean_audio, transform_parameters , audio = None, head = None):
            
        # if transform parameters is a 1d tensor, unsqueeze it
        
        
        
        if isinstance(transform_parameters,dict):
            transform_params_dict = transform_parameters
            for k in transform_parameters.keys():
                transform_parameters[k] = transform_params_dict[k].unsqueeze(-1) if len(transform_params_dict[k].shape) == 1 else transform_params_dict[k]
            transform_parameters = torch.cat([v for k,v in transform_parameters.items()],dim=-1)
                
        else:
            transform_parameters = transform_parameters.unsqueeze(-1) if len(transform_parameters.shape) == 1 else transform_parameters
                
    
                
        assert transform_parameters.shape[-1] == self.input_size, f'Expected input size {self.input_size}, got {transform_parameters.shape[-1]}'
        
        audio = audio[...,self.range]
        clean_audio = clean_audio[...,self.range]
    
    
        parameter_embedding = self.transform_embed(transform_parameters)
        predicted_features = self.embedding_predictor(audio,parameter_embedding)
        
        return {
            'embeddings': clean_audio,
            'transformed_embeddings': audio,
            'predicted_embeddings': predicted_features,
            'gt': clean_audio,
            'preds': predicted_features
        }
    
    def get_losses(self,gt,preds):
        
        # cosine loss is distance, not similarity. Embeddings are of shape batch,embed_dim
        cosine_loss = 1 - torch.nn.functional.cosine_similarity(gt,preds).mean()
        mse_loss = torch.nn.functional.mse_loss(gt,preds)
        return {
            'cosine_loss': cosine_loss,
            'mse_loss': mse_loss,
            'loss': self.cosine_loss_weight * cosine_loss + self.mse_loss_weight * mse_loss
        }



class ParameterEquiMuLOOC(torch.nn.Module):
        
        def __init__(self,
                    predictor_layers=  [2048,2048],
                    chunk_level = True,
                    embed_dim = None,
                    conditioned = False,
                    probing_space = [0],
                    output_size = 1,
                    **kwargs):
            
            
            super(ParameterEquiMuLOOC,self).__init__()
            
            self.embed_dim = embed_dim
            self.range = torch.arange(0,self.embed_dim)
            self.range = torch.cat([self.range + i * self.embed_dim for i in probing_space])
        
            self.embed_dim = self.embed_dim * len(probing_space)
            
            self.output_size = output_size
                
            self.predictor_layers = predictor_layers + [output_size]
            
            self.transformation_predictor = MLP(input_size=self.embed_dim,hidden_sizes=self.predictor_layers,conditional=conditioned,conditioning_size=self.embed_dim)
            
            print(f'Probing space: {probing_space}',f'Range: {self.range}. {self.range.shape}',f'Embed dim: {self.embed_dim}',sep='\n')
        
        
            self.chunk_level = chunk_level
            
        def forward(self, audio,clean_audio, transform_parameters = None, head = None):
            
            features = audio
            clean_features = clean_audio
            
            features = features[...,self.range]
            clean_features = clean_features[...,self.range]
            
            
            if isinstance(transform_parameters,dict):
                transform_params_dict = transform_parameters
                for k in transform_params_dict.keys():
                    transform_params_dict[k] = transform_params_dict[k].unsqueeze(-1) if len(transform_params_dict[k].shape) == 1 else transform_params_dict[k]
                transform_parameters = torch.cat([v for k,v in transform_parameters.items()],dim=-1)
                
            else:
                transform_parameters = transform_parameters.unsqueeze(-1) if len(transform_parameters.shape) == 1 else transform_parameters
            
            assert transform_parameters.shape[-1] == self.output_size, f'Expected output size {self.output_size}, got {transform_parameters.shape[-1]}'
                
            predicted_parameters = self.transformation_predictor(features,conditioning = clean_features)
            
            predicted_parameters_dict = {
                k: v for k,v in zip(transform_params_dict.keys(),torch.split(predicted_parameters,1,dim=-1))
            }
            
            
            return {
                'embeddings': clean_features,
                'transformed_embeddings': features,
                'predicted_parameters': predicted_parameters_dict,
                'transform_parameters': transform_params_dict,
                'gt': transform_params_dict,
                'preds': predicted_parameters_dict
            }
            
        def get_losses(self,gt,preds):
            
            # cosine loss is distance, not similarity. Embeddings are of shape batch,embed_dim
            
            if isinstance(gt,dict):
                # make sure the order is kept here
                gt = torch.cat([v for k,v in gt.items()],dim=-1)
                preds = torch.cat([v for k,v in preds.items()],dim=-1)
            
            mse_loss = torch.nn.functional.mse_loss(gt,preds)
            
            return {
                'mse_loss': mse_loss,
                'loss': mse_loss
            }

class LightningEquiMuLOOC(LightningModule):
    
    def __init__(self, predict = None, *args, **kwargs):
        super().__init__()
        if predict == 'embeddings':
            self.mlp = EmbeddingEquiMuLOOC(**kwargs)
        else:
            self.mlp = ParameterEquiMuLOOC(**kwargs)
        print(f'Using mlp: {self.mlp}')
        
        self.predict = predict
        
        if self.predict == 'parameters':
            self.preds_agg = {}
            self.gt_agg = {}
        else:
            self.preds_agg = []
            self.gt_agg = []
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=5e-4, weight_decay=1e-4)
        return optimizer
    
    def logging(self,gt,preds,split):
        on_step = True if split == 'train' else False
        losses = self.mlp.get_losses(gt,preds)
        for k,v in losses.items():
            self.log(f'{split}_{k}',v,prog_bar=True,on_step=on_step,on_epoch=True)
        
        if self.predict == 'parameters':    
            for k,v in gt.items():
                try:
                    r2 = r2_score(v,preds[k])
                    self.log(f'{k}_{split}_r2',r2,prog_bar=True,on_step=on_step,on_epoch=True)
                except:
                    continue
    
        return losses['loss']
    def training_step(self, batch, batch_idx):
        out_ = self.mlp(**batch)
        
        loss = self.logging(out_['gt'],out_['preds'],'train')
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        
        out_ = self.mlp(**batch)
        loss = self.logging(out_['gt'],out_['preds'],'val')
        
        return loss
    
    def test_step(self, batch, batch_idx):
        out_ = self.mlp(**batch)
        
        if isinstance(out_['gt'],dict):
            for k,v in out_['gt'].items():
                if k not in self.preds_agg:
                    self.preds_agg[k] = []
                    self.gt_agg[k] = []
                self.preds_agg[k].append(out_['preds'][k].detach().cpu())
                self.gt_agg[k].append(out_['gt'][k].detach().cpu())
        else:
            self.preds_agg.append(out_['preds'].detach().cpu())
            self.gt_agg.append(out_['gt'].detach().cpu())
        
    def on_test_epoch_end(self):
        
        if self.predict == 'parameters':
            preds = {k: torch.cat(v) for k,v in self.preds_agg.items()}
            gt = {k: torch.cat(v) for k,v in self.gt_agg.items()}
        else:
            preds = torch.cat(self.preds_agg)
            gt = torch.cat(self.gt_agg)
        
        loss = self.logging(gt,preds,'test')
        
        if self.predict == 'parameters':
            self.preds_agg = {}
            self.gt_agg = {}
        else:
            self.preds_agg = []
            self.gt_agg = []
        
        # data = [[x, y] for (x, y) in zip(class_x_prediction_scores, class_y_prediction_scores)]
        # table = wandb.Table(data=data, columns = ["class_x", "class_y"])
        # wandb.log({"my_custom_id" : wandb.plot.scatter(table, "class_x", "class_y")})

        
        #wandb scatterplot of actual and predicted values
        # only if predicting parameters:
        if self.predict == 'parameters' and self.logger is not None:
            for k in gt.keys():
                data = [[x, y] for (x, y) in zip(gt[k],preds[k])]
                table = wandb.Table(data=data, columns = ["gt", "preds"])
                wandb.log({f"scatter_{k}" : wandb.plot.scatter(table, "gt", "preds")})
        