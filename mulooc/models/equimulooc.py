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
                 **kwargs):
        
        
        super(EmbeddingEquiMuLOOC,self).__init__()
        
        self.embed_dim = embed_dim
            
        self.predictor_layers = predictor_layers + [self.embed_dim]
        self.transformation_embedding_size = transformation_embedding_layers[-1]
        
        self.transform_embed = MLP(input_size=1,hidden_sizes=transformation_embedding_layers,conditional=False)
        self.embedding_predictor = MLP(input_size=self.embed_dim,hidden_sizes=self.predictor_layers,conditional=conditioned,conditioning_size=self.transformation_embedding_size)
        self.chunk_level = chunk_level
        
        # loss function is a linear combination of cosine distance and MSE
        self.cosine_loss_weight = 0.5
        self.mse_loss_weight = 0.5
        
    
        # this is the version that predicts the embedding from the transformation parameters
        
    def forward(self,clean_audio, transform_parameters , audio = None, head = None):
            
        # if transform parameters is a 1d tensor, unsqueeze it
        transform_parameters = transform_parameters.unsqueeze(-1) if len(transform_parameters.shape) == 1 else transform_parameters
        
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
                    **kwargs):
            
            
            super(ParameterEquiMuLOOC,self).__init__()
            
            self.embed_dim = embed_dim
                
            self.predictor_layers = predictor_layers + [1]
            
            self.transformation_predictor = MLP(input_size=self.embed_dim,hidden_sizes=self.predictor_layers,conditional=conditioned,conditioning_size=self.embed_dim)
            self.chunk_level = chunk_level
            
        def forward(self, audio,clean_audio, transform_parameters = None, head = None):
            features = audio
            clean_features = clean_audio
            
            transform_parameters = transform_parameters.unsqueeze(-1) if len(transform_parameters.shape) == 1 else transform_parameters
                
            predicted_parameters = self.transformation_predictor(features,conditioning = clean_features)
            
            return {
                'embeddings': clean_features,
                'transformed_embeddings': features,
                'predicted_parameters': predicted_parameters,
                'transform_parameters': transform_parameters,
                'gt': transform_parameters,
                'preds': predicted_parameters
            }
            
        def get_losses(self,gt,preds):
            
            # cosine loss is distance, not similarity. Embeddings are of shape batch,embed_dim
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
        r2 = r2_score(preds,gt)
        if preds.shape[-1] == 1:
            self.log(f'{split}_r2',r2,prog_bar=True,on_step=on_step,on_epoch=True)
    
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
        
        self.preds_agg.append(out_['preds'].detach().cpu())
        self.gt_agg.append(out_['gt'].detach().cpu())
        
    def on_test_epoch_end(self):
        preds = torch.cat(self.preds_agg)
        gt = torch.cat(self.gt_agg)
        
        loss = self.logging(gt,preds,'test')
        
        self.preds_agg = []
        self.gt_agg = []
        
        # data = [[x, y] for (x, y) in zip(class_x_prediction_scores, class_y_prediction_scores)]
        # table = wandb.Table(data=data, columns = ["class_x", "class_y"])
        # wandb.log({"my_custom_id" : wandb.plot.scatter(table, "class_x", "class_y")})

        
        #wandb scatterplot of actual and predicted values
        # only if predicting parameters:
        if gt.shape[-1] == 1:
            data = [[x, y] for (x, y) in zip(gt,preds)]
            table = wandb.Table(data=data, columns = ["gt", "preds"])
            wandb.log({"scatter" : wandb.plot.scatter(table, "gt", "preds")})
        