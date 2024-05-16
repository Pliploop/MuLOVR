import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.distributed as dist


class GatherLayer(torch.autograd.Function):
    """Gather tensors from all process, supporting backward propagation."""

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        output = [torch.zeros_like(input) for _ in range(dist.get_world_size())]
        dist.all_gather(output, input)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        (input,) = ctx.saved_tensors
        grad_out = torch.zeros_like(input)
        grad_out[:] = grads[dist.get_rank()]
        return grad_out

class NTXent(nn.Module):
    
    def __init__(self, temperature=0.1, contrast_mode='all',
                 base_temperature=0.1, accumulate = None):
        super().__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature
        self.first_run = True
        self.sim_function = nn.CosineSimilarity(2)
        
        self.feature_cache = []
        self.positive_mask_cache = []
        self.negative_mask_cache = []
        self.accumulate = accumulate
        self.accumulate_counter = 0
        
    def get_similarities(self, features, temperature = None):
        if temperature is None:
            temperature = self.temperature  
        return self.sim_function(features.unsqueeze(1),features.unsqueeze(0))/temperature
        
    def forward(self,features, positive_mask, negative_mask):
        
        ## add zeros to negative and positive masks to prevent self-contrasting
        if dist.is_available() and dist.is_initialized(): ## this is not super optimized and doesnt deal with cases beyond bloc diagonal positives
            positive_mask, negative_mask, features = self.dist_gather(positive_mask, negative_mask, features)

        loss, original_cosim = self.compute_loss(features, positive_mask, negative_mask)
        

        if self.accumulate and self.accumulate_counter < self.accumulate:
            self.feature_cache.append(features)
            self.positive_mask_cache.append(positive_mask)
            self.negative_mask_cache.append(negative_mask)
            self.accumulate_counter += 1
            
            return {'loss': loss*0, 'similarity': original_cosim, 'positive_mask': positive_mask, 'negative_mask': negative_mask}
        
        if self.accumulate and self.accumulate_counter == self.accumulate:
            positive_mask, negative_mask, features = self.expand_purge_cache()
            loss, original_cosim = self.compute_loss(features, positive_mask, negative_mask)
            return {'loss': loss, 'similarity': original_cosim, 'positive_mask': positive_mask, 'negative_mask': negative_mask}
        
        if dist.is_available() and dist.is_initialized():
            return {'loss': loss, 'similarity': original_cosim, 'positive_mask': positive_mask, 'negative_mask': negative_mask}
        
        return loss
    
    
    def expand_purge_cache(self):
        
        features = torch.cat(self.feature_cache, dim = 0)
        new_positive_mask = torch.zeros(self.positive_mask_cache[0].shape[0]*len(self.positive_mask_cache), self.positive_mask_cache[0].shape[1]*len(self.positive_mask_cache), device = self.positive_mask_cache[0].device)
        new_negative_mask = torch.zeros(self.negative_mask_cache[0].shape[0]*len(self.negative_mask_cache), self.negative_mask_cache[0].shape[1]*len(self.negative_mask_cache), device = self.negative_mask_cache[0].device)
        
        for i in range(len(self.positive_mask_cache)):
            new_positive_mask[i*self.positive_mask_cache[0].shape[0]:(i+1)*self.positive_mask_cache[0].shape[0], i*self.positive_mask_cache[0].shape[1]:(i+1)*self.positive_mask_cache[0].shape[1]] = self.positive_mask_cache[i]
            new_negative_mask[i*self.negative_mask_cache[0].shape[0]:(i+1)*self.negative_mask_cache[0].shape[0], i*self.negative_mask_cache[0].shape[1]:(i+1)*self.negative_mask_cache[0].shape[1]] = self.negative_mask_cache[i]
            
        self.feature_cache = []
        self.positive_mask_cache = []
        self.negative_mask_cache = []
        self.accumulate_counter = 0
        
        return new_positive_mask, new_negative_mask, features
        
    
    def compute_loss(self, features, positive_mask, negative_mask):
        
        self_contrast = (~(torch.eye(positive_mask.shape[0], device = positive_mask.device).bool())).int()
        
        positive_mask = positive_mask * self_contrast
        # print(positive_mask)
        positive_sums = positive_mask.sum(1)
        positive_sums[positive_sums == 0] = 1
        negative_mask = negative_mask * self_contrast
        
    
        original_cosim = self.get_similarities(features=features)    
        
        original_cosim = torch.exp(original_cosim)   ## remove this when reverting
        
        pos = original_cosim
        neg = torch.sum( original_cosim * negative_mask, dim = 1, keepdim = True)
        
        log_prob = pos/neg
        
        log_prob = -torch.log(log_prob + 1e-6)
        log_prob = log_prob * positive_mask
        log_prob = log_prob.sum(1)
        log_prob = log_prob / positive_sums       
        
        loss = torch.mean(log_prob) 
        
        return loss, original_cosim
    
    def dist_gather(self,positive_mask, negative_mask, features):
        # Gather tensors across all processes
        # new positive mask is a big square matrix with positive masks along the diagonal
        new_positive_mask = torch.zeros(positive_mask.shape[0]*dist.get_world_size(), positive_mask.shape[1]*dist.get_world_size(), device = positive_mask.device)
        new_negative_mask = torch.zeros(negative_mask.shape[0]*dist.get_world_size(), negative_mask.shape[1]*dist.get_world_size(), device = negative_mask.device)
        
        features = torch.cat(GatherLayer.apply(features),dim=0)
        positive_mask_list = GatherLayer.apply(positive_mask)
        negative_mask_list = GatherLayer.apply(negative_mask)
        
        
        for i in range(dist.get_world_size()):
            new_positive_mask[i*positive_mask.shape[0]:(i+1)*positive_mask.shape[0], i*positive_mask.shape[1]:(i+1)*positive_mask.shape[1]] = positive_mask_list[i]
        positive_mask = new_positive_mask
        
        for i in range(dist.get_world_size()):
            new_negative_mask[i*negative_mask.shape[0]:(i+1)*negative_mask.shape[0], i*negative_mask.shape[1]:(i+1)*negative_mask.shape[1]] = negative_mask_list[i]
        negative_mask = new_negative_mask
    
        return positive_mask, negative_mask, features