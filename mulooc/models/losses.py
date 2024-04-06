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
                 base_temperature=0.1):
        super().__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature
        self.first_run = True
        self.sim_function = nn.CosineSimilarity(2)
        
    def get_similarities(self, features, temperature = None):
        if temperature is None:
            temperature = self.temperature  
        return self.sim_function(features.unsqueeze(1),features.unsqueeze(0))/temperature
        
    def forward(self,features, positive_mask, negative_mask):
        
        ## add zeros to negative and positive masks to prevent self-contrasting
        if dist.is_available() and dist.is_initialized(): ## this is not super optimized and doesnt deal with cases beyond bloc diagonal positives
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

        self_contrast = (~(torch.eye(positive_mask.shape[0], device = positive_mask.device).bool())).int()
        
        positive_mask = positive_mask * self_contrast
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
        
        if dist.is_available() and dist.is_initialized():
            return {'loss': loss, 'similarity': original_cosim, 'positive_mask': positive_mask, 'negative_mask': negative_mask}
        
        return loss
    