import torch
import torch.nn as nn

class L1Loss(nn.Module):
    '''
    Compute L1 Loss.
    '''
    def __init__(self, loss_weight=1, data_type=['stereo', 'lidar'], **kwargs):
        super(L1Loss, self).__init__()
        self.loss_weight = loss_weight
        self.data_type = data_type
        self.eps = 1e-6
    
    def forward(self, prediction, target, mask=None, **kwargs):

        diff = torch.abs(prediction[mask] - target[mask])
        loss = torch.sum(diff) / (diff.numel() + self.eps)
        if torch.isnan(loss).item() | torch.isinf(loss).item():
            loss = 0 * torch.sum(prediction)
        
        return loss * self.loss_weight
    