"""Loss functions for temporal action spotting."""
import torch
import torch.nn as nn
import torch.nn.functional as F

def get_class_weights(num_classes=17, bg_weight=0.01):
    """Class weights for CrossEntropyLoss to handle background dominance."""
    weights = torch.ones(num_classes + 1)
    weights[0] = bg_weight
    return weights

class FocalLoss(nn.Module):
    """
    Focal Loss for Extreme Class Imbalance.
    Dynamically scales the loss based on prediction confidence.
    Heavily penalizes the model for missing rare events (goals/fouls) 
    while ignoring easy background frames.
    """
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.alpha = alpha # Tensor of class weights [num_classes + 1]

    def forward(self, inputs, targets):
        # inputs: [Batch, num_classes+1, Time] (Logits)
        # targets: [Batch, Time] (Class indices)
        
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        
        # Calculate pt (the probability of the true class)
        pt = torch.exp(-ce_loss)
        
        # Apply the focal loss formula: (1 - pt)^gamma * log(pt)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss