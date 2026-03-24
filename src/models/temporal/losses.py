"""Loss functions for temporal action spotting."""
import torch


def get_class_weights(num_classes=17, bg_weight=0.05):
    """Class weights for CrossEntropyLoss to handle background dominance.

    Returns a tensor of shape (num_classes+1,) where index 0 is the
    background weight and all event classes get weight 1.0.
    """
    weights = torch.ones(num_classes + 1)
    weights[0] = bg_weight
    return weights
