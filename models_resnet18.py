"""
Model 1: ResNet18 Transfer Learning Baseline Architecture
Author: Verina Fouad Farid Khalil
Role: Primary Architecture Author (Model A & Model B Regimes)

Compares two training paradigms on 741 component images:
- Model A: Fixed Feature Extraction (Frozen Backbone) + Adam Optimizer
- Model B: Partial Fine-Tuning (Layer4 Unfrozen) + SGD with Momentum (0.9)
"""

import torch
import torch.nn as nn
from torchvision import models

def build_resnet18_classifier(num_classes: int = 3, regime: str = 'fine_tune'):
    """
    Builds ResNet18 classifier configured for either Feature Extraction or Fine-Tuning.
    
    Args:
        num_classes: Number of target electronic component classes (e.g. Resistor, Capacitor, Diode)
        regime: 'feature_extract' (freeze all except FC) or 'fine_tune' (unfreeze layer4 + FC)
    """
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    
    if regime == 'feature_extract':
        # Freeze entire convolutional trunk
        for param in model.parameters():
            param.requires_grad = False
    elif regime == 'fine_tune':
        # Freeze early layers, unfreeze layer4 for domain adaptation
        for param in model.parameters():
            param.requires_grad = False
        for param in model.layer4.parameters():
            param.requires_grad = True

    # Custom classification head
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features, num_classes)
    )
    return model


def get_optimizer(model: nn.Module, regime: str = 'fine_tune', lr: float = 0.001):
    """Returns optimal optimizer per regime (Adam for extraction, SGD for fine-tuning)."""
    if regime == 'feature_extract':
        return torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    else:
        # SGD with momentum yields superior generalization without catastrophic forgetting
        return torch.optim.SGD(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=lr,
            momentum=0.9,
            weight_decay=1e-4
        )
