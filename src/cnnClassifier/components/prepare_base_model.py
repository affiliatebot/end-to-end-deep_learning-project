import os
import torch
import torch.nn as nn
import torchvision.models as models
from pathlib import Path
from cnnClassifier.entity.config_entity import PrepareBaseModelConfig


class PrepareBaseModel:

    def __init__(self, config:PrepareBaseModelConfig):
        self.config = config
        


    def get_base_model(self):
        """Load VGG16 + freeze convolution layers + save base model"""
        

        # load pretrained VGG16
        # Convert YAML string → PyTorch Weight Enum
        weights_enum = getattr(models.VGG16_Weights, self.config.params_weights)
    
        model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)

        # freeze feature extractor layers
        for param in model.features.parameters():
            param.requires_grad = False

        # save base model (frozen)
        self.save_model(model,self.config.base_model_path)
        

    def update_base_model(self):
        """Load base model + attach custom classifier + save updated model"""

        # load base model from file (important!)
        model = torch.load(self.config.base_model_path)

        # replace classifier head
        model.classifier = nn.Sequential(
            nn.Linear(25088, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, self.config.params_classes)
        )

        # ensure classifier layers are trainable
        for param in model.classifier.parameters():
            param.requires_grad = True

        # save updated model (ready for training)
        self.save_model(model,self.config.updated_base_model_path)
        print(model)
        


    def save_model(self, model, path):
        """Save full PyTorch model"""
        torch.save(model, path)
