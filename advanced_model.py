import torch
import torch.nn as nn
import torchvision.models as models

class ResNetEmotion(nn.Module):
    def __init__(self, num_classes=7):
        """
        Initializes a pre-trained ResNet18 model and replaces the final fully connected layer 
        to output the number of emotion classes.

        Args:
            num_classes (int): Number of emotion classes (default is 7 for FER2013 dataset).
        """
        super(ResNetEmotion, self).__init__()
        # Load a pre-trained ResNet18 model
        self.model = models.resnet18(pretrained=True)

        # Replace the final fully connected layer with one that outputs 'num_classes'
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, num_classes)

    def forward(self, x):
        """
        Forward pass for the ResNet18 model.

        Args:
            x (Tensor): Input image tensor.

        Returns:
            Tensor: Model output logits.
        """
        return self.model(x)
