import torch
import torch.nn as nn
import timm

class FaceNetModel(nn.Module):
    """
    A model for FaceNet that uses a backbone (like EfficientNet-B0)
    and projects the features to a fixed-size embedding.
    """
    def __init__(self, embedding_size=512, backbone_name='efficientnet_b0', pretrained=True):
        super(FaceNetModel, self).__init__()
        # 1. Load the backbone without its final classifier (num_classes=0)
        self.backbone = timm.create_model(backbone_name, pretrained=pretrained, num_classes=0)
        
        # 2. Get the number of output features from the backbone
        backbone_output_features = self.backbone.num_features # For EfficientNet-B0, this is 1280
        
        # 3. Create a linear layer to project the features to the desired embedding size
        self.embedding_layer = nn.Linear(backbone_output_features, embedding_size)

    def forward(self, image):
        """
        Takes an image and returns its embedding.
        """
        features = self.backbone(image)
        embedding = self.embedding_layer(features)
        return embedding

    def get_embedding(self, image):
        """A helper function to be explicit about getting embeddings."""
        return self.forward(image)