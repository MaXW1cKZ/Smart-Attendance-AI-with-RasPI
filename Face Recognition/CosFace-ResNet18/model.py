import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import timm

class CosFace(nn.Module):
    """
    Implementation of CosFace (Large Margin Cosine Loss)
    Reference: https://arxiv.org/abs/1801.09414
    This is a stable and easy-to-implement loss function.
    """
    def __init__(self, in_features, out_features, s=30.0, m=0.35):
        super(CosFace, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s # Scaling factor
        self.m = m # Additive Cosine Margin
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, embedding, label):
        # Normalize weights and embeddings to compute the cosine of the angle
        cosine = F.linear(F.normalize(embedding), F.normalize(self.weight))
        
        # Create a one-hot mask for the ground truth class
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1), 1.0)
        
        # --- This is the core of CosFace: phi = cos(theta) - m ---
        # We subtract the margin 'm' ONLY from the cosine of the correct class
        phi = cosine - one_hot * self.m
        
        # Scale the final logits. This is the output to be passed to CrossEntropyLoss.
        output = phi * self.s
        
        return output

class FaceRecognitionModel(nn.Module):
    def __init__(self, num_classes, embedding_size=512, backbone_name='resnet18'):
        super(FaceRecognitionModel, self).__init__()
        self.backbone = timm.create_model(backbone_name, pretrained=True, num_classes=0) 
        backbone_output_features = self.backbone.num_features
        
        # --- Using the CosFace Head ---
        self.head = CosFace(in_features=backbone_output_features, out_features=num_classes, s=30.0, m=0.35)
        
    def forward(self, image, label):
        embedding = self.backbone(image)
        output = self.head(embedding, label)
        return output
    
    def get_embedding(self, image):
        return self.backbone(image)