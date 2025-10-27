import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import timm

class SphereFace(nn.Module):
    """
    Implementation of SphereFace (A-Softmax Loss)
    Reference: https://arxiv.org/abs/1704.08063
    """
    def __init__(self, in_features, out_features, m=4):
        super(SphereFace, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        
        # Lambda function for stable training
        self.lambda_min = 5.0
        self.lambda_max = 1500.0
        self.gamma = 0.0001
        self.iteration = 0

    def forward(self, embedding, label):
        # Normalize weights and embeddings
        cosine = F.linear(F.normalize(embedding), F.normalize(self.weight))

        # Lambda annealing schedule
        self.iteration += 1
        current_lambda = max(self.lambda_min, self.lambda_max / (1 + self.gamma * self.iteration))

        # Calculate phi(theta) = (-1)^k * cos(m*theta) - 2k
        # This is the core of SphereFace to make it trainable
        # We need to find k such that m*theta is in [k*pi, (k+1)*pi]
        cosine_angle = cosine.detach() # Don't backprop through this calculation
        theta = torch.acos(torch.clamp(cosine_angle, -1.0 + 1e-7, 1.0 - 1e-7))
        phi_theta = self.m * theta
        k = (phi_theta / math.pi).floor()
        phi = ((-1.0)**k) * torch.cos(phi_theta) - 2.0 * k
        
        # Apply the margin to the correct class
        one_hot = torch.zeros(cosine.size(), device=embedding.device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        
        # Final output calculation using the lambda annealing
        output = (one_hot * (phi + current_lambda * cosine)) + ((1.0 - one_hot) * (current_lambda + 1) * cosine)
        output *= -1.0 # This is part of the original implementation, equivalent to using NLLLoss
        
        return output

class FaceRecognitionModel(nn.Module):
    def __init__(self, num_classes, embedding_size=1280, backbone_name='efficientnet_b0'):
        super(FaceRecognitionModel, self).__init__()
        # Backbone (EfficientNet-B0) gives a 1280-dim feature vector
        self.backbone = timm.create_model(backbone_name, pretrained=False, num_classes=0)
        
        # We will use the native output size of the backbone as embedding size
        backbone_output_features = self.backbone.num_features
        
        # The SphereFace head
        self.sphereface_head = SphereFace(in_features=backbone_output_features, out_features=num_classes)
        
        # Store embedding size for test.py
        self.embedding_size = backbone_output_features

    def forward(self, image, label):
        embedding = self.backbone(image)
        output = self.sphereface_head(embedding, label)
        return output
    
    def get_embedding(self, image):
        return self.backbone(image)