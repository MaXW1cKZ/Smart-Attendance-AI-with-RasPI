import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import timm

class ArcFace(nn.Module):
    """ Implementation of ArcFace loss """
    def __init__(self, in_features, out_features, s=30.0, m=0.50):
        super(ArcFace, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, embedding, label):
        # cos(theta) & phi(theta)
        cosine = F.linear(F.normalize(embedding), F.normalize(self.weight))
        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        # convert label to one-hot
        one_hot = torch.zeros(cosine.size(), device=embedding.device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s
        return output

class FaceRecognitionModel(nn.Module):
    def __init__(self, num_classes, embedding_size=512, backbone_name='resnet18'):
        super(FaceRecognitionModel, self).__init__()
        self.backbone = timm.create_model(backbone_name, pretrained=True, num_classes=0) # num_classes=0 to get features
        self.arcface_head = ArcFace(in_features=self.backbone.num_features, out_features=num_classes)

    def forward(self, image, label):
        embedding = self.backbone(image)
        output = self.arcface_head(embedding, label)
        return output
    
    def get_embedding(self, image):
        return self.backbone(image)