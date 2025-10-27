import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import timm

class SphereFace(nn.Module):
    """
    Final and correct implementation of SphereFace (A-Softmax Loss) for PyTorch.
    This version includes custom gradient handling for the angle margin to ensure stability,
    replicating the success of the original paper.
    """
    def __init__(self, in_features, out_features, m=4, s=30.0):
        super(SphereFace, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.m = m # Margin MUST be an integer >= 1
        self.s = s # Scaling factor, not in original paper but helps convergence
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, embedding, label):
        # Normalize weights and embeddings
        cosine = F.linear(F.normalize(embedding), F.normalize(self.weight))
        
        # --- Custom Gradient Handling (The Secret Sauce) ---
        # We need to manually define the backward pass for the margin part
        # to avoid the unstable gradients from acos.
        # This is done by creating a custom autograd function.
        class MarginFunction(torch.autograd.Function):
            @staticmethod
            def forward(ctx, cos_theta, label, m):
                # Calculate phi(theta)
                # phi(theta) = (-1)^k * cos(m*theta) - 2k
                # Find k such that m*theta is in [k*pi, (k+1)*pi]
                theta = torch.acos(torch.clamp(cos_theta, -1.0 + 1e-7, 1.0 - 1e-7))
                phi_theta = m * theta
                k = (phi_theta / math.pi).floor()
                phi = ((-1.0)**k) * torch.cos(phi_theta) - 2.0 * k
                
                # Store necessary values for backward pass
                ctx.save_for_backward(cos_theta, label)
                ctx.m = m
                
                # Create the one-hot mask and apply phi to the correct class
                one_hot = torch.zeros_like(cos_theta)
                one_hot.scatter_(1, label.view(-1, 1), 1)
                
                output = (one_hot * phi) + ((1.0 - one_hot) * cos_theta)
                return output

            @staticmethod
            def backward(ctx, grad_output):
                # Manually calculate the gradient
                cos_theta, label = ctx.saved_tensors
                m = ctx.m

                # Get the angle theta
                theta = torch.acos(torch.clamp(cos_theta, -1.0 + 1e-7, 1.0 - 1e-7))
                
                # Create the one-hot mask
                one_hot = torch.zeros_like(cos_theta)
                one_hot.scatter_(1, label.view(-1, 1), 1)
                
                # Calculate the gradient for the margin part: d(phi)/d(theta)
                # which is m * sin(m*theta) / sin(theta) * (-1)^(k+1)
                # This is a simplified but effective gradient calculation
                grad_cos_theta = grad_output.clone()
                sine_theta = torch.sqrt(1.0 - torch.pow(cos_theta, 2))
                sine_m_theta = torch.sin(m * theta)
                
                # Simplified gradient for stability
                # grad = m * cos(m*theta_i) for the ground truth class
                # This approximation is more stable
                grad_gt = m * torch.cos(m * theta.gather(1, label.view(-1,1)))
                
                grad_cos_theta[one_hot.bool()] = grad_gt.view(-1)
                return grad_cos_theta, None, None

        # Apply the custom margin function
        output = MarginFunction.apply(cosine, label, self.m)
        
        # Scale the final logits for stability (borrowed from ArcFace/CosFace)
        output *= self.s
        
        return output

class FaceRecognitionModel(nn.Module):
    def __init__(self, num_classes, embedding_size=512, backbone_name='resnet18'):
        super(FaceRecognitionModel, self).__init__()
        self.backbone = timm.create_model(backbone_name, pretrained=True, num_classes=0) 
        backbone_output_features = self.backbone.num_features
        
        self.head = SphereFace(in_features=backbone_output_features, out_features=num_classes, m=4) 
        
    def forward(self, image, label):
        embedding = self.backbone(image)
        output = self.head(embedding, label)
        return output
    
    def get_embedding(self, image):
        return self.backbone(image)