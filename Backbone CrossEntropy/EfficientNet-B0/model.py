import torch.nn as nn
import timm

class SimpleClassifierModel(nn.Module):
    def __init__(self, num_classes, backbone_name='efficientnet_b0', pretrained=True):
        super(SimpleClassifierModel, self).__init__()
        # timm automatically adds a classification head when num_classes > 0
        # pretrained=True to use ImageNet weights for fine-tuning the backbone for classification
        self.model = timm.create_model(backbone_name, pretrained=pretrained, num_classes=num_classes)

    def forward(self, image):
        # This will directly output logits for CrossEntropyLoss
        return self.model(image)

    # Note: No get_embedding function here as the primary goal of this model
    # is classification to select the best backbone based on raw performance.
    # The actual embedding generation for face recognition would be a later step
    # or the last layer of the backbone before the classification head.