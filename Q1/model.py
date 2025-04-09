import torch
import torch.nn as nn
import torchvision.models as models

class AircraftDetector(nn.Module):
    def __init__(self, num_classes=1, backbone_name='resnet34', pretrained_backbone=True):
        super(AircraftDetector, self).__init__()

        if num_classes != 1:
            raise ValueError("This model is designed for 'aircraft' only)")
        self.num_classes = num_classes
        self.num_box_coords = 4

        # Feature Extractor
        if backbone_name == 'resnet34':
            backbone = models.resnet34(pretrained=pretrained_backbone)
            # Output channels from the layer before avgpool
            self.backbone_out_channels = 512
            # Remove the final average pooling and fully connected layer
            self.backbone = nn.Sequential(*list(backbone.children())[:-2])
        else:
            raise NotImplementedError(f"{backbone_name} not supported")

        # Detection gheads + simple convolutional layers
        # channel size for the heads
        head_inter_channels = 256

        # Classification head
        # uses backbone features to predict class scores per location
        self.cls_head = nn.Sequential(
            nn.Conv2d(self.backbone_out_channels, head_inter_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            # Final layer predicts 1 score (aircraft vs background)
            nn.Conv2d(head_inter_channels, self.num_classes, kernel_size=1)
        )

        # Regression head
        # uses backbone features and predicts box coordinates per location
        self.reg_head = nn.Sequential(
            nn.Conv2d(self.backbone_out_channels, head_inter_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            # Final layer predicts 4 box coordinates per location
            nn.Conv2d(head_inter_channels, self.num_box_coords, kernel_size=1)
            # outputs raw coordinates/offsets
        )

        print(f"AircraftDetector started:")
        print(f"  Backbone: {backbone_name} (pretrained={pretrained_backbone})")
        print(f"  Backbone output channels: {self.backbone_out_channels}")
        print(f"  Detection Head intermediate channels: {head_inter_channels}")
        print(f"  Number of classes: {self.num_classes}")
        print(f"  Number of box coordinates: {self.num_box_coords}")


    def forward(self, x):
        # Get features from the backbone
        features = self.backbone(x)

        # predictions from detection heads
        cls_logits = self.cls_head(features)
        box_preds = self.reg_head(features)

        return cls_logits, box_preds

if __name__ == '__main__':
    # Create the model for aircraft class
    model = AircraftDetector(num_classes=1, backbone_name='resnet34', pretrained_backbone=True)

    input = torch.randn(2, 3, 640, 640)

    # model in evaluation mode if not training
    model.eval()
    with torch.no_grad():
        cls_output, box_output = model(input)

    print("\n--- Output Shapes ---")
    print("Classification Shape:", cls_output.shape)
    print("Bounding Box Predictions Shape:", box_output.shape)