"""
ResNet-18 model architecture modified for single-channel (spectrogram) inputs and binary classification.
"""

import torch
import torch.nn as nn


class BasicBlock(nn.Module):
    """Basic residual block with two convolution layers and skip connection."""
    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out


class SpoofDetector(nn.Module):
    """
    Modified ResNet-18 network for binary spoof detection.
    Takes grayscale (single-channel) CQT spectrogram inputs.
    """
    def __init__(self):
        super(SpoofDetector, self).__init__()
        self.in_planes = 64

        # Input is a 1-channel grayscale spectrogram
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Standard ResNet structure (2 blocks per layer)
        self.layer1 = self._make_layer(BasicBlock, 64, 2, stride=1)
        self.layer2 = self._make_layer(BasicBlock, 128, 2, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 256, 2, stride=2)
        self.layer4 = self._make_layer(BasicBlock, 512, 2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * BasicBlock.expansion, 2)

        # Instance variable to keep trace of last conv activations for GradCAM
        self.last_conv_features = None

    def _make_layer(self, block: type[BasicBlock], planes: int, num_blocks: int, stride: int) -> nn.Sequential:
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Standard propagation
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.maxpool(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        
        # Save output of the final convolutional layer block (layer4) for GradCAM evaluation
        self.last_conv_features = self.layer4(out)
        
        out = self.avgpool(self.last_conv_features)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out

    def get_features(self) -> torch.Tensor:
        """
        Returns the multi-channel activations of the final convolutional block.
        Required for generating post-hoc heatmap visualizations via GradCAM.
        """
        return self.last_conv_features

    @classmethod
    def load_pretrained(cls, path: str) -> "SpoofDetector":
        """
        Loads the pre-trained weights from disk.
        
        Args:
            path: String path to the stored state-dictionary (.pt file).
            
        Returns:
            model: An initialized instance of SpoofDetector loaded with learned weights.
        """
        model = cls()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        state_dict = torch.load(path, map_location=device)
        
        # Accommodate dictionary saves vs static weights
        if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
            model.load_state_dict(state_dict["model_state_dict"])
        else:
            model.load_state_dict(state_dict)
            
        model.to(device)
        model.eval()
        return model
