"""PyTorch iResNet100 compatible with InsightFace arcface_torch backbone.pth.

Input: RGB float32 in [-1, 1], aligned to the ArcFace 112px five-point template.
No classifier is instantiated. Weights must be supplied explicitly by the user.
"""
import hashlib
import torch
from torch import nn
from torch.nn import functional as F


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


class IBasicBlock(nn.Module):
    def __init__(self, in_channels, channels, stride=1):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_channels, eps=1e-5)
        self.conv1 = nn.Conv2d(in_channels, channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels, eps=1e-5)
        self.prelu = nn.PReLU(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, stride, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(channels, eps=1e-5)
        self.downsample = None
        if stride != 1 or in_channels != channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, channels, 1, stride, bias=False),
                nn.BatchNorm2d(channels, eps=1e-5),
            )

    def forward(self, x):
        identity = x if self.downsample is None else self.downsample(x)
        x = self.conv1(self.bn1(x))
        x = self.conv2(self.prelu(self.bn2(x)))
        return self.bn3(x) + identity


class IResNet100(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(64, eps=1e-5)
        self.prelu = nn.PReLU(64)
        in_channels = 64
        for index, (channels, depth) in enumerate(zip((64, 128, 256, 512), (3, 13, 30, 3)), 1):
            blocks = [IBasicBlock(in_channels, channels, 2)]
            blocks.extend(IBasicBlock(channels, channels) for _ in range(depth - 1))
            setattr(self, f'layer{index}', nn.Sequential(*blocks))
            in_channels = channels
        self.bn2 = nn.BatchNorm2d(512, eps=1e-5)
        self.dropout = nn.Dropout(0)
        self.fc = nn.Linear(512 * 7 * 7, 512)
        self.features = nn.BatchNorm1d(512, eps=1e-5)

    def forward(self, x):
        x = self.prelu(self.bn1(self.conv1(x)))
        for layer in (self.layer1, self.layer2, self.layer3, self.layer4):
            x = layer(x)
        return self.features(self.fc(self.dropout(self.bn2(x).flatten(1))))


def load_arcface(path, device):
    if str(path).endswith('.onnx'):
        raise ValueError('Training requires arcface_torch iResNet100 backbone.pth, not ONNX.')
    state = torch.load(path, map_location='cpu', weights_only=True)
    if 'state_dict' in state:
        state = state['state_dict']
    state = {key.removeprefix('module.'): value for key, value in state.items()}
    model = IResNet100()
    model.load_state_dict(state, strict=True)
    return model.eval().requires_grad_(False).to(device=device, dtype=torch.float32)


def identity_embedding(model, faces):
    # Do not wrap in no_grad: prediction-side gradients must reach the input.
    with torch.autocast(device_type=faces.device.type, enabled=False):
        return F.normalize(model(faces.float()), dim=-1, eps=1e-8)
