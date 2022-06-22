# packages
import torch
import torch.nn as nn
import torch.nn.functional as F

# standardization layer
class StdLayer(nn.Module):

    def __init__(self, mu, sd):
        super(StdLayer, self).__init__()
        self.register_buffer('mu', torch.Tensor(mu).view(3, 1, 1))
        self.register_buffer('sd', torch.Tensor(sd).view(3, 1, 1))
    
    def forward(self, x):
        return (x - self.mu) / self.sd


""" Model definition for pre activation resnet18 """
class PreActBlock(nn.Module):
    '''Pre-activation version of the BasicBlock.'''
    expansion = 1

    def __init__(self, in_planes, planes, bn_ind=1, resid_conn=1, stride=1):
        super(PreActBlock, self).__init__()

        # convoluational layers
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)

        # batch normalizations
        if bn_ind:
            self.bn1 = nn.BatchNorm2d(in_planes)
            self.bn2 = nn.BatchNorm2d(planes)

        # residual connections
        if resid_conn and (stride != 1 or in_planes != self.expansion*planes):
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):

        # model with optional batch normalizations
        out = self.bn1(x) if hasattr(self, 'bn1') else x
        out = F.relu(out)
        out = self.conv1(out)
        out = self.bn2(out) if hasattr(self, 'bn2')
        out = self.conv2(F.relu(out))

        # residual connections
        shortcut = self.shortcut(x) if hasattr(self, 'shortcut') else 0
        out += shortcut

        return out


class PreActBottleneck(nn.Module):
    '''Pre-activation version of the original Bottleneck module.'''
    expansion = 4

    def __init__(self, in_planes, planes, bn_ind=1, resid_conn=1, stride=1):
        super(PreActBottleneck, self).__init__()

        # convolutional layers
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)

        # batch normalization
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.bn3 = nn.BatchNorm2d(planes)

        # residual connections
        if resid_conn and (stride != 1 or in_planes != self.expansion*planes):
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):

        # model with optional batch normalizations
        out = self.bn1(x) if hasattr(self, 'bn1') else x
        out = F.relu(out)
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else 0
        out = self.conv1(out)
        out = self.bn2(out) if hasattr(self, 'bn2')
        out = self.conv2(F.relu(out))
        out = self.bn3(out) if hasattr(self, 'bn3')
        out = self.conv3(F.relu(out))

        # optional residual connection
        out += shortcut

        return out


class PreActResNet(nn.Module):
    def __init__(self, block, num_blocks, bn_ind=1, resid_conn=1, num_classes=10):
        super(PreActResNet, self).__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], bn_ind, resid_conn, stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], bn_ind, resid_conn, stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], bn_ind, resid_conn, stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], bn_ind, resid_conn, stride=2)

        if bn_ind:
            self.bn = nn.BatchNorm2d(512 * block.expansion)

        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, bn_ind, resid_conn, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, bn_ind, resid_conn))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.bn(out) if hasattr(self, 'bn') else out
        out = F.relu(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def PreActResNet18(bn_ind, resid_conn):
    return PreActResNet(PreActBlock, [2,2,2,2], bn_ind=1, resid_conn=1)

