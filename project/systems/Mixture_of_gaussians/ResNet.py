
import torch
from torch import nn, optim
import torch.nn.functional as F

class ResizeConv1d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, scale_factor, mode='nearest'):
        super().__init__()
        self.scale_factor = scale_factor
        self.mode = mode
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)
        x = self.conv(x)
        return x

class BasicBlockEnc(nn.Module):

    def __init__(self, in_planes, stride=1):
        super().__init__()

        planes = in_planes*stride

        self.conv1 = nn.Conv1d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(planes)

        if stride == 1:
            self.shortcut = nn.Sequential()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes)
            )

    def forward(self, x):
        out = F.leaky_relu(self.bn1(self.conv1(x)), 0.1)
        out = F.dropout(out)
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.leaky_relu(out, 0.1)
        return out

class BasicBlockDec(nn.Module):

    def __init__(self, in_planes, stride=1):
        super().__init__()

        planes = int(in_planes/stride)

        self.conv2 = nn.Conv1d(in_planes, in_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(in_planes)
        # self.bn1 could have been placed here, but that messes up the order of the layers when printing the class

        if stride == 1:
            self.conv1 = nn.Conv1d(in_planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn1 = nn.BatchNorm1d(planes)
            self.shortcut = nn.Sequential()
        else:
            self.conv1 = ResizeConv1d(in_planes, planes, kernel_size=3, scale_factor=stride)
            self.bn1 = nn.BatchNorm1d(planes)
            self.shortcut = nn.Sequential(
                ResizeConv1d(in_planes, planes, kernel_size=3, scale_factor=stride),
                nn.BatchNorm1d(planes)
            )

    def forward(self, x):
        out = F.leaky_relu(self.bn2(self.conv2(x)), 0.1)
        out = F.dropout(out)
        out = self.bn1(self.conv1(out))
        out += self.shortcut(x)
        out = F.leaky_relu(out, 0.1)
        return out

class ResNet18Enc(nn.Module):

    def __init__(self,**args):
        super().__init__()
        num_Blocks=[15,15,15,15]
        self.in_planes = 64
        self.latent_dim = args['latent_dim']
        self.conv1 = nn.Conv1d(8, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.layer1 = self._make_layer(BasicBlockEnc, 64, num_Blocks[0], stride=1)
        self.layer2 = self._make_layer(BasicBlockEnc, 128, num_Blocks[1], stride=2)
        self.layer3 = self._make_layer(BasicBlockEnc, 256, num_Blocks[2], stride=2)
        self.layer4 = self._make_layer(BasicBlockEnc, 512, num_Blocks[3], stride=2)
        self.linear = nn.Linear(512, args['encoder_output_dim'])

    def _make_layer(self, BasicBlockEnc, planes, num_Blocks, stride):
        strides = [stride] + [1]*(num_Blocks-1)
        layers = []
        for stride in strides:
            layers += [BasicBlockEnc(self.in_planes, stride)]
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.leaky_relu(self.bn1(self.conv1(x)), 0.1)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = F.adaptive_avg_pool1d(x, 1)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        #x = torch.tanh(x)
        return x

class ResNet18Dec(nn.Module):

    def __init__(self, **args):
        super().__init__()
        self.in_planes = 512
        num_Blocks=[4,4,4,4]
        nc=8
        self.nc = nc
        if args['data_dir'] == "/raw_data/umcu_median":
            self.input_len = 600

        else:
            self.input_len = 5000

        
        self.first_width = int(self.input_len / 8)

        #self.final_fc = nn.Linear(160 * 8, self.input_len * 8)
        self.linear = nn.Linear(args['latent_dim'], 512 *self.first_width)

        self.layer4 = self._make_layer(BasicBlockDec, 256, num_Blocks[3], stride=2)
        self.layer3 = self._make_layer(BasicBlockDec, 128, num_Blocks[2], stride=2)
        self.layer2 = self._make_layer(BasicBlockDec, 64, num_Blocks[1], stride=2)
        self.layer1 = self._make_layer(BasicBlockDec, 64, num_Blocks[0], stride=1)
        self.conv1 = ResizeConv1d(64, nc, kernel_size=3, scale_factor=1)

    def _make_layer(self, BasicBlockDec, planes, num_Blocks, stride):
        strides = [stride] + [1]*(num_Blocks-1)
        layers = []
        for stride in reversed(strides):
            layers += [BasicBlockDec(self.in_planes, stride)]
        self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, z):
        self.first_width = int(self.input_len / 8)

        #print(z.shape)
        x = self.linear(z)
        x = x.view(z.size(0), 512, self.first_width)
        #print(x.shape)
        #x = F.interpolate(x, scale_factor=20)
        x = self.layer4(x)
        #print(x.shape)
        x = self.layer3(x)
        #print(x.shape)
        x = self.layer2(x)
        #print(x.shape)
        x = self.layer1(x)
        #print(x.shape)
        x = (self.conv1(x))
        #x = torch.tanh(x)
        #print(x.shape)
        #x = torch.flatten(x, 1)
        #x = self.final_fc(x)
        x = x.view(x.size(0), self.nc, self.input_len)
        return x