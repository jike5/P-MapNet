import torch
import torch.nn as nn

from efficientnet_pytorch import EfficientNet
from torchvision.models.resnet import resnet18,resnet50


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super().__init__()

        self.up = nn.Upsample(scale_factor=scale_factor, mode='bilinear',
                              align_corners=True)

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x1 = torch.cat([x2, x1], dim=1)
        return self.conv(x1)


class CamEncode(nn.Module):
    def __init__(self, C):
        super(CamEncode, self).__init__()
        self.C = C

        self.trunk = EfficientNet.from_pretrained("efficientnet-b0")
        self.up1 = Up(320+112, self.C)

    def get_eff_depth(self, x):
        # adapted from https://github.com/lukemelas/EfficientNet-PyTorch/blob/master/efficientnet_pytorch/model.py#L231
        endpoints = dict()

        # Stem
        x = self.trunk._swish(self.trunk._bn0(self.trunk._conv_stem(x)))
        prev_x = x

        # Blocks
        for idx, block in enumerate(self.trunk._blocks):
            drop_connect_rate = self.trunk._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self.trunk._blocks)  # scale drop connect_rate
            x = block(x, drop_connect_rate=drop_connect_rate)
            if prev_x.size(2) > x.size(2):
                endpoints['reduction_{}'.format(len(endpoints)+1)] = prev_x
            prev_x = x

        # Head
        endpoints['reduction_{}'.format(len(endpoints)+1)] = x
        x = self.up1(endpoints['reduction_5'], endpoints['reduction_4'])
        return x

    def forward(self, x):
        return self.get_eff_depth(x)
        


# class BDEncoder(nn.Module):
#     def __init__(self, inC):
#         super(BDEncoder, self).__init__()

#         # self.bevencode = BevEncode(inC=192, outC=4)
#         self.bevencode = BevEncode(inC=128, outC=4)

#         self.convbd = nn.Sequential(
#         nn.Conv2d(3, out_channels=32, kernel_size=3, stride=1, padding=1),
#         nn.ReLU(),
#         nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
#         nn.BatchNorm2d(64),
#         nn.ReLU(inplace=True),
#         nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
#         nn.BatchNorm2d(128),
#         nn.ReLU(inplace=True))

#         # self.conv_osm = nn.Sequential(
#         # nn.Conv2d(1, out_channels=32, kernel_size=3, stride=1, padding=1),
#         # nn.ReLU(),
#         # nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
#         # nn.BatchNorm2d(64),
#         # nn.ReLU(inplace=True))

#     def forward(self, x, x2):
#         masked_map = self.convbd(x)
#         # osm = self.conv_osm(x2)
#         # concat = torch.cat([masked_map, osm], dim=1)
#         # return self.bev(concat)
#         return self.bevencode(masked_map)      

class maeDecode(nn.Module):
    def __init__(self, inC, outC, instance_seg=True, embedded_dim=16, direction_pred=True, direction_dim=37):
        super(maeDecode, self).__init__()
        trunk = resnet50(pretrained=False, zero_init_residual=True)
        self.conv1 = nn.Conv2d(inC, 64, kernel_size=7, stride=2, padding=3, bias=False)

        self.bn1 = trunk.bn1
        self.relu = trunk.relu

        self.layer1 = trunk.layer1
        self.layer2 = trunk.layer2
        self.layer3 = trunk.layer3
        #     self.res50.conv1(),
        #     self.res50.bn1(),
        #     self.res50.relu(),
        #     self.res50.maxpool(),
        #     self.res50.layer1(),
        #     self.res50.layer2(),
        #     self.res50.layer3(),
        #     self.res50.layer4(),
        #     self.res50.avgpool()
        self.up1 = Up(1024 + 256, 256, scale_factor=4)
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear',
                        align_corners=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, outC, kernel_size=1, padding=0),
        )

        self.instance_seg = instance_seg
        if instance_seg:
            self.up1_embedded = Up(1024 + 256, 256, scale_factor=4)
            self.up2_embedded = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear',
                            align_corners=True),
                nn.Conv2d(256, 128, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, embedded_dim, kernel_size=1, padding=0),
            )

        self.direction_pred = direction_pred
        if direction_pred:
            # self.up1_direction = Up(64 + 256, 256, scale_factor=4)
            self.up1_direction = Up(1024 + 256, 256, scale_factor=4)
            self.up2_direction = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear',
                            align_corners=True),
                nn.Conv2d(256, 128, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, direction_dim, kernel_size=1, padding=0),
            )

    def forward(self, x): # x: torch.Size([bs, 128, 200, 400])
        x = self.conv1(x)  # x: torch.Size([bs, 64, 100, 200])
        x = self.bn1(x)
        x = self.relu(x)

        x1 = self.layer1(x) # x1: torch.Size([bs, 256, 100, 200])
        x = self.layer2(x1) # x: torch.Size([bs, 512, 100, 200])
        x2 = self.layer3(x) # x2: torch.Size([bs, 1024, 25, 50])

        x = self.up1(x2, x1) # x: torch.Size([bs, 256, 100, 200])
        x = self.up2(x) # x: torch.Size([bs, 4, 200, 400])

        if self.instance_seg:
            x_embedded = self.up1_embedded(x2, x1)
            x_embedded = self.up2_embedded(x_embedded)
        else:
            x_embedded = None

        if self.direction_pred:
            x_direction = self.up1_embedded(x2, x1)
            x_direction = self.up2_direction(x_direction)
        else:
            x_direction = None

        return x, x_embedded, x_direction

class BDEncoder(nn.Module):
    def __init__(self, inC):
        super(BDEncoder, self).__init__()

        self.maeDecode = maeDecode(inC=128, outC=4)

        self.convbd = nn.Sequential(
        nn.Conv2d(inC, out_channels=32, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(64),
        nn.ReLU(inplace=True),
        nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(128),
        nn.ReLU(inplace=True))

        # self.conv_osm = nn.Sequential(
        # nn.Conv2d(1, out_channels=32, kernel_size=3, stride=1, padding=1),
        # nn.ReLU(),
        # nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
        # nn.BatchNorm2d(64),
        # nn.ReLU(inplace=True))

    def forward(self, x, x2):
        masked_map = self.convbd(x)
        # osm = self.conv_osm(x2)
        # concat = torch.cat([masked_map, osm], dim=1)
        
        return self.maeDecode(masked_map)

# class BDEncoder(nn.Module):
#     def __init__(self, inC):
#         super(BDEncoder, self).__init__()

#         self.bevencode = BevEncode(inC=192, outC=4)
#         # self.bevencode = BevEncode(inC=128, outC=4)

#         self.convbd = nn.Sequential(
#         nn.Conv2d(3, out_channels=32, kernel_size=3, stride=1, padding=1),
#         nn.ReLU(),
#         nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
#         nn.BatchNorm2d(64),
#         nn.ReLU(inplace=True),
#         nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
#         nn.BatchNorm2d(128),
#         nn.ReLU(inplace=True))

#         self.conv_osm = nn.Sequential(
#         nn.Conv2d(1, out_channels=32, kernel_size=3, stride=1, padding=1),
#         nn.ReLU(),
#         nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
#         nn.BatchNorm2d(64),
#         nn.ReLU(inplace=True))

#     def forward(self, x, x2):
#         masked_map = self.convbd(x)
#         osm = self.conv_osm(x2)
#         concat = torch.cat([masked_map, osm], dim=1)
       
#         return self.bevencode(concat)


# class BDEncoder(nn.Module):
#     def __init__(self, inC):
#         super(BDEncoder, self).__init__()
#         self.convbd = nn.Sequential(
#         nn.Conv2d(inC, out_channels=32, kernel_size=3, stride=1, padding=1),
#         nn.ReLU(),
#         nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
#         nn.BatchNorm2d(64),
#         nn.ReLU(inplace=True),
#         nn.Conv2d(64, 192, kernel_size=3, padding=1, bias=False),
#         nn.BatchNorm2d(192),
#         nn.ReLU(inplace=True))

#         self.convosm = nn.Sequential(
#         nn.Conv2d(inC, out_channels=32, kernel_size=3, stride=1, padding=1),
#         nn.ReLU(),
#         nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
#         nn.BatchNorm2d(64),
#         nn.ReLU(inplace=True))
        
#     def forward(self, x):
#         out = self.convosm(x)
#         return out

class BevEncode(nn.Module):
    def __init__(self, inC, outC, instance_seg=True, embedded_dim=16, direction_pred=True, direction_dim=37):
        super(BevEncode, self).__init__()
        trunk = resnet18(pretrained=False, zero_init_residual=True)
        self.conv1 = nn.Conv2d(inC, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = trunk.bn1
        self.relu = trunk.relu

        self.layer1 = trunk.layer1
        self.layer2 = trunk.layer2
        self.layer3 = trunk.layer3

        self.up1 = Up(64 + 256, 256, scale_factor=4)
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear',
                        align_corners=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, outC, kernel_size=1, padding=0),
        )

        self.instance_seg = instance_seg
        if instance_seg:
            self.up1_embedded = Up(64 + 256, 256, scale_factor=4)
            self.up2_embedded = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear',
                            align_corners=True),
                nn.Conv2d(256, 128, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, embedded_dim, kernel_size=1, padding=0),
            )

        self.direction_pred = direction_pred
        if direction_pred:
            self.up1_direction = Up(64 + 256, 256, scale_factor=4)
            self.up2_direction = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear',
                            align_corners=True),
                nn.Conv2d(256, 128, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, direction_dim, kernel_size=1, padding=0),
            )

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x1 = self.layer1(x)
        x = self.layer2(x1)
        x2 = self.layer3(x)

        x = self.up1(x2, x1)
        x = self.up2(x)

        if self.instance_seg:
            x_embedded = self.up1_embedded(x2, x1)
            x_embedded = self.up2_embedded(x_embedded)
        else:
            x_embedded = None

        if self.direction_pred:
            x_direction = self.up1_direction(x2, x1)
            x_direction = self.up2_direction(x_direction)
        else:
            x_direction = None

        return x, x_embedded, x_direction


class BevEncode_bd(nn.Module):
    def __init__(self, inC, outC, instance_seg=True, embedded_dim=16, direction_pred=True, direction_dim=37):
        super(BevEncode_bd, self).__init__()
        trunk = resnet18(pretrained=False, zero_init_residual=True)
        self.conv1 = nn.Conv2d(inC, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = trunk.bn1
        self.relu = trunk.relu

        self.layer1 = trunk.layer1
        self.layer2 = trunk.layer2
        self.layer3 = trunk.layer3

        self.up1 = Up(64 + 256, 256, scale_factor=4)
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear',
                        align_corners=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, outC, kernel_size=1, padding=0),
        )


    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x1 = self.layer1(x)
        x = self.layer2(x1)
        x2 = self.layer3(x)

        x = self.up1(x2, x1)
        x = self.up2(x)

        return x

