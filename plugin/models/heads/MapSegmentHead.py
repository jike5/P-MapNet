import torch
import torch.nn as nn

from efficientnet_pytorch import EfficientNet
from torchvision.models.resnet import resnet18,resnet50
from mmdet.models import HEADS
from mmdet.models import build_loss

def onehot_encoding(logits, dim=1):
    max_idx = torch.argmax(logits, dim, keepdim=True)
    one_hot = logits.new_full(logits.shape, 0)
    one_hot.scatter_(dim, max_idx, 1)
    return one_hot

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


@HEADS.register_module(force=True)
class MapSegmentHead(nn.Module):
    def __init__(
            self,
            embed_dims,
            num_calsses,
            instance_seg=True, 
            embedded_dim=16, 
            direction_pred=True, 
            direction_dim=37,
            loss_seg=dict(),
            loss_ins=dict(),
            loss_dir=dict()):
        super(MapSegmentHead, self).__init__()
        trunk = resnet18(pretrained=False, zero_init_residual=True)
        self.conv1 = nn.Conv2d(embed_dims, 64, kernel_size=7, stride=2, padding=3, bias=False)
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
            nn.Conv2d(128, num_calsses, kernel_size=1, padding=0),
        )
        self.loss_seg = build_loss(loss_seg)

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
            self.loss_ins = build_loss(loss_ins)

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
            self.loss_dir = build_loss(loss_dir)
        

    def forward_train(self, bev_features, img_metas, gts):
        outputs = []
        pred_dict={}
        # forward
        x = self.conv1(bev_features)
        x = self.bn1(x)
        x = self.relu(x)

        x1 = self.layer1(x)
        x = self.layer2(x1)
        x2 = self.layer3(x)

        x = self.up1(x2, x1)
        x_semantic = self.up2(x)
        pred_dict['semantic'] = x_semantic
        if self.instance_seg:
            x_embedded = self.up1_embedded(x2, x1)
            x_embedded = self.up2_embedded(x_embedded)
            pred_dict['instance'] = x_embedded
        else:
            x_embedded = None

        if self.direction_pred:
            x_direction = self.up1_direction(x2, x1)
            x_direction = self.up2_direction(x_direction)
            pred_dict['direction'] = x_direction
        else:
            x_direction = None
        outputs.append(pred_dict)
        loss_dict = self.loss(gts, outputs)
        return outputs, loss_dict
    
    def forward_test(self, bev_features, img_metas):
        outputs = []
        pred_dict={}
        # forward
        x = self.conv1(bev_features)
        x = self.bn1(x)
        x = self.relu(x)

        x1 = self.layer1(x)
        x = self.layer2(x1)
        x2 = self.layer3(x)

        x = self.up1(x2, x1)
        x_semantic = self.up2(x)
        pred_dict['semantic'] = x_semantic
        if self.instance_seg:
            x_embedded = self.up1_embedded(x2, x1)
            x_embedded = self.up2_embedded(x_embedded)
            pred_dict['instance'] = x_embedded
        else:
            x_embedded = None

        if self.direction_pred:
            x_direction = self.up1_direction(x2, x1)
            x_direction = self.up2_direction(x_direction)
            pred_dict['direction'] = x_direction
        else:
            x_direction = None
        outputs.append(pred_dict)
        return outputs
    
    def loss(self,
             gts,
             preds):
        loss_dict = {}
        seg_loss = self.loss_seg(preds[0]['semantic'].permute(0,2,3,1), 
                                 gts['semantic'].permute(0,2,3,1))
        loss_dict['seg_loss'] = seg_loss
        if self.instance_seg:
            ins_loss = self.loss_ins(preds[0]['instance'], gts['instance'])
            loss_dict.update(ins_loss)
        if self.direction_pred:
            dir_loss = self.loss_dir(preds[0]['direction'], gts['direction'])
            loss_dict['dir_loss'] = dir_loss
        return loss_dict
        
    def post_process(self, 
                     preds_dict, 
                     tokens,
                     return_vectoried=False, 
                     thr=0.0,):
        pred_seg = preds_dict['semantic']
        semantic_mask = onehot_encoding(pred_seg)
        # pred_ins = preds_dict['instance']
        # pred_dir = preds_dict['direction']
        bs = len(pred_seg)
        results = []
        for i in range(bs):
            single_mask = semantic_mask[i, 1:]
            single_result = {
                'semantic_mask': single_mask.detach().cpu(),
                'token': tokens[i]
            }
            results.append(single_result)
        
        return results
    
    def forward(self, *args, return_loss=True, **kwargs):
        if return_loss:
            return self.forward_train(*args, **kwargs)
        else:
            return self.forward_test(*args, **kwargs)
        