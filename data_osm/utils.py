import numpy as np

import torch


def get_proj_mat(intrins, rots, trans):
    K = np.eye(4)
    K[:3, :3] = intrins
    R = np.eye(4)
    R[:3, :3] = rots.transpose(-1, -2)
    T = np.eye(4)
    T[:3, 3] = -trans
    RT = R @ T
    return K @ RT


def perspective(cam_coords, proj_mat):
    pix_coords = proj_mat @ cam_coords
    valid_idx = pix_coords[2, :] > 0
    pix_coords = pix_coords[:, valid_idx]
    pix_coords = pix_coords[:2, :] / (pix_coords[2, :] + 1e-7)
    pix_coords = pix_coords.transpose(1, 0)
    return pix_coords


def label_onehot_decoding(onehot):
    return torch.argmax(onehot, axis=0)


def label_onehot_encoding(label, num_classes=4):
    H, W = label.shape
    onehot = torch.zeros((num_classes, H, W))
    onehot.scatter_(0, label[None].long(), 1) # 将label中不同的值映射到不同层里，例如对于方向0~36，将37个类别，对应37个通道对应位置设置为1
    return onehot


def gen_dx_bx(xbound, ybound, zbound):
    dx = torch.Tensor([row[2] for row in [xbound, ybound, zbound]]) # 取xbound ybound zbound的第三维
    bx = torch.Tensor([row[0] + row[2] / 2.0 for row in [xbound, ybound, zbound]]) # 取xbound ybound zbound的第一维加上第三维/2
    nx = torch.LongTensor([(row[1] - row[0]) / row[2] for row in [xbound, ybound, zbound]])
    return dx, bx, nx
