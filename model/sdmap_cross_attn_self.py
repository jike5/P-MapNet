# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR Transformer class.

Copy-paste from torch.nn.Transformer with modifications:
    * positional encodings are passed in MHattention
    * extra LN at the end of encoder is removed
    * decoder returns a stack of activations from all decoding layers
"""
import copy
from typing import Optional, List

import torch
import torch.nn.functional as F
from torch import nn, Tensor


class SDMapCrossAttn(nn.Module):

    def __init__(self, d_model=256, nhead=8, num_encoder_layers=1,
                 num_decoder_layers=1, dim_feedforward=192, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False):
    # def __init__(self, d_model=256, nhead=4, num_encoder_layers=1,
    #              num_decoder_layers=1, dim_feedforward=192, dropout=0.1,
    #              activation="relu", normalize_before=False,
    #              return_intermediate_dec=False):
    # def __init__(self, d_model=256, nhead=4, num_encoder_layers=1,
    #              num_decoder_layers=1, dim_feedforward=128, dropout=0.3,
    #              activation="relu", normalize_before=False,
    #              return_intermediate_dec=False):
        super().__init__()

        self.return_intermediate = return_intermediate_dec
        self.norm = nn.LayerNorm(d_model)



        decoder_layer = SDMapCrossAttnLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation)
        
        self.layers = _get_clones(decoder_layer, num_decoder_layers)
        self.num_layers = num_decoder_layers

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)


    def forward(self, bev, sdmap,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        
        # print(bev.shape, sdmap.shape)
        assert bev.shape == sdmap.shape
        bs, c, h, w = bev.shape
        bev = bev.flatten(2).permute(2, 0, 1)
        sdmap = sdmap.flatten(2).permute(2, 0, 1)
        pos = pos.flatten(2).permute(2, 0, 1)

        output = bev

        intermediate = []
        
        for layer in self.layers:
            output = layer(output, sdmap, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos)
            if self.return_intermediate:
                intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        bew_feat = output.view(h,w,bs,c).permute(2,3,0,1)

        return bew_feat.unsqueeze(0)


class SDMapCrossAttnLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=192, dropout=0.1,
                 activation="relu"):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, bev, sdmap,
                     bev_mask: Optional[Tensor] = None,
                     sdmap_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     sdmap_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):

        q = k = self.with_pos_embed(bev, pos)
        bev2 = self.self_attn(q, k, value=bev, attn_mask=sdmap_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        bev = bev + self.dropout1(bev2)
        bev = self.norm1(bev)
        
        bev2 = self.multihead_attn(query=self.with_pos_embed(bev, pos),
                                   key=self.with_pos_embed(sdmap, pos),
                                #    key=sdmap,
                                   value=sdmap, attn_mask=sdmap_mask,
                                   key_padding_mask=sdmap_key_padding_mask)[0]
        bev = bev + self.dropout2(bev2)
        bev = self.norm2(bev)
        
        bev2 = self.linear2(self.dropout(self.activation(self.linear1(bev))))
        bev = bev + self.dropout3(bev2)
        bev = self.norm3(bev)
        
        return bev


    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        
        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def build_transformer(args):
    return SDMapCrossAttn(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        normalize_before=args.pre_norm,
        return_intermediate_dec=True,
    )


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")