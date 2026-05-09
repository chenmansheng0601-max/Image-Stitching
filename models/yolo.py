"""YOLOv5-specific modules

Usage:
    $ python path/to/models/yolo.py --cfg yolov5s.yaml
"""

import argparse
import logging
import sys
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import time
from collections.abc import Iterable

import yaml

sys.path.append(Path(__file__).parent.parent.absolute().__str__())  # to run '$ python *.py' files in subdirectories
logger = logging.getLogger(__name__)
from models.backbone import Backbone 
import torch
import torch.nn as nn
import torch.nn.functional as F
# from torchvision.transforms.functional import perspective  # crash on cuda tensors. stupid.
from models.common import *
from models.experimental import *
from utils.general import make_divisible, check_file, set_logging
from utils.torch_utils import time_synchronized, fuse_conv_and_bn, model_info, scale_img, initialize_weights, \
    select_device, check_anomaly
from utils.stitching import DLTSolver, STN

try:
    import thop  # for FLOPs computation
except ImportError:
    thop = None
import copy
import math
from typing import List
import torch
import torch.nn.functional as F
from torch import nn
from torchvision.ops.boxes import nms

from util import box_ops
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized, inverse_sigmoid)

from .backbone import build_backbone
from .matcher import build_matcher
from .segmentation import (DETRsegm, PostProcessPanoptic, PostProcessSegm,
                           dice_loss)
# from .deformable_transformer import build_deformable_transformer
from .utils import sigmoid_focal_loss, MLP

from .registry import MODULE_BUILD_FUNCS
from .dn_components import prepare_for_cdn,dn_post_process
# from mmcv.cnn import PLUGIN_LAYERS, Conv2d, ConvModule, kaiming_init
# from mmcv.cnn.bricks.transformer import (build_positional_encoding,
#                                          build_transformer_layer_sequence)
# from mmcv.runner import BaseModule, ModuleList
from timm.models.vision_transformer import trunc_normal_
import itertools

class DINO(nn.Module):
    """ This is the Cross-Attention Detector module that performs object detection """
    def __init__(self, backbone, transformer, num_classes, num_queries, 
                    aux_loss=False, iter_update=False,
                    query_dim=2, 
                    random_refpoints_xy=False,
                    fix_refpoints_hw=-1,
                    num_feature_levels=1,
                    nheads=8,
                    # two stage
                    two_stage_type='no', # ['no', 'standard']
                    two_stage_add_query_num=0,
                    dec_pred_class_embed_share=True,
                    dec_pred_bbox_embed_share=True,
                    two_stage_class_embed_share=True,
                    two_stage_bbox_embed_share=True,
                    decoder_sa_type = 'sa',
                    num_patterns = 0,
                    dn_number = 100,
                    dn_box_noise_scale = 0.4,
                    dn_label_noise_ratio = 0.5,
                    dn_labelbook_size = 100,
                    ):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         Conditional DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.

            fix_refpoints_hw: -1(default): learn w and h for each box seperately
                                >0 : given fixed number
                                -2 : learn a shared w and h
        """
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim = transformer.d_model
        self.num_feature_levels = num_feature_levels
        self.nheads = nheads
        self.label_enc = nn.Embedding(dn_labelbook_size + 1, hidden_dim)

        # setting query dim
        self.query_dim = query_dim
        assert query_dim == 4
        self.random_refpoints_xy = random_refpoints_xy
        self.fix_refpoints_hw = fix_refpoints_hw

        # for dn training
        self.num_patterns = num_patterns
        self.dn_number = dn_number
        self.dn_box_noise_scale = dn_box_noise_scale
        self.dn_label_noise_ratio = dn_label_noise_ratio
        self.dn_labelbook_size = dn_labelbook_size

        # prepare input projection layers
        if num_feature_levels > 1:
            num_backbone_outs = len(backbone.num_channels)
            input_proj_list = []
            for _ in range(num_backbone_outs):
                in_channels = backbone.num_channels[_]
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
            for _ in range(num_feature_levels - num_backbone_outs):
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=3, stride=2, padding=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
                in_channels = hidden_dim
            self.input_proj = nn.ModuleList(input_proj_list)
        else:
            assert two_stage_type == 'no', "two_stage_type should be no if num_feature_levels=1 !!!"
            self.input_proj = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(backbone.num_channels[-1], hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                )])

        self.backbone = backbone
        self.aux_loss = aux_loss
        self.box_pred_damping = box_pred_damping = None

        self.iter_update = iter_update
        assert iter_update, "Why not iter_update?"

        # prepare pred layers
        self.dec_pred_class_embed_share = dec_pred_class_embed_share
        self.dec_pred_bbox_embed_share = dec_pred_bbox_embed_share
        # prepare class & box embed
        _class_embed = nn.Linear(hidden_dim, num_classes)
        _bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        # init the two embed layers
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        _class_embed.bias.data = torch.ones(self.num_classes) * bias_value
        nn.init.constant_(_bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(_bbox_embed.layers[-1].bias.data, 0)

        if dec_pred_bbox_embed_share:
            box_embed_layerlist = [_bbox_embed for i in range(transformer.num_decoder_layers)]
        else:
            box_embed_layerlist = [copy.deepcopy(_bbox_embed) for i in range(transformer.num_decoder_layers)]
        if dec_pred_class_embed_share:
            class_embed_layerlist = [_class_embed for i in range(transformer.num_decoder_layers)]
        else:
            class_embed_layerlist = [copy.deepcopy(_class_embed) for i in range(transformer.num_decoder_layers)]
        self.bbox_embed = nn.ModuleList(box_embed_layerlist)
        self.class_embed = nn.ModuleList(class_embed_layerlist)
        self.transformer.decoder.bbox_embed = self.bbox_embed
        self.transformer.decoder.class_embed = self.class_embed

        # two stage
        self.two_stage_type = two_stage_type
        self.two_stage_add_query_num = two_stage_add_query_num
        assert two_stage_type in ['no', 'standard'], "unknown param {} of two_stage_type".format(two_stage_type)
        if two_stage_type != 'no':
            if two_stage_bbox_embed_share:
                assert dec_pred_class_embed_share and dec_pred_bbox_embed_share
                self.transformer.enc_out_bbox_embed = _bbox_embed
            else:
                self.transformer.enc_out_bbox_embed = copy.deepcopy(_bbox_embed)
    
            if two_stage_class_embed_share:
                assert dec_pred_class_embed_share and dec_pred_bbox_embed_share
                self.transformer.enc_out_class_embed = _class_embed
            else:
                self.transformer.enc_out_class_embed = copy.deepcopy(_class_embed)
    
            self.refpoint_embed = None
            if self.two_stage_add_query_num > 0:
                self.init_ref_points(two_stage_add_query_num)

        self.decoder_sa_type = decoder_sa_type
        assert decoder_sa_type in ['sa', 'ca_label', 'ca_content']
        if decoder_sa_type == 'ca_label':
            self.label_embedding = nn.Embedding(num_classes, hidden_dim)
            for layer in self.transformer.decoder.layers:
                layer.label_embedding = self.label_embedding
        else:
            for layer in self.transformer.decoder.layers:
                layer.label_embedding = None
            self.label_embedding = None

        self._reset_parameters()

    def _reset_parameters(self):
        # init input_proj
        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)

    def init_ref_points(self, use_num_queries):
        self.refpoint_embed = nn.Embedding(use_num_queries, self.query_dim)
        if self.random_refpoints_xy:

            self.refpoint_embed.weight.data[:, :2].uniform_(0,1)
            self.refpoint_embed.weight.data[:, :2] = inverse_sigmoid(self.refpoint_embed.weight.data[:, :2])
            self.refpoint_embed.weight.data[:, :2].requires_grad = False

        if self.fix_refpoints_hw > 0:
            print("fix_refpoints_hw: {}".format(self.fix_refpoints_hw))
            assert self.random_refpoints_xy
            self.refpoint_embed.weight.data[:, 2:] = self.fix_refpoints_hw
            self.refpoint_embed.weight.data[:, 2:] = inverse_sigmoid(self.refpoint_embed.weight.data[:, 2:])
            self.refpoint_embed.weight.data[:, 2:].requires_grad = False
        elif int(self.fix_refpoints_hw) == -1:
            pass
        elif int(self.fix_refpoints_hw) == -2:
            print('learn a shared h and w')
            assert self.random_refpoints_xy
            self.refpoint_embed = nn.Embedding(use_num_queries, 2)
            self.refpoint_embed.weight.data[:, :2].uniform_(0,1)
            self.refpoint_embed.weight.data[:, :2] = inverse_sigmoid(self.refpoint_embed.weight.data[:, :2])
            self.refpoint_embed.weight.data[:, :2].requires_grad = False
            self.hw_embed = nn.Embedding(1, 1)
        else:
            raise NotImplementedError('Unknown fix_refpoints_hw {}'.format(self.fix_refpoints_hw))

    def forward(self, samples: NestedTensor, targets:List=None):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

            It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape= [batch_size x num_queries x num_classes]
               - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                               (center_x, center_y, width, height). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)
        features, poss = self.backbone(samples)

        srcs = []
        masks = []
        for l, feat in enumerate(features):
            src, mask = feat.decompose()
            srcs.append(self.input_proj[l](src))
            masks.append(mask)
            assert mask is not None
        if self.num_feature_levels > len(srcs):
            _len_srcs = len(srcs)
            for l in range(_len_srcs, self.num_feature_levels):
                if l == _len_srcs:
                    src = self.input_proj[l](features[-1].tensors)
                else:
                    src = self.input_proj[l](srcs[-1])
                m = samples.mask
                mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
                pos_l = self.backbone[1](NestedTensor(src, mask)).to(src.dtype)
                srcs.append(src)
                masks.append(mask)
                poss.append(pos_l)

        if self.dn_number > 0 or targets is not None:
            input_query_label, input_query_bbox, attn_mask, dn_meta =\
                prepare_for_cdn(dn_args=(targets, self.dn_number, self.dn_label_noise_ratio, self.dn_box_noise_scale),
                                training=self.training,num_queries=self.num_queries,num_classes=self.num_classes,
                                hidden_dim=self.hidden_dim,label_enc=self.label_enc)
        else:
            assert targets is None
            input_query_bbox = input_query_label = attn_mask = dn_meta = None

        hs, reference, hs_enc, ref_enc, init_box_proposal = self.transformer(srcs, masks, input_query_bbox, poss,input_query_label,attn_mask)
        # In case num object=0
        hs[0] += self.label_enc.weight[0,0]*0.0

        # deformable-detr-like anchor update
        # reference_before_sigmoid = inverse_sigmoid(reference[:-1]) # n_dec, bs, nq, 4
        outputs_coord_list = []
        for dec_lid, (layer_ref_sig, layer_bbox_embed, layer_hs) in enumerate(zip(reference[:-1], self.bbox_embed, hs)):
            layer_delta_unsig = layer_bbox_embed(layer_hs)
            layer_outputs_unsig = layer_delta_unsig  + inverse_sigmoid(layer_ref_sig)
            layer_outputs_unsig = layer_outputs_unsig.sigmoid()
            outputs_coord_list.append(layer_outputs_unsig)
        outputs_coord_list = torch.stack(outputs_coord_list)        

        outputs_class = torch.stack([layer_cls_embed(layer_hs) for
                                     layer_cls_embed, layer_hs in zip(self.class_embed, hs)])
        if self.dn_number > 0 and dn_meta is not None:
            outputs_class, outputs_coord_list = \
                dn_post_process(outputs_class, outputs_coord_list,
                                dn_meta,self.aux_loss,self._set_aux_loss)
        out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord_list[-1]}
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord_list)


        # for encoder output
        if hs_enc is not None:
            # prepare intermediate outputs
            interm_coord = ref_enc[-1]
            interm_class = self.transformer.enc_out_class_embed(hs_enc[-1])
            out['interm_outputs'] = {'pred_logits': interm_class, 'pred_boxes': interm_coord}
            out['interm_outputs_for_matching_pre'] = {'pred_logits': interm_class, 'pred_boxes': init_box_proposal}

            # prepare enc outputs
            if hs_enc.shape[0] > 1:
                enc_outputs_coord = []
                enc_outputs_class = []
                for layer_id, (layer_box_embed, layer_class_embed, layer_hs_enc, layer_ref_enc) in enumerate(zip(self.enc_bbox_embed, self.enc_class_embed, hs_enc[:-1], ref_enc[:-1])):
                    layer_enc_delta_unsig = layer_box_embed(layer_hs_enc)
                    layer_enc_outputs_coord_unsig = layer_enc_delta_unsig + inverse_sigmoid(layer_ref_enc)
                    layer_enc_outputs_coord = layer_enc_outputs_coord_unsig.sigmoid()

                    layer_enc_outputs_class = layer_class_embed(layer_hs_enc)
                    enc_outputs_coord.append(layer_enc_outputs_coord)
                    enc_outputs_class.append(layer_enc_outputs_class)

                out['enc_outputs'] = [
                    {'pred_logits': a, 'pred_boxes': b} for a, b in zip(enc_outputs_class, enc_outputs_coord)
                ]

        out['dn_meta'] = dn_meta

        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_boxes': b}
                for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]


FLOPS_COUNTER = 0

class Conv2d_BN(torch.nn.Sequential):
    def __init__(self, a, b, ks=1, stride=1, pad=0, dilation=1,
                 groups=1, bn_weight_init=1, resolution=-10000):
        super().__init__()
        self.add_module('c', torch.nn.Conv2d(
            a, b, ks, stride, pad, dilation, groups, bias=False))
        bn = torch.nn.BatchNorm2d(b)
        torch.nn.init.constant_(bn.weight, bn_weight_init)
        torch.nn.init.constant_(bn.bias, 0)
        self.add_module('bn', bn)

        global FLOPS_COUNTER
        output_points = ((resolution + 2 * pad - dilation *
                          (ks - 1) - 1) // stride + 1)**2
        FLOPS_COUNTER += a * b * output_points * (ks**2) // groups

    @torch.no_grad()
    def fuse(self):
        c, bn = self._modules.values()
        w = bn.weight / (bn.running_var + bn.eps)**0.5
        w = c.weight * w[:, None, None, None]
        b = bn.bias - bn.running_mean * bn.weight / \
            (bn.running_var + bn.eps)**0.5
        m = torch.nn.Conv2d(w.size(1), w.size(
            0), w.shape[2:], stride=self.c.stride, padding=self.c.padding, dilation=self.c.dilation, groups=self.c.groups)
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m
    
class Linear_BN(torch.nn.Sequential):
    def __init__(self, a, b, bn_weight_init=1, resolution=-100000):
        super().__init__()
        self.add_module('c', torch.nn.Linear(a, b, bias=False))
        bn = torch.nn.BatchNorm1d(b)
        torch.nn.init.constant_(bn.weight, bn_weight_init)
        torch.nn.init.constant_(bn.bias, 0)
        self.add_module('bn', bn)

        global FLOPS_COUNTER
        output_points = resolution**2
        FLOPS_COUNTER += a * b * output_points

    @torch.no_grad()
    def fuse(self):
        l, bn = self._modules.values()
        w = bn.weight / (bn.running_var + bn.eps)**0.5
        w = l.weight * w[:, None]
        b = bn.bias - bn.running_mean * bn.weight / \
            (bn.running_var + bn.eps)**0.5
        m = torch.nn.Linear(w.size(1), w.size(0))
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m

    def forward(self, x):
        l, bn = self._modules.values()
        x = l(x)
        return bn(x.flatten(0, 1)).reshape_as(x)


class BN_Linear(torch.nn.Sequential):
    def __init__(self, a, b, bias=True, std=0.02):
        super().__init__()
        self.add_module('bn', torch.nn.BatchNorm1d(a))
        l = torch.nn.Linear(a, b, bias=bias)
        trunc_normal_(l.weight, std=std)
        if bias:
            torch.nn.init.constant_(l.bias, 0)
        self.add_module('l', l)
        global FLOPS_COUNTER
        FLOPS_COUNTER += a * b

    @torch.no_grad()
    def fuse(self):
        bn, l = self._modules.values()
        w = bn.weight / (bn.running_var + bn.eps)**0.5
        b = bn.bias - self.bn.running_mean * \
            self.bn.weight / (bn.running_var + bn.eps)**0.5
        w = l.weight * w[None, :]
        if l.bias is None:
            b = b @ self.l.weight.T
        else:
            b = (l.weight @ b[:, None]).view(-1) + self.l.bias
        m = torch.nn.Linear(w.size(1), w.size(0))
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m

def b16(n, activation, resolution=224):
    return torch.nn.Sequential(
        Conv2d_BN(1, n // 8, 3, 2, 1, resolution=resolution), # aping num_channel
        activation(),
        Conv2d_BN(n // 8, n // 4, 3, 2, 1, resolution=resolution // 2),
        activation(),
        Conv2d_BN(n // 4, n // 2, 3, 2, 1, resolution=resolution // 4),
        activation(),
        Conv2d_BN(n // 2, n, 3, 2, 1, resolution=resolution // 8))


class Residual(torch.nn.Module):
    def __init__(self, m, drop):
        super().__init__()
        self.m = m
        self.drop = drop

    def forward(self, x):
        if self.training and self.drop > 0:
            return x + self.m(x) * torch.rand(x.size(0), 1, 1,
                                              device=x.device).ge_(self.drop).div(1 - self.drop).detach()
        else:
            x = x[:, :196, :]
            return x + self.m(x)


class Attention(torch.nn.Module):
    def __init__(self, dim, key_dim, num_heads=8,
                 attn_ratio=4,
                 activation=None,
                 resolution=14):
        super().__init__()
        self.num_heads = num_heads
        self.scale = key_dim ** -0.5
        self.key_dim = key_dim
        self.nh_kd = nh_kd = key_dim * num_heads
        self.d = int(attn_ratio * key_dim)
        self.dh = int(attn_ratio * key_dim) * num_heads
        self.attn_ratio = attn_ratio
        h = self.dh + nh_kd * 2
        self.qkv = Linear_BN(dim, h, resolution=resolution)
        self.proj = torch.nn.Sequential(activation(), Linear_BN(
            self.dh, dim, bn_weight_init=0, resolution=resolution))

        points = list(itertools.product(range(resolution), range(resolution)))
        N = len(points)
        attention_offsets = {}
        idxs = []
        for p1 in points:
            for p2 in points:
                offset = (abs(p1[0] - p2[0]), abs(p1[1] - p2[1]))
                if offset not in attention_offsets:
                    attention_offsets[offset] = len(attention_offsets)
                idxs.append(attention_offsets[offset])
        self.attention_biases = torch.nn.Parameter(
            torch.zeros(num_heads, len(attention_offsets)))
        self.register_buffer('attention_bias_idxs',
                             torch.LongTensor(idxs).view(N, N))

        global FLOPS_COUNTER
        #queries * keys
        FLOPS_COUNTER += num_heads * (resolution**4) * key_dim
        # softmax
        FLOPS_COUNTER += num_heads * (resolution**4)
        #attention * v
        FLOPS_COUNTER += num_heads * self.d * (resolution**4)

    @torch.no_grad()
    def train(self, mode=True):
        super().train(mode)
        if mode and hasattr(self, 'ab'):
            del self.ab
        else:
            self.ab = self.attention_biases[:, self.attention_bias_idxs]

    def forward(self, x):  # x (B,N,C)
        B, N, C = x.shape
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.qkv = nn.Linear(384, 768).to(device)
        qkv = self.qkv(x)
        q, k, v = qkv.view(B, N, self.num_heads, -
                           1).split([self.key_dim, self.key_dim, self.d], dim=3)    
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)
        # q = q[:, :, :196, :]
        # k = k[:, :, :196, :]
        # v = v[:, :, :196, :]
        # print(q.shape)
        # print(k.shape)
        # print(v.shape)
        attn = (
            (q @ k.transpose(-2, -1)) * self.scale
            # +
            # (self.attention_biases[:, self.attention_bias_idxs]
            #  if self.training else self.ab)
        )
        attn = attn.softmax(dim=-1)
        # x = (attn @ v).transpose(1, 2).reshape(B, N, self.dh)
        x = (attn @ v).transpose(1, 2).reshape(B, 196, self.dh)
        x = self.proj(x)
        return x


class Subsample(torch.nn.Module):
    def __init__(self, stride, resolution):
        super().__init__()
        self.stride = stride
        self.resolution = resolution

    def forward(self, x):
        B, N, C = x.shape
        x = x.view(B, self.resolution, self.resolution, C)[
            :, ::self.stride, ::self.stride].reshape(B, -1, C)
        return x


class AttentionSubsample(torch.nn.Module):
    def __init__(self, in_dim, out_dim, key_dim, num_heads=8,
                 attn_ratio=2,
                 activation=None,
                 stride=2,
                 resolution=14, resolution_=7):
        super().__init__()
        self.num_heads = num_heads
        self.scale = key_dim ** -0.5
        self.key_dim = key_dim
        self.nh_kd = nh_kd = key_dim * num_heads
        self.d = int(attn_ratio * key_dim)
        self.dh = int(attn_ratio * key_dim) * self.num_heads
        self.attn_ratio = attn_ratio
        self.resolution_ = resolution_
        self.resolution_2 = resolution_**2
        h = self.dh + nh_kd
        self.kv = Linear_BN(in_dim, h, resolution=resolution)

        self.q = torch.nn.Sequential(
            Subsample(stride, resolution),
            Linear_BN(in_dim, nh_kd, resolution=resolution_))
        self.proj = torch.nn.Sequential(activation(), Linear_BN(
            self.dh, out_dim, resolution=resolution_))

        self.stride = stride
        self.resolution = resolution
        points = list(itertools.product(range(resolution), range(resolution)))
        points_ = list(itertools.product(
            range(resolution_), range(resolution_)))
        N = len(points)
        N_ = len(points_)
        attention_offsets = {}
        idxs = []
        for p1 in points_:
            for p2 in points:
                size = 1
                offset = (
                    abs(p1[0] * stride - p2[0] + (size - 1) / 2),
                    abs(p1[1] * stride - p2[1] + (size - 1) / 2))
                if offset not in attention_offsets:
                    attention_offsets[offset] = len(attention_offsets)
                idxs.append(attention_offsets[offset])
        self.attention_biases = torch.nn.Parameter(
            torch.zeros(num_heads, len(attention_offsets)))
        self.register_buffer('attention_bias_idxs',
                             torch.LongTensor(idxs).view(N_, N))

        global FLOPS_COUNTER
        #queries * keys
        FLOPS_COUNTER += num_heads * \
            (resolution**2) * (resolution_**2) * key_dim
        # softmax
        FLOPS_COUNTER += num_heads * (resolution**2) * (resolution_**2)
        #attention * v
        FLOPS_COUNTER += num_heads * \
            (resolution**2) * (resolution_**2) * self.d

    @torch.no_grad()
    def train(self, mode=True):
        super().train(mode)
        if mode and hasattr(self, 'ab'):
            del self.ab
        else:
            self.ab = self.attention_biases[:, self.attention_bias_idxs]

    def forward(self, x):
        B, N, C = x.shape
        k, v = self.kv(x).view(B, N, self.num_heads, -
                               1).split([self.key_dim, self.d], dim=3)
        k = k.permute(0, 2, 1, 3)  # BHNC
        v = v.permute(0, 2, 1, 3)  # BHNC
        q = self.q(x).view(B, self.resolution_2, self.num_heads,
                           self.key_dim).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale + \
            (self.attention_biases[:, self.attention_bias_idxs]
             if self.training else self.ab)
        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, -1, self.dh)
        x = self.proj(x)
        return x

class Conv2dReLU(nn.Sequential):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            padding=0,
            stride=1,
            use_batchnorm=True,
    ):
        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=not (use_batchnorm),
        )
        relu = nn.ReLU(inplace=True)

        bn = nn.BatchNorm2d(out_channels)

        super(Conv2dReLU, self).__init__(conv, bn, relu)

class DecoderBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            #skip_channels=0,
            use_batchnorm=True,
    ):
        super().__init__()
        self.conv1 = Conv2dReLU(
            #in_channels + skip_channels,
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.conv2 = Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, x, skip=None):
        x = self.up(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x

class SegmentationHead(nn.Sequential):

    def __init__(self, in_channels, out_channels, kernel_size=3, upsampling=1):
        conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        upsampling = nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()
        super().__init__(conv2d, upsampling)

class LeViT_UNet_384(torch.nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """

    def __init__(self, img_size=512, ## aping: should change
                 patch_size=16,
                 in_chans=3,
                 num_classes=9,
                 embed_dim=[192],
                 key_dim=[64],
                 depth=[12],
                 num_heads=[3],
                 attn_ratio=[2],
                 mlp_ratio=[2],
                 hybrid_backbone=None,
                 down_ops=[],
                 attention_activation=torch.nn.Hardswish,
                 mlp_activation=torch.nn.Hardswish,
                 distillation=True,
                 drop_path=0):
        super().__init__()
        global FLOPS_COUNTER

        self.num_classes = num_classes
        self.num_features = embed_dim[-1]
        self.embed_dim = embed_dim
        self.distillation = distillation

        # self.patch_embed = hybrid_backbone ## aping: CNN  # aping num_channel
        n = 192 ## cnn num channel
        activation = torch.nn.Hardswish
        self.cnn_b1 = torch.nn.Sequential(
            Conv2d_BN(384, n // 8, 3, 2, 1, resolution=img_size), activation())
        self.cnn_b2 = torch.nn.Sequential(
            Conv2d_BN(n // 8, n // 4, 3, 2, 1, resolution=img_size // 2), activation())
        self.cnn_b3 = torch.nn.Sequential(
            Conv2d_BN(n // 4, n // 2, 3, 2, 1, resolution=img_size // 4), activation())
        self.cnn_b4 = torch.nn.Sequential(    
            Conv2d_BN(n // 2, n, 3, 2, 1, resolution=img_size // 8))


        # self.decoderBlock_1 = DecoderBlock(2048, 512) #->att+cnn:384+512+768+384=2048==1280+256
        # self.decoderBlock_2 = DecoderBlock(704, 256) #->28->56, 512+9+192=713
        # self.decoderBlock_3 = DecoderBlock(352, 128) #->56->112, 256+9+96=361
        # self.segmentation_head = SegmentationHead(176, self.num_classes, kernel_size=3, upsampling=2)


        self.blocks = [] ## attention block
        down_ops.append([''])
        resolution = img_size // patch_size
        for i, (ed, kd, dpth, nh, ar, mr, do) in enumerate(
                zip(embed_dim, key_dim, depth, num_heads, attn_ratio, mlp_ratio, down_ops)):
            print(i)
            for _ in range(dpth):
                self.blocks.append(
                    Residual(Attention(
                        ed, kd, nh,
                        attn_ratio=ar,
                        activation=attention_activation,
                        resolution=resolution,
                    ), drop_path))
                if mr > 0:
                    h = int(ed * mr)
                    self.blocks.append(
                        Residual(torch.nn.Sequential(
                            Linear_BN(ed, h, resolution=resolution),
                            mlp_activation(),
                            Linear_BN(h, ed, bn_weight_init=0,
                                      resolution=resolution),
                        ), drop_path))
            if do[0] == 'Subsample':
                #('Subsample',key_dim, num_heads, attn_ratio, mlp_ratio, stride)
                resolution_ = (resolution - 1) // do[5] + 1
                self.blocks.append(
                    AttentionSubsample(
                        *embed_dim[i:i + 2], key_dim=do[1], num_heads=do[2],
                        attn_ratio=do[3],
                        activation=attention_activation,
                        stride=do[5],
                        resolution=resolution,
                        resolution_=resolution_))
                resolution = resolution_
                if do[4] > 0:  # mlp_ratio
                    h = int(embed_dim[i + 1] * do[4])
                    self.blocks.append(
                        Residual(torch.nn.Sequential(
                            Linear_BN(embed_dim[i + 1], h,
                                      resolution=resolution),
                            mlp_activation(),
                            Linear_BN(
                                h, embed_dim[i + 1], bn_weight_init=0, resolution=resolution),
                        ), drop_path))
        self.blocks = torch.nn.Sequential(*self.blocks)
        ## divid the blocks
        self.block_1 = self.blocks[0:8]
        self.block_2 = self.blocks[8:18]
        self.block_3 = self.blocks[18:28]

        del self.blocks

        ## aping: upsampling
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)


        self.FLOPS = FLOPS_COUNTER
        FLOPS_COUNTER = 0
        

    @torch.jit.ignore
    def no_weight_decay(self):
        return {x for x in self.state_dict().keys() if 'attention_biases' in x}

    def forward(self, x):
 
        x_cnn_1 = self.cnn_b1(x) # torch.Size([4, 1, 512, 512])->([4, 32, 256, 256])

        x_cnn_2 = self.cnn_b2(x_cnn_1) #-> ([4, 64, 128, 128])

        x_cnn_3 = self.cnn_b3(x_cnn_2) #->([4, 128, 64, 64])

        x_cnn = self.cnn_b4(x_cnn_3) # ([4, 256, 32, 32])

        

        x = x_cnn.flatten(2).transpose(1, 2) # torch.Size([4, 196, 256])
        

        ## aping 
        x = self.block_1(x) # torch.Size([4, 196, 384])-->Nx256x14x14
        x_num, x_len = x.shape[0], x.shape[1]

        x_r_1 = x.reshape(x_num, int(x_len**0.5), int(x_len**0.5), -1)
        x_r_1 = x_r_1.permute(0,3,1,2)

        x = self.block_2(x) # downsample + att  torch.Size([4, 49, 384])
        x_num, x_len = x.shape[0], x.shape[1]

        x_r_2 = x.reshape(x_num, int(x_len**0.5), int(x_len**0.5), -1)
        x_r_2 = x_r_2.permute(0,3,1,2)
        ## upsampling
        x_r_2_up = self.up(x_r_2)
        

        x = self.block_3(x) # torch.Size([4, 16, 512])
        x_num, x_len = x.shape[0], x.shape[1]

        x_r_3 = x.reshape(x_num, int(x_len**0.5), int(x_len**0.5), -1)
        x_r_3 = x_r_3.permute(0,3,1,2)
        ## upsampling
        x_r_3_up = self.up(x_r_3)
        x_r_3_up = self.up(x_r_3_up)
        
        # ## aping: resize the feature maps
        # if (x_r_2_up.shape  != x_r_3_up.shape):
        #     x_r_3_up = F.interpolate(x_r_3_up, size=x_r_2_up.shape[2:], mode="bilinear",  align_corners=True)
        # att_all = torch.cat([ x_r_1, x_r_2_up, x_r_3_up], dim=1) # 384+512+768

        # x_att_all = torch.cat([x_cnn, att_all], dim=1) ## torch.Size([4, 1408, 32, 32])
        # decoder_feature = self.decoderBlock_1(x_att_all) # x_att_all: ([4, 1408, 32, 32])->512

        # decoder_feature = torch.cat([decoder_feature, x_cnn_3], dim=1) #:(640+9)x64x64
        # decoder_feature = self.decoderBlock_2(decoder_feature)

        # decoder_feature = torch.cat([decoder_feature, x_cnn_2], dim=1) # ([4, (320+9), 128, 128])
        # decoder_feature = self.decoderBlock_3(decoder_feature)

        # decoder_feature = torch.cat([decoder_feature, x_cnn_1], dim=1) # ([4, 169, 256, 256])
        
        # logits = self.segmentation_head(decoder_feature) ## torch.Size([4, 2, 224, 224])

        return x_r_3[:, :6, :, :]

class Detect(nn.Module):
    stride = None  # strides computed during build
    onnx_dynamic = False  # ONNX export parameter
    
    def __init__(self, nc=80, anchors=(), ch=(), inplace=True):  # detection layer
        super(Detect, self).__init__()
        self.nc = nc  # number of classes
        self.no = nc + 5  # number of outputs per anchor
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.grid = [torch.zeros(1)] * self.nl  # init grid
        a = torch.tensor(anchors).float().view(self.nl, -1, 2)
        self.register_buffer('anchors', a)  # shape(nl,na,2)
        self.register_buffer('anchor_grid', a.clone().view(self.nl, 1, -1, 1, 1, 2))  # shape(nl,1,na,1,1,2)
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv
        self.inplace = inplace  # use in-place ops (e.g. slice assignment)
    
    def forward(self, x):
        # x = x.copy()  # for profiling
        z = []  # inference output
        for i in range(self.nl):
            x[i] = self.m[i](x[i])  # conv
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
            
            if not self.training:  # inference
                if self.grid[i].shape[2:4] != x[i].shape[2:4] or self.onnx_dynamic:
                    self.grid[i] = self._make_grid(nx, ny).to(x[i].device)
                
                y = x[i].sigmoid()
                if self.inplace:
                    y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i]) * self.stride[i]  # xy
                    y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                else:  # for YOLOv5 on AWS Inferentia https://github.com/ultralytics/yolov5/pull/2953
                    xy = (y[..., 0:2] * 2. - 0.5 + self.grid[i]) * self.stride[i]  # xy
                    wh = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i].view(1, self.na, 1, 1, 2)  # wh
                    y = torch.cat((xy, wh, y[..., 4:]), -1)
                z.append(y.view(bs, -1, self.no))
        
        return x if self.training else (torch.cat(z, 1), x)
    
    @staticmethod
    def _make_grid(nx=20, ny=20):
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()

class HEstimator(nn.Module):
    def __init__(self, input_size=128, strides=(2,4,8), keep_prob=0.5, norm='BN', ch=()):
        super(HEstimator, self).__init__()
        self.ch = ch  # channels for multiple feature maps, e.g., [48, 96, 192] for yolov5m
        self.stride = torch.tensor([4, 32])  # fake
        self.input_size = input_size
        self.strides = strides
        self.keep_prob = keep_prob
        self.search_ranges = [16, 8, 4]
        self.patch_sizes = [input_size/4, input_size/2, input_size/1]
        # shape[2, 2, 3, 3]
        self.aux_matrices = torch.stack([self.gen_aux_mat(patch_size) for patch_size in self.patch_sizes])
        self.DLT_solver = DLTSolver()
        self._init_layers(norm=norm)

    def _init_layers(self, norm='BN'):
        m = []
        s = self.input_size // (128 // 8)
        k, p = s, 0
        for i, x in enumerate(self.ch[::-1]):  # manually calculate the channels
            # ch1 = (self.search_ranges[i] * 2 + 1) ** 2
            ch1 = x * 2
            ch_conv = 512 // (2 ** i)
            # ch_flat = (self.input_size // self.strides[-(i+1)] // s) ** 2 * ch_conv
            ch_flat = (self.input_size // self.strides[-1] // s) ** 2 * ch_conv
            # ch_flat = ch_conv * (s ** 2)
            ch_fc = 512 // (2 ** i)
            # print(x, ch1, ch_conv, ch_flat, ch_fc)
            m.append(
                nn.Sequential(
                    Conv(ch1, ch_conv, k=3, s=1, norm=norm),
                    Conv(ch_conv, ch_conv, k=3, s=2 if i >= 2 else 1, norm=norm),  # stage 2
                    Conv(ch_conv, ch_conv, k=3, s=2 if i >= 1 else 1, norm=norm),  # stage 2 & 3
                    DWConv(ch_conv, ch_conv, k=k, s=s, p=p, norm='BN'),  # TODO: DWConv is special. BN seems better.
                    # nn.AvgPool2d(k, s, p),
                    # nn.AdaptiveAvgPool2d((s, s)),
                    nn.Flatten(),
                    # nn.Linear(ch_conv, ch_fc),
                    nn.Linear(ch_flat, ch_fc),
                    nn.SiLU(),
                    # nn.Dropout(keep_prob),
                    nn.Linear(ch_fc, 8, bias=False) 
                )
            )
        self.m = nn.ModuleList(m)
    
    def forward(self, feature1, feature2, image2, mask2):
        bs = image2.size(0)
        assert len(self.search_ranges) == len(feature1) == len(feature2)
        device, dtype = image2.device, image2.dtype
        if self.aux_matrices.device != device:
            self.aux_matrices = self.aux_matrices.to(device)
        if self.aux_matrices.dtype != dtype:
            self.aux_matrices = self.aux_matrices.type(dtype)
            
        vertices_offsets = []
        for i, search_range in enumerate(self.search_ranges):
            x = self._feat_fuse(feature1[-(i+1)], feature2[-(i+1)], i=i, search_range=search_range)
            
            off = self.m[i](x).unsqueeze(-1)  # [bs, 8, 1], for matrix multiplication
            assert torch.isnan(off).sum() == 0
            
            # off, overflow = self.clip_offset(off, vertices_offsets, phase=i)
            vertices_offsets.append(off)
            
            if i == len(self.search_ranges) - 1:
                break
            
            # H = self.DLT_solver.solve(sum(vertices_offsets) / (2 ** (2 - i)), self.patch_sizes[i])  # 2x up-scale
            # M, M_inv = torch.chunk(self.aux_matrices[i], 2, dim=0)
            # 4x down-scale for numerical stability
            H = self.DLT_solver.solve(sum(vertices_offsets) / 4., self.patch_sizes[0])
            M, M_inv = torch.chunk(self.aux_matrices[0], 2, dim=0)

            H = torch.bmm(torch.bmm(M_inv.expand(bs, -1, -1), H), M.expand(bs, -1, -1))

            feature2[-(i + 2)] = self._feat_warp(feature2[-(i + 2)], H, vertices_offsets)

        warped_imgs, warped_msks = [], []
        patch_level = 0
        M, M_inv = torch.chunk(self.aux_matrices[patch_level], 2, dim=0)
        img_with_msk = torch.cat((image2, mask2), dim=1)
        for i in range(len(vertices_offsets)):
            H_inv = self.DLT_solver.solve(sum(vertices_offsets[:i+1]) / (2 ** (2 - patch_level)), self.patch_sizes[patch_level])
            H = torch.bmm(torch.bmm(M_inv.expand(bs, -1, -1), H_inv), M.expand(bs, -1, -1))
            warped_img, warped_msk = STN(img_with_msk, H, vertices_offsets[:i+1]).split([3, 1], dim=1)
            warped_imgs.append(warped_img)
            warped_msks.append(warped_msk)
        
        # the relationship (or definition) between `H` and `H_inv` is confusing
        # H = torch.linalg.inv(H_inv.detach())
        # H /= H[:, -1, -1]  # same as the results from cv2.getPerspectiveTransform(org, dst)
        
        return sum(vertices_offsets), warped_imgs, warped_msks

    def _feat_fuse(self, x1, x2, i, search_range):
        # global_correlation is either time-consuming or memory-consuming, and even leads to divergence
        # concatenation seems to be capable enough for estimation
        # x = self.cost_volume(x1, x2, search_range)
        x = torch.cat((x1, x2), dim=1)
        return x
        
    @staticmethod
    def _feat_warp(x2, H, vertices_offsets):
        return STN(x2, H, vertices_offsets)

    @staticmethod
    def cost_volume(x1, x2, search_range, norm=True, fast=True):
        if norm:
            x1 = F.normalize(x1, p=2, dim=1)
            x2 = F.normalize(x2, p=2, dim=1)
        bs, c, h, w = x1.shape
        padded_x2 = F.pad(x2, [search_range] * 4)  # [b,c,h,w] -> [b,c,h+sr*2,w+sr*2]
        max_offset = search_range * 2 + 1

        if fast:
            # faster(*2) but cost higher(*n) GPU memory
            patches = F.unfold(padded_x2, (max_offset, max_offset)).reshape(bs, c, max_offset ** 2, h, w)
            cost_vol = (x1.unsqueeze(2) * patches).mean(dim=1, keepdim=False)
        else:
            # slower but save memory
            cost_vol = []
            for j in range(0, max_offset):
                for i in range(0, max_offset):
                    x2_slice = padded_x2[:, :, j:j + h, i:i + w]
                    cost = torch.mean(x1 * x2_slice, dim=1, keepdim=True)
                    cost_vol.append(cost)
            cost_vol = torch.cat(cost_vol, dim=1)
        
        cost_vol = F.leaky_relu(cost_vol, 0.1)
        
        return cost_vol
    
    @staticmethod
    def gen_aux_mat(patch_size):
        M = np.array([[patch_size / 2.0, 0., patch_size / 2.0],
                      [0., patch_size / 2.0, patch_size / 2.0],
                      [0., 0., 1.]]).astype(np.float32)
        M_inv = np.linalg.inv(M)
        return torch.from_numpy(np.stack((M, M_inv)))  # [2, 3, 3]


class HEstimatorOrigin(HEstimator):
    def __init__(self, input_size=128, strides=(2,4,8), keep_prob=0.5, norm='None', ch=()):
        super(HEstimatorOrigin, self).__init__(input_size, strides, keep_prob, norm, ch)

    def _init_layers(self, norm='None'):
        m = []
        for i, x in enumerate(self.ch[::-1]):  # manually calculate the channels
            ch1 = (self.search_ranges[i] * 2 + 1) ** 2
            ch_conv = 512 // (2 ** i)
            ch_flat = (self.input_size // self.strides[-1]) ** 2 * ch_conv
            ch_fc = 1024 // (2 ** i)
            # print(x, ch1, ch_conv, ch_flat, ch_fc)
            m.append(
                nn.Sequential(
                    Conv(ch1, ch_conv, k=3, s=1, norm=norm),
                    Conv(ch_conv, ch_conv, k=3, s=2 if i >= 2 else 1, norm=norm),  # stage 2
                    Conv(ch_conv, ch_conv, k=3, s=2 if i >= 1 else 1, norm=norm),  # stage 2 & 3
                    nn.Flatten(),
                    nn.Linear(ch_flat, ch_fc),
                    nn.ReLU(inplace=True),
                    nn.Dropout(self.keep_prob),
                    nn.Linear(ch_fc, 8, bias=False)
                )
            )
        self.m = nn.ModuleList(m)

    def _feat_fuse(self, x1, x2, i, search_range):
        x1, x2 = F.normalize(x1, p=2, dim=1), F.normalize(x2, p=2, dim=1) if i == 0 else x2
        x = self.cost_volume(x1, x2, search_range, norm=False)
        return x
    
    @staticmethod
    def _feat_warp(x2, H, vertices_offsets):
        return STN(F.normalize(x2, p=2, dim=1), H, vertices_offsets)


class Reconstructor(nn.Module):
    def __init__(self, norm='BN', ch=()):
        super(Reconstructor, self).__init__()
        ch_lr = ch[0]
      
        self.m_lr = nn.Sequential(
            Conv2d_BN(ch_lr, ch_lr, ks=3, stride=1, pad=1, bn_weight_init=1, resolution=512),
            nn.Hardswish(),
            nn.Conv2d(ch_lr, 3, kernel_size=3, stride=1, padding=1, bias=False),
        )
        self.m_hr = nn.Sequential(
            Conv(3 * 3, 64, norm=norm),
            C3(64, 64, 3, norm=norm),
            Conv2d_BN(64, 64, ks=3, stride=1, pad=1, bn_weight_init=1, resolution=512),
            nn.Hardswish(),
            C3(64, 64, 3, norm=norm),
            nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1, bias=False),
        )
        self.stride = torch.tensor([4, 32])  # fake
    
    def forward(self, x):
        out_lr, in_hr = x
        # low resolution
        out_lr = self.m_lr(out_lr).sigmoid_()
        # super resolution
        out_lr_sr = F.interpolate(out_lr, mode='bilinear', size=in_hr.shape[2:], align_corners=False)
        # concat
        out_hr = torch.cat((in_hr, out_lr_sr), dim=1)
        # high resolution
        out_hr = self.m_hr(out_hr).sigmoid_()
        return out_lr, out_hr


class Model(nn.Module):
    # def __init__(self, cfg='yolov5s.yaml', ch=3, mode_align=True, dino_params=None):  # model, input channels
    def __init__(self, cfg='yolov5s.yaml', ch=3, mode_align=True):  # model, input channels
        super(Model, self).__init__()
        if isinstance(cfg, dict):
            self.yaml = cfg  # model dict
        else:  # is *.yaml
            import yaml  # for torch hub
            self.yaml_file = Path(cfg).name
            with open(cfg) as f:
                self.yaml = yaml.safe_load(f)  # model dict

        self.mode_align = mode_align
        self.ch = self.yaml['ch'] = self.yaml.get('ch', ch)  # input channels
        ch = 3 if mode_align else 6
        self.model, self.save = parse_model(deepcopy(self.yaml), ch=[ch])  # model, savelist
        self.inplace = self.yaml.get('inplace', True)

        # Build strides, anchors
        m = self.model[-1]  # Detect()
        if isinstance(m, Reconstructor) or isinstance(m, HEstimator):
            self.stride = m.stride
        else:
            self.stride = torch.tensor([4, 32])  # fake

        # Init weights, biases
        initialize_weights(self)
        self.info()
        logger.info('')

    def forward(self, x, profile=False, mode_align=True):
        # x1/m1: right image/mask, x2/m2: left image/mask. warp x2(left image) to x1(right image)
        x1, m1, x2, m2 = torch.split(x, [3, 1, 3, 1], dim=1)  # channel dimension

        mode_align = self.mode_align if hasattr(self, 'mode_align') else mode_align  # TODO: compatible with old api
        if mode_align:
            module_range = (0, -1)
            feature1 = self.forward_once(x1, profile, module_range=module_range)
            feature2 = self.forward_once(x2, profile, module_range=module_range)
          
            return self.model[-1](feature1, feature2, x2, m2) 
        
            
        else:
            x = torch.cat((x1, x2), dim=1)
            out = self.forward_once(x, profile)  # single-scale inference, train
            if not self.training:
                mask = ((m1 + m2) > 0).type_as(x)  # logical_or
                out = (out[0], out[1] * mask)  # higher resolution
            return out

    def forward_once(self, x, profile=False, module_range=None):
        y, dt = [], []  # outputs
        inputs = x
        modules = self.model if module_range is None else self.model[module_range[0]:module_range[1]]
        for m in modules:
            # if m.f != -1:  # if not from previous layer
            if hasattr(m, 'f') and m.f != -1:  # 检查 m 是否有属性 'f' 并且 f 不等于 -1
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers

            if profile:
                o = thop.profile(m, inputs=(x,), verbose=False)[0] / 1E9 * 2 if thop else 0  # FLOPs
                t = time_synchronized()
                for _ in range(10):
                    _ = m(x)
                dt.append((time_synchronized() - t) * 100)
                if m == self.model[0]:
                    logger.info(f"{'time (ms)':>10s} {'GFLOPs':>10s} {'params':>10s}  {'module'}")
                logger.info(f'{dt[-1]:10.2f} {o:10.2f} {m.np:10.0f}  {m.type}')

            if str(m.type) in 'models.yolo.Reconstructor':
                x = (*x, inputs)
            x = m(x)  # run
            y.append(x if m.i in self.save else None)  # save output

        if profile:
            logger.info('%.1fms total' % sum(dt))
        return x

    def fuse(self):  # fuse model Conv2d() + BatchNorm2d() layers
        logger.info('Fusing layers... ')
        for m in self.model.modules():
            if type(m) is Conv and hasattr(m, 'bn') and isinstance(m.bn, nn.BatchNorm2d):
                m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
                delattr(m, 'bn')  # remove batchnorm
                m.forward = m.fuseforward  # update forward
        self.info()
        return self

    def info(self, verbose=False, img_size=640):  # print model information
        model_info(self, verbose, img_size)


# def parse_model(d, ch):  # model_dict, input_channels(3)
#     logger.info('\n%3s%18s%3s%10s  %-40s%-30s' % ('', 'from', 'n', 'params', 'module', 'arguments'))
#     gd, gw = d['depth_multiple'], d['width_multiple']
#     no = 3
#     pvt_channels = []  # 用于记录 PVT 的多尺度通道
#     layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out
#     for i, (f, n, m, args) in enumerate(d['backbone'] + d['head']):  # from, number, module, args
#         m = eval(m) if isinstance(m, str) else m  # eval strings
#         for j, a in enumerate(args):
#             try:
#                 args[j] = eval(a) if isinstance(a, str) else a  # eval strings
#             except:
#                 pass

#         n = max(round(n * gd), 1) if n > 1 else n  # depth gain
#         if m in [Conv, GhostConv, Bottleneck, GhostBottleneck, SPP, DWConv, MixConv2d, Focus, Blur, CrossConv, BottleneckCSP,
#                  C3, C3TR, Focus2, ResBlock]:
#             c1, c2 = ch[f], args[0]
#             if c2 != no:  # if not output
#                 c2 = make_divisible(c2 * gw, 8)

#             args = [c1, c2, *args[1:]]
#             if m in [BottleneckCSP, C3, C3TR]:
#                 args.insert(2, n)  # number of repeats
#                 n = 1
#         elif m is nn.BatchNorm2d:
#             args = [ch[f]]
#         elif m is Concat:
#             c2 = sum([ch[x] for x in f])
#         elif m is Add:
#             c2 = ch[f[0]]
#         elif m is Resizer:
#             c2 = ch[f[0]] if isinstance(f, Iterable) else ch[f]
#         elif m is Reconstructor:
#             args.append([ch[x] for x in f])
#         elif m in [HEstimator, HEstimatorOrigin]:
#             args.append(ch[f])
#         elif m is Contract:
#             c2 = ch[f] * args[0] ** 2
#         elif m is Expand:
#             c2 = ch[f] // args[0] ** 2
        
#         elif m is PVT1:
#             # 假设 PVT1 输出四个多尺度特征图：P1, P2, P3, P4
#             pvt_channels  = [64, 128, 256, 512]  # 你期望的通道数
#             c2 = 0  # PVT1 本身不参与卷积，只占位
#         elif m is Get:
#             idx = args[0]
#             c2 = pvt_channels[idx]
        
#         elif m is nn.Identity:
#             c2 = [ch[x] for x in f] if isinstance(f, Iterable) else ch[f]
#         else:
#             c2 = ch[f]
#         if isinstance(c2, list):
#             ch.extend(c2)
#         else:
#             ch.append(c2)

#         m_ = nn.Sequential(*[m(*args) for _ in range(n)]) if n > 1 else m(*args)  # module
#         t = str(m)[8:-2].replace('__main__.', '')  # module type
#         np = sum([x.numel() for x in m_.parameters()])  # number params
#         m_.i, m_.f, m_.type, m_.np = i, f, t, np  # attach index, 'from' index, type, number params
#         logger.info('%3s%18s%3s%10.0f  %-40s%-30s' % (i, f, n, np, t, args))  # print
#         save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
#         layers.append(m_)
#         if i == 0:
#             ch = []
#         ch.append(c2)
#     return nn.Sequential(*layers), sorted(save)

def parse_model(d, ch):  # model_dict, input_channels(3)
    logger.info('\n%3s%18s%3s%10s  %-40s%-30s' % ('', 'from', 'n', 'params', 'module', 'arguments'))
    gd, gw = d['depth_multiple'], d['width_multiple']
    no = 3

    layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out
    for i, (f, n, m, args) in enumerate(d['backbone'] + d['head']):  # from, number, module, args
        m = eval(m) if isinstance(m, str) else m  # eval strings
        for j, a in enumerate(args):
            try:
                args[j] = eval(a) if isinstance(a, str) else a  # eval strings
            except:
                pass

        n = max(round(n * gd), 1) if n > 1 else n  # depth gain
        if m in [Conv, GhostConv, Bottleneck, GhostBottleneck, SPP, DWConv, MixConv2d, Focus, Blur, CrossConv, BottleneckCSP,
                 C3, C3TR, Focus2, ResBlock]:
            c1, c2 = ch[f], args[0]
            if c2 != no:  # if not output
                c2 = make_divisible(c2 * gw, 8)

            args = [c1, c2, *args[1:]]
            if m in [BottleneckCSP, C3, C3TR]:
                args.insert(2, n)  # number of repeats
                n = 1
        elif m is nn.BatchNorm2d:
            args = [ch[f]]
        elif m is Concat:
            c2 = sum([ch[x] for x in f])
        elif m is Add:
            c2 = ch[f[0]]
        elif m is Resizer:
            c2 = ch[f[0]] if isinstance(f, Iterable) else ch[f]
        elif m is Reconstructor:
            args.append([ch[x] for x in f])
        elif m in [HEstimator, HEstimatorOrigin]:
            args.append(ch[f])
        elif m is Contract:
            c2 = ch[f] * args[0] ** 2
        elif m is Expand:
            c2 = ch[f] // args[0] ** 2
        elif m is nn.Identity:
            c2 = [ch[x] for x in f] if isinstance(f, Iterable) else ch[f]
    
        else:
            c2 = ch[f]

        m_ = nn.Sequential(*[m(*args) for _ in range(n)]) if n > 1 else m(*args)  # module
        t = str(m)[8:-2].replace('__main__.', '')  # module type
        np = sum([x.numel() for x in m_.parameters()])  # number params
        m_.i, m_.f, m_.type, m_.np = i, f, t, np  # attach index, 'from' index, type, number params
        logger.info('%3s%18s%3s%10.0f  %-40s%-30s' % (i, f, n, np, t, args))  # print
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
        layers.append(m_)
        if i == 0:
            ch = []
        ch.append(c2)
    return nn.Sequential(*layers), sorted(save)

# models/yolo.py 里替换掉parse_model()

# def parse_model(d, ch):  # model_dict, input_channels(3)
#     """Parses the model.yaml file to create the model layers."""
#     layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out
#     for i, (f, n, m, args) in enumerate(d['backbone'] + d['head']):
#         m = eval(m) if isinstance(m, str) else m  # eval strings
#         for j, a in enumerate(args):
#             if isinstance(a, str):
#                 try:
#                     args[j] = eval(a)
#                 except:
#                     pass

#         n = max(round(n), 1) if n > 1 else n  # 保证至少1层

#         if not isinstance(f, list):
#             f = [f]

#         # 计算输入通道
#         ch_in = sum([ch[x] for x in f]) if m in [Concat] else ch[f[0]]

#         # 处理输出通道
#         if m.__name__ == 'Get':
#             ch_out = [64, 128, 256, 512][args[0]]  # 按Get的index选通道
#         elif m in [Focus2, Focus, Conv, Bottleneck, C3, nn.Conv2d]:
#             c1 = ch_in
#             c2 = args[0]
#             args = [c1, c2, *args[1:]]
#             ch_out = c2
#         elif m in [nn.Upsample, nn.AdaptiveAvgPool2d, nn.Hardswish, nn.ReLU, nn.SiLU]:
#             ch_out = ch_in
#         elif m in [Concat]:
#             ch_out = ch_in
#         elif m.__name__ == 'Reconstructor':
#             args.append([ch[x] for x in f])
#             ch_out = 0  # Reconstructor输出其实是图像，通道数单独算
#         else:
#             ch_out = ch_in

#         m_ = nn.Sequential(*[m(*args) for _ in range(n)]) if n > 1 else m(*args)  # module
#         t = str(m)[8:-2].replace('__main__.', '')  # module type
#         np = sum(x.numel() for x in m_.parameters())  # number params
#         m_.i, m_.f, m_.type, m_.np = i, f, t, np  # attach index, 'from' index, type, number params
#         layers.append(m_)
#         save.extend(x % i for x in f if x != -1)  # append to savelist
#         ch.append(ch_out)

#     return nn.Sequential(*layers), sorted(save)




# def parse_model(d, ch):  # model_dict, input_channels
#     import models.common as common  # your modules
#     import torch.nn as nn
#     from models.experimental import attempt_load

#     logger.info('\n%3s%18s%3s%10s  %-40s%-30s' % ('', 'from', 'n', 'params', 'module', 'arguments'))
#     anchors, nc, gd, gw = d.get('anchors', []), d['nc'], d['depth_multiple'], d['width_multiple']
#     na = (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors  # number of anchors
#     no = na * (nc + 5)  # number of outputs = anchors * (classes + 5)

#     layers, save, c2 = [], [], ch[-1]  # ch = [6]
    
#     # ✅ 手动指定 backbone 各阶段输出通道
#     #   0: PVT1 (out: 4个feature map)
#     #   1: Get(0) => P1: 64
#     #   2: Get(1) => P2: 128
#     #   3: Get(2) => P3: 320
#     #   4: Get(3) => P4: 512
#     ch = [6, 64, 128, 320, 512]  #  强制写死通道数，防止None报错

#     for i, (f, n, m, args) in enumerate(d['backbone'] + d['head']):
#         m = eval(m) if isinstance(m, str) else m  # eval string
#         for j, a in enumerate(args):
#             if isinstance(a, str) and a.isnumeric():
#                 args[j] = int(a)

#         n = max(round(n * gd), 1) if n > 1 else n
#         if m in [common.Conv, common.C3, common.Bottleneck, common.SPPF]:
#             c1 = ch[f] if isinstance(f, int) else sum([ch[x] for x in f])
#             c2 = args[0]
#             c2 = make_divisible(c2 * gw, 8) if c2 != no else c2
#             args = [c1, c2, *args[1:]]
#         elif m is nn.Upsample:
#             args = [dict(scale_factor=2, mode='nearest')] if len(args) == 0 else args
#         elif m is common.Concat:
#             c2 = sum([ch[x] for x in f])
#         elif m in [common.Reconstructor]:
#             c2 = 2  # 或其他你定义的输出通道

#         m_ = nn.Sequential(*[m(*args) for _ in range(n)]) if n > 1 else m(*args)

#         t = str(m)[8:-2].replace('__main__.', '')  # module type
#         np = sum(x.numel() for x in m_.parameters())  # number params
#         m_.i, m_.f, m_.type, m_.np = i, f, t, np  # attach index, from, type, params
#         logger.info('%3s%18s%3s%10.0f  %-40s%-30s' %
#                     (i, f, n, np, t, args))  # print

#         save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
#         layers.append(m_)
#         ch.append(c2)

#     return nn.Sequential(*layers), sorted(save)

# def parse_model(d, ch):  # model_dict, input_channels
#     import models.common as common
#     import torch.nn as nn
#     from models.experimental import attempt_load

#     logger.info('\n%3s%18s%3s%10s  %-40s%-30s' % ('', 'from', 'n', 'params', 'module', 'arguments'))

#     anchors, nc = d.get('anchors'), d.get('nc')
#     gd, gw = d.get('depth_multiple', 1.0), d.get('width_multiple', 1.0)
#     na = (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors
#     no = na * (nc + 5) if nc else None

#     layers, save = [], []

#     for i, (f, n, m, args) in enumerate(d['backbone'] + d['head']):
#         m = eval(m) if isinstance(m, str) else m

#         for j, a in enumerate(args):
#             if isinstance(a, str) and a.isnumeric():
#                 args[j] = int(a)

#         n = max(round(n * gd), 1) if n > 1 else n

#         if m in [common.Conv, common.C3, common.Bottleneck, common.Focus2]:
#             c1 = ch[f] if isinstance(f, int) else sum([ch[x] for x in f])
#             c2 = args[0]
#             c2 = make_divisible(c2 * gw, 8) if gw != 1.0 else c2
#             args = [c1, c2, *args[1:]]
#         elif m is nn.Upsample:
#             args = [dict(scale_factor=2, mode='nearest')] if len(args) == 0 else args
#         elif m is common.Concat:
#             c2 = sum([ch[x] for x in f])
#         elif m in [Reconstructor]:
#             c2 = 2
#         else:
#             c2 = ch[f] if isinstance(f, int) else ch[f[0]]

#         m_ = nn.Sequential(*[m(*args) for _ in range(n)]) if n > 1 else m(*args)

#         t = str(m)[8:-2].replace('__main__.', '')
#         np = sum(x.numel() for x in m_.parameters())
#         m_.i, m_.f, m_.type, m_.np = i, f, t, np
#         logger.info('%3s%18s%3s%10.0f  %-40s%-30s' % (i, f, n, np, t, args))

#         save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)
#         layers.append(m_)
#         ch.append(c2)  # 动态更新 ch

#     return nn.Sequential(*layers), sorted(save)




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='yolov5s.yaml', help='model.yaml')
    parser.add_argument('--mode', type=str, default='align', choices=['align', 'fuse'], help='model mode')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    opt = parser.parse_args()
    opt.cfg = check_file(opt.cfg)  # check file
    set_logging()
    device = select_device(opt.device)

    # Create model
    model = Model(opt.cfg, mode_align=opt.mode=='align').to(device)
    model.train()
    
    # TODO: replace the `dist-packages/torchsummary/torchsummary.py` with `./models/torchsummary.py`
    #       or apply the corresponding changes in line 20\34\116 of `./models/torchsummary.py` on your `torchsummary/torchsummary.py`
    from torchsummary import summary
    input_size = (8, 128, 128) if opt.mode=='align' else (8, 640, 640)
    summary(model, input_size)

    # Profile
    # img = torch.rand(1, *input_size).to(device)
    # y = model(img, profile=True)

    # Tensorboard (not working https://github.com/ultralytics/yolov5/issues/2898)
    # from torch.utils.tensorboard import SummaryWriter
    # tb_writer = SummaryWriter('.')
    # logger.info("Run 'tensorboard --logdir=models' to view tensorboard at http://localhost:6006/")
    # tb_writer.add_graph(torch.jit.trace(model, img, strict=False), [])  # add model graph
    # tb_writer.add_image('test', img[0], dataformats='CWH')  # add model to tensorboard
