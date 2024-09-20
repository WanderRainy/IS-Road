# Copyright (c) OpenMMLab. All rights reserved.
from mmdet.registry import MODELS
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig
from .maskformer import MaskFormer
import mmseg.models.decode_heads
from mmengine.config import ConfigDict
import warnings
import copy
import torch
from torch import Tensor
from mmdet.structures import SampleList
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict

def resize(input,            size=None,            scale_factor=None,            mode='nearest',            align_corners=None,            warning=True):     
    if warning:         
        if size is not None and align_corners:             
            input_h, input_w = tuple(int(x) for x in input.shape[2:])             
            output_h, output_w = tuple(int(x) for x in size)             
            if output_h > input_h or output_w > output_h:                 
                if ((output_h > 1 and output_w > 1 and input_h > 1                      
                     and input_w > 1) and (output_h - 1) % (input_h - 1)                         
                        and (output_w - 1) % (input_w - 1)):                     
                     warnings.warn(                         
                         f'When align_corners={align_corners}, '                         
                         'the output would more aligned if '                         
                         f'input size {(input_h, input_w)} is `x+1` and '                         
                         f'out size {(output_h, output_w)} is `nx+1`')     
    return F.interpolate(input, size, scale_factor, mode, align_corners)
def _stack_batch_gt(self, batch_data_samples: SampleList) -> Tensor:
    gt_semantic_segs = [data_sample.gt_sem_seg.sem_seg.data for data_sample in batch_data_samples]
    return torch.stack(gt_semantic_segs, dim=0)
def loss_by_feat(self, seg_logits: Tensor,
                    batch_data_samples: SampleList) -> dict:
                    """Compute segmentation loss.
                    Args:
                    seg_logits (Tensor): The output from decode head forward function.
                    batch_data_samples (List[:obj:`SegDataSample`]): The seg
                    data samples. It usually includes information such
                    as `metainfo` and `gt_sem_seg`.
                    Returns:
                    dict[str, Tensor]: a dictionary of loss components
                    """
                    seg_label = self._stack_batch_gt(batch_data_samples)
                    loss = dict()
                    seg_logits = resize(
                        input=seg_logits,
                        size=seg_label.shape[2:],
                        mode='bilinear',
                        align_corners=self.align_corners)
                    if self.sampler is not None:
                        seg_weight = self.sampler.sample(seg_logits, seg_label)
                    else:
                        seg_weight = None
                    seg_label = seg_label.squeeze(1)
                    if not isinstance(self.loss_decode, nn.ModuleList):
                        losses_decode = [self.loss_decode]
                    else:
                        losses_decode = self.loss_decode
                    for loss_decode in losses_decode:
                        if len(loss)==0:
                            loss['loss_point_Focal'] = loss_decode(
                                 seg_logits.squeeze(1),
                                 seg_label)
                        else:
                            loss['loss_point_Dice'] = loss_decode(
                                 seg_logits.squeeze(1),
                                 seg_label)
                    # single loss        
                    # loss['loss_point'] = self.loss_decode(
                    #     seg_logits.squeeze(1),
                    #     seg_label)
                #  loss['acc_seg'] = accuracy(
                #              seg_logits, seg_label, ignore_index=self.ignore_index)
                    return loss



@MODELS.register_module()
class Mask2Former_Point(MaskFormer):
    r"""Implementation of `Masked-attention Mask
    Transformer for Universal Image Segmentation
    <https://arxiv.org/pdf/2112.01527>`_."""

    def __init__(self,
                 backbone: ConfigType,
                 neck: OptConfigType = None,
                 panoptic_head: OptConfigType = None,
                 panoptic_fusion_head: OptConfigType = None,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None):
        super().__init__(
            backbone=backbone,
            neck=neck,
            panoptic_head=panoptic_head,
            panoptic_fusion_head=panoptic_fusion_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            data_preprocessor=data_preprocessor,
            init_cfg=init_cfg)
        self.point_segmentation_branch = mmseg.models.decode_heads.FPNHead(feature_strides=[4, 8, 16, 32],
                                                 in_channels=[192, 384, 768, 1536],#[192, 384, 768, 1536],#[256, 512, 1024, 2048],
                                                 in_index=[0, 1, 2, 3],
                                                 channels=512,
                                                #  dropout_ratio=0.1,
                                                 num_classes=1,
                                                 threshold=0.5,
                                                 norm_cfg=dict(type='BN', requires_grad=True),
                                                 align_corners=False,
                                                #  loss_decode=[dict(
                                                #       type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
                                                #       dict(type='DiceLoss', use_sigmoid=True,loss_weight=1.0)]
                                                #  loss_decode = dict(type='FocalLoss', use_sigmoid=True,loss_weight=1.0)
                                                loss_decode = dict(type='CrossEntropyLoss', use_sigmoid=True,loss_weight=1.0)
                                                # loss_decode = dict(type='MSELoss',loss_weight=10.0)
                                                      )
        self.point_segmentation_branch._stack_batch_gt=_stack_batch_gt.__get__(self.point_segmentation_branch,
                                                                                    self.point_segmentation_branch.__class__)
        self.point_segmentation_branch.loss_by_feat=loss_by_feat.__get__(self.point_segmentation_branch,
                                                                                    self.point_segmentation_branch.__class__)
    def loss(self, batch_inputs: Tensor,
             batch_data_samples: SampleList) -> Dict[str, Tensor]:
        """
        Args:
            batch_inputs (Tensor): Input images of shape (N, C, H, W).
                These should usually be mean centered and std scaled.
            batch_data_samples (list[:obj:`DetDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        x = self.extract_feat(batch_inputs)
        point_losses = self.point_segmentation_branch.loss(x, batch_data_samples,train_cfg=self.train_cfg)
        losses = self.panoptic_head.loss(x, batch_data_samples)
        losses.update(point_losses)
        return losses

    def predict(self,
                batch_inputs: Tensor,
                batch_data_samples: SampleList,
                rescale: bool = True) -> SampleList:
        """Predict results from a batch of inputs and data samples with post-
        processing.

        Args:
            batch_inputs (Tensor): Inputs with shape (N, C, H, W).
            batch_data_samples (List[:obj:`DetDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance`, `gt_panoptic_seg` and `gt_sem_seg`.
            rescale (bool): Whether to rescale the results.
                Defaults to True.

        Returns:
            list[:obj:`DetDataSample`]: Detection results of the
            input images. Each DetDataSample usually contain
            'pred_instances' and `pred_panoptic_seg`. And the
            ``pred_instances`` usually contains following keys.

                - scores (Tensor): Classification scores, has a shape
                    (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape
                    (num_instances, ).
                - bboxes (Tensor): Has a shape (num_instances, 4),
                    the last dimension 4 arrange as (x1, y1, x2, y2).
                - masks (Tensor): Has a shape (num_instances, H, W).

            And the ``pred_panoptic_seg`` contains the following key

                - sem_seg (Tensor): panoptic segmentation mask, has a
                    shape (1, h, w).
        """
        feats = self.extract_feat(batch_inputs)

        point_result = self.point_segmentation_branch.forward(feats)
        point_result = F.interpolate(torch.sigmoid(point_result), (400,400), mode='bilinear', align_corners=False)
        # 保存point结果
        from PIL import Image
        import cv2
        target = '/data1/yry22/Vector/RoadSegment/mmdetection/work_dirs/exp9_4/point_out_vis/'
        import os
        os.makedirs(target,exist_ok=True)
        for i in range(0, point_result.shape[0]):
            predout = point_result[i].cpu()
            imgname = batch_data_samples[i].img_path.split('/')[-1][:-4]#names[i]
            Image.fromarray(cv2.resize((predout.numpy()).transpose(1,2,0),(400,400),interpolation=cv2.INTER_LINEAR)).save(target + imgname + '.tif')
        
        mask_cls_results, mask_pred_results = self.panoptic_head.predict(
            feats, batch_data_samples)
        results_list = self.panoptic_fusion_head.predict(
            mask_cls_results,
            mask_pred_results,
            batch_data_samples,
            rescale=rescale)
        results = self.add_pred_to_datasample(batch_data_samples, results_list)

        return results
