# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.config import ConfigDict
import warnings
import copy
import torch
from torch import Tensor
from mmdet.structures import SampleList
import torch.nn as nn
from mmdet.registry import MODELS
from mmdet.utils import OptConfigType, OptMultiConfig
from .two_stage import TwoStageDetector
import torch.nn.functional as F
# from mmseg.models import FPNHead
import mmseg.models.decode_heads

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

# def cls_seg(self, feat):
#     if self.dropout is not None:
#         feat = self.dropout(feat)
#         output = self.conv_seg(feat)
#     # return torch.sigmoid(output)
#     return output

@MODELS.register_module()
class Mask_PointRCNN(TwoStageDetector):
    """Based on `Mask R-CNN <https://arxiv.org/abs/1703.06870>`_,在maskrcnn的基础上添加一个点提取分支，利用backbone的特征图做语义分割"""

    def __init__(self,
                 backbone: ConfigDict,
                 rpn_head: ConfigDict,
                 roi_head: ConfigDict,
                 train_cfg: ConfigDict,
                 test_cfg: ConfigDict,
                 neck: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__(
            backbone=backbone,
            neck=neck,
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            init_cfg=init_cfg,
            data_preprocessor=data_preprocessor)
        # from mmseg.models import FPNHead
        self.point_segmentation_branch = mmseg.models.decode_heads.FPNHead(feature_strides=[4, 8, 16, 32, 64],
                                                 in_channels=[256, 256, 256, 256, 256],
                                                 in_index=[0, 1, 2, 3, 4],
                                                 channels=128,
                                                #  dropout_ratio=0.1,
                                                 num_classes=1,
                                                 threshold=0.5,
                                                 norm_cfg=dict(type='BN', requires_grad=True),
                                                 align_corners=False,
                                                #  loss_decode=[dict(
                                                #       type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
                                                #       dict(type='DiceLoss', use_sigmoid=True,loss_weight=1.0)]
                                                 loss_decode = dict(type='FocalLoss', use_sigmoid=True,loss_weight=1.0)
                                                      )
        self.point_segmentation_branch._stack_batch_gt=_stack_batch_gt.__get__(self.point_segmentation_branch,
                                                                                    self.point_segmentation_branch.__class__)
        self.point_segmentation_branch.loss_by_feat=loss_by_feat.__get__(self.point_segmentation_branch,
                                                                                    self.point_segmentation_branch.__class__)
        # self.point_segmentation_branch.cls_seg=cls_seg.__get__(self.point_segmentation_branch,
        #                                                                  self.point_segmentation_branch.__class__)
    def _forward(self, batch_inputs: Tensor,
                 batch_data_samples: SampleList) -> tuple:
        """Network forward process. Usually includes backbone, neck and head
        forward without any post-processing.

        Args:
            batch_inputs (Tensor): Inputs with shape (N, C, H, W).
            batch_data_samples (list[:obj:`DetDataSample`]): Each item contains
                the meta information of each image and corresponding
                annotations.

        Returns:
            tuple: A tuple of features from ``rpn_head`` and ``roi_head``
            forward.
        """
        results = ()
        x = self.extract_feat(batch_inputs)
        
        if self.with_rpn:
            rpn_results_list = self.rpn_head.predict(
                x, batch_data_samples, rescale=False)
        else:
            assert batch_data_samples[0].get('proposals', None) is not None
            rpn_results_list = [
                data_sample.proposals for data_sample in batch_data_samples
            ]
        roi_outs = self.roi_head.forward(x, rpn_results_list,
                                         batch_data_samples)
        results = results + (roi_outs, )
        return results
    
    def loss(self, batch_inputs: Tensor,
             batch_data_samples: SampleList) -> dict:
        """Calculate losses from a batch of inputs and data samples.

        Args:
            batch_inputs (Tensor): Input images of shape (N, C, H, W).
                These should usually be mean centered and std scaled.
            batch_data_samples (List[:obj:`DetDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.

        Returns:
            dict: A dictionary of loss components
        """
        x = self.extract_feat(batch_inputs)

        losses = dict()
        # point 掩膜预测
        point_losses = self.point_segmentation_branch.loss(x, batch_data_samples,train_cfg=self.train_cfg)
        losses.update(point_losses)

        # RPN forward and loss
        if self.with_rpn:
            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                              self.test_cfg.rpn)
            rpn_data_samples = copy.deepcopy(batch_data_samples)
            # set cat_id of gt_labels to 0 in RPN
            for data_sample in rpn_data_samples:
                data_sample.gt_instances.labels = \
                    torch.zeros_like(data_sample.gt_instances.labels)

            rpn_losses, rpn_results_list = self.rpn_head.loss_and_predict(
                x, rpn_data_samples, proposal_cfg=proposal_cfg)
            # avoid get same name with roi_head loss
            keys = rpn_losses.keys()
            for key in list(keys):
                if 'loss' in key and 'rpn' not in key:
                    rpn_losses[f'rpn_{key}'] = rpn_losses.pop(key)
            losses.update(rpn_losses)
        else:
            assert batch_data_samples[0].get('proposals', None) is not None
            # use pre-defined proposals in InstanceData for the second stage
            # to extract ROI features.
            rpn_results_list = [
                data_sample.proposals for data_sample in batch_data_samples
            ]

        roi_losses = self.roi_head.loss(x, rpn_results_list,
                                        batch_data_samples)
        losses.update(roi_losses)

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
            list[:obj:`DetDataSample`]: Return the detection results of the
            input images. The returns value is DetDataSample,
            which usually contain 'pred_instances'. And the
            ``pred_instances`` usually contains following keys.

                - scores (Tensor): Classification scores, has a shape
                    (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape
                    (num_instances, ).
                - bboxes (Tensor): Has a shape (num_instances, 4),
                    the last dimension 4 arrange as (x1, y1, x2, y2).
                - masks (Tensor): Has a shape (num_instances, H, W).
        """

        assert self.with_bbox, 'Bbox head must be implemented.'
        x = self.extract_feat(batch_inputs)
        point_result = self.point_segmentation_branch.forward(x)
        point_result = F.interpolate(point_result, (400,400), mode='bilinear', align_corners=False)
        # 保存point结果
        from PIL import Image
        import cv2
        target = '/home/yry22/Vector/RoadSegment/mmdetection/work_dirs/exp4_18/point_out_vis/'
        import os
        os.makedirs(target,exist_ok=True)
        for i in range(0, point_result.shape[0]):
            predout = point_result[i].cpu()
            imgname = batch_data_samples[i].img_path.split('/')[-1][:-4]#names[i]
            Image.fromarray(cv2.resize((predout.numpy()).transpose(1,2,0),(400,400),interpolation=cv2.INTER_LINEAR)).save(target + imgname + '.tif')
        
        # If there are no pre-defined proposals, use RPN to get proposals
        if batch_data_samples[0].get('proposals', None) is None:
            rpn_results_list = self.rpn_head.predict(
                x, batch_data_samples, rescale=False)
        else:
            rpn_results_list = [
                data_sample.proposals for data_sample in batch_data_samples
            ]

        results_list = self.roi_head.predict(
            x, rpn_results_list, batch_data_samples, rescale=rescale)

        batch_data_samples = self.add_pred_to_datasample(
            batch_data_samples, results_list)
        return batch_data_samples    
