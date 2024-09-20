_base_ = [
    '../_base_/models/cascade-mask-rcnn_r50_fpn_road.py',
    '../_base_/datasets/coco_instance_road_point.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]
model = dict(
    backbone=dict(
        type='ResNet',
        depth=101,
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet101')))