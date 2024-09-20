_base_ = ['./mask2former_r50_8xb2-lsj-50e_coco_road.py']

# model = dict(
#     backbone=dict(
#         depth=101,
#         init_cfg=dict(type='Pretrained',
#                       checkpoint='torchvision://resnet101')))
model = dict(
    backbone=dict(
        type='ResNeXt',
        depth=101,
        groups=64,
        base_width=4,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        style='pytorch',
        init_cfg=dict(
            type='Pretrained', checkpoint='open-mmlab://resnext101_64x4d')),
    )
