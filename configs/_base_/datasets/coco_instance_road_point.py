# dataset settings
dataset_type = 'CocoDataset'
data_root = 'data/city_scale/'

# Example to use different file client
# Method 1: simply set the data root and let the file I/O module
# automatically infer from prefix (not support LMDB and Memcache yet)

# data_root = 's3://openmmlab/datasets/detection/coco/'

# Method 2: Use `backend_args`, `file_client_args` in versions before 3.0.0rc6
# backend_args = dict(
#     backend='petrel',
#     path_mapping=dict({
#         './data/': 's3://openmmlab/datasets/detection/',
#         'data/': 's3://openmmlab/datasets/detection/'
#     }))
backend_args = None

albu_train_transforms = [
    dict(
        type = 'RandomRotate90',
        p=0.5),
    dict(
        type = 'RandomBrightnessContrast',
        p=0.5)
#    dict(
#        type = 'RandomResizedCrop',
#        height=800,
#        width=800,
#        scale=(0.8,1.0),
#        ratio=(0.7,1.3),
#        p=0.5)
]

train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotationswithPointMask', with_bbox=True, with_mask=True),
    dict(type='Resize', scale=(800, 800), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
#    dict(type='Albu',         
#        transforms=albu_train_transforms,         
#        bbox_params=dict(
#                        type='BboxParams',             
#                        format='pascal_voc',
#                        label_fields=['gt_bboxes_labels'],
#                        min_visibility=0.1)),
    dict(type='PackDetInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='Resize', scale=(800, 800), keep_ratio=True),
    # If you don't have a gt annotation, delete the pipeline
    # dict(type='LoadAnnotationswithPointMask', with_bbox=True, with_mask=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]
train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/instances_train2017.json',
        data_prefix=dict(img='train2017/'),
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=train_pipeline,
        backend_args=backend_args))
val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/instances_test2017.json',
        data_prefix=dict(img='test2017/'),
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=backend_args))
# test_dataloader = val_dataloader

val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'annotations/instances_test2017.json',
    metric=['bbox', 'segm'],
    format_only=False,
    backend_args=backend_args)
# test_evaluator = val_evaluator

# inference on test dataset and
# format the output results for submission.
test_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/instances_test2017.json',
        data_prefix=dict(img='test2017/'),
        # ------overpass 推理
        # data_root='/home/yry22/Vector/RoadSegment/mmdetection/data/overpass/',
        # ann_file='annotations/out.json',
        # data_prefix=dict(img='image/'),
        test_mode=True,
        pipeline=test_pipeline))
test_evaluator = dict(
    type='CocoMetric',
    metric=['bbox', 'segm'],
    format_only=True,
    ann_file=data_root + 'annotations/instances_test2017.json',
    outfile_prefix='./work_dirs/exp7_1/test_infer')
    # ann_file="/home/yry22/Vector/RoadSegment/mmdetection/data/overpass/annotations/out.json",
    # outfile_prefix='./work_dirs/exp5_4_2/overpass_inferout')