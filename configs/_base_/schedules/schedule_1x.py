# training schedule for 1x
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=12, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# learning rate
param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=500),
    dict(
        type='MultiStepLR',
        begin=0,
        end=12,
        by_epoch=True,
        milestones=[9, 11],
        gamma=0.1)
    # dict(
    #     type='PolyLR',
    #     power=0.9,
    #     eta_min=1e-4,
    #     begin=0,
    #     end=12,
    #     by_epoch=True
    # )
    # dict(type='CosineAnnealingLR',
    #      T_max=12,
    #      eta_min=1e-4,
    #      begin=0,
    #      end=12,
    #      by_epoch=True)
]

# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
#    optimizer=dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001))
    optimizer=dict(
        type='AdamW',
        lr=0.0001,
        weight_decay=0.0002,
        eps=1e-8,
        betas=(0.9, 0.999)))
# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (8 GPUs) x (2 samples per GPU).
auto_scale_lr = dict(enable=False, base_batch_size=16)
