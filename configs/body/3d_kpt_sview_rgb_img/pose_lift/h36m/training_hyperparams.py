_base_ = [
    '../../../../_base_/default_runtime.py',
    '../../../../_base_/datasets/h36m.py',
    './hrnet_end2end_3d_2branch.py',
]

data = dict(
    samples_per_gpu=128,
)

lr_config = dict(
    policy='step',
    by_epoch=False,
    step=[30,40],
    gamma=0.96,
)

total_epochs = 50

optimizer = dict(
    type='Adam',
    lr=5e-4,
)