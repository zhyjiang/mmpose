_base_ = [
    '../../../../_base_/default_runtime.py',
    '../../../../_base_/datasets/h36m.py'
]
evaluation = dict(interval=4, metric=['mpjpe', 'p-mpjpe'], save_best='MPJPE')

checkpoint_config = dict(interval=10)

log_config = dict(
    interval=100,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
        # dict(type='PaviLoggerHook') # for internal services
    ])

# optimizer settings
optimizer = dict(
    type='Adam',
    lr=1e-4,
)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='exp',
    by_epoch=True,
    gamma=0.98,
)

total_epochs = 200

channel_cfg = dict(
    num_output_channels=17,
    dataset_joints=17,
    dataset_channel=[
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
    ],
    inference_channel=[
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16
    ])

# model settings
load_from = 'checkpoint/best_MPJPE_epoch_72.pth'
model = dict(
    type='TopDown3D',
    img_inference=False,
    pretrained=None,
    backbone=dict(
        type='HRNet',
        in_channels=3,
        extra=dict(
            stage1=dict(
                num_modules=1,
                num_branches=1,
                block='BOTTLENECK',
                num_blocks=(4, ),
                num_channels=(64, )),
            stage2=dict(
                num_modules=1,
                num_branches=2,
                block='BASIC',
                num_blocks=(4, 4),
                num_channels=(32, 64)),
            stage3=dict(
                num_modules=4,
                num_branches=3,
                block='BASIC',
                num_blocks=(4, 4, 4),
                num_channels=(32, 64, 128)),
            stage4=dict(
                num_modules=3,
                num_branches=4,
                block='BASIC',
                num_blocks=(4, 4, 4, 4),
                num_channels=(32, 64, 128, 256),
                multiscale_output=True)),
    ),
    keypoint_head=dict(
        type='TopdownHeatmapSimpleHead',
        in_channels=32,
        out_channels=channel_cfg['num_output_channels'],
        num_deconv_layers=0,
        extra=dict(final_conv_kernel=1, ),
        loss_keypoint=dict(type='JointsMSELoss', use_target_weight=True, loss_weight=0)),
    fix_keypoint_head=True,
    keypoint3d_head=dict(
        type='Topdown3DLateFuseHead',
        in_channels=32,
        posemb_dim=[1024, 1024],
        imgfeat_dim=[512, 512, 256],
        final_dim=[1024, 1024],
        extra=dict(
            global_feat_size=[8, 8],
            global_feat_dim=256,
            pose_branch_dim=[512, 256]
        ),
        loss_keypoint=dict(type='MPJPELoss', use_target_weight=True)
    ),
    train_cfg=dict(
        heatmap_size=[64, 64]),
    test_cfg=dict(
        heatmap_size=[64, 64],
        flip_test=False,
        post_process='default',
        restore_global_position=True,
        shift_heatmap=True,
        modulate_kernel=11))

# data settings
data_root = 'data/h36m'
data_cfg = dict(
    image_size=[256, 256],
    heatmap_size=[64, 64],
    num_joints=17,
    seq_len=1,
    seq_frame_interval=1,
    causal=True,
    joint_2d_src='gt',
    need_camera_param=True,
    camera_param_file=f'{data_root}/annotation_body3d/cameras.pkl',
)

# 3D joint normalization parameters
# From file: '{data_root}/annotation_body3d/fps10/joint3d_rel_stats.pkl'
joint_3d_normalize_param = dict(
    mean=[[0, 0, 0] for i in range(17)],
    std=[[1, 1, 1] if i == 0 else [1, 1, 1] for i in range(17)])

# 2D joint normalization parameters
# From file: '{data_root}/annotation_body3d/fps10/joint2d_stats.pkl'
joint_2d_normalize_param = dict(
    mean=[[532.08351635, 419.74137558], [531.80953144, 418.2607141],
          [530.68456967, 493.54259285], [529.36968722, 575.96448516],
          [532.29767646, 421.28483336], [531.93946631, 494.72186795],
          [529.71984447, 578.96110365], [532.93699382, 370.65225054],
          [534.1101856, 317.90342311], [534.55416813, 304.24143901],
          [534.86955004, 282.31030885], [534.11308566, 330.11296796],
          [533.53637525, 376.2742511], [533.49380107, 391.72324565],
          [533.52579142, 330.09494668], [532.50804964, 374.190479],
          [532.72786934, 380.61615716]],
    std=[[107.73640054, 63.35908715], [119.00836213, 64.1215443],
         [119.12412107, 50.53806215], [120.61688045, 56.38444891],
         [101.95735275, 62.89636486], [106.24832897, 48.41178119],
         [108.46734966, 54.58177071], [109.07369806, 68.70443672],
         [111.20130351, 74.87287863], [111.63203838, 77.80542514],
         [113.22330788, 79.90670556], [105.7145833, 73.27049436],
         [107.05804267, 73.93175781], [107.97449418, 83.30391802],
         [121.60675105, 74.25691526], [134.34378973, 77.48125087],
         [131.79990652, 89.86721124]])

train_pipeline = [
    # dict(type='LoadImageFromFile'),
    # dict(type='TopDownGetBboxCenterScale', padding=1.25),
    # dict(type='TopDownRandomFlip', flip_prob=0.5),
    # dict(type='TopDownGetRandomScaleRotation', rot_factor=0, scale_factor=0),
    # dict(type='TopDownAffine'),
    # dict(type='ToTensor'),
    # dict(
    #     type='NormalizeTensor',
    #     mean=[0.485, 0.456, 0.406],
    #     std=[0.229, 0.224, 0.225]),
    # dict(type='TopDownGenerateTarget', sigma=2),
    dict(
        type='GetRootCenteredPose',
        item='target_3d',
        visible_item='target_visible',
        root_index=0,
        root_name='root_position'),
    dict(
        type='NormalizeJointCoordinate',
        item='target_3d',
        mean=joint_3d_normalize_param['mean'],
        std=joint_3d_normalize_param['std']),
    # dict(
    #     type='Joint3DFlip',
    #     item=['target_3d'],
    #     flip_cfg=[
    #         dict(center_mode='static', center_x=0.)
    #     ],
    #     visible_item=['target_weight']),
    dict(
        type='Collect',
        keys=['target_3d'],
        meta_keys=[
            'target_image_path', 'flip_pairs', 'root_position',
            'root_position_index', 'target_3d_mean', 'target_3d_std',
            'bbox', 'ann_info', 'image_width', 'image_height',
            'center', 'scale', 'image_file', 'input_2d'
        ])
]

val_pipeline = [
    # dict(type='LoadImageFromFile'),
    # dict(type='TopDownGetBboxCenterScale', padding=1.25),
    # dict(type='TopDownGetRandomScaleRotation', rot_factor=0, scale_factor=0),
    # dict(type='TopDownAffine'),
    # dict(type='ToTensor'),
    # dict(
    #     type='NormalizeTensor',
    #     mean=[0.485, 0.456, 0.406],
    #     std=[0.229, 0.224, 0.225]),
    dict(
        type='GetRootCenteredPose',
        item='target_3d',
        visible_item='target_visible',
        root_index=0,
        root_name='root_position'),
    dict(
        type='NormalizeJointCoordinate',
        item='target_3d',
        mean=joint_3d_normalize_param['mean'],
        std=joint_3d_normalize_param['std']),
    dict(
        type='Collect',
        keys=['target_3d'],
        meta_keys=[
            'target_image_path', 'flip_pairs', 'root_position',
            'root_position_index', 'target_3d_mean', 'target_3d_std',
            'bbox', 'ann_info', 'image_width', 'image_height',
            'center', 'scale', 'image_file', 'input_2d'
        ])
]
test_pipeline = [
    # dict(type='LoadImageFromFile'),
    # dict(type='TopDownGetBboxCenterScale', padding=1.25),
    # dict(type='TopDownGetRandomScaleRotation', rot_factor=0, scale_factor=0),
    # dict(type='TopDownAffine'),
    # dict(type='ToTensor'),
    # dict(
    #     type='NormalizeTensor',
    #     mean=[0.485, 0.456, 0.406],
    #     std=[0.229, 0.224, 0.225]),
    dict(
        type='GetRootCenteredPose',
        item='target_3d',
        visible_item='target_visible',
        root_index=0,
        root_name='root_position'),
    dict(
        type='NormalizeJointCoordinate',
        item='target_3d',
        mean=joint_3d_normalize_param['mean'],
        std=joint_3d_normalize_param['std']),
    dict(
        type='Collect',
        keys=['target_3d'],
        meta_keys=[
            'target_image_path', 'flip_pairs', 'root_position',
            'root_position_index', 'target_3d_mean', 'target_3d_std',
            'bbox', 'ann_info', 'image_width', 'image_height',
            'center', 'scale', 'image_file', 'input_2d'
        ])
]

data = dict(
    samples_per_gpu=512,
    workers_per_gpu=32,
    val_dataloader=dict(samples_per_gpu=24),
    test_dataloader=dict(samples_per_gpu=48),
    train=dict(
        type='Body3DSViewH36MDataset',
        ann_file=f'{data_root}/annotation_body3d/fps10/h36m_train.npz',
        img_prefix=f'{data_root}/images/',
        data_cfg=data_cfg,
        pipeline=train_pipeline,
        dataset_info={{_base_.dataset_info}}),
    val=dict(
        type='Body3DSViewH36MDataset',
        ann_file=f'{data_root}/annotation_body3d/fps10/h36m_test.npz',
        img_prefix=f'{data_root}/images/',
        data_cfg=data_cfg,
        pipeline=val_pipeline,
        dataset_info={{_base_.dataset_info}}),
    test=dict(
        type='Body3DSViewH36MDataset',
        ann_file=f'{data_root}/annotation_body3d/fps10/h36m_test.npz',
        img_prefix=f'{data_root}/images/',
        data_cfg=data_cfg,
        pipeline=test_pipeline,
        dataset_info={{_base_.dataset_info}}),
)
