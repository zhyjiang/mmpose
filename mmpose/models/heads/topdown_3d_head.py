# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcv.cnn import (build_conv_layer, build_norm_layer, build_upsample_layer,
                      constant_init, normal_init)

from mmpose.core.evaluation import keypoint_mpjpe
from mmpose.core.post_processing import flip_back
from mmpose.models.builder import build_loss
from mmpose.models.utils.ops import resize
from ..builder import HEADS
from .topdown_heatmap_base_head import TopdownHeatmapBaseHead


@HEADS.register_module()
class Topdown3DHead(nn.Module):
    """Top-down heatmap simple head. paper ref: Bin Xiao et al. ``Simple
    Baselines for Human Pose Estimation and Tracking``.

    TopdownHeatmapSimpleHead is consisted of (>=0) number of deconv layers23
    and a simple conv2d layer.

    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        num_deconv_layers (int): Number of deconv layers.
            num_deconv_layers should >= 0. Note that 0 means
            no deconv layers.
        num_deconv_filters (list|tuple): Number of filters.
            If num_deconv_layers > 0, the length of
        num_deconv_kernels (list|tuple): Kernel sizes.
        in_index (int|Sequence[int]): Input feature index. Default: 0
        input_transform (str|None): Transformation type of input features.
            Options: 'resize_concat', 'multiple_select', None.
            Default: None.

            - 'resize_concat': Multiple feature maps will be resized to the
                same size as the first one and then concat together.
                Usually used in FCN head of HRNet.
            - 'multiple_select': Multiple feature maps will be bundle into
                a list and passed into decode head.
            - None: Only one select feature map is allowed.
        align_corners (bool): align_corners argument of F.interpolate.
            Default: False.
        loss_keypoint (dict): Config for keypoint loss. Default: None.
    """

    def __init__(self,
                 in_channels,
                 num_keypoints=17,
                 posemb_dim=[64],
                 mlp_dim=[128, 256],
                 final_dim=[1024, 512],
                 loss_keypoint=None,
                 train_cfg=None,
                 test_cfg=None):
        super().__init__()

        self.in_channels = in_channels
        self.loss = build_loss(loss_keypoint)

        self.train_cfg = {} if train_cfg is None else train_cfg
        self.test_cfg = {} if test_cfg is None else test_cfg
        
        self.num_keypoints = num_keypoints
        self.posemb_dim = posemb_dim
        self.mlp_dim = mlp_dim
        
        posemb_layers = []
        for lid in range(len(self.posemb_dim)):
            if lid == 0:
                posemb_layers.append(nn.Linear(6, self.posemb_dim[lid]))
            else:
                posemb_layers.append(nn.Linear(self.posemb_dim[lid-1], self.posemb_dim[lid]))
        if len(posemb_layers) > 1:
            self.posemb_layers = nn.Sequential(*posemb_layers)
        else:
            self.posemb_layers = posemb_layers[0]
                
        mlp_layers = []
        for lid in range(len(self.mlp_dim)):
            if lid == 0:
                mlp_layers.append(nn.Conv1d(in_channels=in_channels + self.posemb_dim[-1], 
                                            out_channels=self.mlp_dim[lid],
                                            kernel_size=1))
            else:
                mlp_layers.append(nn.Conv1d(in_channels=self.mlp_dim[lid-1], 
                                            out_channels=self.mlp_dim[lid],
                                            kernel_size=1))
        if len(mlp_layers) > 1:
            self.mlp_layers = nn.Sequential(*mlp_layers)
        else:
            self.mlp_layers = mlp_layers[0]
                
        self.merge_layer = nn.Conv2d(in_channels=self.mlp_dim[-1],
                                     out_channels=self.mlp_dim[-1],
                                     kernel_size=(self.num_keypoints, 1))
        
        final_layers = []
        for lid in range(len(self.final_dim)):
            if lid == 0:
                final_layers.append(nn.Linear(self.mlp_dim[-1], self.final_dim[lid]))
            else:
                final_layers.append(nn.Linear(self.final_dim[lid-1], self.final_dim[lid]))
        final_layers.append(nn.Linear(self.final_dim[-1], self.num_keypoints * 3))
        self.final_layers = nn.Sequential(*final_layers)

    def get_loss(self, output, target):
        """Calculate top-down keypoint loss.

        Note:
            - batch_size: N
            - num_keypoints: K

        Args:
            output (torch.Tensor[N,K,3]): Output keypoints.
            target (torch.Tensor[N,K,3]): Target keypoints.
        """

        losses = dict()

        assert not isinstance(self.loss, nn.Sequential)
        assert target.dim() == 4 and target_weight.dim() == 3
        losses['Key3D_loss'] = self.loss(output, target)

        return losses

    def get_mpjpe(self, output, target):
        """Calculate MPJPE.

        Note:
            - batch_size: N
            - num_keypoints: K

        Args:
            output (torch.Tensor[N,K,3]): Output keypoints.
            target (torch.Tensor[N,K,3]): Target keypoints.
        """

        mpjpe = dict()

        _, avg_acc, _ = keypoint_mpjpe(
            output.detach().cpu().numpy(),
            target.detach().cpu().numpy(),
            target_weight.detach().cpu().numpy().squeeze(-1) > 0)
        mpjpe['mpjpe'] = float(avg_acc)

        return mpjpe

    def forward(self, x):
        """Forward function."""
        x = self._transform_inputs(x)
        x = self.deconv_layers(x)
        x = self.final_layer(x)
        return x

    def inference_model(self, x, flip_pairs=None):
        """Inference function.

        Returns:
            output_heatmap (np.ndarray): Output heatmaps.

        Args:
            x (torch.Tensor[N,K,H,W]): Input features.
            flip_pairs (None | list[tuple]):
                Pairs of keypoints which are mirrored.
        """
        output = self.forward(x)

        if flip_pairs is not None:
            output_heatmap = flip_back(
                output.detach().cpu().numpy(),
                flip_pairs,
                target_type=self.target_type)
            # feature is not aligned, shift flipped heatmap for higher accuracy
            if self.test_cfg.get('shift_heatmap', False):
                output_heatmap[:, :, :, 1:] = output_heatmap[:, :, :, :-1]
        else:
            output_heatmap = output.detach().cpu().numpy()
        return output_heatmap

    def _make_deconv_layer(self, num_layers, num_filters, num_kernels):
        """Make deconv layers."""
        if num_layers != len(num_filters):
            error_msg = f'num_layers({num_layers}) ' \
                        f'!= length of num_filters({len(num_filters)})'
            raise ValueError(error_msg)
        if num_layers != len(num_kernels):
            error_msg = f'num_layers({num_layers}) ' \
                        f'!= length of num_kernels({len(num_kernels)})'
            raise ValueError(error_msg)

        layers = []
        for i in range(num_layers):
            kernel, padding, output_padding = \
                self._get_deconv_cfg(num_kernels[i])

            planes = num_filters[i]
            layers.append(
                build_upsample_layer(
                    dict(type='deconv'),
                    in_channels=self.in_channels,
                    out_channels=planes,
                    kernel_size=kernel,
                    stride=2,
                    padding=padding,
                    output_padding=output_padding,
                    bias=False))
            layers.append(nn.BatchNorm2d(planes))
            layers.append(nn.ReLU(inplace=True))
            self.in_channels = planes

        return nn.Sequential(*layers)

    def init_weights(self):
        """Initialize model weights."""
        for _, m in self.posemb_layers.named_modules():
            if isinstance(m, nn.Linear):
                normal_init(m, std=0.001)
            elif isinstance(m, nn.BatchNorm2d):
                constant_init(m, 1)
        for m in self.final_layer.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.001, bias=0)
            elif isinstance(m, nn.BatchNorm2d):
                constant_init(m, 1)
