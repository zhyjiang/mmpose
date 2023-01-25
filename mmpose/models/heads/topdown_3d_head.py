# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
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
        self.final_dim = final_dim
        
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

    def get_loss(self, output, target, target_weight):
        """Calculate top-down keypoint loss.

        Note:
            - batch_size: N
            - num_keypoints: K

        Args:
            output (torch.Tensor[N,K,3]): Output keypoints.
            target (torch.Tensor[N,K,3]): Target keypoints.
            target_weight (torch.Tensor[N,K,1]):
                Weights across different joint types.
        """

        losses = dict()

        assert not isinstance(self.loss, nn.Sequential)
        assert target.dim() == 3 and target_weight.dim() == 3
        losses['Key3D_loss'] = self.loss(output, target, target_weight)

        return losses

    def get_mpjpe(self, output, target, target_weight):
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

    def forward(self, x, heatmap, img_metas):
        """Forward function."""
        bbox = self.get_bbox(img_metas)
        keypoint, keyIdx = self.get_keypoint(heatmap)
        
        posemb = torch.cat([keypoint, bbox], dim=2)
        posemb = self.posemb_layers(posemb)
        
        featemb = torch.zeros((heatmap.shape[0], heatmap.shape[1], self.in_channels)).cuda()
        for i in range(heatmap.shape[0]):
            for j in range(heatmap.shape[1]):
                featemb[i, j, :] = x[i, :, keyIdx[i, j, 1], keyIdx[i, j, 0]]
                
        keyemb = torch.cat([posemb, featemb], dim=2)
        keyemb = keyemb.permute(0, 2, 1)
        keyemb = self.mlp_layers(keyemb)
        keyemb = keyemb[:, :, :, None]
        keyemb = self.merge_layer(keyemb)
        keyemb = torch.flatten(keyemb, start_dim=1)
        key3d = self.final_layers(keyemb)
        key3d = key3d.view(-1, self.num_keypoints, 3)
        return key3d

    def get_bbox(self, img_metas):
        bbox = np.zeros((len(img_metas), 4))
        for i in range(len(img_metas)):
            bbox[i] = img_metas[i]['bbox']
            bbox[i, 0::2] /= img_metas[i]['image_width']
            bbox[i, 1::2] /= img_metas[i]['image_height'] 
            bbox[i, :2] = bbox[i, :2] * 2 - 1
        bbox = torch.FloatTensor(bbox).cuda()[:, None, :]
        bbox = bbox.repeat(1, self.num_keypoints, 1)
        return bbox

    def get_keypoint(self, heatmap):
        heatmap = torch.flatten(heatmap, start_dim=2)
        keypoint = torch.zeros((heatmap.shape[0], heatmap.shape[1], 2)).cuda().long()
        key_loc = torch.argmax(heatmap, dim=2).long()
        keypoint[:, :, 0] = key_loc % self.train_cfg['heatmap_size'][1]
        keypoint[:, :, 1] = key_loc // self.train_cfg['heatmap_size'][1]
        
        keyIdx = keypoint.cpu().numpy()
        keypoint[:, :, 0] = keypoint[:, :, 0] / self.train_cfg['heatmap_size'][1] * 2 - 1
        keypoint[:, :, 1] = keypoint[:, :, 1] / self.train_cfg['heatmap_size'][0] * 2 - 1
        return keypoint, keyIdx

    def inference_model(self, x, heatmap, img_metas):
        """Inference function.

        Returns:
            output_key3d (np.ndarray): Output keypoints.

        Args:
            x (torch.Tensor[N,C,H,W]): Input features.
            heatmap (torch.Tensor[N,K,H,W]): Heatmap from the 2D head.
            img_metas (List): Label info.
        """
        output = self.forward(x, heatmap, img_metas)
        output_key3d = output.cpu().numpy()
        return output_key3d
    
    def decode(self, img_metas, output):
        """Decode the keypoints from output regression.

        Args:
            metas (list(dict)): Information about data augmentation.
                By default this includes:

                - "target_image_path": path to the image file
            output (np.ndarray[N, K, 3]): predicted regression vector.
            metas (list(dict)): Information about data augmentation including:

                - target_image_path (str): Optional, path to the image file
                - target_mean (float): Optional, normalization parameter of
                    the target pose.
                - target_std (float): Optional, normalization parameter of the
                    target pose.
                - root_position (np.ndarray[3,1]): Optional, global
                    position of the root joint.
                - root_index (torch.ndarray[1,]): Optional, original index of
                    the root joint before root-centering.
        """
        # Denormalize the predicted pose
        if 'target_mean' in img_metas[0] and 'target_std' in img_metas[0]:
            target_mean = np.stack([m['target_mean'] for m in img_metas])
            target_std = np.stack([m['target_std'] for m in img_metas])
            output = self._denormalize_joints(output, target_mean, target_std)

        target_image_paths = [m.get('target_image_path', None) for m in img_metas]
        result = {'preds': output, 'target_image_paths': target_image_paths}

        return result

    def init_weights(self):
        """Initialize model weights."""
        for _, m in self.posemb_layers.named_modules():
            if isinstance(m, nn.Linear):
                normal_init(m, std=0.001)
            elif isinstance(m, nn.BatchNorm2d):
                constant_init(m, 1)
        for m in self.merge_layer.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.001, bias=0)
            elif isinstance(m, nn.BatchNorm2d):
                constant_init(m, 1)
        for m in self.mlp_layers.modules():
            if isinstance(m, nn.Conv1d):
                normal_init(m, std=0.001, bias=0)
            elif isinstance(m, nn.BatchNorm2d):
                constant_init(m, 1)
        for m in self.final_layers.modules():
            if isinstance(m, nn.Linear):
                normal_init(m, std=0.001, bias=0)
            elif isinstance(m, nn.BatchNorm2d):
                constant_init(m, 1)
