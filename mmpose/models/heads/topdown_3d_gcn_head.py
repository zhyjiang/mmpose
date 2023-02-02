# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch
import torch.nn as nn
from mmcv.cnn import (build_conv_layer, build_norm_layer, build_upsample_layer,
                      constant_init, normal_init)

from mmpose.core import compute_similarity_transform
from mmpose.core.evaluation import keypoint_mpjpe
from mmpose.core.post_processing import flip_back
from mmpose.models.builder import build_loss
from mmpose.models.utils.ops import resize
from ..builder import HEADS
from .topdown_heatmap_base_head import TopdownHeatmapBaseHead
from ..necks.sem_gcn import SemGCN


@HEADS.register_module()
class Topdown3DGCNHead(nn.Module):
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
                 hid_dim=128,
                 num_layers=4,
                 p_dropout=0.15,
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
                
        self.sem_gcn = SemGCN(self.num_keypoints, hid_dim, coords_dim=(posemb_dim[-1] + in_channels, 3), 
                              num_layers=num_layers, p_dropout=p_dropout)

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
        key3d = self.sem_gcn(keyemb)
        return key3d

    def get_accuracy(self, output, target, target_weight, metas):
        """Calculate accuracy for keypoint loss.

        Note:
            - batch_size: N
            - num_keypoints: K

        Args:
            output (torch.Tensor[N, K, 3]): Output keypoints.
            target (torch.Tensor[N, K, 3]): Target keypoints.
            target_weight (torch.Tensor[N, K, 3]):
                Weights across different joint types.
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
        
        accuracy = dict()
        
        N = output.shape[0]
        output_ = output.detach().cpu().numpy()
        target_ = target.detach().cpu().numpy()
        # Denormalize the predicted pose
        if 'target_mean' in metas[0] and 'target_std' in metas[0]:
            target_mean = np.stack([m['target_mean'] for m in metas])
            target_std = np.stack([m['target_std'] for m in metas])
            output_ = self._denormalize_joints(output_, target_mean,
                                               target_std)
            target_ = self._denormalize_joints(target_, target_mean,
                                               target_std)

        # Restore global position
        if self.test_cfg.get('restore_global_position', False):
            root_pos = np.stack([m['root_position'] for m in metas])
            root_idx = metas[0].get('root_position_index', None)
            output_ = self._restore_global_position(output_, root_pos,
                                                    root_idx)
            target_ = self._restore_global_position(target_, root_pos,
                                                    root_idx)
        # Get target weight
        if target_weight is None:
            target_weight_ = np.ones_like(target_)
        else:
            target_weight_ = target_weight.detach().cpu().numpy()
            if self.test_cfg.get('restore_global_position', False):
                root_idx = metas[0].get('root_position_index', None)
                root_weight = metas[0].get('root_joint_weight', 1.0)
                target_weight_ = self._restore_root_target_weight(
                    target_weight_, root_weight, root_idx)

        mpjpe = np.mean(
            np.linalg.norm((output_ - target_) * target_weight_, axis=-1))

        transformed_output = np.zeros_like(output_)
        for i in range(N):
            transformed_output[i, :, :] = compute_similarity_transform(
                output_[i, :, :], target_[i, :, :])
        p_mpjpe = np.mean(
            np.linalg.norm(
                (transformed_output - target_) * target_weight_, axis=-1))

        accuracy['mpjpe'] = output.new_tensor(mpjpe)
        accuracy['p_mpjpe'] = output.new_tensor(p_mpjpe)

        return accuracy

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
        
        output = output[:, 1:, :]
        if self.test_cfg.get('restore_global_position', False):
            output = output[:, 1:, :]
            root_pos = np.stack([m['root_position'] for m in img_metas])
            root_idx = img_metas[0].get('root_position_index', None)
            output = self._restore_global_position(output, root_pos, root_idx)
        else:
            output = output + output[:, img_metas[0]['root_position_index']:img_metas[0]['root_position_index'] + 1, :]
            output[:, img_metas[0]['root_position_index'], :] /= 2.0
            
        target_image_paths = [m.get('target_image_path', None) for m in img_metas]
        result = {'preds': output, 'target_image_paths': target_image_paths}

        return result
    
    @staticmethod
    def _denormalize_joints(x, mean, std):
        """Denormalize joint coordinates with given statistics mean and std.

        Args:
            x (np.ndarray[N, K, 3]): Normalized joint coordinates.
            mean (np.ndarray[K, 3]): Mean value.
            std (np.ndarray[K, 3]): Std value.
        """
        assert x.ndim == 3
        assert x.shape == mean.shape == std.shape

        return x * std + mean

    @staticmethod
    def _restore_global_position(x, root_pos, root_idx=None):
        """Restore global position of the root-centered joints.

        Args:
            x (np.ndarray[N, K, 3]): root-centered joint coordinates
            root_pos (np.ndarray[N,1,3]): The global position of the
                root joint.
            root_idx (int|None): If not none, the root joint will be inserted
                back to the pose at the given index.
        """
        x = x + root_pos
        if root_idx is not None:
            x = np.insert(x, root_idx, root_pos.squeeze(1), axis=1)
        return x

    @staticmethod
    def _restore_root_target_weight(target_weight, root_weight, root_idx=None):
        """Restore the target weight of the root joint after the restoration of
        the global position.

        Args:
            target_weight (np.ndarray[N, K, 1]): Target weight of relativized
                joints.
            root_weight (float): The target weight value of the root joint.
            root_idx (int|None): If not none, the root joint weight will be
                inserted back to the target weight at the given index.
        """
        if root_idx is not None:
            root_weight = np.full(
                target_weight.shape[0], root_weight, dtype=target_weight.dtype)
            target_weight = np.insert(
                target_weight, root_idx, root_weight[:, None], axis=1)
        return target_weight

    def init_weights(self):
        """Initialize model weights."""
        for _, m in self.posemb_layers.named_modules():
            if isinstance(m, nn.Linear):
                normal_init(m, std=0.001)
            elif isinstance(m, nn.BatchNorm2d):
                constant_init(m, 1)
