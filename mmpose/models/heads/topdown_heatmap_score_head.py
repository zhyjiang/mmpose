# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import numpy as np
from mmcv.cnn import (build_conv_layer, build_norm_layer, build_upsample_layer,
                      constant_init, normal_init)

from mmpose.core.evaluation import pose_pck_accuracy
from mmpose.core.post_processing import flip_back, flip_score_back
from mmpose.models.builder import build_loss
from mmpose.models.utils.ops import resize
from ..builder import HEADS
from .topdown_heatmap_base_head import TopdownHeatmapBaseHead


@HEADS.register_module()
class TopdownHeatmapScoreHead(TopdownHeatmapBaseHead):
    """Top-down heatmap simple head. paper ref: Bin Xiao et al. ``Simple
    Baselines for Human Pose Estimation and Tracking``.

    TopdownHeatmapSimpleHead is consisted of (>=0) number of deconv layers
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
                 out_channels,
                 num_deconv_layers=3,
                 num_deconv_filters=(256, 256, 256),
                 num_deconv_kernels=(4, 4, 4),
                 fix_heatmap=False,
                 extra=None,
                 in_index=0,
                 input_transform=None,
                 align_corners=False,
                 loss_keypoint=None,
                 loss_score=None,
                 loss_cls_score=None,
                 train_cfg=None,
                 test_cfg=None):
        super().__init__()

        self.in_channels = in_channels

        if loss_keypoint is not None:
            self.loss = build_loss(loss_keypoint)
        else:
            self.loss = None
        
        if loss_score is not None:
            self.score_loss = build_loss(loss_score)
        else:
            self.score_loss = None
        
        if loss_cls_score is not None:
            self.cls_score_loss = build_loss(loss_cls_score)
        else:
            self.cls_score_loss = None

        self.fix_heatmap = fix_heatmap
        self.train_cfg = {} if train_cfg is None else train_cfg
        self.test_cfg = {} if test_cfg is None else test_cfg
        self.target_type = self.test_cfg.get('target_type', 'GaussianHeatmap')

        self._init_inputs(in_channels, in_index, input_transform)
        self.in_index = in_index
        self.align_corners = align_corners

        if extra is not None and not isinstance(extra, dict):
            raise TypeError('extra should be dict or None.')

        if num_deconv_layers > 0:
            self.deconv_layers = self._make_deconv_layer(
                num_deconv_layers,
                num_deconv_filters,
                num_deconv_kernels,
            )
        elif num_deconv_layers == 0:
            self.deconv_layers = nn.Identity()
        else:
            raise ValueError(
                f'num_deconv_layers ({num_deconv_layers}) should >= 0.')

        identity_final_layer = False
        if extra is not None and 'final_conv_kernel' in extra:
            assert extra['final_conv_kernel'] in [0, 1, 3]
            if extra['final_conv_kernel'] == 3:
                padding = 1
            elif extra['final_conv_kernel'] == 1:
                padding = 0
            else:
                # 0 for Identity mapping.
                identity_final_layer = True
            kernel_size = extra['final_conv_kernel']
        else:
            kernel_size = 1
            padding = 0
        
        if extra is not None and 'score_conv_kernel' in extra:
            assert extra['score_conv_kernel'] in [0, 1, 3]
            if extra['score_conv_kernel'] == 3:
                score_padding = 1
            elif extra['score_conv_kernel'] == 1:
                score_padding = 0
            score_kernel_size = extra['score_conv_kernel']
        else:
            score_kernel_size = 1
            score_padding = 0

        if identity_final_layer:
            self.final_layer = nn.Identity()
        else:
            conv_channels = num_deconv_filters[
                -1] if num_deconv_layers > 0 else self.in_channels

            layers = []
            if extra is not None:
                num_conv_layers = extra.get('num_conv_layers', 0)
                num_conv_kernels = extra.get('num_conv_kernels',
                                             [1] * num_conv_layers)

                for i in range(num_conv_layers):
                    layers.append(
                        build_conv_layer(
                            dict(type='Conv2d'),
                            in_channels=conv_channels,
                            out_channels=conv_channels,
                            kernel_size=num_conv_kernels[i],
                            stride=1,
                            padding=(num_conv_kernels[i] - 1) // 2))
                    layers.append(
                        build_norm_layer(dict(type='BN'), conv_channels)[1])
                    layers.append(nn.ReLU(inplace=True))

            layers.append(
                build_conv_layer(
                    cfg=dict(type='Conv2d'),
                    in_channels=conv_channels,
                    out_channels=out_channels,
                    kernel_size=1,
                    stride=1,
                    padding=0))

            if len(layers) > 1:
                self.final_layer = nn.Sequential(*layers)
            else:
                self.final_layer = layers[0]
        
        layers = []
        if extra is not None:
            layers.append(
                build_conv_layer(
                    dict(type='Conv2d'),
                    in_channels=conv_channels + out_channels,
                    out_channels=extra['score_channels'][0],
                    kernel_size=1,
                    stride=1,
                    padding=0))
            layers.append(
                    build_norm_layer(dict(type='BN'), extra['score_channels'][0])[1])
            layers.append(nn.ReLU(inplace=True))

            for i in range(len(extra['score_channels']) - 1):
                layers.append(
                    build_conv_layer(
                        dict(type='Conv2d'),
                        in_channels=extra['score_channels'][i],
                        out_channels=extra['score_channels'][i + 1],
                        kernel_size=score_kernel_size,
                        stride=2,
                        padding=score_padding))
                layers.append(
                    build_norm_layer(dict(type='BN'), extra['score_channels'][i + 1])[1])
                layers.append(nn.ReLU(inplace=True))

        if len(layers) > 1:
            self.score_layer = nn.Sequential(*layers)
        else:
            self.score_layer = layers[0]
        
        layers = []
        layers.append(nn.Linear(extra['score_channels'][-1] * self.train_cfg['heatmap_size'][0] * self.train_cfg['heatmap_size'][1] // 
                                2**((len(extra['score_channels']) - 1) * 2), extra['score_linear_dim']))
        
        for i in range(extra['score_linear_layers']):
            layers.append(nn.Linear(extra['score_linear_dim'], extra['score_linear_dim']))
        
        if len(layers) > 1:
            self.score_fc = nn.Sequential(*layers)
        else:
            self.score_fc = layers[0]
        
        self.score_head = nn.Linear(extra['score_linear_dim'], out_channels)
        self.cls_head = nn.Linear(extra['score_linear_dim'], out_channels)
        self.sigmoid = nn.Sigmoid()

    def get_loss(self, output, score, cls_score, target, target_weight, img_metas):
        """Calculate top-down keypoint loss.

        Note:
            - batch_size: N
            - num_keypoints: K
            - heatmaps height: H
            - heatmaps weight: W

        Args:
            output (torch.Tensor[N,K,H,W]): Output heatmaps.
            target (torch.Tensor[N,K,H,W]): Target heatmaps.
            target_weight (torch.Tensor[N,K,1]):
                Weights across different joint types.
        """

        losses = dict()

        assert not isinstance(self.loss, nn.Sequential)
        assert target.dim() == 4 and target_weight.dim() == 3
        if self.loss is not None:
            losses['heatmap_loss'] = self.loss(output, target, target_weight)
        
        if self.score_loss is not None:
            pred_keypoint = self.decode(img_metas, output.detach().cpu().numpy())['preds']
            gt_keypoint = np.zeros_like(pred_keypoint)
            area = []
            bboxes = []
            for img_id, img_meta in enumerate(img_metas):
                bbox = img_meta['bbox']
                gt_keypoint[img_id, :, :2] = img_meta['joints_3d_ori'][:, :2]
                gt_keypoint[img_id, :, 2] = img_meta['joints_3d_visible'][:, 0]
                area.append([bbox[2] * bbox[3]])
                bboxes.append(bbox)
            area = np.array(area)
            bboxes = np.array(bboxes)
            
            oks_score = self.computeOKSPerJoint(pred_keypoint, gt_keypoint, area, bboxes)
            target_weight_seq = target_weight.squeeze()

            losses['oks_score_loss'] = self.score_loss(score * target_weight_seq, 
                                                       torch.Tensor(oks_score).cuda() * target_weight_seq)
            losses['cls_score_loss'] = self.cls_score_loss(cls_score, target_weight.squeeze())

        return losses
    
    def computeOKSPerJoint(self, dts, gts, area, bbox):
        num_points = np.zeros((len(dts), 1))
        sigmas = np.array([.26, .25, .25, .35, .35, .79, .79, .72, .72, .62,.62, 1.07, 1.07, .87, .87, .89, .89])/10.0
        ious = np.zeros((len(dts), len(sigmas)),dtype=np.float32)
        vars = (sigmas * 2)**2
        k = len(sigmas)
        # compute oks between each detection and ground truth object
        for j in range(gts.shape[0]):
            g = gts[j,:,:]
            d = dts[j,:,:]
            # create bounds for ignore regions(double the gt bbox)
            xg = g[:,0]; yg = g[:,1]; vg = g[:,2]
            k1 = np.count_nonzero(vg > 0)
            bb = bbox[j,:]
            a = area[j,:]
            x0 = bb[0] - bb[2]; x1 = bb[0] + bb[2] * 2
            y0 = bb[1] - bb[3]; y1 = bb[1] + bb[3] * 2
            
            xd = d[:,0]; yd = d[:,1]
            if k1>0:
                # measure the per-keypoint distance if keypoints visible
                dx = xd - xg
                dy = yd - yg
            else:
                # measure minimum distance to keypoints in (x0,y0) & (x1,y1)
                z = np.zeros((k))
                dx = np.max((z, x0-xd),axis=0)+np.max((z, xd-x1),axis=0)
                dy = np.max((z, y0-yd),axis=0)+np.max((z, yd-y1),axis=0)
            e = (dx**2 + dy**2) / vars / (a+np.spacing(1)) / 2
            # if k1 > 0:
            #     e=e[vg > 0]
            ious[j] = np.exp(-e)
            ious[j][vg == 0] = 0
            num_points[j] = k1
        return ious

    def get_accuracy(self, output, target, target_weight):
        """Calculate accuracy for top-down keypoint loss.

        Note:
            - batch_size: N
            - num_keypoints: K
            - heatmaps height: H
            - heatmaps weight: W

        Args:
            output (torch.Tensor[N,K,H,W]): Output heatmaps.
            target (torch.Tensor[N,K,H,W]): Target heatmaps.
            target_weight (torch.Tensor[N,K,1]):
                Weights across different joint types.
        """

        accuracy = dict()

        if self.target_type == 'GaussianHeatmap':
            _, avg_acc, _ = pose_pck_accuracy(
                output.detach().cpu().numpy(),
                target.detach().cpu().numpy(),
                target_weight.detach().cpu().numpy().squeeze(-1) > 0)
            accuracy['acc_pose'] = float(avg_acc)

        return accuracy

    def forward(self, x):
        """Forward function."""
        x = self._transform_inputs(x)
        heatmap_feature = self.deconv_layers(x)
        heatmap = self.final_layer(heatmap_feature)
        if self.fix_heatmap:
            x_score = self.score_layer(torch.cat((x.detach(), heatmap.detach()), dim=1))
        else:
            x_score = self.score_layer(torch.cat((x, heatmap), dim=1))
        x_score = x_score.view(x_score.size(0),-1)
        x_score = self.score_fc(x_score)
        score = self.score_head(x_score)
        cls_score = self.sigmoid(self.cls_head(x_score))
        return heatmap, score, cls_score

    def inference_model(self, x, flip_pairs=None):
        """Inference function.

        Returns:
            output_heatmap (np.ndarray): Output heatmaps.

        Args:
            x (torch.Tensor[N,K,H,W]): Input features.
            flip_pairs (None | list[tuple]):
                Pairs of keypoints which are mirrored.
        """
        output, score, cls_score = self.forward(x)

        if flip_pairs is not None:
            output_heatmap = flip_back(
                output.detach().cpu().numpy(),
                flip_pairs,
                target_type=self.target_type)
            score, cls_score = flip_score_back(
                score.detach().cpu().numpy(),
                cls_score.detach().cpu().numpy(),
                flip_pairs)
            # feature is not aligned, shift flipped heatmap for higher accuracy
            if self.test_cfg.get('shift_heatmap', False):
                output_heatmap[:, :, :, 1:] = output_heatmap[:, :, :, :-1]
        else:
            output_heatmap = output.detach().cpu().numpy()
            score = score.detach().cpu().numpy()
            cls_score = cls_score.detach().cpu().numpy()
        return output_heatmap, score, cls_score

    def _init_inputs(self, in_channels, in_index, input_transform):
        """Check and initialize input transforms.

        The in_channels, in_index and input_transform must match.
        Specifically, when input_transform is None, only single feature map
        will be selected. So in_channels and in_index must be of type int.
        When input_transform is not None, in_channels and in_index must be
        list or tuple, with the same length.

        Args:
            in_channels (int|Sequence[int]): Input channels.
            in_index (int|Sequence[int]): Input feature index.
            input_transform (str|None): Transformation type of input features.
                Options: 'resize_concat', 'multiple_select', None.

                - 'resize_concat': Multiple feature maps will be resize to the
                    same size as first one and than concat together.
                    Usually used in FCN head of HRNet.
                - 'multiple_select': Multiple feature maps will be bundle into
                    a list and passed into decode head.
                - None: Only one select feature map is allowed.
        """

        if input_transform is not None:
            assert input_transform in ['resize_concat', 'multiple_select']
        self.input_transform = input_transform
        self.in_index = in_index
        if input_transform is not None:
            assert isinstance(in_channels, (list, tuple))
            assert isinstance(in_index, (list, tuple))
            assert len(in_channels) == len(in_index)
            if input_transform == 'resize_concat':
                self.in_channels = sum(in_channels)
            else:
                self.in_channels = in_channels
        else:
            assert isinstance(in_channels, int)
            assert isinstance(in_index, int)
            self.in_channels = in_channels

    def _transform_inputs(self, inputs):
        """Transform inputs for decoder.

        Args:
            inputs (list[Tensor] | Tensor): multi-level img features.

        Returns:
            Tensor: The transformed inputs
        """
        if not isinstance(inputs, list):
            return inputs

        if self.input_transform == 'resize_concat':
            inputs = [inputs[i] for i in self.in_index]
            upsampled_inputs = [
                resize(
                    input=x,
                    size=inputs[0].shape[2:],
                    mode='bilinear',
                    align_corners=self.align_corners) for x in inputs
            ]
            inputs = torch.cat(upsampled_inputs, dim=1)
        elif self.input_transform == 'multiple_select':
            inputs = [inputs[i] for i in self.in_index]
        else:
            inputs = inputs[self.in_index]

        return inputs

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
        for _, m in self.deconv_layers.named_modules():
            if isinstance(m, nn.ConvTranspose2d):
                normal_init(m, std=0.001)
            elif isinstance(m, nn.BatchNorm2d):
                constant_init(m, 1)
        for m in self.final_layer.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.001, bias=0)
            elif isinstance(m, nn.BatchNorm2d):
                constant_init(m, 1)
