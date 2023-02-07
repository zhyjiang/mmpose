# Copyright (c) OpenMMLab. All rights reserved.
import warnings

import mmcv
import numpy as np
from mmcv.image import imwrite
from mmcv.utils.misc import deprecated_api_warning
from mmcv.visualization.image import imshow

from mmpose.core import imshow_bboxes, imshow_keypoints, imshow_keypoints_3d
from .. import builder
from ..builder import POSENETS
from .base import BasePose

try:
    from mmcv.runner import auto_fp16
except ImportError:
    warnings.warn('auto_fp16 from mmpose will be deprecated from v0.15.0'
                  'Please install mmcv>=1.1.4')
    from mmpose.core import auto_fp16


@POSENETS.register_module()
class TopDown3D(BasePose):
    """Top-down pose detectors.

    Args:
        backbone (dict): Backbone modules to extract feature.
        keypoint_head (dict): Keypoint head to process feature.
        train_cfg (dict): Config for training. Default: None.
        test_cfg (dict): Config for testing. Default: None.
        pretrained (str): Path to the pretrained models.
        loss_pose (None): Deprecated arguments. Please use
            `loss_keypoint` for heads instead.
    """

    def __init__(self,
                 backbone,
                 neck=None,
                 keypoint_head=None,
                 keypoint3d_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 fix_keypoint_head=False,
                 pretrained=None,
                 inference3d=True,
                 loss_pose=None,
                 loss_pose3d=None,):
        super().__init__()
        self.fp16_enabled = False

        self.backbone = builder.build_backbone(backbone)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.inference3d = inference3d
        self.fix_keypoint_head = fix_keypoint_head

        if neck is not None:
            self.neck = builder.build_neck(neck)

        if keypoint_head is not None:
            keypoint_head['train_cfg'] = train_cfg
            keypoint_head['test_cfg'] = test_cfg

            if 'loss_keypoint' not in keypoint_head and loss_pose is not None:
                warnings.warn(
                    '`loss_pose` for TopDown is deprecated, '
                    'use `loss_keypoint` for heads instead. See '
                    'https://github.com/open-mmlab/mmpose/pull/382'
                    ' for more information.', DeprecationWarning)
                keypoint_head['loss_keypoint'] = loss_pose

            self.keypoint_head = builder.build_head(keypoint_head)
        
        if keypoint3d_head is not None:
            keypoint3d_head['train_cfg'] = train_cfg
            keypoint3d_head['test_cfg'] = test_cfg

            if 'loss_keypoint' not in keypoint3d_head and loss_pose3d is not None:
                warnings.warn(
                    '`loss_pose` for TopDown is deprecated, '
                    'use `loss_keypoint` for heads instead. See '
                    'https://github.com/open-mmlab/mmpose/pull/382'
                    ' for more information.', DeprecationWarning)
                keypoint3d_head['loss_keypoint3d'] = loss_pose3d

            self.keypoint3d_head = builder.build_head(keypoint3d_head)
        
        self.pretrained = pretrained
        self.init_weights()

    @property
    def with_neck(self):
        """Check if has neck."""
        return hasattr(self, 'neck')

    @property
    def with_keypoint(self):
        """Check if has keypoint_head."""
        return hasattr(self, 'keypoint_head')
    
    @property
    def with_keypoint3d(self):
        """Check if has keypoint3d_head."""
        return hasattr(self, 'keypoint3d_head')

    def init_weights(self, pretrained=None):
        """Weight initialization for model."""
        if pretrained is not None:
            self.pretrained = pretrained
        self.backbone.init_weights(self.pretrained)
        if self.with_neck:
            self.neck.init_weights()
        if self.with_keypoint:
            self.keypoint_head.init_weights()
        if self.with_keypoint3d:
            self.keypoint3d_head.init_weights()

    @auto_fp16(apply_to=('img', ))
    def forward(self,
                img,
                target=None,
                target_3d=None,
                target_weight=None,
                img_metas=None,
                return_loss=True,
                return_heatmap=False,
                **kwargs):
        """Calls either forward_train or forward_test depending on whether
        return_loss=True. Note this setting will change the expected inputs.
        When `return_loss=True`, img and img_meta are single-nested (i.e.
        Tensor and List[dict]), and when `resturn_loss=False`, img and img_meta
        should be double nested (i.e.  List[Tensor], List[List[dict]]), with
        the outer list indicating test time augmentations.

        Note:
            - batch_size: N
            - num_keypoints: K
            - num_img_channel: C (Default: 3)
            - img height: imgH
            - img width: imgW
            - heatmaps height: H
            - heatmaps weight: W

        Args:
            img (torch.Tensor[NxCximgHximgW]): Input images.
            target (torch.Tensor[NxKxHxW]): Target heatmaps.
            target_3d (torch.Tensor[NxKx3]): Target 3D poses.
            target_weight (torch.Tensor[NxKx1]): Weights across
                different joint types.
            img_metas (list(dict)): Information about data augmentation
                By default this includes:

                - "image_file: path to the image file
                - "center": center of the bbox
                - "scale": scale of the bbox
                - "rotation": rotation of the bbox
                - "bbox_score": score of bbox
            return_loss (bool): Option to `return loss`. `return loss=True`
                for training, `return loss=False` for validation & test.
            return_heatmap (bool) : Option to return heatmap.

        Returns:
            dict|tuple: if `return loss` is true, then return losses. \
                Otherwise, return predicted poses, boxes, image paths \
                and heatmaps.
        """
        if return_loss:
            return self.forward_train(img, target, target_3d, target_weight, img_metas,
                                      **kwargs)
        return self.forward_test(
            img, img_metas, return_heatmap=return_heatmap, **kwargs)

    def forward_train(self, img, target, target_3d, target_weight, img_metas, **kwargs):
        """Defines the computation performed at every call when training."""
        output = self.backbone(img)
        if self.with_neck:
            output = self.neck(output)
        if self.with_keypoint:
            heatmap = self.keypoint_head(output)
            if self.fix_keypoint_head:
                output[0] = output[0].detach()
                output[3] = output[3].detach()
                pose3d = self.keypoint3d_head(output, heatmap.detach(), img_metas)
                # pose3d = self.keypoint3d_head(output[0], heatmap, img_metas)
            else:
                pose3d = self.keypoint3d_head(output, heatmap, img_metas)

        # if return loss
        losses = dict()
        if self.with_keypoint:
            keypoint_losses = self.keypoint_head.get_loss(
                heatmap, target, target_weight)
            losses.update(keypoint_losses)
            keypoint3d_losses = self.keypoint3d_head.get_loss(
                pose3d, target_3d, target_weight)
            losses.update(keypoint3d_losses)
            keypoint_accuracy = self.keypoint_head.get_accuracy(
                heatmap, target, target_weight)
            losses.update(keypoint_accuracy)
            keypoint_3d_accuracy = self.keypoint3d_head.get_accuracy(
                pose3d, target_3d, target_weight, img_metas
            )
            losses.update(keypoint_3d_accuracy)

        return losses

    def forward_test(self, img, img_metas, return_heatmap=False, **kwargs):
        """Defines the computation performed at every call when testing."""
        assert img.size(0) == len(img_metas)
        batch_size, _, img_height, img_width = img.shape

        result = {}

        features = self.backbone(img)
        if self.with_neck:
            features = self.neck(features)
        if self.with_keypoint:
            output_heatmap = self.keypoint_head(features)
            output_pose3d = self.keypoint3d_head.inference_model(
                features[0], output_heatmap, img_metas)
            output_heatmap = output_heatmap.cpu().numpy()

        if self.with_keypoint:
            keypoint_result = self.keypoint_head.decode(
                img_metas, output_heatmap, img_size=[img_width, img_height])
            result.update(keypoint_result)
            if self.inference3d:
                result['preds2d'] = result['preds']
                keypoint3d_result = self.keypoint3d_head.decode(
                    img_metas, output_pose3d
                )
                result.update(keypoint3d_result)

            if not return_heatmap:
                output_heatmap = None

            result['output_heatmap'] = output_heatmap
            
        return result

    def forward_dummy(self, img):
        """Used for computing network FLOPs.

        See ``tools/get_flops.py``.

        Args:
            img (torch.Tensor): Input image.

        Returns:
            Tensor: Output heatmaps.
        """
        output = self.backbone(img)
        if self.with_neck:
            output = self.neck(output)
        if self.with_keypoint:
            output = self.keypoint_head(output)
        return output

    @deprecated_api_warning({'pose_limb_color': 'pose_link_color'},
                            cls_name='TopDown')
    def show_result(self,
                    result,
                    img,
                    skeleton=None,
                    kpt_score_thr=0.3,
                    bbox_color='green',
                    pose_kpt_color=None,
                    pose_link_color=None,
                    text_color='white',
                    radius=4,
                    thickness=1,
                    font_scale=0.5,
                    bbox_thickness=1,
                    vis_height=400,
                    num_instances=-1,
                    axis_azimuth=-115,
                    win_name='',
                    show=False,
                    show_keypoint_weight=False,
                    wait_time=0,
                    out_file=None):
        """Draw `result` over `img`.

        Args:
            img (str or Tensor): The image to be displayed.
            result (list[dict]): The results to draw over `img`
                (bbox_result, pose_result).
            skeleton (list[list]): The connection of keypoints.
                skeleton is 0-based indexing.
            kpt_score_thr (float, optional): Minimum score of keypoints
                to be shown. Default: 0.3.
            bbox_color (str or tuple or :obj:`Color`): Color of bbox lines.
            pose_kpt_color (np.array[Nx3]`): Color of N keypoints.
                If None, do not draw keypoints.
            pose_link_color (np.array[Mx3]): Color of M links.
                If None, do not draw links.
            text_color (str or tuple or :obj:`Color`): Color of texts.
            radius (int): Radius of circles.
            thickness (int): Thickness of lines.
            font_scale (float): Font scales of texts.
            win_name (str): The window name.
            show (bool): Whether to show the image. Default: False.
            show_keypoint_weight (bool): Whether to change the transparency
                using the predicted confidence scores of keypoints.
            wait_time (int): Value of waitKey param.
                Default: 0.
            out_file (str or None): The filename to write the image.
                Default: None.

        Returns:
            Tensor: Visualized img, only if not `show` or `out_file`.
        """
        if num_instances < 0:
            assert len(result) > 0
        result = sorted(result, key=lambda x: x.get('track_id', 0))
        # draw image and 2d poses
        if img is not None:
            img = mmcv.imread(img)

            bbox_result = []
            pose_2d = []
            for res in result:
                if 'bbox' in res:
                    bbox = np.array(res['bbox'])
                    if bbox.ndim != 1:
                        assert bbox.ndim == 2
                        bbox = bbox[-1]  # Get bbox from the last frame
                    bbox_result.append(bbox)
                if 'keypoints' in res:
                    kpts = np.array(res['keypoints'])
                    if kpts.ndim != 2:
                        assert kpts.ndim == 3
                        kpts = kpts[-1]  # Get 2D keypoints from the last frame
                    pose_2d.append(kpts)

            if len(bbox_result) > 0:
                bboxes = np.vstack(bbox_result)
                mmcv.imshow_bboxes(
                    img,
                    bboxes,
                    colors=bbox_color,
                    top_k=-1,
                    thickness=2,
                    show=False)
            if len(pose_2d) > 0:
                imshow_keypoints(
                    img,
                    pose_2d,
                    skeleton,
                    kpt_score_thr=kpt_score_thr,
                    pose_kpt_color=pose_kpt_color,
                    pose_link_color=pose_link_color,
                    radius=radius,
                    thickness=thickness)
            img = mmcv.imrescale(img, scale=vis_height / img.shape[0])
        
        img_vis = imshow_keypoints_3d(
            result,
            img,
            skeleton,
            pose_kpt_color,
            pose_link_color,
            vis_height,
            axis_limit=300,
            axis_azimuth=axis_azimuth,
            axis_elev=15,
            kpt_score_thr=kpt_score_thr,
            num_instances=num_instances)

        if show:
            mmcv.visualization.imshow(img_vis, win_name, wait_time)

        if out_file is not None:
            mmcv.imwrite(img_vis, out_file)

        return img_vis