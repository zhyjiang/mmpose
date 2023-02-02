# Copyright (c) OpenMMLab. All rights reserved.
import os
import os.path as osp
import warnings
from argparse import ArgumentParser
import json
import copy

import numpy as np
import mmcv
from xtcocotools.coco import COCO

from mmpose.apis import (inference_top_down_pose_hybird_model, init_pose_model,
                         vis_pose_result, vis_3d_pose_result)
from mmpose.core.camera import SimpleCamera
from mmpose.datasets import DatasetInfo

def _keypoint_camera_to_world(keypoints,
                              camera_params,
                              image_name=None,
                              dataset='Body3DH36MDataset'):
    """Project 3D keypoints from the camera space to the world space.

    Args:
        keypoints (np.ndarray): 3D keypoints in shape [..., 3]
        camera_params (dict): Parameters for all cameras.
        image_name (str): The image name to specify the camera.
        dataset (str): The dataset type, e.g. Body3DH36MDataset.
    """
    cam_key = None
    if dataset == 'Body3DH36MDataset':
        subj, rest = osp.basename(image_name).split('_', 1)
        _, rest = rest.split('.', 1)
        camera, rest = rest.split('_', 1)
        cam_key = (subj, camera)
    else:
        raise NotImplementedError

    camera = SimpleCamera(camera_params[cam_key])
    keypoints_world = keypoints.copy()
    keypoints_world[..., :3] = camera.camera_to_world(keypoints[..., :3])

    return keypoints_world

def main():
    """Visualize the demo images.

    Require the json_file containing boxes.
    """
    parser = ArgumentParser()
    parser.add_argument('pose_config', help='Config file for detection')
    parser.add_argument('pose_checkpoint', help='Checkpoint file')
    parser.add_argument('--img-root', type=str, default='', help='Image root')
    parser.add_argument(
        '--json-file',
        type=str,
        default='',
        help='Json file containing image info.')
    parser.add_argument(
        '--camera-param-file',
        type=str,
        default=None,
        help='Camera parameter file for converting 3D pose predictions from '
        ' the camera space to the world space. If None, no conversion will be '
        'applied.')
    parser.add_argument(
        '--show',
        action='store_true',
        default=False,
        help='whether to show img')
    parser.add_argument(
        '--out-img-root',
        type=str,
        default='',
        help='Root of the output img file. '
        'Default not saving the visualization images.')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--kpt-thr', type=float, default=0.3, help='Keypoint score threshold')
    parser.add_argument(
        '--radius',
        type=int,
        default=4,
        help='Keypoint radius for visualization')
    parser.add_argument(
        '--thickness',
        type=int,
        default=1,
        help='Link thickness for visualization')

    args = parser.parse_args()

    assert args.show or (args.out_img_root != '')

    json_file = json.load(open(args.json_file))
    coco = COCO(args.json_file)
    # build the pose model from a config file and a checkpoint file
    pose_model = init_pose_model(
        args.pose_config, args.pose_checkpoint, device=args.device.lower())

    dataset = pose_model.cfg.data['test']['type']
    dataset_info = pose_model.cfg.data['test'].get('dataset_info', None)
    if dataset_info is None:
        warnings.warn(
            'Please set `dataset_info` in the config.'
            'Check https://github.com/open-mmlab/mmpose/pull/663 for details.',
            DeprecationWarning)
    else:
        dataset_info = DatasetInfo(dataset_info)

    img_keys = list(coco.imgs.keys())

    # optional
    return_heatmap = False

    # e.g. use ('backbone', ) to return backbone feature
    output_layer_names = None
    pose_jsons = []
    # process each image
    for i in mmcv.track_iter_progress(range(len(img_keys))):
        # get bounding box annotations
        image_id = img_keys[i]
        image = coco.loadImgs(image_id)[0]
        image_name = os.path.join(args.img_root, image['file_name'])
        ann_ids = coco.getAnnIds(image_id)

        # make person bounding boxes
        person_results = []
        anns = []
        for ann_id in ann_ids:
            person = {}
            ann = coco.anns[ann_id]
            anns.append(ann)
            # ann: dict_keys(['id', 'category_id', 'image_id', 'iscrowd', 'bbox', 'area', 'num_keypoints', 'keypoints', 'keypoints_3d'])
            
            # bbox format is 'xywh'
            person['bbox'] = ann['bbox']
            
            person_results.append(person)

        # test a single image, with a list of bboxes
        pose_results, returned_outputs = inference_top_down_pose_hybird_model(
            pose_model,
            image_name,
            person_results,
            bbox_thr=None,
            format='xywh',
            dataset=dataset,
            dataset_info=dataset_info,
            return_heatmap=return_heatmap,
            outputs=output_layer_names)

        if args.out_img_root == '':
            out_file = None
        else:
            os.makedirs(args.out_img_root, exist_ok=True)
            out_file = os.path.join(args.out_img_root, f'vis_{i}.jpg')
        pose_jsons.append(copy.deepcopy(pose_results[0]))
        pose_results[0]['keypoints_3d'] -= pose_results[0]['keypoints_3d'][:1, :]
        pose_results[0]['keypoints_3d'] /= 10

        vis_3d_pose_result(
            pose_model,
            result=pose_results,
            img=image_name,
            dataset_info=dataset_info,
            out_file=out_file)

    np.save(os.path.join(args.out_img_root, 'output.npy'), pose_jsons)

if __name__ == '__main__':
    main()
