import numpy as np
import json
import matplotlib.pyplot as plt
import cv2
from argparse import ArgumentParser
import os

def main():
    """Visualize the demo images.

    Require the json_file containing boxes.
    """
    parser = ArgumentParser()
    parser.add_argument(
        '--gt-npy-file',
        type=str,
        default='',
        help='npy file containing gt info.')
    parser.add_argument(
        '--gt-json-file',
        type=str,
        default='',
        help='Json file containing gt info.')
    parser.add_argument(
        '--pred-npy-file',
        type=str,
        default='',
        help='npy file containing prediction info.')
    parser.add_argument(
        '--out',
        type=str,
        default='',
        help='output path for visualization')

    args = parser.parse_args()


    gt_json = json.load(open(args.gt_json_file))
    pred_output = np.load(args.pred_npy_file, allow_pickle=True)
    gt_npy = np.load(args.gt_npy_file)
    # print(gt_npy['center'])
    # exit()

    for i in range(len(gt_json['annotations'])):
        gt_json['annotations'][i]['keypoints_3d'] = np.array(gt_json['annotations'][i]['keypoints_3d']).reshape(-1,4)[:, :3]
        gt_json['annotations'][i]['keypoints'] = np.array(gt_json['annotations'][i]['keypoints']).reshape(-1,3)

    skeletons = gt_json['categories'][0]['skeleton']
    mpjpe = []
    root_mpjpe = []
    for i in range(10):
        ax = plt.axes(projection='3d')
        # ax.view_init(elev=0, azim=0)
        keypoints = gt_npy['S'][i][:, :3]
        # keypoints = keypoints - keypoints[:1, :]
        gt_keypoints = keypoints.copy()

        max_range = np.array([2, 2, 2]).max() / 2.0
        mid_x = keypoints[0, 0]
        mid_y = keypoints[0, 2]
        mid_z = -keypoints[0, 1]
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
        for index, skeleton in enumerate(skeletons):
            ax.plot([keypoints[skeleton[0], 0], keypoints[skeleton[1], 0]],
                    [keypoints[skeleton[0], 2], keypoints[skeleton[1], 2]],
                    [-keypoints[skeleton[0], 1], -keypoints[skeleton[1], 1]], 'r')

        keypoints = pred_output[i]['keypoints_3d']
        root = keypoints[:1, :] / 1000
        keypoints = keypoints - keypoints[:1, :]
        keypoints /= 1000
        keypoints = keypoints + root

        max_range = np.array([2, 2, 2]).max() / 2.0
        # mid_x = 0.0
        # mid_y = 0.0
        # mid_z = 0.0
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
        for index, skeleton in enumerate(skeletons):
            ax.plot([keypoints[skeleton[0], 0], keypoints[skeleton[1], 0]],
                    [keypoints[skeleton[0], 2], keypoints[skeleton[1], 2]],
                    [-keypoints[skeleton[0], 1], -keypoints[skeleton[1], 1]], 'b')
        
        mpjpe.append(np.average(np.linalg.norm(keypoints - gt_keypoints, axis=1)))
        root_mpjpe.append(np.average(np.linalg.norm(keypoints - gt_keypoints - keypoints[0:1] + gt_keypoints[0:1], axis=1)))
        plt.savefig(os.path.join(args.out, 'compare_{}.jpg'.format(i)))
        
    print('mpjpe: ', np.average(mpjpe) * 1000)
    print('root mpjpe: ', np.average(root_mpjpe) * 1000)
    
if __name__ == '__main__':
    main()