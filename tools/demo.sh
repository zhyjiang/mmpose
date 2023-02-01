python demo/ours_demo.py \
    configs/body/3d_kpt_sview_rgb_img/pose_lift/h36m/visualization.py \
    work_dirs/hrnet_end2end_3d_2branch/latest.pth \
    --json-file tests/data/h36m/h36m_coco.json \
    --img-root tests/data/h36m \
    --camera-param-file tests/data/h36m/cameras.pkl \
    --out-img-root work_dirs/hrnet_end2end_3d_2branch/vis_results \