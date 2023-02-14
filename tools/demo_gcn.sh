echo "start testing hrnet_end2end_3d_gcn..."
python demo/ours_demo.py \
    configs/body/3d_kpt_sview_rgb_img/pose_lift/h36m/hrnet_end2end_3d_gcn_2branch.py \
    work_dirs/hrnet_end2end_3d_gcn_2branch/best_MPJPE_epoch_28.pth \
    --json-file tests/data/h36m/h36m_sampled_coco.json \
    --img-root tests/data/h36m \
    --camera-param-file tests/data/h36m/cameras.pkl \
    --out-img-root work_dirs/hrnet_end2end_3d_gcn_2branch/vis_results
wait
python tools/test_global.py \
    --gt-npy-file tests/data/h36m/sampled_vis.npz \
    --pred-npy-file work_dirs/hrnet_end2end_3d_gcn/vis_results/output.npy \
    --gt-json-file tests/data/h36m/h36m_sampled_coco.json \
    --out work_dirs/hrnet_end2end_3d_gcn_2branch/vis_results