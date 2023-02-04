echo "sample dataset..."
python tools/sample_data4test.py
wait
echo "start testing $1..."
python demo/ours_demo.py \
    configs/body/3d_kpt_sview_rgb_img/pose_lift/h36m/visualization.py \
    work_dirs/rgb/latest.pth \
    --json-file tests/data/h36m/h36m_sampled_coco.json \
    --img-root tests/data/h36m \
    --camera-param-file tests/data/h36m/cameras.pkl \
    --out-img-root work_dirs/$1/vis_results
wait
python tools/test_global.py \
    --gt-npy-file tests/data/h36m/sampled_vis.npz \
    --pred-npy-file work_dirs/$1/vis_results/output.npy \
    --gt-json-file tests/data/h36m/h36m_sampled_coco.json \
    --out work_dirs/$1/vis_results