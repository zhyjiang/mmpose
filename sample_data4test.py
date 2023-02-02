import numpy as np
import random
import os
import json
import copy

def _get_bbox_xywh(center, scale, w=200, h=200):
    w = w * scale
    h = h * scale
    x = center[0] - w / 2
    y = center[1] - h / 2
    return [x, y, w, h]

data_test = np.load('data/h36m/annotation_body3d/fps10/h36m_test.npz')
print(data_test.files)
index_list = [i for i in range(data_test['scale'].shape[0])]
random.shuffle(index_list)
data_sampled = dict()
for key in data_test.files:
    data_sampled[key] = data_test[key][index_list[:10]]
    
for file in data_sampled['imgname']:
    Sname, Aname, imgName = file.split('/')
    os.system('mkdir -p %s && cp -r %s %s' % (os.path.join('tests/data/h36m', Sname, Aname),
                                              os.path.join('data/h36m/images', Sname, Aname, imgName),
                                              os.path.join('tests/data/h36m', Sname, Aname, imgName)))


np.savez('tests/data/h36m/sampled_vis.npz', imgname=data_sampled['imgname'], 
                                            center=data_sampled['center'], 
                                            scale=data_sampled['scale'], 
                                            part=data_sampled['part'], 
                                            S=data_sampled['S'])

ori_json = json.load(open('tests/data/h36m/h36m_coco.json'))
new_json = copy.copy(ori_json)
new_json['images'] = []
new_json['annotations'] = []
for i in range(10):
    new_json['images'].append({
        'file_name': data_sampled['imgname'][i],
        'id': i
    })
    bbox = _get_bbox_xywh(data_sampled['center'][i], data_sampled['scale'][i])
    new_json['annotations'].append({
        'id': i,
        'category_id': 1, 
        'image_id': i,
        'iscrowd': 0,
        'bbox': bbox,
        'area': bbox[2] * bbox[3],
        'num_keypoints': 17,
        'keypoints': data_sampled['part'][i].flatten().tolist(),
        'keypoints_3d': data_sampled['S'][i].flatten().tolist(),
    })
json.dump(new_json, open('tests/data/h36m/h36m_sampled_coco.json', 'w'))
