import os, sys, time
import numpy as np
import open3d as o3d
import torch
import pickle as pkl

dataset_path = 'data/NSP_dataset/'

shape_name_list = []
shape_sion_list = []

class_list = os.listdir(dataset_path)
for class_name in class_list:
    print(class_name)
    class_path = os.path.join(dataset_path, class_name)
    if not os.path.isdir(class_path):
        print(f'skipping {class_path}')
        continue
    shape_list = os.listdir(class_path)
    for shape_name in shape_list:
        print(class_name, shape_name)
        assert shape_name[-4:] == '.ply', shape_name

        shape_path = os.path.join(class_path, shape_name)
        pcd = o3d.io.read_point_cloud(shape_path)
        pcd_points = torch.tensor(np.asarray(pcd.points, dtype=np.float32)).cuda()
        pcd_normals = torch.tensor(np.asarray(pcd.normals, dtype=np.float32)).cuda()

        violated = torch.zeros(pcd_points.shape[0]).cuda()
        
        for i in range(pcd_points.shape[0]):
            normal = pcd_normals[i].unsqueeze(0) # (1,3)
            disp = pcd_points - pcd_points[i].unsqueeze(0) # (100000, 3)
            dist_in_normal_dir = disp@normal.T
            positive_mask = dist_in_normal_dir.squeeze() > 0.02
            if positive_mask.any():
                dist2normal = (disp[positive_mask] - dist_in_normal_dir[positive_mask] * normal).norm(dim=-1)
                violated[i] = dist2normal.min() <= 0.01
        violated_mean = violated.mean().item()
        print(f'{shape_name} {violated_mean:.4f}')

        shape_name_list.append(shape_path)
        shape_sion_list.append(violated_mean)

sions = np.array(shape_sion_list)
paths = np.array(shape_name_list)
print(sions.min(), sions.mean(), np.median(sions), sions.max())
ind = np.argpartition(sions, -10)[-10:]
sorted_inds = np.argsort(sions)
top20sions = sions[sorted_inds[-20:][::-1]]
top20paths = paths[sorted_inds[-20:][::-1]]

concavity_dict = {}
concavity_dict['sions'] = sions
concavity_dict['paths'] = paths
concavity_dict['top20paths'] = top20paths
concavity_dict['top20sions'] = top20sions

pkl.dump(concavity_dict, open('sion.pkl','wb'))