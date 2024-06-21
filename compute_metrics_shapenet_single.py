import os,sys,time
import numpy as np

import torch
import trimesh

from utils import DotDict, PointsScaler, compute_shapenet_metrics, make_print_also_log
from sdf_field import SDFField_for_PC, SDFField_for_PC_Config, FieldHeadNames

# Manually set the following
dataset_path = '/home/chamin/data/NSP_dataset'
raw_dataset_path = '/home/chamin/data/ShapeNetNSP'
shape = 'cbd0b4a0d264c856f35cb5c94d4343bf'
shape_class = 'car'
recon_folder = 'vis/exp0/' # Experiment folder

shape_class_name_dict = \
        {
            "04256520": "sofa",
            "02691156": "airplane",
            "03636649": "lamp",
            "04401088": "telephone",
            "04530566": "watercraft",
            "03691459": "loudspeaker",
            "03001627": "chair",
            "02933112": "cabinet",
            "04379243": "table",
            "03211117": "display",
            "02958343": "car",
            "02828884": "bench",
            "04090263": "rifle",
        }

shape_class_name2id = {v:k for k,v in shape_class_name_dict.items()}

device = torch.device("cuda")
sdf_field_config: SDFField_for_PC_Config = SDFField_for_PC_Config()
sdf_model: SDFField_for_PC = sdf_field_config.setup(in_dims=3).to(device)
implicit_func = lambda points : sdf_model(points.to(device), return_grad=False)[FieldHeadNames('sdf')].detach()

# Print metrics to file in the results directory
out_path = os.path.join(recon_folder, 'shapenet_metric_summary.txt')
print = make_print_also_log(out_path) # make print also save to log file

print("Metrics on ShapeNet")
print('Path', recon_folder)
print(shape_class, shape)

try:
    log_file = os.path.join(recon_folder, 'log.txt')
    last_init = 98
    with open(log_file, 'r') as fp:
        txt = fp.readlines()
    seconds = float(txt[-1].strip().split()[-1][:-1])
    print(f'Took {seconds:.2f}s i.e., {seconds/60:.2f}mins or {seconds/3600:.2f}hrs')
except:
    pass

files = os.listdir(recon_folder)
recon_shape_fname = max([x for x in files if x[-4:]=='.ply'], key=lambda x : int(x.split('it')[-1].split('.')[0]))
recon_shape_path = os.path.join(recon_folder, recon_shape_fname)
recon_state_dict_fname = max([x for x in files if x[-4:]=='.pth'])
recon_state_dict_path = os.path.join(recon_folder, recon_state_dict_fname)

sdf_model.load_state_dict(torch.load(recon_state_dict_path, map_location=device))

shape_file = f"{shape}.ply"
gt_shape_class_path = os.path.join(dataset_path, shape_class)
gt_shape_path = os.path.join(gt_shape_class_path, shape_file)
shape_class_id = shape_class_name2id[shape_class]
gt_raw_shape_class_path = os.path.join(raw_dataset_path, shape_class_id)
gt_raw_shape_path = os.path.join(gt_raw_shape_class_path, shape)
metrics_dict = compute_shapenet_metrics(
    recon_shape_path=recon_shape_path, gt_shape_path=gt_shape_path,
    implicit_func=implicit_func, gt_raw_shape_path=gt_raw_shape_path, scale_obj=None)
print(recon_shape_fname, recon_state_dict_path)
print(metrics_dict)
# print(f"{shape} IoU {metrics_dict['IoU']:.4f} SqChamfer {metrics_dict['SqChamfer']:.4e} {recon_shape_fname.split('_')[-1]}")

