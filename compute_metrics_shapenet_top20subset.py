import os,sys,time
import numpy as np

import torch
import trimesh

from utils import DotDict, PointsScaler, compute_shapenet_metrics, make_print_also_log
from sdf_field import SDFField_for_PC, SDFField_for_PC_Config, FieldHeadNames

# Manually set the following
dataset_path = '/home/chamin/data/NSP_dataset'
raw_dataset_path = '/home/chamin/data/ShapeNetNSP'
result_path = 'vis/top_shapes_exp0' # Experiment folder

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

top_shapes = [('lamp', 'd284b73d5983b60f51f77a6d7299806'), ('sofa', 'cde1943d3681d3dc4194871f9a6dae1d'), ('car', 'cc39c510115f0dadf774728aa9ef18b6'), ('car', 'cc32e1fb3be07a942ea8e4c752a397ac'), ('chair', 'cc2930e7ceb24691febad4f49b26ec52'), ('sofa', 'cc906e84c1a985fe80db6871fa4b6f35'), ('loudspeaker', 'cb25ea4938c1d31a1a5a6c4a1d8120d4'), ('bench', 'ca4bb0b150fa55eff3cd1fcb0edd1e8f'), ('table', 'cd91028b64fbfac2733464f54a85f798'), ('lamp', 'd1b15263933da857784a45ea6efa1d77'), ('car', 'cc6309e59ebc35b8becf71e2e014ff6f'), ('car', 'cbd0b4a0d264c856f35cb5c94d4343bf'), ('lamp', 'd1948365bbed9de270bb6bcb670ecdf5'), ('cabinet', 'b28d1c49cfe93a3f79368d1198f406e7'), ('table', 'cd82d526e58e3e7f4fb85ea6fd426098'), ('chair', 'cbdfedbf2dbeed3a91f6e3ed62ffa5d1'), ('bench', 'c97683ce5dcea25ad1d789f3b2120d0'), ('table', 'cde43e92674a66f46f3f02fb4d0927eb'), ('bench', 'c8802eaffc7e595b2dc11eeca04f912e'), ('airplane', 'd1b1c13fdec4d69ccfd264a25791a5e1')]
shape_name2class = {v:k for k,v in top_shapes}

device = torch.device("cuda")
sdf_field_config: SDFField_for_PC_Config = SDFField_for_PC_Config()
sdf_model: SDFField_for_PC = sdf_field_config.setup(in_dims=3).to(device)
implicit_func = lambda points : sdf_model(points.to(device), return_grad=False)[FieldHeadNames('sdf')].detach()

# Print metrics to file in the results directory
out_path = os.path.join(result_path, 'shapenet_metric_summary.txt')
print = make_print_also_log(out_path) # make print also save to log file

print("Metrics on ShapeNet")
print('Path', result_path)

ious = []
chamfers = []
times = []

for _, shape in top_shapes:
    shape_class = shape_name2class[shape]
    print(shape)
    recon_folder = f'{result_path}/{shape_class}_{shape}'
    if not os.path.isdir(recon_folder):
        print(f'{recon_folder} not found')
        continue

    try:
        log_file = os.path.join(recon_folder, 'log.txt')
        last_init = 98
        with open(log_file, 'r') as fp:
            txt = fp.readlines()
        seconds = float(txt[-1].strip().split()[-1][:-1])
        times.append(seconds)
        # print(f'Took {seconds:.2f}s i.e., {seconds/60:.2f}mins or {seconds/3600:.2f}hrs')
    except:
        continue

    files = os.listdir(recon_folder)
    if len([x for x in files if x[-4:]=='.ply']) == 0:
        print('no meshes')
        continue
    recon_shape_fname = max([x for x in files if x[-4:]=='.ply'], key=lambda x : int(x.split('it')[-1].split('.')[0]))
    recon_shape_path = os.path.join(recon_folder, recon_shape_fname)
    if len([x for x in files if x[-4:]=='.pth']) == 0:
        print('no saved weights')
        continue
    recon_state_dict_fname = max([x for x in files if x[-4:]=='.pth'])
    recon_state_dict_path = os.path.join(recon_folder, recon_state_dict_fname)
    print(recon_shape_fname, recon_state_dict_fname)

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
    # print(shape_class, shape, metrics_dict)
    # print(recon_shape_path, recon_state_dict_path)
    print(f"{shape} IoU {metrics_dict['IoU']:.4f} SqChamfer {metrics_dict['SqChamfer']:.4e} {recon_shape_fname.split('_')[-1]}")
    ious.append(metrics_dict['IoU'])
    chamfers.append(metrics_dict['SqChamfer'])

ious = np.array(ious)
chamfers = np.array(chamfers)
times = np.array(times)

print()
print('Summary')
print('IoU & mean & median & std')
print(f'& {ious.mean():.4f} & {np.median(ious):.4f} & {np.std(ious):.4f} ')
print('SqChamfer & mean & median & std')
print(f'& {chamfers.mean():.2e} & {np.median(chamfers):.2e} & {np.std(chamfers):.2e} ')
print('Time (min) & mean & median & std & range')
print(f'{times.mean()/60:.2f} & {np.median(times)/60:.2f} & {times.std()/60:.2f} & ({times.min()/60:.2f}, {times.max()/60:.2f})')

