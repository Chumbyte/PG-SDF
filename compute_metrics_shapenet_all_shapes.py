import os,sys,time
import numpy as np

import torch
import trimesh

from utils import DotDict, PointsScaler, compute_shapenet_metrics, make_print_also_log
from sdf_field import SDFField_for_PC, SDFField_for_PC_Config, FieldHeadNames

# Manually set the following
dataset_path = '/home/chamin/data/NSP_dataset'
raw_dataset_path = '/home/chamin/data/ShapeNetNSP'
result_path = 'vis/all_shapes_exp0' # Experiment folder

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

order = sorted(["car","chair","airplane","display","table","rifle","cabinet","loudspeaker","telephone","bench","sofa","watercraft","lamp",])

device = torch.device("cuda")
sdf_field_config: SDFField_for_PC_Config = SDFField_for_PC_Config()
sdf_model: SDFField_for_PC = sdf_field_config.setup(in_dims=3).to(device)
implicit_func = lambda points : sdf_model(points.to(device), return_grad=False)[FieldHeadNames('sdf')].detach()

# Print metrics to file in the results directory
out_path = os.path.join(result_path, 'shapenet_metric_summary.txt')
print = make_print_also_log(out_path) # make print also save to log file

print("Metrics on ShapeNet")
print('Path', result_path)

IoUs = {}
chamfers = {}
times = {}

for shape_class in order:
    if shape_class not in os.listdir(result_path):
        continue
    shape_class_id = shape_class_name2id[shape_class]
    gt_shape_class_path = os.path.join(dataset_path, shape_class)
    gt_raw_shape_class_path = os.path.join(raw_dataset_path, shape_class_id)

    results_shape_class_path = os.path.join(result_path, shape_class)
    if not os.path.exists(results_shape_class_path):
        print("Did not find folder for {}".format(shape_class))
        continue
    
    shape_folders = [f for f in os.listdir(results_shape_class_path)]
    print("Found {} folders for {}".format(len(shape_folders), shape_class))

    chamfers[shape_class] = []
    IoUs[shape_class] = []
    times[shape_class] = []
    for fldr_name in shape_folders:
        recon_folder = os.path.join(results_shape_class_path, fldr_name)
        shape = fldr_name.split('_')[-1]
        print(f'{shape_class} {shape}')
        
        try:
            log_file = os.path.join(recon_folder, 'log.txt')
            last_init = 98
            with open(log_file, 'r') as fp:
                txt = fp.readlines()
            seconds = float(txt[-1].strip().split()[-1][:-1])
            times[shape_class].append(seconds)
            print(f'\tTook {seconds:.2f}s i.e., {seconds/60:.2f}mins or {seconds/3600:.2f}hrs')
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
        print('\t', recon_shape_fname, recon_state_dict_fname)

        sdf_model.load_state_dict(torch.load(recon_state_dict_path, map_location=device))

        shape_file = f"{shape}.ply"
        gt_shape_class_path = os.path.join(dataset_path, shape_class)
        gt_shape_path = os.path.join(gt_shape_class_path, shape_file)
        shape_class_id = shape_class_name2id[shape_class]
        gt_raw_shape_class_path = os.path.join(raw_dataset_path, shape_class_id)
        gt_raw_shape_path = os.path.join(gt_raw_shape_class_path, shape)
        try:
            metrics_dict = compute_shapenet_metrics(
                recon_shape_path=recon_shape_path, gt_shape_path=gt_shape_path,
                implicit_func=implicit_func, gt_raw_shape_path=gt_raw_shape_path, scale_obj=None)
        except Exception as e:
            print('error when computing metrics')
            import traceback
            traceback.print_exc()
            continue

        iou = metrics_dict['IoU']
        chamfer_dist = metrics_dict['SqChamfer']
        print(f"\t{shape} IoU {metrics_dict['IoU']:.4f} SqChamfer {metrics_dict['SqChamfer']:.4e} {recon_shape_fname.split('_')[-1]}")
        IoUs[shape_class].append(iou)
        chamfers[shape_class].append(chamfer_dist)

    dists_np = np.array(chamfers[shape_class])
    print("{}: Mean: {:e}, Median: {:e}, Std: {:e}".format(shape_class,  dists_np.mean(), np.median(dists_np), dists_np.std()))

print()
print('Summary')
print('Final IoUs')
for key in order:
    if key not in IoUs:
        continue
    IoUs_np = np.array(IoUs[key])
    print("{:<15} : Mean: {:.4f}, Median: {:.4f}, Std: {:.4f}".format(key, IoUs_np.mean(), np.median(IoUs_np), IoUs_np.std()))
all_IoUs = np.concatenate([IoUs[k] for k in IoUs]).reshape(-1)
print("Overall: Mean: {:.4f}, Median: {:.4f}, Std: {:.4f}".format(all_IoUs.mean(), np.median(all_IoUs), all_IoUs.std()))

print('Final Chamfers')
for key in order:
    if key not in chamfers:
        continue
    dists_np = np.array(chamfers[key])
    print("{:<15} : Mean: {:e}, Median: {:e}, Std: {:e}".format(key, dists_np.mean(), np.median(dists_np), dists_np.std()))
all_chamfers = np.concatenate([chamfers[k] for k in chamfers]).reshape(-1)
print("Overall: Mean: {:e}, Median: {:e}, Std: {:e}".format(all_chamfers.mean(), np.median(all_chamfers), all_chamfers.std()))

print('Final Times')
for key in order:
    if key not in times:
        continue
    times_np = np.array(times[key])
    print("{:<15} : Mean: {:e}, Median: {:e}, Std: {:e}".format(key, times_np.mean()/60, np.median(times_np)/60, times_np.std()/60))
all_times = np.concatenate([times[k] for k in times]).reshape(-1)
print("Overall: Mean: {:e}, Median: {:e}, Std: {:e}".format(all_times.mean()/60, np.median(all_times)/60, all_times.std()/60))
