import os, sys, time
from typing import Any, Callable, Dict, List

import math
import numpy as np
import torch
# from einops import rearrange, reduce, einsum

import dataclasses
from dataclasses import dataclass
import pyrallis
from pyrallis import field

import open3d as o3d

# import basic_shape_dataset2d
from dataset3d import PointGuidedReconDataset
from utils import DotDict, make_print_also_log, keops_knn
from pgsdf import initialisation, update_estimated_surface, move_points, \
    finetune, PGSDF_Vis

from sdf_field import SDFField_for_PC, SDFField_for_PC_Config


@dataclass
class PGSDF_Config:
    shapenet_path : str = '/home/chamin/data/NSP_dataset'
    shape_class : str = 'car'
    shape_name : str = 'cbd0b4a0d264c856f35cb5c94d4343bf' # Monster truck
    
    log_dir: str = 'vis/exp0'

    batch_size: int = 256**2
    seed: int = 1

    # save_vis: bool = True # whether to save a mesh of the current reconstruction
    use_wandb : bool = False # Whether to log using wandb
    use_rerun : bool = True

    M : int = 30000 # Number of est pc points
    D : int = 3 # Domain dimensions

    # the radius at initialisation as a multiple of pcd_rad
    init_rad_mult : int = 16
    # the radius at the start of opt_loop as a multiple of pcd_rad
    start_rad_mult : int = 4
    # the radius at finetune (after opt loop) as a multiple of pcd_rad
    final_rad_mult : int = 1

    notes : str = "nothing much" # Notes to log
    extra_tags : List[str] = field(default_factory=lambda : []) # Extra tags to add
    extra : dict = field(default_factory=lambda : DotDict()) # A place to add extra values

    def __post_init__(self):
        os.makedirs(self.log_dir, exist_ok=True)
        self.print = make_print_also_log(f'{self.log_dir}/log.txt')
        self.shape_path = os.path.join(self.shapenet_path, 
                                       self.shape_class,f'{self.shape_name}.ply') 

def run_pgsdf(cfg):
    ###########################################################
    ### Set up logging and visualisation
    ###########################################################
    tags = [cfg.shape_class, cfg.shape_name, *cfg.extra_tags]

    # Whether to visualise graphs in wandb
    if cfg.use_wandb:
        import wandb
        wandb.login()
        run = wandb.init(
            # Set the project where this run will be logged
            # project="chamfer_loss_2D",
            project="pointGuidedSDF_3D",
            name=cfg.log_dir.split('/')[-1],
            # Track hyperparameters and run metadata
            config={
                "shape_class": cfg.shape_class,
                "shape_name": cfg.shape_name,
            })

    # Whether to visualise with rerun. 
    # rr.init(.., spawn=True) will show visualisation in real time, but doesnm't save at the end as it crashes a lot, so save manually
    # rr.init(); rr.save() will not spawn a real time viewer and will save updates, which can be viewed from the saves later
    if cfg.use_rerun:
        import rerun as rr
        # rr.init(f"pgsdf_{cfg.log_dir.split('/')[-1]}", spawn=True)
        rr.init(f"pgsdf_{cfg.log_dir.split('/')[-1]}")
        rr.save(f'{cfg.log_dir}/data.rrd')


    # Save current code state
    os.system('cp %s %s' % (__file__, cfg.log_dir))  # backup this file
    os.system('cp %s %s' % ('pgsdf.py', cfg.log_dir))  # backup algorithm code
    os.system('cp %s %s' % ('dataset3d.py', cfg.log_dir))  # backup dataset code
    os.system('cp %s %s' % ('utils.py', cfg.log_dir))  # backup utils code
    os.system('cp %s %s' % ('sdf_field.py', cfg.log_dir))  # backup sdf_field code
    os.system('cp %s %s' % ('sample_farthest_points.py', cfg.log_dir))  # backup sdf_field code
    
    cfg.print('Starting...')
    cfg.print(cfg)
    cfg.print("Tags:", tags)
    initial_t0 = time.time()

    ###########################################################
    ### Set up dataset
    ###########################################################
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed(cfg.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    t0 = time.time()

    dataset = PointGuidedReconDataset(cfg, cfg.shape_name, cfg.shape_path, None, 
                                      cfg.batch_size, grid_res=256, grid_radius=1.1)
    
    cfg.print(f'Loaded dataset {time.time()-t0:.3f}s'); t0 = time.time()

    ################################
    ## Set up Model and Optimiser
    ################################

    sdf_field_config: SDFField_for_PC_Config = SDFField_for_PC_Config()

    sdf_model: SDFField_for_PC = sdf_field_config.setup(
                in_dims=3,
            ).cuda()
    
    model_params = sum(p.numel() for p in sdf_model.parameters() if p.requires_grad)
    encoding_params = sum(p.numel() for p in sdf_model.encoding.parameters() if p.requires_grad)
    cfg.print(f'Number of training parameters is {model_params}, {encoding_params}'
        f'={encoding_params/model_params*100}% are for the hash encoding')

    sdf_optimiser = torch.optim.Adam(sdf_model.parameters(), lr=1e-4)

    cfg.print(f'Loaded Model and Optimiser {time.time()-t0:.3f}s'); t0 = time.time()

    ################################
    ## Optimisation
    ################################

    # Set up visualisation object
    vis_obj = PGSDF_Vis(cfg, dataset)
    cfg.print(f'Set up visualisation object {time.time()-t0:.3f}s'); t0 = time.time()

    # Our input points \X
    mnfld_pnts_cuda = torch.tensor(dataset.pcd_points.astype(np.float32)).cuda()

    # find point cloud radius
    dists, idxes = keops_knn(mnfld_pnts_cuda, mnfld_pnts_cuda, 5, return_numpy=False)
    sorted_dists, sorted_idxes = torch.sort(dists, dim=0)
    p95 = int(dists.shape[0]*.95)
    pcd_rad = sorted_dists[p95:,1:5].mean().item()
    del dists, idxes, sorted_dists, sorted_idxes, p95
    cfg.print(f'pcd_rad is {pcd_rad:.5f}, took {time.time()-t0:.3f}s to compute'); t0 = time.time()

    cfg.pcd_rad = pcd_rad
    cfg.mnfld_pnts_cuda = mnfld_pnts_cuda
    cfg.vis_obj = vis_obj
    cfg.sdf_model = sdf_model
    cfg.sdf_optimiser = sdf_optimiser
    cfg.dataset = dataset

    ### Initialise
    cfg.print(f'Starting initialisation')
    rad_mult = cfg.init_rad_mult
    rad = pcd_rad * rad_mult
    cfg.print(f'rad as initialisation is {rad:.5f}={rad_mult}*pcd_rad')
    # Guiding points \Y and its normals
    est_pc_cuda, normals = initialisation(cfg, rad)
    cfg.print(f'Finished initialisation {time.time()-t0:.3f}s'); t0 = time.time()

    ### Main Loop
    rad_mult = cfg.start_rad_mult
    rad = pcd_rad * rad_mult
    cfg.print(f'rad at start of main loop is {rad:.5f}={rad_mult}*pcd_rad')
    iters_since_last_decrease = 0
    for it in range(100000):
        t0 = time.time()
        ### Update estimated surface based on est_pc_cuda, normals. Internally updates sdf_model.
        cfg.print('Update phi starting')
        est_pc_cuda, normals = update_estimated_surface(cfg, it, est_pc_cuda, normals, rad)
        cfg.print(f'it {it}: Updated est surface {time.time()-t0:.3f}s'); t0 = time.time()

        # Save points and normals to file
        est_pc_save_path = '{}/{}_it{:05d}_est_pc.pt'.format(cfg.log_dir,cfg.shape_name, it)
        torch.save({'est_pc_cuda': est_pc_cuda, 'normals': normals}, est_pc_save_path)
        cfg.print(f"Saved est_pc w normals to file : {est_pc_save_path}, took {time.time()-t0:.3f}s"); t0 = time.time()

        ### Determine if to reduce rad and if to exit loop
        # Look at distance between Y=est_pc_cuda and X=mnfld_pnts_cuda, stop if every y\in Y is approx within rad of an x\in X
        tol = 1e-2
        dists_Y2X, _ = keops_knn(est_pc_cuda, mnfld_pnts_cuda, 1, return_numpy=False)
        dists_Y2X_min = dists_Y2X.min().item(); dists_Y2X_mean = dists_Y2X.mean().item(); dists_Y2X_max = dists_Y2X.max().item()
        cfg.print(f'\tstop_crit: rad+tol {rad+tol:.5f}, dists_Y2X min {dists_Y2X_min:.5f}, mean {dists_Y2X_mean:.5f}, max {dists_Y2X_max:.5f}')
        del dists_Y2X

        cfg.print(f'\tstop_crit: iters_since_last_decrease {iters_since_last_decrease}')

        # Determine if we have reached the next level set
        if rad_mult == cfg.start_rad_mult:
            cond = (iters_since_last_decrease > 5) or (dists_Y2X_max < rad + tol)
        elif rad_mult > cfg.final_rad_mult:
            cond = (iters_since_last_decrease > 10) or (dists_Y2X_max < rad + tol)
        else:
            cond = (iters_since_last_decrease > 200) or (dists_Y2X_max < rad + tol)

        if cond:
            # Reached level set
            cfg.print(f'\tstop_crit: stopped moving!!!!!!, iters_since_last_decrease {iters_since_last_decrease}')
            rad_mult /= 2
            rad = pcd_rad * rad_mult
            cfg.print(f"\tstop_crit: rad is now {rad:.5f} (rad_mult {rad_mult})!!!!!!!!!!!!!!!")

            sdf_save_path = '{}/{}_it{:05d}.pth'.format(cfg.log_dir,cfg.shape_name, it)
            torch.save(sdf_model.state_dict(), sdf_save_path)
            cfg.print(f"Saved SDF model params to file : {sdf_save_path}")

            if rad_mult < 1:
                # Finished the final level set, exit loop
                last_init_iter = it
                cfg.print(f'\tstop_crit: last_init_iter {last_init_iter}, final opt step!!!!!!')
                break
            iters_since_last_decrease = 0
        else: # Haven't changed the rad
            iters_since_last_decrease += 1

        ### Move `est_pc_cuda` towards `mnfld_pnts_cuda` in the direction of `normals` with buffer `rad`
        t0 = time.time()
        cfg.print('Move points starting')
        est_pc_cuda, normals, correspondences, move_dists = move_points(cfg, est_pc_cuda, normals, rad)
        cfg.print(f'it {it}: Moved est pc, current rad {rad:.5f} (rad_mult {rad_mult}), {time.time()-t0:.3f}s')
        vis_obj.plot_figure(it, est_pc=est_pc_cuda, normals=normals, 
                            correspondences=correspondences)
        vis_obj.it_ended()

    vis_obj.it_ended()

    ### Finetune
    rad_mult = cfg.final_rad_mult
    rad = pcd_rad * rad_mult
    cfg.print(f'it {last_init_iter+1}: Finetune started!!!, rad is {rad:.5f}={rad_mult}*pcd_rad')
    it = finetune(cfg, last_init_iter, rad=rad)

    ### Save result
    sdf_save_path = '{}/{}_it{:05d}.pth'.format(cfg.log_dir,cfg.shape_name, it)
    est_pc_save_path = '{}/{}_it{:05d}_est_pc.pt'.format(cfg.log_dir,cfg.shape_name, it)
    torch.save(sdf_model.state_dict(), sdf_save_path)
    torch.save({'est_pc_cuda': est_pc_cuda, 'normals': normals}, est_pc_save_path)
    cfg.print(f"Saved SDF model params to file : {sdf_save_path} and edt_pc w normals to file : {est_pc_save_path}")

    cfg.print(f"Total running time {time.time() - initial_t0:.2f}s")

    # Force GC before next shape
    import gc
    gc.collect()
    torch.cuda.empty_cache()



if __name__ == '__main__':

    ###########################################################
    ## Run a single shape fron ShapeNet
    ###########################################################

    # cfg = pyrallis.parse(config_class=PGSDF_Config)
    cfg = pyrallis.parse(config_class=PGSDF_Config, args=[
        '--log_dir', 'vis/exp0',
        '--shape_class', 'car',
        '--shape_name', 'cbd0b4a0d264c856f35cb5c94d4343bf'
        ])
    run_pgsdf(cfg)

    ###########################################################
    ## Run on the shapes with 20 highest SIONs
    ###########################################################

    # top_shapes = [('lamp', 'd284b73d5983b60f51f77a6d7299806'), ('sofa', 'cde1943d3681d3dc4194871f9a6dae1d'), ('car', 'cc39c510115f0dadf774728aa9ef18b6'), ('car', 'cc32e1fb3be07a942ea8e4c752a397ac'), ('chair', 'cc2930e7ceb24691febad4f49b26ec52'), ('sofa', 'cc906e84c1a985fe80db6871fa4b6f35'), ('loudspeaker', 'cb25ea4938c1d31a1a5a6c4a1d8120d4'), ('bench', 'ca4bb0b150fa55eff3cd1fcb0edd1e8f'), ('table', 'cd91028b64fbfac2733464f54a85f798'), ('lamp', 'd1b15263933da857784a45ea6efa1d77'), ('car', 'cc6309e59ebc35b8becf71e2e014ff6f'), ('car', 'cbd0b4a0d264c856f35cb5c94d4343bf'), ('lamp', 'd1948365bbed9de270bb6bcb670ecdf5'), ('cabinet', 'b28d1c49cfe93a3f79368d1198f406e7'), ('table', 'cd82d526e58e3e7f4fb85ea6fd426098'), ('chair', 'cbdfedbf2dbeed3a91f6e3ed62ffa5d1'), ('bench', 'c97683ce5dcea25ad1d789f3b2120d0'), ('table', 'cde43e92674a66f46f3f02fb4d0927eb'), ('bench', 'c8802eaffc7e595b2dc11eeca04f912e'), ('airplane', 'd1b1c13fdec4d69ccfd264a25791a5e1')]
    # for shape_class, shape_name in top_shapes[:]:
    #     exp_name = 'top_shapes_exp0'
    #     if os.path.exists(f'vis/{exp_name}/{shape_class}_{shape_name}'):
    #         continue
    #     cfg = pyrallis.parse(config_class=PGSDF_Config, args=[
    #         '--log_dir', f'vis/{exp_name}/{shape_class}_{shape_name}',
    #         '--shape_class', shape_class,
    #         '--shape_name', shape_name,
    #         ])
    #     run_pgsdf(cfg)

    ###########################################################
    ## Run on all shapes fron ShapeNet
    ###########################################################

    # dataset_path = '/home/chamin/data/NSP_dataset'
    # for shape_class in sorted(os.listdir(dataset_path)):
    #     shape_class_dir = os.path.join(dataset_path, shape_class)
    #     if not os.path.isdir(shape_class_dir):
    #         continue
    #     print(shape_class, len(sorted(os.listdir(shape_class_dir))))
    #     for shape_name in sorted(os.listdir(shape_class_dir)):
    #         exp_name = 'all_shapes_exp0'
    #         shape_name = shape_name.split('.')[0]
    #         if os.path.exists(f'vis/{exp_name}/{shape_class}/{shape_class}_{shape_name}'):
    #             continue
    #         print(shape_class, shape_name)

    #         cfg = pyrallis.parse(config_class=PGSDF_Config, args=[
    #             '--log_dir', f'vis/{exp_name}/{shape_class}/{shape_class}_{shape_name}',
    #             '--shape_class', shape_class,
    #             '--shape_name', shape_name,
    #             ])
    #         run_pgsdf(cfg)
