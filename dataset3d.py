import os, sys, time
import numpy as np
import pickle

import torch
import torch.utils.data as data

import open3d as o3d
import trimesh

from utils import PointsScaler, CenteredGrid, keops_knn

sys.path.append('octree_base')
import octree_base.octree as octree

class PointGuidedReconDataset(data.Dataset):
    def __init__(self, cfg, shape_name, pcd_path, scaling_path, batch_size, grid_res=128, grid_radius=1.1, estimate_normals=False):
        self.cfg = cfg
        self.shape_name = shape_name
        self.pcd_path = pcd_path # path to the point cloud
        self.scaling_path = scaling_path
        self.batch_size = batch_size
        self.grid_res = grid_res # g_res
        self.grid_radius = grid_radius
        self.is2D = False
        
        self.cfg.print('Loading PCD from {}'.format(self.pcd_path)); t0 = time.time()
        # load data
        o3d_point_cloud = o3d.io.read_point_cloud(self.pcd_path)
        if estimate_normals:
            t1 = time.time()
            self.cfg.print('Running Normal Estimation')
            o3d_point_cloud.normals = o3d.utility.Vector3dVector(np.zeros(
                (1, 3)))  # invalidate existing normals
            o3d_point_cloud.estimate_normals()
            o3d_point_cloud.orient_normals_consistent_tangent_plane(100)
            self.cfg.print(f'Normal Estimation took {time.time() - t1:.5f}s')
            
        # Returns points on the manifold
        self.pcd_points = np.asarray(o3d_point_cloud.points, dtype=np.float32)
        self.pcd_normals = np.asarray(o3d_point_cloud.normals, dtype=np.float32) # might be empty
        # center and scale point cloud
        if scaling_path is None:
            self.scale_obj = PointsScaler(self.pcd_points)
            self.cp = self.scale_obj.cp
            self.max_norm = self.scale_obj.max_norm
        else:
            scaling = np.load(scaling_path)
            self.cp = scaling['cp']
            self.max_norm = scaling['max_norm']
            self.scale_obj = PointsScaler(self.pcd_points, self.cp, self.max_norm)
        self.pcd_points = self.scale_obj.scaled_points
        self.cfg.print('PCD loaded and scaled ({:.5f}s)'.format(time.time()-t0))

        # Make a grid obj for saving meshes
        self.cfg.print('Making grid'); t0 = time.time()
        self.grid_obj = CenteredGrid(2 if self.is2D else 3, grid_res=self.grid_res, radius=self.grid_radius)
        self.cfg.print('Grid made ({:.5f}s)'.format(time.time()-t0))
        self.gp = self.grid_obj.grid_points_flattened # (g_res*g_res*g_res,3)
        self.num_gp = self.gp.shape[0]

        self.gp_idxes = np.random.permutation(self.num_gp)
        self.randomised_gp = self.gp[self.gp_idxes]
        # self.gp_bs = 150000
        # self.gp_bs = 50000
        self.gp_bs = 15000
        self.gp_num_iters = self.num_gp // self.gp_bs

        self.cfg.print('data loaded'); t0 = time.time()

    def __len__(self):
        # there is no length, keep running over data
        return int(1e6)
    
    def buildOctree(self, zls_points):
        # given the current zls_points, make an octree around it and compute adaptive samples from it

        self.zls_points = zls_points
        zls_point_np = zls_points.detach().cpu().numpy() if torch.is_tensor(zls_points) else zls_points

        t0 = time.time()
        ot = octree.py_createTree(6, zls_point_np, 
                                2*1.1, # cube length
                                -1.1, # min_x
                                -1.1, # min_y
                                -1.1, # min_z
                                False)
        
        self.num_nonsurf = ot.numNonSurfaceLeaves
        self.num_surf = ot.numSurfaceLeaves
        
        self.surf_lvs_centers = []
        for node in ot.surfaceLeaves:
            l = node.length
            x,y,z = node.top_pos_x, node.top_pos_y, node.top_pos_z
            self.surf_lvs_centers.append((x+l/2, y+l/2, z+l/2))
        self.surf_lvs_centers = np.array(self.surf_lvs_centers, dtype=np.float32)
        self.surf_leaf_length = l # should all be the same length

        self.nonsurf_lvs_centers = []
        self.nonsurf_lvs_lengths = []
        for node in ot.nonSurfaceLeaves:
            l = node.length
            x,y,z = node.top_pos_x, node.top_pos_y, node.top_pos_z
            self.nonsurf_lvs_centers.append((x+l/2, y+l/2, z+l/2))
            self.nonsurf_lvs_lengths.append(l)
        self.nonsurf_lvs_centers = np.array(self.nonsurf_lvs_centers, dtype=np.float32)
        self.nonsurf_lvs_lengths = np.array(self.nonsurf_lvs_lengths, dtype=np.float32)
        # self.cfg.print(f"# non-surface leafs {self.num_nonsurf}, # surface leafs {self.num_surf}, ({time.time()-t0:.5f}s)")
        t0 = time.time()
        # self.cfg.print(f'{12*(self.num_nonsurf+self.num_surf)}')

        samples_per_leaf = 20
        nonsurf_samples = self.nonsurf_lvs_centers[:,None,:] + \
            (np.random.rand(self.num_nonsurf,samples_per_leaf,3).astype(np.float32)-0.5) * self.nonsurf_lvs_lengths[:,None,None]
        nonsurf_samples = nonsurf_samples.reshape(-1, 3)
        surf_samples = self.surf_lvs_centers[:,None,:] + \
            (np.random.rand(self.num_surf,samples_per_leaf,3).astype(np.float32)-0.5) * self.surf_leaf_length
        surf_samples = surf_samples.reshape(-1, 3)

        # leaf points
        self.lp = torch.tensor(np.concatenate((nonsurf_samples, surf_samples), axis=0), dtype=torch.float32)
        self.num_lp = self.lp.shape[0]

        self.lp_idxes = np.random.permutation(self.num_lp)
        self.randomised_lp = self.lp[self.lp_idxes]
        # self.lp_bs = min(self.num_lp, 50000)
        self.lp_bs = min(self.num_lp, 15000)
        # self.lp_bs = min(self.num_lp, 5000)
        self.lp_num_iters = self.num_lp // self.lp_bs
        
        # self.cfg.print(f"leaf points made, ({time.time()-t0:.5f}s)")
 

    def __getitem__(self, index):

        gp_ind = index % self.gp_num_iters
        batch_gp = self.randomised_gp[gp_ind*self.gp_bs : (gp_ind+1)*self.gp_bs]

        lp_ind = index % self.lp_num_iters
        batch_lp = self.randomised_lp[lp_ind*self.lp_bs : (lp_ind+1)*self.lp_bs]


        domain_samples = torch.cat((batch_lp, batch_gp), axis=0)

        return {
                'domain_samples': domain_samples,
                }

