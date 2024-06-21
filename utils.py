import os, sys, time
import pickle
import typing
from tqdm import tqdm
import builtins as __builtin__

import numpy as np
import torch
from torch.autograd import grad

from pykeops.torch import LazyTensor

import trimesh
from skimage import measure

import plotly.graph_objects as go

from sdf_field import FieldHeadNames

class DotDict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
    __name__ = 'DotDict'
    def __getstate__(self): return self.__dict__
    def __setstate__(self, d): self.__dict__.update(d)

def make_print_also_log(log_filepath):
    def print(*args, **kwargs):
        __builtin__.print(*args, **kwargs)
        with open(log_filepath, 'a') as fp:
            __builtin__.print(*args, file=fp, **kwargs)
    return print

def count_parameters(model):
    #count the number of parameters in a given model
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class CenteredGrid():
    def __init__(self, grid_dims, grid_res=128, radius=1):
        # Make a grid in n dimensions. It spans a hypercube centred at the origin with lengths of 2*radius in each 
        # dimension, and has grid_res points in each dimension. Thus there are grid_res**grid_dims points in the grid.
        # Short names are n, g_res, g_rad
        self.grid_dims = grid_dims # n=2 or n=3
        self.grid_res = grid_res # g_res is usually a power of 2: 128, 256, 512 etc.
        self.radius = radius
        assert self.grid_dims == 2 or self.grid_dims == 3, self.grid_dims
        self.grid_range = np.stack([[-1.0*radius,1.0*radius] for _ in range(self.grid_dims)]) # (n,2)

        self.axes = [np.linspace(self.grid_range[i,0], self.grid_range[i,1], grid_res) for i in range(self.grid_dims)] # list of n (g_res,)
        grid_points_np = np.stack(np.meshgrid(*self.axes, indexing='ij'), axis=-1) # (g_res,g_res,2) or (g_res,g_res,g_res,3)
        self.grid_spacing = self.axes[0][1] - self.axes[0][0]

        self.grid_points = torch.tensor(grid_points_np, dtype=torch.float32) # (g_res,g_res,2) or (g_res,g_res,g_res,3), i.e. an (x,y[,z]) at each x,y[,z] point
        self.grid_points_flattened = self.grid_points.reshape(-1,self.grid_dims) # (g_res*g_res, 2) or (g_res*g_res*g_res,3)

class PointsScaler():
    def __init__(self, initial_points, cp=None, max_norm = None):
        # assert torch.is_tensor(initial_points), type(initial_points)
        assert len(initial_points.shape) == 2 and initial_points.shape[-1] in [2,3], initial_points.shape

        if cp is None:
            cp = initial_points.mean(axis=0, keepdims=True)
        scaled_points = initial_points - cp
        if max_norm is None:
            max_norm = np.linalg.norm(scaled_points, axis=-1).max(-1) * 1.2
        scaled_points /= max_norm

        self.initial_points = initial_points
        self.cp = cp
        self.max_norm = max_norm
        self.scaled_points = scaled_points
    
    def scale_points(self, points):
        return (points - self.cp) / self.max_norm
    
    def unscale_points(self, points):
        return points*self.max_norm + self.cp

def pointsTensor2index3D(points, g_rad, g_res):
    # points should be a torch Tensor of shape (num_points, 2 or 3)
    assert torch.is_tensor(points), type(points)
    assert len(points.shape) == 2 and points.shape[-1] in [2,3], points.shape

    # normalise points from [-g_rad,g_rad]^n to [0,2*g_rad]^n, then to [0,1]^n, 
    # then to [0, g_res]^n, then to {0,1,...,g_res-1}^n
    return ((points+g_rad)*(g_res)/(2*g_rad)).floor().int().clamp(0,g_res-1)

def pointsTensor2octreeSigns(points, ot, ot_rad, ot_depth):
    # points should be a torch Tensor of shape (num_points, 2 or 3)
    assert torch.is_tensor(points), type(points)
    assert len(points.shape) == 2 and points.shape[-1] in [2,3], points.shape
    res = 2**ot_depth

    index3ds = pointsTensor2index3D(points, ot_rad, res).numpy()
    signs = ot.signsFromIndex3ds(index3ds, ot_depth) # returns an int list
    signs = np.array(signs, dtype = np.int32)
    return signs # (num_points, )

def implicitFunc2mesh(grid_obj, implicit_func, unscaling_func, chunk_size = 100000, use_tqdm=True):
    # grid_obj is an instance of CenteredGrid
    # implicit_func takes a pointsTensor (num_points, 2 or 3) and returns a value Tensor (num_points, )
    # unscaling func takes scaled points and unscales them
    points = grid_obj.grid_points_flattened

    z = []
    if use_tqdm:
        generator = tqdm(torch.split(points, chunk_size, dim=0))
    else:
        generator = torch.split(points, chunk_size, dim=0)
    for pnts in generator:
        # pnts: (chunk_size, 3)
        vals = implicit_func(pnts)
        if torch.is_tensor(vals):
            vals = vals.cpu().numpy()
        z.append(vals)
    z = np.concatenate(z, axis=0) # (num_pnts, )
    z = z.reshape(grid_obj.grid_res, grid_obj.grid_res, grid_obj.grid_res) # (g_res, g_res, g_res)

    verts, faces, normals, values = measure.marching_cubes(volume=z, level=0.0, 
                                spacing=(grid_obj.grid_spacing, grid_obj.grid_spacing, grid_obj.grid_spacing))

    verts = verts + np.array([-grid_obj.radius, -grid_obj.radius, -grid_obj.radius])
    verts = unscaling_func(verts)
    mesh = trimesh.Trimesh(verts, faces, vertex_normals=normals, vertex_colors=values, validate=True)
    return mesh

def computeImplicitFuncOnGrid(grid_obj, implicit_func, chunk_size = 100000, use_tqdm=True):
    # grid_obj is an instance of CenteredGrid
    # implicit_func takes a pointsTensor (num_points, 2 or 3) and returns a value Tensor (num_points, )
    # unscaling func takes scaled points and unscales them
    points = grid_obj.grid_points_flattened

    z = []
    if use_tqdm:
        generator = tqdm(torch.split(points, chunk_size, dim=0))
    else:
        generator = torch.split(points, chunk_size, dim=0)
    for pnts in generator:
        # pnts: (chunk_size, 3)
        vals = implicit_func(pnts)
        if torch.is_tensor(vals):
            vals = vals.cpu().numpy()
        z.append(vals)
    z = np.concatenate(z, axis=0) # (num_pnts, )
    # z = z.reshape(grid_obj.grid_res, grid_obj.grid_res, grid_obj.grid_res) # (g_res, g_res, g_res)
    return z

def plot_distance_map(x_grid, y_grid, z_grid, out_path, title_text='', colorscale='Geyser', show_ax=True):
    # plot contour and scatter plot given input points

    traces = []
    # plot implicit function contour
    traces.append(go.Contour(x=x_grid, y=y_grid, z=z_grid, colorscale=colorscale,
                            #  contours=dict(start=-1, end=1, size=0.025), showscale=True))
                             contours=dict(start=-0.5, end=1, size=0.01), showscale=True))
    traces.append(go.Contour(x=x_grid, y=y_grid, z=z_grid,
                             contours=dict(start=0,  end=0, coloring='lines'),
                             line=dict(width=5),
                             showscale=False,
                             colorscale=[[0, 'rgb(100, 100, 100)'], [1, 'rgb(100, 100, 100)']]))  # contour trace

    layout = go.Layout(width=1200, height=1200,
                       xaxis=dict(side="bottom", range=[-1, 1], showgrid=show_ax, zeroline=show_ax, visible=show_ax),
                       yaxis=dict(side="left", range=[-1, 1], showgrid=show_ax, zeroline=show_ax, visible=show_ax),
                       scene=dict(xaxis=dict(range=[-1, 1], autorange=False),
                                  yaxis=dict(range=[-1, 1], autorange=False),
                                  aspectratio=dict(x=1, y=1)),
                       showlegend=False,
                       title=dict(text=title_text, y=0.95, x=0.5, xanchor='center', yanchor='middle',
                                  font=dict(family='Sherif', size=24, color='red'))
                       )

    fig = go.Figure(data=traces, layout=layout)
    # fig.show()
    # print("Writing image to vis/temp.png")
    if out_path is None:
        out_path = os.path.join('vis', "temp.png")
    fig.write_image(out_path)
    return fig

def plot_distance_map_3(x_grid, y_grid, z_grid, title_text='', colorscale='Geyser', show_ax=True, out_path='vis/temp.png',range_lim=(-1,1, 0.025),other={}):
    # plot contour and scatter plot given input points

    traces = []
    for colour, points in other.items():
        if points is None:
            continue
        traces.append(go.Scatter(x=points[:, 0], y=points[:, 1],
                                    mode='markers', marker=dict(size=6, 
                                    # color='rgb(0, 0, 0)')))  # points scatter
                                    color=colour)))  # points scatter

    # plot implicit function contour
    traces.append(go.Contour(x=x_grid, y=y_grid, z=z_grid, colorscale=colorscale,
                             contours=dict(start=range_lim[0], end=range_lim[1], size=range_lim[2]), showscale=True))
    traces.append(go.Contour(x=x_grid, y=y_grid, z=z_grid,
                             contours=dict(start=0,  end=0, coloring='lines'),
                             line=dict(width=5),
                             showscale=False,
                             colorscale=[[0, 'rgb(100, 100, 100)'], [1, 'rgb(100, 100, 100)']]))  # contour trace


    layout = go.Layout(width=1200, height=1200,
                       xaxis=dict(side="bottom", range=[-1, 1], showgrid=show_ax, zeroline=show_ax, visible=show_ax),
                       yaxis=dict(side="left", range=[-1, 1], showgrid=show_ax, zeroline=show_ax, visible=show_ax),
                       scene=dict(xaxis=dict(range=[-1, 1], autorange=False),
                                  yaxis=dict(range=[-1, 1], autorange=False),
                                  aspectratio=dict(x=1, y=1)),
                       showlegend=False,
                       title=dict(text=title_text, y=0.95, x=0.5, xanchor='center', yanchor='middle',
                                  font=dict(family='Sherif', size=24, color='red'))
                       )

    fig = go.Figure(data=traces, layout=layout)
    # fig.show()
    print(f"Writing image to {out_path}")
    fig.write_image(out_path)
    
    # fig_bytes = fig.to_image(format="png")
    # buf = io.BytesIO(fig_bytes)
    # img = Image.open(buf)
    # arr = np.asarray(img)
    arr = None
    return fig, arr

def plot_3d_distance_map(sdf_model, dataset, out_dir):
    implicit_func = lambda points : sdf_model(points.to(next(sdf_model.parameters()).device))[FieldHeadNames('sdf')].detach()
    z = computeImplicitFuncOnGrid(dataset.grid_obj, implicit_func, chunk_size = 100000, use_tqdm=False)
    # import pdb; pdb.set_trace()
    z = z.reshape(256, 256, 256) # (g_res, g_res, g_res)
    # out_dir = './contour_vis/after_finetune'
    os.makedirs(out_dir, exist_ok=True)
    for i in range(256):
        if i % 20 == 0:
            print(i)
        plot_z = z[:,:,i].reshape(-1)
        points = dataset.grid_points.reshape(256,256,256,3)[:,:,i].reshape(-1,3)
        fig = plot_distance_map(points[:, 0], points[:, 1], 
                        plot_z, os.path.join(out_dir, f'level{i:03d}.png')
                    )

def keops_knn(pts_a, pts_b, k, force_cuda=False, return_numpy=False):
    if not torch.is_tensor(pts_a):
        pts_a = torch.tensor(pts_a)
    if not torch.is_tensor(pts_b):
        pts_b = torch.tensor(pts_b)
    
    if force_cuda:
        pts_a_i = LazyTensor(pts_a.cuda()[:, None, :])  # (M, 1, D) LazyTensor
        pts_b_j = LazyTensor(pts_b.cuda()[None, :, :])  # (1, N, D) LazyTensor
    else:
        pts_a_i = LazyTensor(pts_a[:, None, :])  # (M, 1, D) LazyTensor
        pts_b_j = LazyTensor(pts_b[None, :, :])  # (1, N, D) LazyTensor

    D_ij = ((pts_a_i - pts_b_j) ** 2).sum(dim=-1).sqrt() # (M, N)
    dists_a2b, idxes_a2b = D_ij.Kmin_argKmin(k, dim=1) # (M, k), (M, k)
    dists_a2b = dists_a2b.squeeze()
    idxes_a2b = idxes_a2b.squeeze()
    if return_numpy:
        return dists_a2b.cpu().numpy(), idxes_a2b.cpu().numpy()
    else:
        return dists_a2b, idxes_a2b

def sq_chamfer_distance_3d(pts_a, pts_b):
    dists_a2b, _ = keops_knn(pts_a, pts_b, 1, return_numpy=True)
    dists_b2a, _ = keops_knn(pts_b, pts_a, 1, return_numpy=True)
    return (dists_a2b**2).mean() + (dists_b2a**2).mean()

def compute_shapenet_metrics(recon_shape_path=None, gt_shape_path=None, implicit_func=None, gt_raw_shape_path=None, scale_obj=None):
    # t0 = time.time()
    metrics_dict = {}

    if (implicit_func is not None) and (gt_raw_shape_path is not None):
        # points.npz has [('points', (100000, 3)), ('occupancies', (12500,)), ('loc', (3,)), ('scale', ())]
        points = np.load(os.path.join(gt_raw_shape_path, 'points.npz')) 
        # pointcloud.npz has [('points', (100000, 3)), ('normals', (100000, 3)), ('loc', (3,)), ('scale', ())]
        # pointcloud should be the same as gt_pc
        pointcloud = np.load(os.path.join(gt_raw_shape_path, 'pointcloud.npz'))['points']
        gen_points = points['points'] # (100000,3)
        occupancies = np.unpackbits(points['occupancies']) # (100000)

        # center and scale point cloud
        if scale_obj is None:
            scale_obj = PointsScaler(pointcloud)

        eval_points = scale_obj.scale_points(gen_points) # (100000,3)
        eval_points = torch.tensor(eval_points, device=torch.device("cuda"), dtype=torch.float32)
        res = implicit_func(eval_points)

        pred_occupancies = (res.reshape(-1)<0).int().detach().cpu().numpy()
        iou = (occupancies & pred_occupancies).sum() / (occupancies | pred_occupancies).sum()

        metrics_dict['IoU'] = iou
        # print(f'iou took {time.time()-t0:.5f}s'); t0 = time.time()
    
    if (recon_shape_path is not None) and (gt_shape_path is not None):
        recon_mesh = trimesh.load(recon_shape_path)
        gt_pc = trimesh.load(gt_shape_path)

        # Compute Chamfer and Hausdorff Dist on input point cloud and samples on the reconstructed mesh
        gt_points = gt_pc.vertices
        recon_points = trimesh.sample.sample_surface(recon_mesh, 30000)[0]
        sq_chamfer_dist = sq_chamfer_distance_3d(recon_points, gt_points)
        metrics_dict['SqChamfer'] = sq_chamfer_dist
        # print(f'chamfer took {time.time()-t0:.5f}s'); t0 = time.time()
    
    return metrics_dict