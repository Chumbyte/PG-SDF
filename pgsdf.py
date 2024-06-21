import os, sys, time

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange, reduce, einsum

from pykeops.torch import LazyTensor
from sample_farthest_points import sample_farthest_points_naive_new

from utils import implicitFunc2mesh, keops_knn

from sdf_field import SDFField_for_PC, SDFField_for_PC_Config, FieldHeadNames

import rerun as rr
import wandb as wandb


################################
## Low Level Algorithm Components
################################

def generate_samples_on_sphere(n, r):
    # # Generate n random points within the unit sphere
    z = np.random.uniform(-1, 1, n)
    theta = np.random.uniform(-np.pi, np.pi, n)
    x = np.sqrt(1-z**2) * np.sin(theta)
    y = np.sqrt(1-z**2) * np.cos(theta)

    return r*np.stack((x, y, z), axis=1)

def move_to_lvlset(cfg, mnfld_pnts_cuda, est_pc_cuda, normals, vis_obj, rad=0.2, step_size=0.05, steps=100, randomness=0.3):
    # get points on the exterior level set of radius rad
    points = est_pc_cuda
    direction = normals + randomness * torch.randn(normals.shape, device=normals.device)
    direction = direction / direction.norm(dim=-1, keepdim=True)
    t0 = time.time()
    for its in range(steps):
        # check if current points are still further than rad away from surface
        dists_Y2X, idxes_Y2X = keops_knn(points, mnfld_pnts_cuda, 1, return_numpy=False)
        move_mask = (dists_Y2X > rad).float().unsqueeze(-1)
        # also check if points are within the domain (sphere of rad 1)
        outside_D_mask = (points.norm(dim=-1) < 1.05).float().unsqueeze(-1)
        # move points if both conditions hold
        points = points - direction * move_mask * outside_D_mask * step_size
        # vis_obj.plot_figure(None, est_pc=points)
        if (move_mask*outside_D_mask).sum() == 0:
            # no points left to move, converged
            break
    
    # only get the ones that stay within the domain and are within rad of the points
    dists_Y2X, idxes_Y2X = keops_knn(points, mnfld_pnts_cuda, 1, return_numpy=False)
    valid_mask = (dists_Y2X <= rad) * (points.norm(dim=-1) < 1.05)

    setX_cl2_setY = mnfld_pnts_cuda[idxes_Y2X[valid_mask]]
    disps = points[valid_mask] - setX_cl2_setY # (Ma, D)
    lvlset_normals = disps / disps.norm(dim=-1, keepdim=True) # (Ma, D)
    lvlset_points = setX_cl2_setY + lvlset_normals * rad # (Ma, D)

    return lvlset_points, lvlset_normals


def find_exterior_level_set(cfg, mnfld_pnts_cuda, est_pc_cuda, normals, 
                            rad=0.2, step_size=0.05, initial_repeats=3, bounce_repeats=3, steps=100, randomness=0.3):
    vis_obj = cfg.vis_obj
    
    # currently est_pc_cuda are sampled on a sphere surrounding mnfld_pnts_cuda
    # move est_pc_cuda points towards the rad level set of mnfld_pnts_cuda by moving in the
    #  inward normal direction + some randomness
    t0 = time.time()
    all_lvlset_points = []
    all_lvlset_normals = []
    for _ in range(initial_repeats):
        lvlset_points, lvlset_normals = move_to_lvlset(cfg, mnfld_pnts_cuda, est_pc_cuda, normals, vis_obj, 
                                                rad=rad, step_size=step_size, steps=steps, randomness=randomness)
        all_lvlset_points.append(lvlset_points)
        all_lvlset_normals.append(lvlset_normals)
        vis_obj.plot_figure(None, est_pc=lvlset_points)
    all_lvlset_points = torch.cat(all_lvlset_points, axis=0)
    all_lvlset_normals = torch.cat(all_lvlset_normals, axis=0)
    cfg.print(f'\tinit: {initial_repeats} move_to_lvlset repeats took {time.time() - t0:.3}s'); t0 = time.time()
    
    # FPS sampled from these points
    fps_points,fps_indices = sample_farthest_points_naive_new(all_lvlset_points, K=80000)
    cfg.print(f'\tinit: FPS took {time.time() - t0:.3}s'); t0 = time.time()
    lvlset_points = fps_points
    lvlset_normals = all_lvlset_normals[fps_indices]

    # do the same thing starting from the current level set points, but moving in the outwards normal direction
    # this is because not all surfaces are reached from moving inwards from a sphere, can get more coverage by
    # moving outwards again multiple times
    all_lvlset_points = []
    all_lvlset_normals = []
    all_lvlset_points.append(lvlset_points)
    all_lvlset_normals.append(lvlset_normals)
    lvlset_points, lvlset_normals = move_to_lvlset(cfg, mnfld_pnts_cuda, lvlset_points+lvlset_normals*1e-3, -lvlset_normals, vis_obj, 
                                            rad=rad, step_size=step_size, steps=steps, randomness=0.1)
    all_lvlset_points.append(lvlset_points)
    all_lvlset_normals.append(lvlset_normals)
    vis_obj.plot_figure(None, est_pc=lvlset_points)
    cfg.print(f'\tinit: Reverse move_to_lvlset took {time.time() - t0:.3}s'); t0 = time.time()

    # repeat moving out (above) more times
    for _ in range(bounce_repeats):
        if lvlset_points.shape[0] == 0:
            break
        multiplier = max(100000 // lvlset_points.shape[0], 1)
        lvlset_points = rearrange(lvlset_points.unsqueeze(1).repeat(1,multiplier,1), 'n p d -> (n p) d')
        lvlset_normals = rearrange(lvlset_normals.unsqueeze(1).repeat(1,multiplier,1), 'n p d -> (n p) d')
        lvlset_points, lvlset_normals = move_to_lvlset(cfg, mnfld_pnts_cuda, lvlset_points+lvlset_normals*1e-3, -lvlset_normals, vis_obj, 
                                                rad=rad, step_size=step_size, steps=steps, randomness=0.5)
        all_lvlset_points.append(lvlset_points)
        all_lvlset_normals.append(lvlset_normals)
        vis_obj.plot_figure(None, est_pc=lvlset_points)
        
    all_lvlset_points = torch.cat(all_lvlset_points, axis=0)
    all_lvlset_normals = torch.cat(all_lvlset_normals, axis=0)
    cfg.print(f'\tinit: {bounce_repeats} move_to_lvlset repeats in reverse direction took {time.time() - t0:.3}s'); t0 = time.time()
    fps_points,fps_indices = sample_farthest_points_naive_new(all_lvlset_points, K=cfg.M)
    cfg.print(f'\tinit: FPS took {time.time() - t0:.3}s'); t0 = time.time()

    new_points = fps_points
    new_normals = all_lvlset_normals[fps_indices]

    return new_points, new_normals        


def optimise_sdf(cfg, it, zls_points_cuda, num_sdf_its,
                 weights = [10, 0.5, 2], rad=0.1,
                 normals=None, lr=None, subset_size = None,
                 use_mnfld=False,
                 ):
    sdf_model = cfg.sdf_model
    sdf_optimiser = cfg.sdf_optimiser
    dataset = cfg.dataset


    if subset_size is not None:
        subset_idxes = np.random.permutation(zls_points_cuda.shape[0])[:subset_size]
        zls_points_cuda = zls_points_cuda[subset_idxes]
        if normals is not None:
            normals = normals[subset_idxes]

    if lr is not None:
        for g in sdf_optimiser.param_groups:
            g['lr'] = lr
    
    t0 = time.time()

    dataset.buildOctree(zls_points_cuda.detach())
    cfg.print(f'\tBuilt octree {time.time()-t0:.5f}s')

    gp_num_iters = dataset.gp_num_iters
    lp_num_iters = dataset.lp_num_iters
    gp_bs = dataset.gp_bs
    lp_bs = dataset.lp_bs
    gp_use_num = gp_bs * min(num_sdf_its, gp_num_iters)
    lp_use_num = lp_bs * min(num_sdf_its, lp_num_iters)
    gp_rem = dataset.num_gp - gp_use_num
    lp_rem = dataset.num_lp - lp_use_num
    gp_start = np.random.randint(gp_rem) if gp_rem > 0 else 0
    lp_start = np.random.randint(lp_rem) if lp_rem > 0 else 0

    randomised_gp = dataset.randomised_gp[gp_start:gp_start+gp_use_num].cuda()
    randomised_lp = dataset.randomised_lp[lp_start:lp_start+lp_use_num].cuda()
    dists_gp2zls, idxes_gp2zls = keops_knn(randomised_gp, zls_points_cuda, 1)
    dists_lp2zls, idxes_lp2zls = keops_knn(randomised_lp, zls_points_cuda, 1)


    sdf_optimiser.zero_grad()
    for sdf_it in range(num_sdf_its):
        # 200k points for ds, 30k for zls, 100k for mnfld
        gp_ind = sdf_it % gp_num_iters
        batch_gp = randomised_gp[gp_ind*gp_bs : (gp_ind+1)*gp_bs]
        lp_ind = sdf_it % lp_num_iters
        batch_lp = randomised_lp[lp_ind*lp_bs : (lp_ind+1)*lp_bs]
        domain_samples_cuda = torch.cat((batch_lp, batch_gp), axis=0)

        dists_ds2zls = torch.cat((dists_lp2zls[lp_ind*lp_bs : (lp_ind+1)*lp_bs], dists_gp2zls[gp_ind*gp_bs : (gp_ind+1)*gp_bs]), axis=0)
        idxes_ds2zls = torch.cat((idxes_lp2zls[lp_ind*lp_bs : (lp_ind+1)*lp_bs], idxes_gp2zls[gp_ind*gp_bs : (gp_ind+1)*gp_bs]), axis=0)

        weighting = 0.3 + 0.7*(dists_ds2zls<5*rad) + 0.3*(dists_ds2zls<2*rad) # (N_ds)
        
        zls_pred_dict = sdf_model(zls_points_cuda, return_divergence=False)
        domain_pred_dict = sdf_model(domain_samples_cuda, return_divergence=False)

        zls_sdf_pred = zls_pred_dict[FieldHeadNames('sdf')].reshape(-1) # (N_zls,)
        zls_sdf_grad = zls_pred_dict[FieldHeadNames('gradient')] # (N_zls, 2)

        domain_sdf_pred = domain_pred_dict[FieldHeadNames('sdf')].reshape(-1) # (N_ds,)
        domain_sdf_grad = domain_pred_dict[FieldHeadNames('gradient')] # (N_ds, D)

        if use_mnfld:
            mnfld_pred_dict = sdf_model(cfg.mnfld_pnts_cuda, return_divergence=False, return_grad=False)
            cfg.mnfld_pnts_cuda.requires_grad = False
            mnfld_sdf_pred = mnfld_pred_dict[FieldHeadNames('sdf')].reshape(-1) # (N_mnfld,)

        if (zls_sdf_pred.isnan().any() or zls_sdf_grad.isnan().any() 
            or domain_sdf_pred.isnan().any() or domain_sdf_grad.isnan().any()
            ):
            import pdb; pdb.set_trace()

        loss_dict = {}

        loss_dict['zls_term'] = (zls_sdf_pred).abs().mean()

        if use_mnfld:
            loss_dict['mnfld_term'] = (mnfld_sdf_pred + rad).clamp(0, 1).mean()
        else:
            loss_dict['mnfld_term'] = loss_dict['zls_term'] * 0

        grad_norm = 1
        loss_dict['zls_eik_term'] = (zls_sdf_grad.norm(dim=1) - grad_norm).abs().mean()
        
        loss_dict['domain_eik_term'] = ((domain_sdf_grad.norm(dim=1) - grad_norm).abs()
                                             * weighting).mean()
        
        if normals is not None:
            # normals and zls_sdf_grad are (M, 3)
            loss_dict['normal_term'] = (1-F.cosine_similarity(normals, zls_sdf_grad, dim=-1)).mean() # (M,)
        else:
            loss_dict['normal_term'] = loss_dict['zls_term'] * 0

        if normals is None:
            normals = zls_sdf_grad.detach()
        disps = zls_points_cuda.detach()[idxes_ds2zls] - domain_samples_cuda.detach() # (N_ds, D)
        dists = disps.norm(dim=-1) # (N_ds, )
        signs = ((-disps) * (normals[idxes_ds2zls])).sum(axis=-1).sign()
        sdfs = dists * signs
        # loss_dict['sdf_term'] = (domain_sdf_pred - sdfs).abs().mean()
        # loss_dict['sdf_term'] = ((domain_sdf_pred - sdfs).abs() * weighting).mean()
        sdf_scaling = weighting * 1 / torch.maximum(domain_sdf_pred.abs(), sdfs.abs()).clamp(1e-5, 10).detach()
        loss_dict['sdf_term'] = ((domain_sdf_pred - sdfs).abs() * sdf_scaling).mean()
        

        grad_dir = domain_sdf_grad/domain_sdf_grad.norm(dim=-1, keepdim=True) # (N_ds, D)
        pulled_points = domain_samples_cuda - grad_dir * domain_sdf_pred.unsqueeze(-1) # (N_ds, D)
        closest_points = zls_points_cuda[idxes_ds2zls]
        # loss_dict['npull_term'] = ((pulled_points - closest_points).norm(dim=-1)*weighting).mean()
        npull_scaling = weighting * 1 / ((domain_samples_cuda - closest_points).detach().norm(dim=-1) + 1e-5)
        loss_dict['npull_term'] = ((pulled_points - closest_points).norm(dim=-1) * npull_scaling).mean()

        
        loss_dict['weighted_zls_term'] = weights[0] * loss_dict['zls_term']
        loss_dict['weighted_zls_eik_term'] = weights[1] * loss_dict['zls_eik_term']
        loss_dict['weighted_domain_eik_term'] = weights[2] * loss_dict['domain_eik_term']
        loss_dict['weighted_sdf_term'] = weights[3] * loss_dict['sdf_term']
        loss_dict['weighted_normal_term'] = weights[4] * loss_dict['normal_term']
        loss_dict['weighted_npull_term'] = weights[5] * loss_dict['npull_term']
        loss_dict['weighted_mnfld_term'] = weights[0] * loss_dict['mnfld_term']

        loss_dict['loss'] = loss_dict['weighted_zls_term'] + loss_dict['weighted_zls_eik_term'] + \
            loss_dict['weighted_domain_eik_term'] + loss_dict['weighted_sdf_term'] + \
            loss_dict['weighted_normal_term'] + loss_dict['weighted_npull_term'] + loss_dict['weighted_mnfld_term']

        if sdf_it % 20 == 19 or sdf_it == 0:
        # if sdf_it % 1 == 0:
            cfg.print(f'\t{it} {sdf_it}', 
                f"Loss = {loss_dict['loss'].item():.5f}" + 
                f", PC ZLS = {loss_dict['weighted_zls_term'].item():.5f}" +
                    f"({loss_dict['zls_term']:.5f})"+
                f", PC Mlfld = {loss_dict['weighted_mnfld_term'].item():.5f}"+
                    f"({loss_dict['mnfld_term']:.5f})"+
                f", PC Eik = {loss_dict['weighted_zls_eik_term'].item():.5f}"+
                    f"({loss_dict['zls_eik_term']:.5f})"+
                f", Dom Eik = {loss_dict['weighted_domain_eik_term'].item():.5f}"+
                    f"({loss_dict['domain_eik_term']:.5f})"+
                f", SDF = {loss_dict['weighted_sdf_term'].item():.5f}"+
                    f"({loss_dict['sdf_term']:.5f})"+
                f", Normal = {loss_dict['weighted_normal_term'].item():.5f}"+
                    f"({loss_dict['normal_term']:.5f})"+
                f", NPull = {loss_dict['weighted_npull_term'].item():.5f}"+
                    f"({loss_dict['npull_term']:.5f})"+
                "")
        if cfg.use_wandb: wandb.log({'it':it, **loss_dict})

        loss_dict['loss'].backward()
        sdf_optimiser.step()
        sdf_optimiser.zero_grad()
    zls_points_cuda.requires_grad = False # sdf_model set it to true, so set it back now
    return domain_sdf_pred.detach(), domain_sdf_grad.detach()


def project_points_to_zls(cfg, points_cuda):
    sdf_model = cfg.sdf_model

    # points_cuda: (n, D)
    # iterate sdf projection z = z - \phi(z)*n_z until converged
    # n_z = nabla_z \phi(z) / ||nabla_z \phi(z)||
    # converged when max diff between its is < 1e-4
    with torch.no_grad():
        old_sdf_pred = None
        diff_max = 10
        for proj_iter in range(50):
            pred_dict = sdf_model(points_cuda)
            sdf_pred = pred_dict[FieldHeadNames('sdf')].reshape(-1) # (n,)
            sdf_grad = pred_dict[FieldHeadNames('gradient')] # (n, D)
            normals = (sdf_grad/sdf_grad.norm(dim=1,keepdim=True))
            points_cuda = (points_cuda - sdf_pred.unsqueeze(-1) * normals)
            if old_sdf_pred is None:
                old_sdf_pred = sdf_pred.detach().clone()
            else:
                diff_max = (sdf_pred - old_sdf_pred).abs().max()
                if diff_max.item() < 1e-4:
                    # cfg.print(f'\tBreaking out of projection loop at iter {proj_iter}')
                    old_sdf_pred = sdf_pred.detach().clone()
                    break
                old_sdf_pred = sdf_pred.detach().clone()
        
        # Only keep points that are actually on the zls. First want to make sure
        # diff of last iteration is also <1e-4. Then to ensure its on the zls, 
        # specify that sdf_value.abs() < 1e-4. Finally to ensure it didn't get stuck in
        # a saddle point, ensure that sdf_grad > 0.5.
        pred_dict = sdf_model(points_cuda)
        sdf_pred = pred_dict[FieldHeadNames('sdf')].reshape(-1) # (n,)
        sdf_grad_norm = pred_dict[FieldHeadNames('gradient')].norm(dim=-1) # (n,)
        diff = (sdf_pred - old_sdf_pred).abs()

        # sdf_min = sdf_pred.min(); sdf_mean = sdf_pred.mean(); sdf_max = sdf_pred.max()
        # sdf_grad_norm_min = sdf_grad_norm.min(); sdf_grad_norm_mean = sdf_grad_norm.mean(); sdf_grad_norm_max = sdf_grad_norm.max()
        # cfg.print(f'\t Projection took {proj_iter} iters, diff_max {diff_max:.6f}, sdf_min {sdf_min:.8f}, sdf_mean {sdf_mean:.8f}, sdf_max {sdf_max:.8f}')
        # cfg.print(f'\t sdf_grad_norm_min {sdf_grad_norm_min:.8f}, sdf_grad_norm_mean {sdf_grad_norm_mean:.8f}, sdf_grad_norm_max {sdf_grad_norm_max:.8f}')

        diff_idxes = diff < 1e-4
        sdf_idxes = sdf_pred.abs() < 1e-4
        grad_idxes = sdf_grad_norm > 0.5
        both_idxes = diff_idxes & sdf_idxes & grad_idxes
        cfg.print(f'\t Projection removed {sdf_pred.shape[0] - both_idxes.float().sum():.1f} points')
        points_cuda = points_cuda[both_idxes]
        normals = normals[both_idxes]

        # pred_dict = sdf_model(points_cuda)
        # sdf_pred = pred_dict[FieldHeadNames('sdf')].reshape(-1) # (n,)
        # sdf_grad_norm = pred_dict[FieldHeadNames('gradient')].norm(dim=-1) # (n,)
        # sdf_min = sdf_pred.min(); sdf_mean = sdf_pred.mean(); sdf_max = sdf_pred.max()
        # sdf_grad_norm_min = sdf_grad_norm.min(); sdf_grad_norm_mean = sdf_grad_norm.mean(); sdf_grad_norm_max = sdf_grad_norm.max()
        # cfg.print(f'\t After removing, sdf_min {sdf_min:.8f}, sdf_mean {sdf_mean:.8f}, sdf_max {sdf_max:.8f}')
    return points_cuda.detach(), normals.detach()

def resample_zls(cfg, est_pc_cuda, normals, downsample=None, oversample_crit_region=True):
    M_d = cfg.M
    mnfld_pnts_cuda = cfg.mnfld_pnts_cuda.detach()

    if downsample is not None:
        M_d = downsample
        fps_points,fps_indices = sample_farthest_points_naive_new(est_pc_cuda, K=M_d)
        est_pc_cuda = fps_points # (M_d, D)
        normals = normals[fps_indices] # (M_d,)
        # print(f'Applied first Farthest Point Sampling, took {time.time()-t0:.3f}s'); t0 = time.time()

    est_pc_cuda = est_pc_cuda.unsqueeze(1) + (0.05*torch.randn(M_d,30,cfg.D).clamp(-1,1)).cuda() # (M_d, 30, 3)
    est_pc_cuda = rearrange(est_pc_cuda, 'n p d -> (n p) d') # (M_d * 30, 3)
    # if cfg.use_rerun:
    #     res_rand1 = est_pc_cuda.detach().cpu().numpy()
    #     rr.log_points("res_rand1", positions=res_rand1, colors=(100,225,0), radii=0.003)
    est_pc_cuda, normals = project_points_to_zls(cfg, est_pc_cuda) # (M_d*30, 3), (M_d*30, 3)
    # if cfg.use_rerun:
    #     res_proj1 = est_pc_cuda.detach().cpu().numpy()
    #     rr.log_points("res_proj1", positions=res_proj1, colors=(100,225,0), radii=0.003)
    # print(f'Oversampled and projected back to ZLS, took {time.time()-t0:.3f}s'); t0 = time.time()

    
    if oversample_crit_region:
        crit_region_num = 1000
        crit_region_upsample = 10000
        fps_points,fps_indices = sample_farthest_points_naive_new(est_pc_cuda, K=cfg.M-crit_region_upsample)
        est_pc_cuda = fps_points # (M-5000, D)
        normals = normals[fps_indices] # (M-5000,D)
        # print(f'Applied second Farthest Point Sampling, took {time.time()-t0:.3f}s'); t0 = time.time()
        # if cfg.use_rerun:
        #     res_fps1 = est_pc_cuda.detach().cpu().numpy()
        #     rr.log_points("res_fps1", positions=res_fps1, colors=(100,225,0), radii=0.003)

        dists, close_idxes = keops_knn(est_pc_cuda, mnfld_pnts_cuda, 1, return_numpy=False)
        furthest_dists_idxes = torch.argsort(dists)[-crit_region_num:]
        closest_points = est_pc_cuda[furthest_dists_idxes]
        # closest_points = closest_points.unsqueeze(1) + (0.01*torch.randn(crit_region_num,30,cfg.D).clamp(-1,1)).cuda() # (500, 30, 3)
        closest_points = closest_points.unsqueeze(1) + (0.01*torch.randn(crit_region_num,10,cfg.D).clamp(-1,1)).cuda() # (500, 30, 3)
        closest_points = rearrange(closest_points, 'n p d -> (n p) d') # (500 * 30, 3)
        # if cfg.use_rerun:
        #     res_rand2 = closest_points.detach().cpu().numpy()
        #     rr.log_points("res_rand2", positions=res_rand2, colors=(100,225,0), radii=0.003)
        closest_points, cl_normals = project_points_to_zls(cfg, closest_points) # (500*30, 3), (500*30, 3)
        # if cfg.use_rerun:
        #     res_proj2 = closest_points.detach().cpu().numpy()
        #     rr.log_points("res_proj2", positions=res_proj2, colors=(100,225,0), radii=0.003)

        # fps_points2,fps_indices2 = sample_farthest_points_naive_new(closest_points, K=crit_region_upsample)
        # closest_points = fps_points2 # (5000, D)
        # cl_normals = cl_normals[fps_indices2] # (5000, D)
        # # if cfg.use_rerun:
        # #     res_fps2 = closest_points.detach().cpu().numpy()
        # #     rr.log_points("res_fps2", positions=res_fps2, colors=(100,225,0), radii=0.003)

        est_pc_cuda = torch.cat((est_pc_cuda, closest_points), dim=0)
        normals = torch.cat((normals, cl_normals), dim=0)
    else:
        fps_points,fps_indices = sample_farthest_points_naive_new(est_pc_cuda, K=cfg.M)
        est_pc_cuda = fps_points # (M, D)
        normals = normals[fps_indices] # (M,)
        # print(f'Applied second Farthest Point Sampling, took {time.time()-t0:.3f}s'); t0 = time.time()

    return est_pc_cuda, normals

################################
## Main Algorithm Components
################################

def initialisation(cfg, initial_rad):
    mnfld_pnts_cuda = cfg.mnfld_pnts_cuda.detach()
    vis_obj = cfg.vis_obj
    sdf_model = cfg.sdf_model

    initial_est_pc = generate_samples_on_sphere(cfg.M*3, 1.0)

    est_pc_cuda = torch.tensor(initial_est_pc.astype(np.float32)).cuda() # (M, D)
    normals = est_pc_cuda / est_pc_cuda.norm(dim=-1, keepdim=True)
    
    vis_obj.plot_figure(-10, mnfld_pnts=mnfld_pnts_cuda, 
                        est_pc=est_pc_cuda, normals=normals)
    vis_obj.it_ended()

    #### New Initialisation #####
    t0 = time.time()
    step_size = initial_rad / 10
    steps = int(2.5 / step_size)
    
    est_pc_cuda, normals = find_exterior_level_set(cfg, mnfld_pnts_cuda, est_pc_cuda, normals,  
                                # rad=initial_rad, step_size=step_size, initial_repeats=3, bounce_repeats=3, steps=steps)
                                rad=initial_rad, step_size=step_size, initial_repeats=1, bounce_repeats=0, steps=steps)
    vis_obj.plot_figure(-5, est_pc=est_pc_cuda, normals=normals)
    vis_obj.it_ended()

    cfg.print(f'\tinit: Approximate dilated surface made, rad is now {initial_rad}, took {time.time()-t0:.3f}s'); t0 = time.time()

    optimise_sdf(
        cfg, -2, est_pc_cuda, 100, weights = [100, 10, 50, 50, 10, 20], rad=initial_rad, normals=normals, lr=1e-4)
    cfg.print(f'\tinit: SDF Optimised, took {time.time()-t0:.3f}s'); t0 = time.time()
    vis_obj.plot_figure(-2, sdf_model=sdf_model)
    vis_obj.it_ended()

    optimise_sdf(
        cfg, -1, est_pc_cuda, 100, weights = [100, 10, 50, 50, 10, 20], rad=initial_rad, normals=normals, lr=3e-5)
    cfg.print(f'\tinit: SDF Optimised, took {time.time()-t0:.3f}s'); t0 = time.time()
    vis_obj.plot_figure(-1, sdf_model=sdf_model)
    vis_obj.it_ended()
    return est_pc_cuda, normals

def update_estimated_surface(cfg, it, est_pc_cuda, normals, rad):
    
    vis_obj = cfg.vis_obj
    sdf_model = cfg.sdf_model

    t0 = time.time()
    # SDF Iterations
    optimise_sdf(
        cfg, it, est_pc_cuda, 100, weights = [500, 10, 2, 100, 0, 1], rad=rad, use_mnfld=True, lr=3e-5)
    cfg.print(f'\tupdate_phi: SDF Optimised, took {time.time()-t0:.3f}s'); t0 = time.time()
    # vis_obj.plot_figure(it, sdf_model=sdf_model)

    # Project back to SDF Zero Level Set
    est_pc_cuda, normals = project_points_to_zls(cfg, est_pc_cuda)
    cfg.print(f'\tupdate_phi: Projected back to ZLS, took {time.time()-t0:.3f}s'); t0 = time.time()
    vis_obj.plot_figure(it, est_pc=est_pc_cuda, normals=normals)

    # SDF Iterations
    optimise_sdf(
        cfg, it, est_pc_cuda, 100, weights = [100, 50, 50, 10, 0, 50], rad=rad, use_mnfld=True, lr=3e-5)
    cfg.print(f'\tupdate_phi: SDF Optimised, took {time.time()-t0:.3f}s'); t0 = time.time()
    vis_obj.plot_figure(it, sdf_model=sdf_model)

    # Resample the Zero Level Set
    # add noise, project back, and then do farthest point sampling
    M_d = cfg.M//20 # Downsample before oversampling to resample surface
    est_pc_cuda, normals = resample_zls(cfg, est_pc_cuda, normals, 
                                downsample=M_d, oversample_crit_region=True)
    cfg.print(f'\tupdate_phi: Resampled ZLS, took {time.time()-t0:.3f}s'); t0 = time.time()
    vis_obj.plot_figure(it, est_pc=est_pc_cuda, normals=normals)
    return est_pc_cuda, normals


def move_points(cfg, est_pc_cuda, normals, rad):
    mnfld_pnts_cuda = cfg.mnfld_pnts_cuda.detach()
    sdf_model = cfg.sdf_model
    pcd_rad = cfg.pcd_rad

    setY = est_pc_cuda
    setX = mnfld_pnts_cuda
    dirY = normals
    # setY is (M, D), setX is (N, D), dirY is (M, D)

    with torch.no_grad():
        setY_i = LazyTensor(setY.unsqueeze(1)) # (M, 1, D) LT
        setX_j = LazyTensor(setX.unsqueeze(0)) # (1, N, D) LT
        D_ij = ((setY_i - setX_j)**2).sum(dim=-1).sqrt() # (M, N) LT
        unit_disp_ji = ((setY_i - setX_j) / D_ij).T # (N, M, D) LT
        dir = dirY.unsqueeze(0) # (1, M, D)
        # cos sim between y-x and outward normal, equal to cos sim between x-y and inward normal
        cos_sim_ji = (unit_disp_ji*dir).sum(dim=-1) # (N, M) LT

        # 0 if cos_sim>0.9, 0.5 if cos_sim=0.9, 1 otherwise
        within_cone_constraint = (0.9 - cos_sim_ji).sign()*0.5 + 0.5

        # 0 if cos_sim>0.0, 0.5 if cos_sim=0.0, 1 otherwise
        within_halfspace_constraint = (0.0 - cos_sim_ji).sign()*0.5 + 0.5

        # 0 if dist<2*pcd_rad, 0.5 if dist=2*pcd_rad, 1 otherwise
        within_ball_constraint = (D_ij.T - 2*pcd_rad).sign()*0.5 + 0.5

        within_halfball_constraint = within_halfspace_constraint + within_ball_constraint

        dist_within_inward_dir = D_ij.T + within_cone_constraint * within_halfball_constraint # (N, M) LT

        min_dists, min_idxes = dist_within_inward_dir.min_argmin(dim=0) # (M, 1), (M, 1)
        min_dists = min_dists.squeeze().detach()
        min_idxes = min_idxes.squeeze().detach()

    matched_idxes = min_idxes.clone()

    # remove if not within cone or half ball
    matched_mask = min_dists <= 0.9999
    num_not_matched = matched_mask.shape[0] - matched_mask.sum()

    cfg.print(f'\tmove_points: Removing {num_not_matched.item()} as no match within cone or halfball')
    if cfg.use_rerun:
        removed_points = est_pc_cuda[~matched_mask].detach().cpu().numpy()
        rr.log_points("removed_points", positions=removed_points, colors=(100,225,0), radii=0.003)

    # matched X for each Y
    setX_cl2_setY = setX[matched_idxes[matched_mask]] # (Ma, D)
    # displacements, distances and directions of matched X from Y
    disps = setX_cl2_setY - setY[matched_mask] # (Ma, D)
    disps_dist = disps.norm(dim=-1, keepdim=True) # (Ma, 1)
    disps_dir = disps / disps_dist # (Ma, D)
    target_vec = disps_dir * (disps_dist - rad) # (Ma, 3)

    # cossim = (disps_dir * (-normals[matched_mask])).sum(axis=-1, keepdims=True)
    # noprog = torch.logical_and(cossim.squeeze() < 0.1, disps_dist.squeeze() < 0.005)
    # cfg.print(f'\tdists: {disps_dist.min().item():.3f}, {disps_dist.max().item():.3f}; cossim: {cossim.min().item():.3f}, {cossim.max().item():.3f}')
    # cfg.print(f'\tnum no progress: {noprog.sum()}')

    # distance of target_vec projected onto the normal direction
    target_dist = (target_vec * (-normals[matched_mask])).sum(axis=-1, keepdims=True) # (Ma, 1)
    
    correspondences = torch.stack((setY[matched_mask], setX[matched_idxes[matched_mask]]), axis=1)
    correspondences = correspondences.reshape(-1,cfg.D)

    adjusted_dist = target_dist.clone().detach() # (Ma, 1)
    min_val = adjusted_dist.min().item()
    # min_val = max(min_val, 0.05)
    min_val = max(min_val, 0.1)
    # min_val = max(min_val, 2*pcd_rad)
    # min_val = max(min_val, 4*pcd_rad)
    cfg.print(f'\tmove_points: min_val {min_val:.5f}')
    # min_val = 0.1
    adjusted_dist = torch.minimum(adjusted_dist, torch.zeros(1,1).cuda()+min_val) # (Ma, 1)
    adjusted_dist = torch.maximum(adjusted_dist, torch.zeros(1,1).cuda()-min_val) # (Ma, 1)
    new_est_pc_cuda = est_pc_cuda[matched_mask] - normals[matched_mask] * adjusted_dist # Points moved to new location! (Ma, 3)
    new_normals = normals[matched_mask]

    with torch.no_grad():
        pred_dict = sdf_model(new_est_pc_cuda)
        sdf_pred = pred_dict[FieldHeadNames('sdf')].reshape(-1) # (n,)
        # for points far away, they should always be moving inwards, if they move outward they probably closed up a hole, e.g.
        # the hole in a torus
        # for points close to the level set, i.e. adjusted_dist < 1e-2, the new point might move in or out of the current shape
        # as per the current sdf_pred values, so don't remove for these points
        went_inside_mask = torch.logical_or(sdf_pred.detach() <= 1e-3 , adjusted_dist.squeeze() < 1e-2)
        new_est_pc_cuda = new_est_pc_cuda.detach()

    cfg.print(f'\tmove_points: Removing {new_est_pc_cuda.shape[0] - went_inside_mask.float().sum():.1f} points as they moved to outside region')
    
    new_est_pc_cuda = new_est_pc_cuda[went_inside_mask]
    new_normals = new_normals[went_inside_mask]

    return new_est_pc_cuda, new_normals, correspondences, adjusted_dist


def finetune(cfg, last_init_iter, rad):
    mnfld_pnts_cuda = cfg.mnfld_pnts_cuda.detach()
    vis_obj = cfg.vis_obj
    sdf_model = cfg.sdf_model
    # pcd_rad = cfg.pcd_rad

    it = last_init_iter
    cfg.print(f'\tfinetune: Training on input points now')
    
    # num_finetune = 50
    # num_finetune = 15
    num_finetune = 10
    for f_it in range(num_finetune):
        cfg.print(f'\tfinetune: {f_it+1}/{num_finetune}'); t0 = time.time()
        it = it+1
        optimise_sdf(
            # cfg, it, mnfld_pnts_cuda.detach(), 200, weights = [100, 2, 1, 0, 0, 1], rad=rad, lr=1e-5,
            # cfg, it, mnfld_pnts_cuda.detach(), 200, weights = [10, 2, 1, 0, 0, 0], rad=rad, lr=1e-5,
            cfg, it, mnfld_pnts_cuda.detach(), 200, weights = [100, 2, 1, 0, 0, 0], rad=rad, lr=1e-5,
            subset_size = None,
            # subset_size = 30000,
            )
        cfg.print(f'\tfinetune: SDF Optimised, took {time.time()-t0:.3f}s'); t0 = time.time()
        vis_obj.plot_figure(it, sdf_model=sdf_model)
        vis_obj.it_ended()
    return it

################################
## Visualisation
################################

class PGSDF_Vis():
    def __init__(self, cfg, dataset):
        self.cfg = cfg
        self.dataset = dataset
        self.plot_count = 0
    
    def it_ended(self):
        # reset plot count
        self.plot_count = 0
    
    def plot_figure(self, it,
                mnfld_pnts = None, sdf_model = None, est_pc = None, 
                correspondences = None, normals = None, mnfld_normals = None):
        if sdf_model is not None:
            vis_path = '{}/{}_{}_it{}.ply'.format(self.cfg.log_dir, self.cfg.shape_name, self.plot_count, it)
            implicit_func = lambda points : sdf_model(points.to(next(sdf_model.parameters()).device))[FieldHeadNames('sdf')].detach()
            mesh = implicitFunc2mesh(self.dataset.grid_obj, implicit_func, self.dataset.scale_obj.unscale_points, chunk_size = 100000, use_tqdm=True)
            # mesh is scaled back to original coords using self.dataset.scale_obj.unscale_points
            mesh.export(vis_path)

        if self.cfg.use_rerun:
            if it is not None:
                rr.set_time_sequence("iteration", it*10 + self.plot_count)
            if mnfld_pnts is not None:
                if torch.is_tensor(mnfld_pnts):
                    mnfld_pnts = mnfld_pnts.detach().cpu().numpy()
                rr.log_points("mnfld_pnts", positions=mnfld_pnts, colors=(255,0,255), radii=0.003)
                if mnfld_normals is not None:
                    if torch.is_tensor(mnfld_normals):
                        mnfld_normals = mnfld_normals.detach().cpu().numpy()
                    mnfld_normals_start = mnfld_pnts
                    mnfld_normals_end = mnfld_pnts + mnfld_normals*0.05
                    norm_vecs = np.stack((mnfld_normals_start, mnfld_normals_end), axis=1).reshape(-1,self.cfg.D)
                    # rr.log_arrow("simple", origin=mnfld_pnts, vector=0.1*mnfld_normals, width_scale=0.002) # causes errors when run
                    rr.log_line_segments('mnfld_normals', norm_vecs, color=(255,0,255), stroke_width=0.0008)
            if est_pc is not None:
                if torch.is_tensor(est_pc):
                    est_pc = est_pc.detach().cpu().numpy()
                rr.log_points("est_pc", positions=est_pc, colors=(0,225,0), radii=0.003)
                if normals is not None:
                    if torch.is_tensor(normals):
                        normals = normals.detach().cpu().numpy()
                    normals_start = est_pc
                    normals_end = est_pc + normals*0.05
                    norm_vecs = np.stack((normals_start, normals_end), axis=1).reshape(-1,self.cfg.D)
                    # rr.log_arrow("simple", origin=est_pc, vector=0.1*normals, width_scale=0.002) # causes errors when run
                    rr.log_line_segments('normals', norm_vecs, color=(50,255,0), stroke_width=0.0008)
            if sdf_model is not None:
                # scale mesh to recon domain for visualisation
                mesh_pos = self.dataset.scale_obj.scale_points(mesh.vertices)
                mesh_norms = self.dataset.scale_obj.scale_points(mesh.vertex_normals)
                rr.log_mesh('mesh', positions=mesh_pos, indices=mesh.faces, normals=mesh_norms)
            if correspondences is not None:
                if torch.is_tensor(correspondences):
                    correspondences = correspondences.detach().cpu().numpy()
                # correspondences = correspondences.reshape(-1,self.cfg.D)
                
                num_points = correspondences.shape[0] // 2
                corr_idxes = np.zeros((num_points, 2), dtype=np.int32)
                corr_idxes[np.random.permutation(num_points)[:10000]] = np.ones(2,dtype=np.int32)
                corr_idxes = corr_idxes.reshape(-1).astype(bool)
                correspondences = correspondences.reshape(-1,self.cfg.D)[corr_idxes]\
                
                rr.log_line_segments('correspondences', correspondences, color=(255,255,255), stroke_width=0.0008)


        if it is not None:
            self.plot_count += 1
            if self.plot_count > 9:
                print('plot count is larger than 9 for this iteration', self.plot_count, it)
