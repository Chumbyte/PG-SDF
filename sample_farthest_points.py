# From Pytorch3D

from random import randint
from typing import Optional, Tuple, Union, List

import torch

# from .utils import masked_gather
def masked_gather(points: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """
    Helper function for torch.gather to collect the points at
    the given indices in idx where some of the indices might be -1 to
    indicate padding. These indices are first replaced with 0.
    Then the points are gathered after which the padded values
    are set to 0.0.
    Args:
        points: (N, P, D) float32 tensor of points
        idx: (N, K) or (N, P, K) long tensor of indices into points, where
            some indices are -1 to indicate padding
    Returns:
        selected_points: (N, K, D) float32 tensor of points
            at the given indices
    """

    if len(idx) != len(points):
        raise ValueError("points and idx must have the same batch dimension")

    N, P, D = points.shape

    if idx.ndim == 3:
        # Case: KNN, Ball Query where idx is of shape (N, P', K)
        # where P' is not necessarily the same as P as the
        # points may be gathered from a different pointcloud.
        K = idx.shape[2]
        # Match dimensions for points and indices
        idx_expanded = idx[..., None].expand(-1, -1, -1, D)
        points = points[:, :, None, :].expand(-1, -1, K, -1)
    elif idx.ndim == 2:
        # Farthest point sampling where idx is of shape (N, K)
        idx_expanded = idx[..., None].expand(-1, -1, D)
    else:
        raise ValueError("idx format is not supported %s" % repr(idx.shape))

    idx_expanded_mask = idx_expanded.eq(-1)
    idx_expanded = idx_expanded.clone()
    # Replace -1 values with 0 for gather
    idx_expanded[idx_expanded_mask] = 0
    # Gather points
    selected_points = points.gather(dim=1, index=idx_expanded)
    # Replace padded values
    selected_points[idx_expanded_mask] = 0.0
    return selected_points

def sample_farthest_points_naive(
    points: torch.Tensor,
    lengths: Optional[torch.Tensor] = None,
    K: Union[int, List, torch.Tensor] = 50,
    random_start_point: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Iterative farthest point sampling algorithm [1] to subsample a set of
    K points from a given pointcloud. At each iteration, a point is selected
    which has the largest nearest neighbor distance to any of the
    already selected points.
    Farthest point sampling provides more uniform coverage of the input
    point cloud compared to uniform random sampling.
    [1] Charles R. Qi et al, "PointNet++: Deep Hierarchical Feature Learning
        on Point Sets in a Metric Space", NeurIPS 2017.
    Args:
        points: (N, P, D) array containing the batch of pointclouds
        lengths: (N,) number of points in each pointcloud (to support heterogeneous
            batches of pointclouds)
        K: samples you want in each sampled point cloud (this is typically << P). If
            K is an int then the same number of samples are selected for each
            pointcloud in the batch. If K is a tensor is should be length (N,)
            giving the number of samples to select for each element in the batch
        random_start_point: bool, if True, a random point is selected as the starting
            point for iterative sampling.
    Returns:
        selected_points: (N, K, D), array of selected values from points. If the input
            K is a tensor, then the shape will be (N, max(K), D), and padded with
            0.0 for batch elements where k_i < max(K).
        selected_indices: (N, K) array of selected indices. If the input
            K is a tensor, then the shape will be (N, max(K), D), and padded with
            -1 for batch elements where k_i < max(K).
    """
    N, P, D = points.shape
    device = points.device

    # Validate inputs
    if lengths is None:
        lengths = torch.full((N,), P, dtype=torch.int64, device=device)

    if lengths.shape[0] != N:
        raise ValueError("points and lengths must have same batch dimension.")

    # TODO: support providing K as a ratio of the total number of points instead of as an int
    if isinstance(K, int):
        K = torch.full((N,), K, dtype=torch.int64, device=device)
    elif isinstance(K, list):
        K = torch.tensor(K, dtype=torch.int64, device=device)

    if K.shape[0] != N:
        raise ValueError("K and points must have the same batch dimension")

    # Find max value of K
    max_K = torch.max(K)

    # List of selected indices from each batch element
    all_sampled_indices = []

    for n in range(N):
        # Initialize an array for the sampled indices, shape: (max_K,)
        sample_idx_batch = torch.full(
            (max_K,), fill_value=-1, dtype=torch.int64, device=device
        )

        # Initialize closest distances to inf, shape: (P,)
        # This will be updated at each iteration to track the closest distance of the
        # remaining points to any of the selected points
        # pyre-fixme[16]: `torch.Tensor` has no attribute new_full.
        closest_dists = points.new_full(
            (lengths[n],), float("inf"), dtype=torch.float32
        )

        # Select a random point index and save it as the starting point
        selected_idx = randint(0, lengths[n] - 1) if random_start_point else 0
        sample_idx_batch[0] = selected_idx

        # If the pointcloud has fewer than K points then only iterate over the min
        k_n = min(lengths[n], K[n])

        # Iteratively select points for a maximum of k_n
        for i in range(1, k_n):
            # Find the distance between the last selected point
            # and all the other points. If a point has already been selected
            # it's distance will be 0.0 so it will not be selected again as the max.
            dist = points[n, selected_idx, :] - points[n, : lengths[n], :]
            dist_to_last_selected = (dist ** 2).sum(-1)  # (P - i)

            # If closer than currently saved distance to one of the selected
            # points, then updated closest_dists
            closest_dists = torch.min(dist_to_last_selected, closest_dists)  # (P - i)

            # The aim is to pick the point that has the largest
            # nearest neighbour distance to any of the already selected points
            selected_idx = torch.argmax(closest_dists)
            sample_idx_batch[i] = selected_idx

        # Add the list of points for this batch to the final list
        all_sampled_indices.append(sample_idx_batch)

    all_sampled_indices = torch.stack(all_sampled_indices, dim=0)

    # Gather the points
    all_sampled_points = masked_gather(points, all_sampled_indices)

    # Return (N, max_K, D) subsampled points and indices
    return all_sampled_points, all_sampled_indices

def sample_farthest_points_naive_new(
    points: torch.Tensor,
    K: int = 50,
    selected_idx = 0
) -> Tuple[torch.Tensor, torch.Tensor]:
    P, D = points.shape
    device = points.device

    # Idxes of sampled points, initialised to -1, shape: (K,)
    sample_idx_batch = torch.full(
        (K,), fill_value=-1, dtype=torch.int64, device=device
    )

    # Choose the first point as the first chosen/sampled point
    selected_idx = selected_idx
    sample_idx_batch[0] = selected_idx

    # Closest dist of each point to all chosen points, initialised to inf, shape: (P,)
    closest_dists = torch.full(
        (P,), float("inf"), dtype=torch.float32, device=device
    )

    # Iteratively select points for 1...K-1
    for i in range(1, K):
        # Calc dist of last chosen point to all points
        dist_to_last_selected = ((points[selected_idx, :] - points) ** 2).sum(-1)  # (P - i)

        # Update dist of each point to all chosen points
        closest_dists = torch.min(dist_to_last_selected, closest_dists)  # (P - i)

        # Sample the point with maximum dist. Note chosen points have dist 0
        selected_idx = torch.argmax(closest_dists)
        sample_idx_batch[i] = selected_idx

    # Return (K, D) subsampled points and  (K,indices
    return points[sample_idx_batch], sample_idx_batch