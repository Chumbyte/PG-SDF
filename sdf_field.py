# Adapted from NerfStudio's NerfFacto model

import math
from dataclasses import dataclass, field
from typing import Optional, Type, Union
from enum import Enum

import math
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.parameter import Parameter
from typing_extensions import Literal

from nerfstudio.field_components.encodings import (NeRFEncoding)
from nerfstudio.fields.base_field import Field, FieldConfig

try:
    import tinycudann as tcnn
except ImportError:
    # tinycudann module doesn't exist
    pass

class FieldHeadNames(Enum):
    """Possible field outputs"""

    RGB = "rgb"
    SH = "sh"
    DENSITY = "density"
    NORMALS = "normals"
    PRED_NORMALS = "pred_normals"
    UNCERTAINTY = "uncertainty"
    TRANSIENT_RGB = "transient_rgb"
    TRANSIENT_DENSITY = "transient_density"
    SEMANTICS = "semantics"
    NORMAL = "normal"
    SDF = "sdf"
    ALPHA = "alpha"
    GRADIENT = "gradient"
    OCCUPANCY = "occupancy"


@dataclass
class SDFField_for_PC_Config(FieldConfig):
    """Nerfacto Model Config"""

    _target: Type = field(default_factory=lambda: SDFField_for_PC)
    num_layers: int = 2 # 8, only 2 needed for hash encoding, 8 needed for Periodic Encoding
    """Number of layers for geometric network"""
    hidden_dim: int = 256
    """Number of hidden dimension of geometric network"""
    bias: float = 0.5 # 0.8
    """sphere size of geometric initializaion"""
    geometric_init: bool = True
    # geometric_init: bool = False
    """Whether to use geometric initialization"""
    inside_outside: bool = False # True
    """whether to revert signed distance value, set to True for indoor scene"""
    weight_norm: bool = True
    """Whether to use weight norm for linear laer"""
    use_grid_feature: bool = True # False
    """Whether to use multi-resolution feature grids"""
    divide_factor: float = 2.0
    """Normalization factor for multi-resolution grids"""
    beta_init: float = 0.02
    """Init learnable beta value for transformation of sdf to density"""
    encoding_type: Literal["hash", "periodic", "tensorf_vm"] = "hash"
    """feature grid encoding type"""
    position_encoding_max_degree: int = 6
    """positional encoding max degree"""
    off_axis: bool = False
    """whether to use off axis encoding from mipnerf360"""

class SDFField_for_PC(Field):
    """_summary_

    Args:
        Field (_type_): _description_
    """

    config: SDFField_for_PC_Config

    def __init__(
        self,
        config: SDFField_for_PC_Config,
        in_dims: int = 3,
    ) -> None:
        super().__init__()
        self.config = config
        self.in_dims = in_dims

        self.use_grid_feature = self.config.use_grid_feature
        self.divide_factor = self.config.divide_factor

        # num_levels = 16
        num_levels = 7
        max_res = 2048
        base_res = 16
        # base_res = 1
        log2_hashmap_size = 19
        features_per_level = 2
        # use_hash = True
        use_hash = False
        smoothstep = True
        # growth_factor = np.exp((np.log(max_res) - np.log(base_res)) / (num_levels - 1))
        growth_factor = 1.5

        assert self.config.encoding_type == "hash", f"Used {self.config.encoding_type} encoding, but non-hash encodings not implemented for dim!= 3"

        if self.config.encoding_type == "hash":
            # feature encoding
            self.encoding = tcnn.Encoding(
                n_input_dims=self.in_dims,
                encoding_config={
                    "otype": "HashGrid" if use_hash else "DenseGrid",
                    "n_levels": num_levels,
                    "n_features_per_level": features_per_level,
                    "log2_hashmap_size": log2_hashmap_size,
                    "base_resolution": base_res,
                    "per_level_scale": growth_factor,
                    "interpolation": "Smoothstep" if smoothstep else "Linear",
                },
            )

        # we concat inputs position ourselves
        self.position_encoding = NeRFEncoding(
            in_dim=self.in_dims,
            num_frequencies=self.config.position_encoding_max_degree,
            min_freq_exp=0.0,
            max_freq_exp=self.config.position_encoding_max_degree - 1,
            include_input=False,
        )

        # TODO move it to field components
        # MLP with geometric initialization
        dims = [self.config.hidden_dim for _ in range(self.config.num_layers)]
        in_dim = self.in_dims + self.position_encoding.get_out_dim() + self.encoding.n_output_dims
        dims = [in_dim] + dims + [1,]
        self.num_layers = len(dims)
        # TODO check how to merge skip_in to config
        self.skip_in = [4]

        for l in range(0, self.num_layers - 1):
            if l + 1 in self.skip_in:
                out_dim = dims[l + 1] - dims[0]
            else:
                out_dim = dims[l + 1]

            lin = nn.Linear(dims[l], out_dim)

            if self.config.geometric_init:
                if l == self.num_layers - 2:
                    if not self.config.inside_outside:
                        torch.nn.init.normal_(lin.weight, mean=np.sqrt(np.pi) / np.sqrt(dims[l]), std=0.0001)
                        torch.nn.init.constant_(lin.bias, -self.config.bias)
                    else:
                        torch.nn.init.normal_(lin.weight, mean=-np.sqrt(np.pi) / np.sqrt(dims[l]), std=0.0001)
                        torch.nn.init.constant_(lin.bias, self.config.bias)
                elif l == 0:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.constant_(lin.weight[:, 3:], 0.0)
                    torch.nn.init.normal_(lin.weight[:, :3], 0.0, np.sqrt(2) / np.sqrt(out_dim))
                elif l in self.skip_in:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
                    torch.nn.init.constant_(lin.weight[:, -(dims[0] - 3) :], 0.0)
                else:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))

            if self.config.weight_norm:
                lin = nn.utils.weight_norm(lin)
                # print("=======", lin.weight.shape)
            setattr(self, "glin" + str(l), lin)

        self.softplus = nn.Softplus(beta=100)
        self.relu = nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()

    def forward_geonetwork(self, inputs):
        """forward the geonetwork"""
        if self.use_grid_feature:
            # positions = (inputs + 1.0) / 2.0
            positions = (inputs - 0.5) / 1.0
            feature = self.encoding(positions)
        else:
            feature = torch.zeros_like(inputs[:, :1].repeat(1, self.encoding.n_output_dims))

        pe = self.position_encoding(inputs)

        inputs = torch.cat((inputs, pe, feature), dim=-1)

        x = inputs

        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "glin" + str(l))

            if l in self.skip_in:
                x = torch.cat([x, inputs], 1) / np.sqrt(2)

            x = lin(x)

            if l < self.num_layers - 2:
                x = self.softplus(x)
        return x

    def get_occupancy(self, sdf):
        """compute occupancy as in UniSurf"""
        occupancy = self.sigmoid(-10.0 * sdf)
        return occupancy

    def get_outputs(self, positions, return_grad=True, return_occupancy=False, return_divergence=False):
        """compute output of samples"""

        outputs = {}

        inputs = positions # should be (..., 3)
        
        if return_grad:
            inputs.requires_grad_(True)

        with torch.enable_grad():
            sdf = self.forward_geonetwork(inputs)
        outputs.update({
                FieldHeadNames.SDF: sdf,
            })

        if return_grad:
            d_output = torch.ones_like(sdf, requires_grad=False, device=sdf.device)
            gradients = torch.autograd.grad(
                outputs=sdf, inputs=inputs, grad_outputs=d_output, create_graph=True, retain_graph=True, only_inputs=True
            )[0] # (N, 2)
            normals = F.normalize(gradients, p=2, dim=-1)
            outputs.update({
                    FieldHeadNames.NORMAL: normals,
                    FieldHeadNames.GRADIENT: gradients,
                })

        if return_occupancy:
            occupancy = self.get_occupancy(sdf)
            outputs.update({FieldHeadNames.OCCUPANCY: occupancy})
        
        if return_divergence:
            # # # import pdb; pdb.set_trace()
            # d_x = torch.autograd.grad(outputs=gradients[:,0:1], inputs=inputs, grad_outputs=d_output, create_graph=True, retain_graph=True, only_inputs=True)[0]
            # d_y = torch.autograd.grad(outputs=gradients[:,1:2], inputs=inputs, grad_outputs=d_output, create_graph=True, retain_graph=True, only_inputs=True)[0]
            # if self.in_dims == 3:
            #     d_z = torch.autograd.grad(outputs=gradients[:,2:3], inputs=inputs, grad_outputs=d_output, create_graph=True, retain_graph=True, only_inputs=True)[0]
            #     div = d_x[:,0:1] + d_y[:,1:2] + d_z[:,2:3] # (N, 1)
            # else:
            #     div = d_x[:,0:1] + d_y[:,1:2] # (N, 1)
            # div = torch.abs(div).mean()
            # outputs.update({'div': div})

            dot_grad = (gradients * gradients).sum(dim=-1, keepdim=True) # (N, 1)
            hvp = torch.ones_like(dot_grad)
            hvp = 0.5 * torch.autograd.grad(dot_grad, inputs, hvp, 
                                            retain_graph=True, create_graph=True)[0] # (N, 2)
            div = (gradients * hvp).sum(dim=-1) / (torch.sum(gradients ** 2, dim=-1) + 1e-5)
            div = torch.abs(div).mean()
            outputs.update({'div': div})
            # outputs.update({'hvp': hvp})

        return outputs

    def forward(self, positions, return_grad=True, return_occupancy=False, return_divergence=False):
        """Evaluates the field at points.

        Args:
            ray_samples: Samples to evaluate field on.
        """
        field_outputs = self.get_outputs(positions, return_grad=return_grad, return_occupancy=return_occupancy, return_divergence=return_divergence)
        return field_outputs