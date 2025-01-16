import torch
from torch import nn
import numpy as np

def get_rays(H, W, focal, pose):
    i ,j = np.meshgrid(np.arange(W), np.arange(H), indexing='xy')
    d = np.stack([(i-W/2)/focal, -(j-H/2)/focal, -np.ones_like(i)],-1)
    rays_d = np.sum(d[..., np.newaxis, :] * pose[:3, :3], -1)
    rays_o = np.broadcast_to(pose[:3, -1], np.shape(rays_d))
    return torch.tensor(rays_d.reshape(-1,3),dtype=torch.float32), torch.tensor(rays_o.reshape(-1,3),dtype=torch.float32)

def render_rays(model, rays_d, rays_o, near, far, N):
    device = rays_d.device

    # Compute sampling points
    ts = torch.linspace(near, far, N,device=device).expand(rays_o.shape[0], N)
    mid = (ts[:, :-1] + ts[:, 1:]) / 2.
    lower = torch.cat((ts[:, :1], mid), -1)
    upper = torch.cat((mid, ts[:, -1:]), -1)
    u = torch.rand(ts.shape, device=device)
    ts = lower + (upper - lower) * u
    delta = torch.cat((ts[:, 1:] - ts[:, :-1], torch.tensor([1e10], device=device).expand(rays_o.shape[0], 1)), -1)
    
    # Prepare input for the model
    inputs = rays_o.unsqueeze(1) + rays_d.unsqueeze(1) * ts.unsqueeze(2)
    direcation = rays_d.expand(N, rays_d.shape[0], 3).transpose(0, 1).reshape(-1,3)
    inputs_flat = inputs.reshape(-1, 3)
    #print(.shape)

    # Process in batches
    colors, sigma = model(inputs_flat,direcation)
    colors = colors.reshape(inputs.shape)
    sigma = sigma.reshape(inputs.shape[:-1])

    # Volume rendering
    alpha = 1 - torch.exp(-sigma*delta)
    llf = torch.cumprod(1.0 - alpha + 1e-10, dim=-1)
    llf = torch.cat([torch.ones_like(llf[..., :1],device=device), llf[..., :-1]], dim=-1)
    weights = llf.unsqueeze(2) * alpha.unsqueeze(2)
    c = (weights * colors).sum(dim=1)
    weight_sum = weights.sum(-1).sum(-1)  # Regularization for white background 
    return c + 1 - weight_sum.unsqueeze(-1)

