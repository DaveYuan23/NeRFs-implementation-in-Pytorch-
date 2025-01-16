import argparse
import torch
from torch import nn
import numpy as np
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from PIL import Image
import time

from utils import render_rays
from model import nerf, NGP, Plenoxels

def train(model, optimizer, data_loader, num_epoch, device='cpu', near = 2., far = 6., 
          num_samples = 128, H = 100, W = 100):
    for idx, batch in enumerate(data_loader):
        num_batches = len(data_loader)
        rays_d = batch[:,:3].to(device)
        rays_o = batch[:,3:6].to(device)
        target = batch[:, 6:].to(device)
        rgb = render_rays(model, rays_d, rays_o, near, far, N=num_samples)
        loss = ((rgb-target)**2).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"\r[Epoch {num_epoch+1}]\t [Train {idx+1}/{num_batches}]\t loss={loss:.3f}", end='')
    print()

@torch.no_grad()
def test(model, data_loader,  num_epoch, device,num_samples = 128, near = 2., far = 6., H = 100, W = 100):
    for idx, batch in enumerate(data_loader):
        num_batches = len(data_loader)
        rays_d = batch[:,:3].to(device)
        rays_o = batch[:,3:6].to(device)
        target = batch[:, 6:].to(device)
        rgb = render_rays(model, rays_d, rays_o, near, far, N=num_samples)
        loss = ((rgb-target)**2).mean().cpu().detach().numpy()
        psnr = -10. * np.log(loss) / np.log(10.)
        print(f"\r[Epoch {num_epoch+1}]\t [Test {idx+1}/{num_batches}]\t loss={loss.mean():.3f}\t psnr={psnr:.3f}", end='')
    print()

@torch.no_grad()
def save_image(model, dataset, epoch,img_index, device,chunk_size = 20,num_samples = 128, near = 2., far = 6., H = 100, W = 100):
    ray_directions = training_data[img_index * H * W: (img_index + 1) * H * W, :3]
    ray_origins = training_data[img_index * H * W: (img_index + 1) * H * W, 3:6]
    px_values = []   # list of regenerated pixel values
    for i in range(int(np.ceil(H / chunk_size))):   # iterate over chunks
        ray_origins_ = ray_origins[i * W * chunk_size: (i + 1) * W * chunk_size].to(device)
        ray_directions_ = ray_directions[i * W * chunk_size: (i + 1) * W * chunk_size].to(device)
        px_values.append(render_rays(model, ray_directions_, ray_origins_,
                                        near=near, far=far, N=num_samples))
    img = torch.cat(px_values).data.cpu().numpy().reshape(H, W, 3)
    img = (img.clip(0, 1)*255).astype(np.uint8)
    img = Image.fromarray(img)
    plt.figure(figsize=(10,4))
    img.save(f'output/img_Epoch{epoch}.png')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a NeRF model")

    # Common arguments
    parser.add_argument('--model', type=str, required=True, choices=['nerf', 'NGP', 'Plenoxels'], help="Model type")
    parser.add_argument('--epochs', type=int, default=10, help="Number of epochs")
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help="Device to use")
    parser.add_argument('--near', type=float, default=2.0, help="Near plane distance")
    parser.add_argument('--far', type=float, default=6.0, help="Far plane distance")
    parser.add_argument('--num_samples', type=int, default=128, help="Number of samples per ray")
    parser.add_argument('--H', type=int, default=100, help="Image height")
    parser.add_argument('--W', type=int, default=100, help="Image width")

    # Model-specific arguments
    subparsers = parser.add_subparsers(dest="model_args", help="Model-specific arguments")

    # NeRF arguments
    nerf_parser = subparsers.add_parser('nerf', help="Arguments for NeRF model")
    nerf_parser.add_argument('--lr', type=float, default=1e-3, help="Learning rate for NeRF")


    # NGP arguments
    ngp_parser = subparsers.add_parser('NGP', help="Arguments for NGP model")
    ngp_parser.add_argument('--T', type=int, default=2**19, help="Number of hash table entries")
    ngp_parser.add_argument('--Nmin', type=int, default=16, help="Coarsest resolution")
    ngp_parser.add_argument('--Nmax', type=int, default=2048, help="Finsest resolution")
    ngp_parser.add_argument('--L', type=int, default=16, help="Number of levels")
    ngp_parser.add_argument('--scale', type=float, default=8.0, help="Scale factor for NGP")
    ngp_parser.add_argument('--lr', type=float, default=1e-2, help="Learning rate for NGP")

    # Plenoxels arguments
    plenoxels_parser = subparsers.add_parser('Plenoxels', help="Arguments for Plenoxels model")
    plenoxels_parser.add_argument('--scale', type=float, default=1.5, help="Scale factor for Plenoxels")
    plenoxels_parser.add_argument('--Nl', type=int, default=256, help="Grid size for Plenoxels")
    plenoxels_parser.add_argument('--lr', type=float, default=1e-2, help="Learning rate for Plenoxels")

    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Using device: {device}")

    # Load dataset
    loaded = torch.load('dataset.pt')
    training_data = loaded['training']
    testing_data = loaded['testing']

    # Initialize model based on user selection
    if args.model == 'nerf':
        model = nerf().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    elif args.model == 'NGP':
        T = args.T
        N_min, N_max = args.Nmin, args.Nmax
        L = args.L
        b = np.exp((np.log(N_max) - np.log(N_min)) / (L - 1))
        Nl = [int(np.floor(N_min * b**l)) for l in range(L)]
        model = NGP(T=T, Nl=Nl, device=device, scale=args.scale).to(device)
        optimizer = torch.optim.Adam(
            [{"params": model.lookup_table.parameters(), "lr": args.lr, "betas": (0.9, 0.99), "eps": 1e-15, "weight_decay": 0.},
            {"params": model.ffn1.parameters(), "lr": args.lr,  "betas": (0.9, 0.99), "eps": 1e-15, "weight_decay": 10**-6},
            {"params": model.ffn2.parameters(), "lr": args.lr,  "betas": (0.9, 0.99), "eps": 1e-15, "weight_decay": 10**-6}])
    elif args.model == 'Plenoxels':
        model = Plenoxels(N=args.Nl, scale=args.scale).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Training parameters
    data_loader = DataLoader(training_data, batch_size=2**14, shuffle=True)
    test_loader = DataLoader(testing_data, batch_size=2**14, shuffle=False)

    for epoch in range(args.epochs):
        train(model, optimizer, data_loader, epoch, device=device, near=args.near, far=args.far, num_samples=args.num_samples, H=args.H, W=args.W)
        test(model, test_loader, num_epoch=epoch, device=device, num_samples=args.num_samples, near=args.near, far=args.far, H=args.H, W=args.W)
        save_image(model, training_data, epoch,img_index=8, device=device,chunk_size = 20,num_samples = args.num_samples, near = args.near, far = args.far, H = args.H, W = args.W)
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    model_name = args.model
    identifier = f"{model_name}_epochs{args.epochs}_lr{args.lr}_{timestamp}"

    torch.save({
        'model_state_dict': model.state_dict(),
        'args': vars(args),
    }, f'./pretrained/model_{identifier}.pth')