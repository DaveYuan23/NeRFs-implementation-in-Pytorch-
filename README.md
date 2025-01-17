# NeRFs-implementation-in-Pytorch

This is a simple PyTorch implementation of the original [NeRF model](https://github.com/bmild/nerf), [Instant-NGP](https://github.com/NVlabs/instant-ngp) and [Plenoxels](https://github.com/sxyu/svox2).

## Introduction
NeRF (Neural Radiance Field) is a method for reconstructing 3D scenes from 2D images. This repository provides simple Pytorch implementations of NeRF, Instant-NGP, and Plenoxels, covering key steps such as preprocessing, model formulation, and rendering.

## output
### NGP

https://github.com/DaveYuan23/NeRFs-implementation-in-Pytorch-/issues/1#issue-2794574670

### Plenoxels

https://github.com/DaveYuan23/NeRFs-implementation-in-Pytorch-/issues/2#issue-2794575967

## Preprocessing
To preprocess the image dataset, run the following command:

```bash
python preprocess.py --input_path < > --output_path < > --H < > --W < > --focal < >
```

Image dataset should look like: 
```plaintext
│
├── data/                  
│   ├── images
        ├── 1.jpg
        ├── 2.jpg
│   ├── cameras.json  # camera information
```

## Run
Example to run NeRF model:
```bash
python train.py --model nerf --input_path ./data/ --epochs 20 --device cuda --near 2.0 --far 6.0 --num_samples 128 --H 200 --W 200 nerf --lr 5e-4 
```

Example to run Instant NGP model:
```bash
python train.py --model NGP --input_path ./data/ --epochs 15 --device cuda --near 2.0 --far 6.0 --num_samples 128 --H 256 --W 256 NGP --T 524288 --Nmin 16 --Nmax 2048 --L 16 --scale 8.0 --lr 1e-2 
```

Example to run Plenoxels model:
```bash
python train.py --model Plenoxels --input_path ./data/ --epochs 10 --device cuda --near 1.5 --far 5.5 --num_samples 128 --H 300 --W 300 Plenoxels --Nl 256 --scale 1.5 --lr 1e-2
```
