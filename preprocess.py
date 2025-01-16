import os, sys
import torch
import numpy as np
from utils import get_rays
import json
from PIL import Image

def process_json_and_save(file_path, output_path, H, W,focal):
    with open(file_path+'./cameras.json', 'r') as f:
        data = json.load(f)
    N = len(data['frames'])-1
    poses = np.empty((N,4,4))
    images = []
    for i, entry in enumerate(data['frames'][:-1]):
        pose = np.array(entry['transform_matrix'])
        poses[i,:,:] = pose
        img_path = file_path + entry['file_path']
        with Image.open(img_path) as img:
            img = img.resize((H, W))  
            images.append(np.asarray(img.convert("RGB")) / 255.0)  

    images_stack = np.stack(images, axis=0)

    np.savez(output_path, images=images_stack, poses=poses,focal=focal)   

def get_train_test_data(images, focal, poses,ratio):
    # training dataset
    train_rays_d = []
    train_rays_o = []
    test_rays_d = []
    test_rays_o = []
    H = images.shape[1]
    W = images.shape[2]
    l = int(np.floor(images.shape[0]*ratio))
    for i,image in enumerate(images[:l]):
        rays_d,rays_o = get_rays(H,W,focal,pose=poses[i])
        train_rays_d.append(rays_d)
        train_rays_o.append(rays_o)
    train_rays_d = torch.concat(train_rays_d,0)
    train_rays_o = torch.concat(train_rays_o,0)
    target = torch.tensor(images[:l].reshape(-1,3),dtype=torch.float32)
    training_data = torch.concat([train_rays_d,train_rays_o, target],-1)

    # testing dataset
    for i,image in enumerate(images[l:]):
        rays_d,rays_o = get_rays(H,W,focal,pose=poses[i+l])
        test_rays_d.append(rays_d)
        test_rays_o.append(rays_o)
    test_rays_d = torch.concat(test_rays_d,0)
    test_rays_o = torch.concat(test_rays_o,0)
    target = torch.tensor(images[l:].reshape(-1,3),dtype=torch.float32)
    testing_data = torch.concat([test_rays_d,test_rays_o,target],-1)
    return training_data, testing_data

if __name__ == "__main__":

    path = 'tiny_nerf_data.npz'
    data = np.load(path)
    images = data['images']
    poses = data['poses']
    focal = data['focal']

    training_data, testing_data = get_train_test_data(images, focal, poses, ratio= 0.95)
    torch.save({'training': training_data, 'testing': testing_data}, 'dataset.pt')

