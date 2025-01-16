import torch
from torch import nn

class positionalEncoding(nn.Module):
    def __init__(self, L_embed = 10):
        super().__init__()
        self.L_embed = L_embed
    
    def forward(self,x):
        pe = [x]
        for i in range(self.L_embed):
            for fn in [torch.sin, torch.cos]:
                pe.append(fn(2.**i * x))
        return torch.concat(pe,-1)
    
class nerf(nn.Module):
    def __init__(self, num_hiddens=256, L_embed_1 = 10, L_embed_2 = 4):
        super().__init__()
        self.pe1 = positionalEncoding(L_embed_1)
        self.ffn1 = nn.Sequential(
            nn.Linear(3+3*2*L_embed_1,num_hiddens),
            nn.ReLU(),
            nn.Linear(num_hiddens,num_hiddens),
            nn.ReLU(),
            nn.Linear(num_hiddens,num_hiddens),
            nn.ReLU(),
            nn.Linear(num_hiddens,num_hiddens),
            nn.ReLU(),
        )
        self.ffn2 = nn.Sequential(
            nn.Linear(3+3*2*L_embed_1 + num_hiddens,num_hiddens),
            nn.ReLU(),
            nn.Linear(num_hiddens,num_hiddens),
            nn.ReLU(),
            nn.Linear(num_hiddens,num_hiddens),
            nn.ReLU(),
            nn.Linear(num_hiddens,num_hiddens + 1),
        )
        self.ffn3 = nn.Sequential(
            nn.Linear(3+3*2*L_embed_2 + num_hiddens,num_hiddens//2),
            nn.ReLU(),
            nn.Linear(num_hiddens//2,3),
            nn.Sigmoid(),
        )
        self.pe2 = positionalEncoding(L_embed_2)
        self.relu = nn.ReLU()

    def forward(self, inputs, direction):
        inputs = self.pe1(inputs)
        outputs = self.ffn1(inputs)
        outputs = torch.concat([outputs,inputs],-1)
        outputs = self.ffn2(outputs)
        tmp,sigma = outputs[:,:-1], self.relu(outputs[:,-1])
        tmp = torch.concat([tmp,self.pe2(direction)],-1)
        C = self.ffn3(tmp)
        return C, sigma
    

class NGP(nn.Module):
    def __init__(self, T, Nl,device, scale ,F = 2, L_embeded = 4):
        super(NGP, self).__init__()
        self.T = T
        self.Nl = Nl
        self.F = F
        self.scale = scale

        self.pi1, self.pi2, self.pi3 = 1, 2654435761, 805459861
        self.pe = positionalEncoding(L_embed=L_embeded)
        self.ffn1 = nn.Sequential(
            nn.Linear(F*len(Nl), 64),
            nn.ReLU(),
            nn.Linear(64, 16),
        ).to(device)
        self.ffn2 = nn.Sequential(
            nn.Linear(3+6*L_embeded+16, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 3),
            nn.Sigmoid(),
        ).to(device)
        self.lookup_table = torch.nn.ParameterDict(
                {str(i): torch.nn.Parameter((torch.rand(
                (T, 2),device=device) * 2 - 1) * 1e-4) for i in range(len(Nl))})

    def forward(self, x, d):
        # x is a batchsize * 3
        # d is a batchsize *3
        x /= self.scale
        mask = (x[:, 0].abs() < .5) & (x[:, 1].abs() < .5) & (x[:, 2].abs() < .5) # keep the central points
        #mask = x.abs() >= 0
        x += 0.5 # move points to [0,1]^3, but why?
        color = torch.zeros((x.shape[0], 3), device=x.device)
        log_sigma = torch.zeros((x.shape[0]), device=x.device) - 100000 # why log first?
        features = self.get_features(x, mask)
        dir = self.pe(d[mask])
        h = self.ffn1(features)
        log_sigma[mask] = h[:,0]
        color[mask] = self.ffn2(torch.concat([h, dir],-1))
        return color, torch.exp(log_sigma) 


    def get_features(self, x, mask):
        features = torch.empty((x[mask].size(0), self.F * len(self.Nl)),device=x.device)
        for i, N in enumerate(self.Nl):
            x_floor = torch.floor(x[mask] * N)
            x_ceil = torch.ceil(x[mask]* N) 
            vertices = torch.zeros((x[mask].size(0), 8, 3), dtype=torch.int64,device=x.device)
            vertices[:,0] = x_floor
            vertices[:,1] = torch.concat([x_ceil[:,0, None], x_floor[:,1,None], x_floor[:,2,None]], dim=1)
            vertices[:,2] = torch.concat([x_floor[:,0, None], x_ceil[:,1,None], x_floor[:,2,None]], dim=1)
            vertices[:,3] = torch.concat([x_floor[:,0, None], x_floor[:,1,None], x_ceil[:,2,None]], dim=1)
            vertices[:,4] = torch.concat([x_floor[:,0, None], x_ceil[:,1,None], x_ceil[:,2,None]], dim=1)
            vertices[:,5] = torch.concat([x_ceil[:,0, None], x_floor[:,1,None], x_ceil[:,2,None]], dim=1)
            vertices[:,6] = torch.concat([x_ceil[:,0, None], x_ceil[:,1,None], x_floor[:,2,None]], dim=1)
            vertices[:,7] = x_ceil

            a = vertices[:,:,0] * self.pi1
            b = vertices[:,:,1] * self.pi2
            c = vertices[:,:,2] * self.pi3

            h_x = torch.remainder(torch.bitwise_xor(torch.bitwise_xor(a,b),c), self.T)
            v = self.lookup_table[str(i)][h_x].transpose(-1,-2) # b*8*2 -> b*2*8
            v = v.reshape((v.size(0), 2,2,2,2))
            features[:, i*self.F:(i+1)*self.F] = torch.nn.functional.grid_sample(
                v,
                ((x[mask]*N - x_floor)-0.5).unsqueeze(1).unsqueeze(1).unsqueeze(1) # interpolation weights
            ).squeeze(-1).squeeze(-1).squeeze(-1)
        return features

class Plenoxels(nn.Module):
    def __init__(self,device = 'cpu', Nl = [256], scale = 1.5):
        super(Plenoxels, self).__init__()
        self.Nl  = Nl
        self.scale = scale
        self.voxel_coefficients = torch.nn.Parameter(torch.ones(
                (Nl,Nl,Nl, 27+1),device=device) /100)
        self.relu = nn.ReLU()
        self.SHLayer = SH_function()

    def forward(self, x, d):
        color = torch.zeros_like(x)
        sigma = torch.zeros((x.shape[0]), device=x.device)

        x /= self.scale
        mask = (x[:, 0].abs() < .5) & (x[:, 1].abs() < .5) & (x[:, 2].abs() < .5) # keep the central points
        output = self.get_features(x,mask)
        sigma[mask], k = self.relu(output[:,0]), output[:,1:]
        color[mask] = self.SHLayer(k.reshape((-1,3,9)), d[mask])
        return color, sigma


    def get_features(self, x, mask):
        features = torch.empty((x[mask].size(0), 28),device=x.device)
        idx = (x[mask] / (2 / self.Nl) + self.Nl / 2).clip(0, self.Nl - 1)
        x_floor = torch.floor(idx)
        x_ceil = torch.ceil(idx) 
        vertices = torch.zeros((idx.size(0), 8, 3), dtype=torch.int64,device=x.device)
        vertices[:,0] = x_floor
        vertices[:,1] = torch.concat([x_ceil[:,0, None], x_floor[:,1,None], x_floor[:,2,None]], dim=1)
        vertices[:,2] = torch.concat([x_floor[:,0, None], x_ceil[:,1,None], x_floor[:,2,None]], dim=1)
        vertices[:,3] = torch.concat([x_floor[:,0, None], x_floor[:,1,None], x_ceil[:,2,None]], dim=1)
        vertices[:,4] = torch.concat([x_floor[:,0, None], x_ceil[:,1,None], x_ceil[:,2,None]], dim=1)
        vertices[:,5] = torch.concat([x_ceil[:,0, None], x_floor[:,1,None], x_ceil[:,2,None]], dim=1)
        vertices[:,6] = torch.concat([x_ceil[:,0, None], x_ceil[:,1,None], x_floor[:,2,None]], dim=1)
        vertices[:,7] = x_ceil

        v1 = vertices[:,:,0]
        v2 = vertices[:,:,1]
        v3 = vertices[:,:,2]

        v = self.voxel_coefficients[v1,v2,v3].transpose(-1,-2) # b*8*2 -> b*2*8
        v = v.reshape((v.size(0), 28,2,2,2))
        features = torch.nn.functional.grid_sample(
            v,
            ((idx - x_floor)-0.5).unsqueeze(1).unsqueeze(1).unsqueeze(1) # interpolation weights
        ).squeeze(-1).squeeze(-1).squeeze(-1)
 
        return features
        

class SH_function(nn.Module):
    def __init__(self):
        super().__init__()
        self.SH_C0 = 0.28209479177387814
        self.SH_C1 = 0.4886025119029199
        self.SH_C2 = [
                1.0925484305920792,
                -1.0925484305920792,
                0.31539156525252005,
                -1.0925484305920792,
                0.5462742152960396
                ]
    def forward(self,k,d):
        x, y, z = d[...,0:1], d[...,1:2], d[...,2:3]

        return self.SH_C0 * k[..., 0] +\
            - self.SH_C1 * y * k[..., 1] + self.SH_C1 * z * k[..., 2] - self.SH_C1 * x * k[..., 3] + \
            (self.SH_C2[0] * x * y * k[..., 4] + self.SH_C2[1] * y * z * k[..., 5] + self.SH_C2[2] * (2.0 * z * z - x * x - y * y) * k[
               ..., 6] + self.SH_C2[3] * x * z * k[..., 7] + self.SH_C2[4] * (x * x - y * y) * k[..., 8])

