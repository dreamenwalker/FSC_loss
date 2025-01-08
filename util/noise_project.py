import torch

def downsampleImageGPU(x, dx, dy): #Box downsampling
#    y = torch.zeros(x.shape[0], x.shape[1]//dx, x.shape[2]//dy)
    y = torch.cuda.FloatTensor(x.shape[0], x.shape[1], x.shape[2]//dx, x.shape[3]//dy).fill_(0)
    for ii in range(dx):
        for jj in range(dy):
            y += x[:,:, ii::dx, jj::dy]
    return y / (dx * dy)


def upsampleImageGPU(x, dx, dy): #Box upsampling
#    y = torch.zeros(x.shape[0], x.shape[1] * dx, x.shape[2] * dy)
    y = torch.cuda.FloatTensor(x.shape[0], x.shape[1], x.shape[2] * dx, x.shape[3] * dy)
    for ii in range(dx):
        for jj in range(dy):
            y[:, :, ii::dx, jj::dy] = x
    return y


def noisePowerNormalize(nsPwr,min,max,up):
    return nsPwr / (max - min) * up

def noisePowerDeNormalize(nsPwr,min,max,up):
    return nsPwr * (max - min) / up

def denormalize(nsPwr,min,max,up, down):
    return (nsPwr - down) / up * (max - min) + min

def projectToNoiseLevel(x, y, epsList, dx ,dy):
    y_hat = y - downsampleImageGPU(x, dx ,dy)
    yNorm = (y_hat**2).sum(axis = 3).sum(axis = 2).sum(axis = 1).sqrt()
    workWithIndices = yNorm > epsList[:,0]
    x_est = x.clone()
    x_est[workWithIndices, :, :, :] += (1 - epsList[workWithIndices,:] / yNorm[workWithIndices,None])[:,:,None,None] * upsampleImageGPU(y_hat[workWithIndices,:,:,:], dx ,dy)
    return x_est



