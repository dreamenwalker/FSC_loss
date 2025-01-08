import torch
import numpy as np
def calculate_rmse_psnr(vol_predict, vol_true):
    # calculate volumen metrics

    mse = torch.square(vol_true - vol_predict).mean(dim=(1,2,3)) # .mean()不同维度算平均
    rmse = torch.sqrt(mse)
    if rmse.mean() == 0:
        psnr = 100.0
    else:
        psnr = 20.0 * torch.log10( torch.amax(vol_true, dim=(1,2,3)) / (rmse + 1e-10)) # torch.amax() 求最大值，是个数值


    nrmse = torch.sqrt(torch.square(vol_true - vol_predict).sum(dim = (1,2,3))) / torch.sqrt(torch.square(vol_true).sum(dim = (1,2,3)))
    #nrmsediff = rmse/(rmse.max()-rmse.min())

    #nrmsemaxmin = torch.norm(vol_true - vol_predict)/ torch.norm(vol_true)   #和上面等价  #torch.norm 是范数
    #print("yf result:", nrmse)
    nrmse_Tnorm = torch.norm(vol_predict-vol_true) / torch.norm(vol_true)# nvalidDS 是什么东西


    # 上面计算的没有0维，是样本数，求 mean算一个值
    return rmse.mean(), psnr.mean(), nrmse.mean(), nrmse_Tnorm
