import torch.utils.data as data
import numpy as np
import torch
from scipy.ndimage import zoom
class SM_dataset(data.Dataset):
    def __init__(self, data_path, args, transform=None):
        self.args = args
        self.scale = args.scale
        self.inputzoom = args.inputzoom
        self.hr = np.load(data_path) # n * 2 * h * w origin size
        # tailor the size to 32*32 if is 7.mdf need 37-5  else -1
        if self.hr.shape[-1] == 33:
            self.hr = self.hr[:, :, :-1, :-1]# zui hou weidu shi image size height width last channel -1 
        elif self.hr.shape[-1] == 37:
            self.hr = self.hr[:, :, :-5, :-5]
        elif self.hr.shape[-1] ==32:
            pass
        else:
            raise ValueError(f"wrong dimention for  {self.hr.shape[-1]}")



        self.lr = self.downsample(self.hr, args.scale) / self.scale ** 2  # low resolution input
        # 2023520new1020
        # if self.inputzoom:
        #    self.lr = self.upsample(self.lr, args.scale)
        # import pdb
        # pdb.set_trace()
        # compute the max value for the last three dimention, output is four dimention
        self.value_max = torch.from_numpy(self.lr.max(axis = (1, 2, 3), keepdims=True))
        self.value_min = torch.from_numpy(self.lr.min(axis = (1, 2, 3), keepdims=True))#keepdims=True参数保持维度数不变。
        self.transform = transform
        self.up = args.up
        self.down = args.down


    def __getitem__(self, index):
        hr = self.hr[index]
        lr = self.lr[index]
        value_max = self.value_max[index]
        value_min = self.value_min[index]
        if self.transform:
            lr, hr = self.transform([lr, hr]) # transform for list，items in list keep same transform
        lr = self.normalize(lr, value_min, value_max, self.up, self.down)
        hr = self.normalize(hr, value_min, value_max, self.up, self.down)

        return lr, hr, value_min, value_max

    def __len__(self):
        return len(self.hr)

    def downsample(self, x, scale):
        if self.args.downsample_type == 'bicubic':
            return x[:, :, ::scale, ::scale]
        elif self.args.downsample_type == 'box-car':
            y = np.zeros([x.shape[0], x.shape[1], x.shape[2] // scale, x.shape[3] // scale],dtype=x.dtype)
            for ii in range(scale):
                for jj in range(scale):
                    y += x[:, :, ii::scale, jj::scale]
            return y

    def normalize(self, data, min, max, up, down):
        return (data - min) / (max - min) * up + down


    def denormalize(self, data, max, min, up, down):
        return ((data - down) / up) * (max - min) + min


    def upsample(self, x, scale):
            zoom_image = zoom(x, zoom=(1,1,scale,scale), order=1)
            return zoom_image


