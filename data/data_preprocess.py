import os
import numpy as np



def normalize(data):
    data_pr = (data - data.mean(axis=(2, 3),keepdims=True)) / data.std(axis=(2, 3),keepdims=True)
    return data_pr


input_path = r'/data/yangfan/MPI/SM_SR/data/OpenMPI/origin'
output_path = r'/data/yangfan/MPI/SM_SR/data/OpenMPI/preprocess'

train_data_path = os.path.join(input_path, 'train_data.npy')
val_data_path = os.path.join(input_path, 'val_data.npy')

train_pr_data_path = os.path.join(output_path,'train_pr_data.npy')
val_pr_data_path = os.path.join(output_path,'val_pr_data.npy')


train_data = np.load(train_data_path)
val_data = np.load(val_data_path)
train_data = train_data[:,:,:-1,:-1]
val_data = val_data[:,:,:-1,:-1]

train_data_pr = normalize(train_data)
val_data_pr = normalize(val_data)

np.save(train_pr_data_path,train_data)
np.save(val_pr_data_path,val_data)




