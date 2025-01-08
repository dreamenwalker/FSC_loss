import datetime
import json
import numpy as np
import os
import time
import math
from pathlib import Path
import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import Compose
from torchvision.transforms import transforms
from torch import nn
from lib.loss.SSIM_self1207 import SSIM_ori, SSIM_MSE_Pro, SSIM_MSE_Plus, SSIM_L1_Pro, SSIM_L1_Plus, SSIM_ori_self2
import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from util.transform import RandomFlip,RandomRotate,ToTensor
from data.dataset import SM_dataset

from engine_train import train_one_epoch, evaluate
from model.tranSMS import par_cvt_rdnDualNonSq
from model.SwinIR import SwinIR
from model.compareMIC.smrNet import vdsr2channels
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
import sys
sys.path.append('./util')
from util.config_parser import get_args_parser, load_configs_yaml
from util.optimizer_surv import get_optimizer
from util.warmup_scheduler_surv import WarmupScheduler
def main(args):
    misc.init_distributed_mode(args)
    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(args.device,0)
    torch.cuda.set_device(args.useGPUno)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    train_transform = Compose([
        RandomFlip(),
        RandomRotate(),
        ToTensor()
    ])
    train_transform1 = transforms.Compose([
        transforms.ToPILImage(),
        #transforms.Resize([32, 32]),  # 将输入图片resize成统一尺寸
        transforms.RandomCrop(16, padding=4), #从输入中随机裁剪出一个大小为32×3的子图像，并在裁剪前向图像的四周填充4个像素
        transforms.RandomRotation(degrees=(-10, 10)),  # 随机旋转，-10到10度之间随机选
        transforms.RandomHorizontalFlip(p=1.0),  # 1 的概率随机水平翻转 p选择一个概率概率
        transforms.RandomVerticalFlip(p=1.0),  # 随机垂直翻转
        transforms.RandomPerspective(distortion_scale=0.6, p=1.0),  # 随机视角
        transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),  # 随机选择的高斯模糊模糊图像
        transforms.ToTensor(),  # 将PIL Image或numpy.ndarray转换为tensor，并归一化到[0,1]之间
        transforms.Normalize(  # 标准化处理-->转换为标准正太分布（高斯分布），使模型更容易收敛
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])  # 其中 mean=[0.485,0.456,0.406]与std=[0.229,0.224,0.225] 从数据集中随机抽样计算得到的。
    ])

    val_transform = Compose([
        #transforms.Resize([32, 32]),  # 将输入图片resize成统一尺寸
        ToTensor()
    ])
    #%%%%%%#%%%% dataset load#########%%%%
    trainFileList = []
    valFileList = []
    testFileList = []

    trainFileList = ([file for file in os.listdir(args.data_path) if 'train' and '80' in file and 'snr5' in file])
    if len(trainFileList) == 1:
        trainFileList = trainFileList[0]
    valFileList = ([file for file in os.listdir(args.data_path) if 'val' and '20' in file and 'snr5' in file])
    if len(valFileList) == 1:
        valFileList = valFileList[0]
    testFileList = ([file for file in os.listdir(args.data_path) if 'test' in file and  "Tran" in file])
    if len(testFileList) == 1:
        testFileList = testFileList[0]

    train_data_path = os.path.join(args.data_path, trainFileList)
    val_data_path = os.path.join(args.data_path, valFileList)
    test_data_path = os.path.join(args.data_path,testFileList)
    args.trainfile = train_data_path
    args.testfile = test_data_path

    dataset_train = SM_dataset(train_data_path, args, transform = train_transform)

    #print len( dataset_train) 167022  Out[22]: dataset_train.hr.shape 167022, 2, 32, 32)
    dataset_val = SM_dataset(val_data_path, args, transform = val_transform)
    # len(dataset_val) = 82485
    dataset_test = SM_dataset(test_data_path, args, transform=val_transform)

    total_iters = args.total_iters
    train_size = int(math.ceil(len(dataset_train) / args.batch_size))# len(dataset_tr)=167022, bt = 128
    tempepoch = int(math.ceil(total_iters / train_size))
    if  tempepoch > 130:
        args.epochs = tempepoch #154
    configs = load_configs_yaml(args.configs)
    timestr = time.strftime('%Y-%m-%d-%H-%M')
    testname = configs['name']


    if True:  # args.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        print("Sampler_train = %s" % str(sampler_train))
        if args.dist_eval:
            if len(dataset_val) % num_tasks != 0:
                print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                      'This will slightly alter validation results as extra duplicate entries are added to achieve '
                      'equal num of samples per-process.')
            sampler_val = torch.utils.data.DistributedSampler(
                dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False)  # shuffle=True to reduce monitor bias
        else:
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)
        
        sampler_test = torch.utils.data.SequentialSampler(dataset_test)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    
    
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )
    #
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=128,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, sampler=sampler_test,
        batch_size=128,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )
    # args.epochs = 2000
    args.lr_size = args.input_size // args.scale # lr:low resolution
    if (args.scale == 2):
        config={'inChannel': 2, 'outChannel': 2, 'initialConvFeatures': 24, 'scaleFactor': 2, 'rdn_nb_of_features': 24, 'rdn_nb_of_blocks': 4, 'rdn_layer_in_each_block': 5, 'rdn_growth_rate': 6, 'img_size1': args.lr_size, 'img_size2': args.lr_size , 'cvt_out_channels': 64, 'cvt_dim': 64, 'num_attention_heads': 8, 'convAfterConcatLayerFeatures': 48}
    elif (args.scale == 4):
        config = {'inChannel': 2, 'outChannel': 2, 'initialConvFeatures': 24, 'scaleFactor': 4, 'rdn_nb_of_features': 24, 'rdn_nb_of_blocks': 4, 'rdn_layer_in_each_block': 8, 'rdn_growth_rate': 6, 'img_size1': args.lr_size, 'img_size2': args.lr_size, 'cvt_out_channels': 64, 'cvt_dim': 64, 'num_attention_heads': 8, 'convAfterConcatLayerFeatures': 48}
    elif (args.scale == 8):
        config = {'inChannel': 2, 'outChannel': 2, 'initialConvFeatures': 64, 'scaleFactor': 8, 'rdn_nb_of_features': 24, 'rdn_nb_of_blocks': 4, 'rdn_layer_in_each_block': 9, 'rdn_growth_rate': 6, 'img_size1': args.lr_size, 'img_size2': args.lr_size, 'cvt_out_channels': 64, 'cvt_dim': 64, 'num_attention_heads': 8, 'convAfterConcatLayerFeatures': 48}
#%%%
    #model = par_cvt_rdnDualNonSq(config)
    #model = SwinIR()
    if args.inputzoom:
         upscale = 1
    else:
        upscale = args.scale
    model = SwinIR(upscale=upscale, img_size=(32, 32), in_chans=2,
                       window_size=4, img_range=1., depths=[6, 6, 6, 2],
                       embed_dim=60, num_heads=[4, 4, 4, 4], mlp_ratio=2, upsampler='pixelshuffledirect')
    model.to(device)

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # print("Model = %s" % str(model_without_ddp))
    print('number of params (M): %.2f' % (n_parameters / 1.e6))
    args.n_parameters = n_parameters / 1.e6
    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()

    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256  #blr: base lr

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module
    param_groups = model.parameters()
    #optimizer = torch.optim.Adam(param_groups, lr=args.lr)
    optimizer = get_optimizer(param_groups, configs['train']['optimizer'])  #2023704 add from surv
    lr_scheduler = WarmupScheduler(optimizer,
                                   total_epochs = args.epochs,
                                   warmup_param = configs['train']['scheduler']['warmup'],
                                   scheduler_param = configs['train']['scheduler'])
    args.lr=configs['train']['optimizer']['lr']
    #lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.epochs // 100, gamma=0.8) # 2023704 from tranSMS
    loss_scaler = NativeScaler()
    #criterion = nn.L1Loss()
    #criterion =SSIM_ori()
    #criterion = nn.MSELoss()
    #criterion =SSIM_L1_Pro()
    criterion = SSIM_ori_self2(type = "None",onlychar=False)
    print("criterion = %s" % str(criterion))

    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    if args.eval:
        test_stats = evaluate(data_loader=data_loader_val, model = model, criterion = criterion, device = device, args = args)
        print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['dice_metric']:.3%}")
        exit(0)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    best_index = 1.0
    val_best_index = 1.0
    best_epochnum = 0
    dirname = timestr + '_' + args.remote_server + '_' + model._get_name() 
    args.output_dir = os.path.join(args.output_dir,
                                  dirname +
                                   '_bs_' + str(args.batch_size) +
                                   '_lr_' + str(args.lr) +
                                   '_ep_' + str(args.epochs) +
                                   args.downsample_type +
                                   '_sc_' + str(args.scale) +
                                   '_useNP_' + str(args.useNoisyProjection)+
                                   '_' + criterion._get_name())
    if misc.is_main_process():
        if args.output_dir:
            Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    if global_rank == 0 and args.log_dir is not None and not args.eval:
        #os.makedirs(args.log_dir, exist_ok=True)
        #time_str = time.strftime('%Y-%m-%d-%H-%M')
        args.log_dir = args.output_dir
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    if args.output_dir and misc.is_main_process():

        if os.path.exists(os.path.join(args.output_dir, "args.txt")):
            os.remove(os.path.join(args.output_dir, "args.txt"))
        if os.path.exists(os.path.join(args.output_dir, "log.txt")):
            os.remove(os.path.join(args.output_dir, "log.txt"))

        argsDict = args.__dict__
        with open(os.path.join(args.output_dir, "args.txt"), mode="a", encoding="utf-8") as f:
            for eachArg, value in argsDict.items():
                f.writelines(eachArg + ': ' + str(value) + '\n')
        f.close()


    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        train_stats = train_one_epoch(model, criterion, data_loader_train,
                                    optimizer, lr_scheduler, device, epoch, loss_scaler,
                                    args.clip_grad,
                                    log_writer=log_writer,
                                    args=args)
        # back to epoch
        #scheduler.step() 2023704
        lr_scheduler.step()
        
        if args.output_dir and (epoch % args.save_epoch == 0 or epoch + 1 == args.epochs):
            if args.output_dir and args.save_weight:
                misc.save_model(
                    args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                    loss_scaler=loss_scaler, epoch=epoch)
        print("********************************result***********************")
        print("*************Result for validation set****************")
        val_stats = evaluate(data_loader=data_loader_val, model=model, epoch = epoch,  log_writer=log_writer,
                             criterion=criterion, device=device, args=args, stage="val")
        print(f"nrmse_all_denorm of the {len(dataset_val)} validation images: {val_stats['nrmse_all_denorm']:.4%}")
        '''
        val_best_index = min(val_best_index, val_stats["nrmse_all_denorm"])
        print(f'validation best nrmse: {val_best_index:.4f}')
        '''
        print("*************Result for test set****************")
        test_stats = evaluate(data_loader=data_loader_test, model=model, epoch = epoch, log_writer=log_writer,
                              criterion=criterion, device=device, args=args, stage="test")
        print(f"nrmse_all_denorm of the {len(dataset_test)} test images: {test_stats['nrmse_all_denorm']:.4%}")
        if best_index > test_stats["nrmse_all_denorm"] or epoch % 10 == 0:
            best_index = min(best_index, test_stats["nrmse_all_denorm"])
            test_best_index = best_index
            best_epochnum = epoch
            print(f"nrmse_all_denorm of the {len(dataset_test)} test images: {test_stats['nrmse_all_denorm']:.5%}")
            print(f'Test set best nrmse: {best_index:.5f}')
            print(f'Epoch for best result is {best_epochnum}') 
        '''     
        if test_stats["nrmse_all_denorm"] <= best_index:
            test_best_index = min(best_index, test_stats["nrmse_all_denorm"]) # 这块有问题，上述判断为假则变量不存在
            best_index = test_best_index
            print(f'Test set best nrmse: {test_best_index:.4f}')
        '''  

        log_stats = {'epoch': epoch,
                     **{f'train_{k}': f'{v:.6f}' for k, v in train_stats.items()},
                     **{f'val_{k}': f'{v:.6f}' for k, v in val_stats.items()},
                     **{f'test_{k}': f'{v:.6f}' for k, v in test_stats.items()},
                    }

        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

    if args.output_dir and misc.is_main_process():
        log_stats = {'best result of nrmse_all_denorm ': test_best_index}
        with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
            f.write(json.dumps(log_stats) + "\n")
        f.close()



if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    main(args)



