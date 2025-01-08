# create 2023601 author:zlw
import argparse
import os
import sys
import yaml
import json
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(parent_dir)
sys.path.insert(0, parent_dir)
def get_args_parser():
    parser = argparse.ArgumentParser('MAE fine-tuning for image classification', add_help=False)
    parser.add_argument("--inputzoom", type=int, default=0,
                        help="0: Don't use zoom for input, 1: use zoom")   
    parser.add_argument('--configs', type=str, default='configs/train.yaml',help='configs file path')
    parser.add_argument('--remote_server', default='R38SM712',type=str, choices=('R36SM712','R36SM530'))
    parser.add_argument('--batch_size', default=128, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--total_iters', default=3e4,
                        type=int, help='the number of iteration')
    parser.add_argument("--useGPUno", type=int, default=0, help="Selected GPU no")
    parser.add_argument('--downsample_type', default='box-car', type=str,choices=('bicubic','box-car'),
                        help='the method can be used downsample system matrix')
    #useNoisyProjection: 0 ablates the data consistency block from TranSMS, 1 is regular TranSMS with data consistency
    parser.add_argument("--useNoisyProjection", type=int, default=0,
                        help="0: Don't use noise projection, 1: use noise projection")
    parser.add_argument('--scale', default=2, type=int)
    #in matri the up and down are used  in ./data/dataset.py
    parser.add_argument('--up', default=0.7, type=int)
    parser.add_argument('--down', default=0.15, type=int)
    parser.add_argument('--epochs', default=200, type=int)
    # Dataset parameters
    parser.add_argument('--data_path', default='./data/simulation_split_data/', type=str,
                        help='dataset path')

    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    parser.add_argument('--save_epoch', default=50, type=int, metavar='N',
                        help='start epoch')

    # Model parameters
    parser.add_argument('--model', default='tranSMS', type=str, metavar='MODEL',
                        help='Name of model to train')

    parser.add_argument('--input_size', default=32, type=int,
                        help='images size for input')

    parser.add_argument('--drop_path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')

    # Optimizer parameters
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--layer_decay', type=float, default=0.75,
                        help='layer-wise lr decay from ELECTRA/BEiT')

    parser.add_argument('--min_lr', type=float, default=1e-7, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=int, default=5, metavar='N',
                        help='epochs to warmup LR')

    parser.add_argument('--output_dir', default='./result',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./tensorboard_dir',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')
    #metavar参数来指定在帮助文本中显示参数的名称。
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true',
                        help='Perform evaluation only')
    parser.add_argument('--dist_eval', action='store_true', default=False,
                        help='Enabling distributed evaluation (recommended during training for faster monitor')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--distributed', default=0, type=int,
                        help='1：distributed processes, 0:no')
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')
    #new add 
    parser.add_argument("--save_weight", type=int, default=1,
                        help="0: Don't save weight, 1: save")
    # * Random Erase params
    # parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
    #                     help='Random erase prob (default: 0.25)')
    # parser.add_argument('--remode', type=str, default='pixel',
    #                     help='Random erase mode (default: "pixel")')
    # parser.add_argument('--recount', type=int, default=1,
    #                     help='Random erase count (default: 1)')
    # parser.add_argument('--resplit', action='store_true', default=False,
    #                     help='Do not random erase first (clean) augmentation split')

    # * Finetuning params
    # parser.add_argument('--finetune', default='',
    #                     help='finetune from checkpoint')
    # parser.add_argument('--global_pool', action='store_true')
    # parser.set_defaults(global_pool=True)
    # parser.add_argument('--cls_token', action='store_false', dest='global_pool',
    #                     help='Use class token instead of global pool for classification')

    return parser


def update_configs(configs,  args):
    #TODO
    pass
#023704
def load_configs_yaml(config_file):
        parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        config_file = os.path.join(parent_dir, config_file)
        assert os.path.exists(config_file)
        with open(config_file, 'rb') as f:
            #configs = json.load(f)
            configs = yaml.safe_load(f)
            #configs = json.dumps(configs)
        return configs

def load_configs_json(self, config_file: object) -> object:
        parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        config_file = os.path.join(parent_dir, config_file)
        assert os.path.exists(config_file)
        json_str = ''
        with open(config_file, 'r') as f: # fonfig_file is json file
            for line in f:
                line = line.split('//')[0] + '\n'
                json_str += line
        configs = json.loads(json_str)
        return configs
