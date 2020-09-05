import os
import sys
import time
import torch
import random
import argparse
import numpy as np

from src.GAL import *

parser = argparse.ArgumentParser(description='pytorch version of GraphSAGE')

parser.add_argument('--cuda', type=int, default=-1, help='Which GPU to run on (-1 for using CPU, 9 for not specifying which GPU to use.)')
parser.add_argument('--dataSet', type=str, default='weibo_s')
parser.add_argument('--file_paths', type=str, default='file_paths.json')
parser.add_argument('--config_dir', type=str, default='./configs')
parser.add_argument('--logs_dir', type=str, default='./logs')
parser.add_argument('--out_dir', default='./results')
parser.add_argument('--name', type=str, default='debug')

parser.add_argument('--seed', type=int, default=2019)
parser.add_argument('--epochs', type=int, default=5)
parser.add_argument('--b_sz', type=int, default=100)
parser.add_argument('--n_gnnlayer', type=int, default=2)
parser.add_argument('--out_emb_size', type=int, default=128)
parser.add_argument('--tvt_split', type=int, default=0, help='Which of the 5 presets of train-validation-test data splits to use. (0~4), used for bitcoin dataset.')
parser.add_argument('--C', type=float, default=20)
parser.add_argument('--n_block', type=int, default=-1)
parser.add_argument('--thresh', type=float, default=-1)
parser.add_argument('--a_loss_weight', type=float, default=4)
parser.add_argument('--max_vali_f1', type=float, default=0)
parser.add_argument('--agg_func', type=str, default='MEAN')
parser.add_argument('--simi_func', type=str, default='cos')
parser.add_argument('--learn_method', type=str, default='bigal')
parser.add_argument('--loss', type=str, default='1010')
parser.add_argument('--a_loss', type=str, default='none')
parser.add_argument('--unsup_loss', type=str, default='normal')
parser.add_argument('--over_sample', type=str, default='none')
parser.add_argument('--feature', type=str, default='all')
parser.add_argument('--biased_rw', action='store_true')
parser.add_argument('--cluster_aloss', action='store_true')
parser.add_argument('--best_rw', action='store_true')
parser.add_argument('--gcn', action='store_true')
parser.add_argument('--gat', action='store_true')
parser.add_argument('--no_save_embs', action='store_true')
parser.add_argument('--batch_output', action='store_true')
parser.add_argument('--nognn', action='store_true')
parser.add_argument('--noblock', action='store_true')
args = parser.parse_args()
args.argv = sys.argv

if torch.cuda.is_available():
    if args.cuda == -1:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    else:
        device_id = torch.cuda.current_device()
        print('using device', device_id, torch.cuda.get_device_name(device_id))

args.device = torch.device(f"cuda:{args.cuda}" if args.cuda>=0 else "cpu")
if args.cuda == 9:
    args.device = torch.device('cuda')

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

def main():
    args.name = f'{args.name}_{args.dataSet}_{args.learn_method}_{args.feature}_loss{args.loss}_{args.n_gnnlayer}layers_simi-{args.simi_func}_{args.a_loss}_{time.strftime("%m-%d_%H-%M")}'
    args.out_path  = args.out_dir  + '/' + args.name
    if not os.path.isdir(args.out_path): os.mkdir(args.out_path)
    args.biased_rw = True
    args.best_rw = True
    args.embedding_ready_methods = set(['feature', 'rand', 'rand2', 'svdgnn', 'lr'])

    GAL(args)

if __name__ == '__main__':
    main()
