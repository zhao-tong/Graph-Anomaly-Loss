import os
import sys
import time
import torch
import random
import argparse
import numpy as np

from src.utils import *
from src.models import *
from src.dataCenter import *

def GAL(args):
    # TODO: finish check_args in utils.py
    check_args(args)
    Dc = DataCenter(args)
    Dc.logger.info(f'Input argument vector: {args.argv[1:]}')
    ds = args.dataSet
    device = args.device
    features = torch.FloatTensor(getattr(Dc, ds+'_feats')).to(device)

    gnn = GNN(args.n_gnnlayer, features.size(1), args.out_emb_size, features, Dc, args.a_loss, device, gcn=args.gcn, gat=args.gat, agg_func=args.agg_func)

    gnn.to(device)

    num_labels = 2
    if args.learn_method in args.embedding_ready_methods:
        args.out_emb_size = features.size(1)

    args.cls_path = f'configs/cls-model_state-dict_in{args.out_emb_size}_out{num_labels}.torch'
    classification = Classification(args.out_emb_size, num_labels)
    if os.path.isfile(args.cls_path):
        classification.load_state_dict(torch.load(args.cls_path))
    else:
        classification.init_params()
        torch.save(classification.state_dict(), args.cls_path)
    classification.to(device)

    unsupervised_loss = UnsupervisedLoss(Dc, ds, device, args.biased_rw, args.a_loss, args.C)

    if args.batch_output:
        args.batch_output_b_cnt = 0
        args.no_save_embs = True

    Dc.logger.info(f'Device: {device}')

    if args.learn_method in args.embedding_ready_methods:
        Dc.logger.info('----------------------EPOCH 0-----------------------')
        classification, args.max_vali_f1 = train_classification(Dc, gnn, classification, ds, device, args.max_vali_f1, 0, epochs=500)
    else:
        for epoch in range(args.epochs):
            Dc.logger.info(f'----------------------EPOCH {epoch}-----------------------')
            # if args.batch_output:
            #     record_process(args, epoch)
            gnn = train_model(Dc, args, gnn, classification, unsupervised_loss, device, epoch)
            if not args.batch_output:
                classification, args.max_vali_f1 = train_classification(Dc, gnn, classification, ds, device, args.max_vali_f1, epoch)

    if args.dataSet.startswith('time'):
        run_time = time.time() - getattr(Dc, 'time_tic')
        Dc.logger.info(f'Total running time is: {run_time:.4f} seconds.')
