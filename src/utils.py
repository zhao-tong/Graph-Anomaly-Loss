import os
import sys
import copy
import json
import math
import torch
import pickle
import random
import logging
import logging.config

import numpy as np
import torch.nn as nn

from collections import Counter
from numba import guvectorize
from scipy.sparse import csr_matrix
from imblearn.over_sampling import SMOTE, ADASYN
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import f1_score, precision_recall_fscore_support, precision_recall_curve, roc_curve, average_precision_score

def check_args(args):
    if args.learn_method not in ['gnn', 'bigal', 'feature', 'rand', 'rand2', 'svdgnn', 'lr']:
        sys.exit('ERROR: invalid learning method.')
    if args.tvt_split not in [0, 1, 2, 3, 4]:
        sys.exit('ERROR: invalid train-vali-test data split selection.')

def record_process(args, epoch):
    record_file = f'{args.out_path}/current_batch.txt'
    with open(record_file, 'w') as fw:
        fw.write(f'Current at batch {epoch}.')

def get_simi_single_iter(params):
    entries_batch, feats, _get_simi = params
    ii, jj = entries_batch.T
    if isinstance(feats, np.ndarray):
        simi = _get_simi(feats[ii], feats[jj])
    else:
        simi = _get_simi(feats[ii].toarray(), feats[jj].toarray())
    return ii, jj, simi

@guvectorize(['(float64[:], float64[:], float64[:])'], '(n),(n)->()')
def getPCC(u1, u2, simi_score):
    eps = 1e-12
    nz_u1 = u1.nonzero()[0]
    nz_u2 = u2.nonzero()[0]
    nz_inter = np.array(list(set(nz_u1) & set(nz_u2)))
    assert len(nz_inter) > 0
    mean_u1 = u1.sum() / len(nz_u1)
    mean_u2 = u2.sum() / len(nz_u2)
    nume = np.sum((u1[nz_inter] - mean_u1) * (u2[nz_inter] - mean_u2))
    deno = np.sqrt(max(eps, np.sum((u1[nz_inter] - mean_u1) ** 2)) * max(eps, np.sum((u2[nz_inter] - mean_u2) ** 2)))
    # deno = np.sqrt(np.sum((u1[nz_u1] - mean_u1) ** 2) * np.sum((u2[nz_u2] - mean_u2) ** 2))
    assert deno > 0
    simi_score[0] = nume / deno
    simi_score[0] = max(min(simi_score[0], 1.0), -1.0)

@guvectorize(['(float64[:], float64[:], float64[:])'], '(n),(n)->()')
def getACOS(u1, u2, simi_score):
    eps = 1e-12
    nz_u1 = u1.nonzero()[0]
    nz_u2 = u2.nonzero()[0]
    nz_inter = np.intersect1d(nz_u1, nz_u2)
    assert len(nz_inter) > 0
    nume = np.sum(u1[nz_inter] * u2[nz_inter])
    deno = np.sqrt(max(eps, np.sum(u1[nz_inter] ** 2)) * max(eps, np.sum(u2[nz_inter] ** 2)))
    # deno = np.sqrt(np.sum(u1[nz_u1] ** 2) * np.sum(u2[nz_u2] ** 2))
    simi_score[0] = nume / deno
    simi_score[0] = max(min(simi_score[0], 1.0), 0.0)
    simi_score[0] = 2 * simi_score[0] - 1

@guvectorize(['(float64[:], float64[:], float64[:])'], '(n),(n)->()')
def getCOS(u1, u2, simi_score):
    eps = 1e-12
    nz_u1 = u1.nonzero()[0]
    nz_u2 = u2.nonzero()[0]
    nz_inter = np.intersect1d(nz_u1, nz_u2)
    assert len(nz_inter) > 0
    nume = np.sum(u1[nz_inter] * u2[nz_inter])
    deno = np.sqrt(max(eps, np.sum(u1[nz_inter] ** 2)) * max(eps, np.sum(u2[nz_inter] ** 2)))
    simi_score[0] = nume / deno
    simi_score[0] = max(min(simi_score[0], 1.0), 0.0)

def getLogger(name, out_path, config_dir):
    config_dict = json.load(open(config_dir + '/log_config.json'))

    config_dict['handlers']['file_handler']['filename'] = f'{out_path}/log-{name}.txt'
    logging.config.dictConfig(config_dict)
    logger = logging.getLogger(name)

    std_out_format = '%(asctime)s - [%(levelname)s] - %(message)s'
    consoleHandler = logging.StreamHandler(sys.stdout)
    consoleHandler.setFormatter(logging.Formatter(std_out_format))
    logger.addHandler(consoleHandler)

    return logger

def evaluate(Dc, ds, features, gnn, classification, device, max_vali_f1, epoch, assigned=None):
    test_nodes = getattr(Dc, ds+'_test')
    if assigned is None:
        val_nodes = getattr(Dc, ds+'_val')
    else:
        val_nodes = assigned
    labels = getattr(Dc, ds+'_labels')
    if Dc.args.learn_method == 'rand2':
        labels_test = labels[test_nodes]
        logists = np.random.rand(len(labels_test), 2)
        predicts = np.zeros(len(labels_test))
        logists_file = f'{Dc.args.out_path}/logists_test.txt'
        results = eval_n_save(test_nodes, labels_test, logists, predicts, logists_file)
        Dc.logger.info(results)
        return classification, results['f1']
    elif Dc.args.learn_method == 'lr':
        train_nodes = getattr(Dc, ds+'_train_cls')
        features = features.numpy()
        feats_train = features[train_nodes]
        label_train = labels[train_nodes]
        feats_test = features[test_nodes]
        label_test = labels[test_nodes]
        clf = LogisticRegression(random_state=0).fit(feats_train, label_train)
        logists = clf.predict_proba(feats_test)
        logists_file = f'{Dc.args.out_path}/logists_test.txt'
        results = eval_n_save(test_nodes, label_test, logists, np.round(logists[:,1]), logists_file, exp=False)
        Dc.logger.info(results)
        sys.exit(0)
        return classification, results['f1']

    if Dc.args.learn_method in Dc.args.embedding_ready_methods:
        features = torch.Tensor(getattr(Dc, ds+'_feats')).to(device)
    else:
        features = features.to(device)

    embs = features[val_nodes]
    with torch.no_grad():
        logists = classification(embs)
    _, predicts = torch.max(logists, 1)
    labels_val = labels[val_nodes]

    assert len(labels_val) == len(predicts)
    comps = zip(labels_val, predicts.data)

    logists_file = f'{Dc.args.out_path}/logists_vali.txt'
    vali_results = eval_n_save(val_nodes, labels_val, logists.cpu().numpy(), predicts.cpu().numpy(), logists_file)

    Dc.logger.info('Epoch [{}], Validation F1: {:.6f}'.format(epoch, vali_results['f1']))

    if vali_results['f1'] > max_vali_f1:
        max_vali_f1 = vali_results['f1']
        embs = features[test_nodes]
        with torch.no_grad():
            logists = classification(embs)
        _, predicts = torch.max(logists, 1)
        labels_test = labels[test_nodes]
        assert len(labels_test) == len(predicts)
        comps = zip(labels_test, predicts.data)

        logists_file = f'{Dc.args.out_path}/logists_test.txt'
        test_results = eval_n_save(test_nodes, labels_test, logists.cpu().numpy(), predicts.cpu().numpy(), logists_file)

        Dc.logger.info('Epoch [{}], Current best test F1: {:.6f}, AUC: {:.6f}'.format(epoch, test_results['f1'], test_results['roc_auc']))

        resultfile = f'{Dc.args.out_path}/result.txt'
        torch.save(gnn.state_dict(), f'{Dc.args.out_path}/model_gnn.torch')
        torch.save(classification.state_dict(), f'{Dc.args.out_path}/model_classifier.torch')
        with open(resultfile, 'w') as fr:
            fr.write(f'Epoch {epoch}\n')
            fr.write('     \t pre  \t rec  \t  f1  \t  ap  \tpr_auc\troc_auc\tpre100\tpre300\tpre500\tpre1k \t h_pre\t h_rec\t h_f1 \n')
            fr.write('vali:\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\n'.format(vali_results['pre'],vali_results['rec'],vali_results['f1'],vali_results['ap'],vali_results['pr_auc'],vali_results['roc_auc'],vali_results['pre100'],vali_results['pre300'],vali_results['pre500'],vali_results['pre1k'],vali_results['h_pre'],vali_results['h_rec'],vali_results['h_f1']))
            fr.write('test:\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\n'.format(test_results['pre'],test_results['rec'],test_results['f1'],test_results['ap'],test_results['pr_auc'],test_results['roc_auc'],test_results['pre100'],test_results['pre300'],test_results['pre500'],test_results['pre1k'],test_results['h_pre'],test_results['h_rec'],test_results['h_f1']))

    return max_vali_f1

def eval_n_save(nodes, labels, logists, predicts, filename, exp=True):
    assert len(nodes) == len(labels) == len(logists) == len(predicts)
    assert np.shape(logists)[1] == 2
    logists = logists.T[1]
    if exp:
        logists = np.exp(logists)
    pre, rec, f1, _ = precision_recall_fscore_support(labels, predicts, average='binary')
    fpr, tpr, _ = roc_curve(labels, logists, pos_label=1)
    roc_auc = metrics.auc(fpr, tpr)
    precisions, recalls, _ = precision_recall_curve(labels, logists, pos_label=1)
    pr_auc = metrics.auc(recalls, precisions)
    ap = average_precision_score(labels, logists)
    f1s = np.nan_to_num(2*precisions*recalls/(precisions+recalls))
    best_comb = np.argmax(f1s)
    best_f1 = f1s[best_comb]
    best_pre = precisions[best_comb]
    best_rec = recalls[best_comb]
    # t100 = logists.argpartition(-100)[-100:]
    # pre100 = np.sum(labels[t100]) / 100
    # t300 = logists.argpartition(-300)[-300:]
    # pre300 = np.sum(labels[t300]) / 300
    # t500 = logists.argpartition(-500)[-500:]
    # pre500 = np.sum(labels[t500]) / 500
    # t1k = logists.argpartition(-1000)[-1000:]
    # pre1k = np.sum(labels[t1k]) / 1000
    pre100 = 0
    pre300 = 0
    pre500 = 0
    pre1k = 0
    results = {
        'h_pre': pre,
        'h_rec': rec,
        'h_f1': f1,
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'ap': ap,
        'pre': best_pre,
        'rec': best_rec,
        'f1': best_f1,
        'pre100': pre100,
        'pre300': pre300,
        'pre500': pre500,
        'pre1k': pre1k
    }
    with open(filename, 'w') as fw:
        fw.write('node  \tlogist\tpredict\tlabel\n')
        for i in range(len(nodes)):
            fw.write(f'{nodes[i]:6}\t{logists[i]:.4f}\t{predicts[i]:7}\t{labels[i]:5}\n')

    return results

def eval_simple(nodes, labels, predicts):
    assert len(nodes) == len(labels) == len(predicts)
    TP, TN, FP, FN = 0., 0., 0., 0.
    pre, rec, f1 = 0., 0., 0.
    for i in range(len(nodes)):
        if int(labels[i]) == 1:
            if predicts[i] == 1:
                TP += 1
            else:
                FN += 1
        else:
            if predicts[i] == 1:
                FP += 1
            else:
                TN += 1
    if TP+FP > 0: pre = TP / (TP+FP)
    if TP+FN > 0: rec = TP / (TP+FN)
    if pre+rec > 0: f1 = 2*pre*rec / (pre+rec)
    results = {
        'TP': TP,
        'TN': TN,
        'FP': FP,
        'FN': FN,
        'pre': pre,
        'rec': rec,
        'f1': f1
    }
    return results

def get_gnn_embeddings(gnn_model, Dc):
    Dc.logger.info('Loading embeddings from trained GNN model.')
    features = np.zeros((getattr(Dc, Dc.args.dataSet+'_useridmax'), gnn_model.out_size))
    nodes = np.arange(getattr(Dc, Dc.args.dataSet+'_useridmax')).tolist()
    b_sz = 500
    batches = math.ceil(len(nodes) / b_sz)
    embs = []
    for index in range(batches):
        nodes_batch = nodes[index*b_sz:(index+1)*b_sz]
        with torch.no_grad():
            embs_batch = gnn_model(nodes_batch)
        assert len(embs_batch) == len(nodes_batch)
        embs.append(embs_batch)
        if ((index+1)*b_sz) % 10000 == 0:
            Dc.logger.info(f'Dealed Nodes [{(index+1)*b_sz}/{len(nodes)}]')

    assert len(embs) == batches
    embs = torch.cat(embs, 0)
    assert len(embs) == len(nodes)
    Dc.logger.info('Embeddings loaded.')
    return embs.detach().cpu()

def save_gnn_embeddings(embs, Dc, outer_epoch):
    pickle.dump(embs, open(f'{Dc.args.out_path}/gnn_embs_ep{outer_epoch}.pkl', 'wb'))
    Dc.logger.info('Embeddings saved.')

def train_classification(Dc, gnn, classification, ds, device, max_vali_f1, outer_epoch, epochs=500):
    Dc.logger.info('Training Classification ...')
    # train classification, detached from the current graph
    classification.load_state_dict(torch.load(Dc.args.cls_path))
    classification.zero_grad()
    c_optimizer = torch.optim.SGD(classification.parameters(), lr=0.5)
    c_optimizer.zero_grad()
    b_sz = 100
    train_nodes = getattr(Dc, ds+'_train_cls')
    labels = getattr(Dc, ds+'_labels')
    if Dc.args.learn_method == 'rand2' or Dc.args.learn_method == 'lr':
        evaluate(Dc, ds, torch.Tensor(getattr(Dc, ds+'_feats')), gnn, classification, device, max_vali_f1, 0)
        return

    if Dc.args.learn_method in Dc.args.embedding_ready_methods:
        features = torch.Tensor(getattr(Dc, ds+'_feats'))
        Dc.logger.info(f'Loaded features from {Dc.args.learn_method}.')
    else:
        features = get_gnn_embeddings(gnn, Dc)
        Dc.logger.info('Loaded features from GNN model.')
        if not Dc.args.no_save_embs:
            save_gnn_embeddings(features.numpy(), Dc, outer_epoch)
            Dc.logger.info('Saved features from GNN model.')

    if Dc.args.over_sample == 'smote':
        _features = features[train_nodes]
        _labels = labels[train_nodes]
        features_train, labels_train = SMOTE().fit_resample(_features, _labels)
        Dc.logger.info(f'Oversampled training data with SMOTE from {dict(Counter(_labels))} to {dict(Counter(labels_train))}.')
        features_train = torch.Tensor(features_train)
        train_nodes = np.arange(len(labels_train))
    elif Dc.args.over_sample == 'adasyn':
        _features = features[train_nodes]
        _labels = labels[train_nodes]
        features_train, labels_train = ADASYN().fit_resample(_features, _labels)
        Dc.logger.info(f'Oversampled training data with ADASYN from {dict(Counter(_labels))} to {dict(Counter(labels_train))}.')
        features_train = torch.Tensor(features_train)
        train_nodes = np.arange(len(labels_train))
    else:
        Dc.logger.info('Not using any oversampling.')
        features_train = features
        labels_train = labels

    features_train = features_train.to(device)

    for epoch in range(epochs):
        # train_nodes = shuffle(train_nodes)
        np.random.shuffle(train_nodes)
        batches = math.ceil(len(train_nodes) / b_sz)
        # visited_nodes = set()

        for index in range(batches):
            nodes_batch = train_nodes[index*b_sz:(index+1)*b_sz]
            # visited_nodes |= set(nodes_batch)
            labels_batch = labels_train[nodes_batch]
            embs_batch = features_train[nodes_batch]
            logists = classification(embs_batch)
            loss = -torch.sum(logists[range(logists.size(0)), labels_batch], 0)
            loss /= len(nodes_batch)
            loss.backward()

            nn.utils.clip_grad_norm_(classification.parameters(), 5)
            c_optimizer.step()
            c_optimizer.zero_grad()
            classification.zero_grad()

        old_best_vali_f1 = max_vali_f1
        features_tmp = copy.deepcopy(features)
        max_vali_f1 = evaluate(Dc, ds, features, gnn, classification, device, max_vali_f1, 1000*outer_epoch+epoch)
        if max_vali_f1 != old_best_vali_f1:
            save_gnn_embeddings(features_tmp.cpu().numpy(), Dc, 'BEST')
            Dc.logger.info('Saved best features from GNN model.')

    return classification, max_vali_f1

def train_model(Dc, args, gnn, classification, unsupervised_loss, device, outer_epoch):
    ds = args.dataSet
    test_nodes = getattr(Dc, ds+'_test')
    val_nodes = getattr(Dc, ds+'_val')
    train_nodes = getattr(Dc, ds+'_train')
    labels = getattr(Dc, ds+'_labels')

    np.random.shuffle(train_nodes)

    params = []
    for param in gnn.parameters():
        if param.requires_grad:
            params.append(param)

    optimizer = torch.optim.SGD(params, lr=0.7)
    optimizer.zero_grad()
    gnn.zero_grad()

    batches = math.ceil(len(train_nodes) / args.b_sz)

    visited_nodes = set()
    for index in range(batches):
        if args.batch_output:
            args.batch_output_b_cnt += 1
            if args.batch_output_b_cnt % 3 == 0:
                record_process(args, args.batch_output_b_cnt)
                classification, args.max_vali_f1 = train_classification(Dc, gnn, classification, ds, device, args.max_vali_f1, args.batch_output_b_cnt)

        nodes_batch = train_nodes[index*args.b_sz:(index+1)*args.b_sz]
        nodes_batch = np.asarray(list(unsupervised_loss.extend_nodes(nodes_batch, num_neg=50)))
        visited_nodes |= (set(nodes_batch) & set(train_nodes))
        labels_batch = labels[nodes_batch]

        embs_batch = gnn(nodes_batch)

        # if args.learn_method == 'gnn':
        #     # unsupervised loss in graphSAGE
        #     loss = 0
        #     loss_unsup, _ = unsupervised_loss.get_loss_unsup(embs_batch, nodes_batch)
        #     loss += loss_unsup
        # elif args.learn_method == 'bigal':
        #     # unsupervised loss
        loss = 0
        loss_maxmin, loss_mean = unsupervised_loss.get_loss_unsup(embs_batch, nodes_batch)
        if int(args.loss[0]):
            loss += loss_maxmin
        if int(args.loss[1]):
            loss += loss_mean
        if args.a_loss != 'none':
            aloss_maxmin, aloss_mean = unsupervised_loss.get_loss_anomaly(embs_batch, nodes_batch)
            if int(args.loss[2]):
                loss += aloss_maxmin * args.a_loss_weight
            if int(args.loss[3]):
                loss += aloss_mean * args.a_loss_weight
        # else:
        #     Dc.logger.error("Invalid learn_method.")
        #     sys.exit(1)

        Dc.logger.info(f'EP[{outer_epoch}], Batch [{index+1}/{batches}], Loss: {loss.item():.4f}, Dealed Nodes [{len(visited_nodes)}/{len(train_nodes)}]')

        loss.backward()

        # for model in models:
        #     nn.utils.clip_grad_norm_(model.parameters(), 5)
        nn.utils.clip_grad_norm_(gnn.parameters(), 5)
        optimizer.step()

        optimizer.zero_grad()
        # for model in models:
        #     model.zero_grad()
        gnn.zero_grad()

    return gnn
