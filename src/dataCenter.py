import os
import sys
import copy
import time
import math
import torch
import pickle
import pathlib
import powerlaw
import numpy as np
from multiprocessing import Pool
from scipy.sparse import csr_matrix
from collections import defaultdict, Counter

from src.utils import *

class DataCenter(object):
    """docstring for DataCenter"""
    def __init__(self, args):
        self.args = args
        self.file_paths = json.load(open(f'{args.config_dir}/{args.file_paths}'))
        self.logger = getLogger(args.name, args.out_path, args.config_dir)

        self.load_dataSet(args.dataSet)

    def load_dataSet(self, dataSet):
        self.logger.info(f'Dataset: {dataSet}')

        if dataSet.startswith('bitcoin'):
            if self.args.learn_method == 'gnn':
                ds = dataSet
                graph_u2p_file = self.file_paths[ds]['graph_u2p']
                labels_file = self.file_paths[ds]['labels']
                features_file = self.file_paths[ds]['features']
                data_split_file = self.file_paths[ds]['trainvalitest_indexes']

                graph_u2p = pickle.load(open(graph_u2p_file, 'rb'))
                labels = pickle.load(open(labels_file, 'rb'))
                feat_data = pickle.load(open(features_file, 'rb'))

                m, n = np.shape(graph_u2p)
                adj_lists = {}
                for i in range(m):
                    adj_lists[i] = set(m + graph_u2p[i,:].nonzero()[1])
                for j in range(n):
                    adj_lists[j+m] = set(graph_u2p[:,j].nonzero()[0])

                feat_data = np.concatenate((feat_data, np.zeros((n, np.shape(feat_data)[1]))), 0)
                assert np.shape(feat_data)[0] == m+n

                train_indexs = np.arange(len(labels))

                if os.path.isfile(data_split_file):
                    self.logger.info('loaded train-vali-test data splits.')
                    data_splits = pickle.load(open(data_split_file, 'rb'))
                else:
                    node_w_labels = np.where(labels>=0)[0]
                    data_splits = self._split_data_limited(node_w_labels)
                    pickle.dump(data_splits, open(data_split_file, "wb"))
                    self.logger.info('generated train-vali-test data splits and saved for future use.')

                self.logger.info(f'using data split {self.args.tvt_split}')
                test_indexs, val_indexs, train_indexs_cls = data_splits[self.args.tvt_split]

                assert len(feat_data) == len(labels)+n == len(adj_lists)
                train_indexs = np.arange(len(labels))

                user_id_max = len(labels)
                graph_simi = np.ones((m+n, m+n))

                setattr(self, dataSet+'_test', test_indexs)
                setattr(self, dataSet+'_val', val_indexs)
                setattr(self, dataSet+'_train', train_indexs)
                setattr(self, dataSet+'_train_cls', train_indexs_cls)
                setattr(self, dataSet+'_trainable', train_indexs)
                setattr(self, dataSet+'_useridmax', user_id_max)

                setattr(self, dataSet+'_feats', feat_data)
                setattr(self, dataSet+'_labels', labels)
                setattr(self, dataSet+'_adj_lists', adj_lists)
                setattr(self, dataSet+'_best_adj_lists', adj_lists)
                setattr(self, dataSet+'_simis', graph_simi)
            else:
                ds = dataSet
                graph_u2p_file = self.file_paths[ds]['graph_u2p']
                graph_u2u_file = self.file_paths[ds]['graph_u2u']
                labels_file = self.file_paths[ds]['labels']
                features_file = self.file_paths[ds]['features']
                data_split_file = self.file_paths[ds]['trainvalitest_indexes']

                if self.args.simi_func == 'pcc' or self.args.simi_func == 'rpcc':
                    _get_simi = getPCC
                    graph_simi_file = self.file_paths[ds]['graph_u2u_pcc']
                elif self.args.simi_func == 'acos':
                    _get_simi = getACOS
                    graph_simi_file = self.file_paths[ds]['graph_u2u_acos']
                elif self.args.simi_func == 'cos':
                    _get_simi = getCOS
                    graph_simi_file = self.file_paths[ds]['graph_u2u_cos']
                else:
                    self.logger.error('Invalid user-user similarity function.')

                graph_u2p = pickle.load(open(graph_u2p_file, 'rb'))
                graph_u2u = pickle.load(open(graph_u2u_file, 'rb'))
                labels = pickle.load(open(labels_file, 'rb'))
                feat_data = pickle.load(open(features_file, 'rb'))

                if self.args.feature == 'deepwalk':
                    embs_file = self.file_paths[ds]['deepwalk']
                    feat_data = pickle.load(open(embs_file, 'rb'))
                elif self.args.feature == 'bine':
                    embs_file = self.file_paths[ds]['bine']
                    feat_data = pickle.load(open(embs_file, 'rb'))
                elif self.args.feature == 'line':
                    embs_file = self.file_paths[ds]['line']
                    feat_data = pickle.load(open(embs_file, 'rb'))
                elif self.args.feature.startswith('n2v'):
                    n2v_index = self.args.feature.split('_')[-1]
                    embs_file = self.file_paths[ds]['n2v'][n2v_index]
                    feat_data = pickle.load(open(embs_file, 'rb'))

                graph_simi = copy.deepcopy(graph_u2u)
                if self.args.learn_method not in self.args.embedding_ready_methods:
                    self.logger.info(f'Using {self.args.simi_func} user-user similarity.')
                    if pathlib.Path(graph_simi_file).is_file():
                        graph_simi = pickle.load(open(graph_simi_file, 'rb'))
                        self.logger.info('Loaded user-user similarity graph.')
                    else:
                        nz_entries = np.asarray(graph_u2u.nonzero()).T
                        self.logger.info(f'Calculating user-user {self.args.simi_func} similarity graph, {len(nz_entries)} edges to go...')
                        sz = 1000
                        n_batch = math.ceil(len(nz_entries) / sz)
                        batches = np.array_split(nz_entries, n_batch)
                        pool = Pool()
                        results = pool.map(get_simi_single_iter, [(entries_batch, graph_u2p, _get_simi) for entries_batch in batches])
                        results = list(zip(*results))
                        row = np.concatenate(results[0])
                        col = np.concatenate(results[1])
                        dat = np.concatenate(results[2])
                        graph_simi = csr_matrix((dat, (row, col)), shape=np.shape(graph_u2u))
                        if np.max(graph_simi) > 1:
                            graph_simi = graph_simi / np.max(graph_simi)
                        pickle.dump(graph_simi, open(graph_simi_file, "wb"))
                        self.logger.info('Calculated user-user similarity and saved it for catch.')

                if self.args.simi_func == 'rpcc':
                    graph_simi[graph_simi<0] = 0

                # get the user-user adjacency list
                adj_lists = {}
                for i in range(np.shape(graph_u2u)[0]):
                    adj_lists[i] = set(graph_u2u[i,:].nonzero()[1])

                # best neighbors list
                best_adj_lists = {}
                for i in range(np.shape(graph_u2u)[0]):
                    if len(adj_lists[i]) <= 15:
                        best_adj_lists[i] = adj_lists[i]
                    else:
                        adjs = graph_u2u[i].toarray()[0]
                        best_adj_lists[i] = set(np.argpartition(adjs, -15)[-15:])

                if self.args.a_loss != 'none':
                    self.logger.info(f'using {self.args.a_loss} anomaly loss.')
                    label_a_file = self.file_paths[ds][self.args.a_loss]
                    labels_a = pickle.load(open(label_a_file, 'rb'))
                    clusters_a = labels_a
                    u2size_of_cluster = {}
                    for u in range(np.shape(graph_u2p)[0]):
                        u2size_of_cluster[u] = np.sum(labels_a == labels_a[u])

                assert len(feat_data) == len(labels) == len(adj_lists)
                train_indexs = np.arange(len(labels))

                if os.path.isfile(data_split_file):
                    self.logger.info('loaded train-vali-test data splits.')
                    data_splits = pickle.load(open(data_split_file, 'rb'))
                else:
                    node_w_labels = np.where(labels>=0)[0]
                    data_splits = self._split_data_limited(node_w_labels)
                    pickle.dump(data_splits, open(data_split_file, "wb"))
                    self.logger.info('generated train-vali-test data splits and saved for future use.')

                self.logger.info(f'using data split {self.args.tvt_split}')
                test_indexs, val_indexs, train_indexs_cls = data_splits[self.args.tvt_split]

                user_id_max = len(labels)
                self.logger.info(f'distr: train {np.sum(labels[train_indexs_cls])}/{len(train_indexs_cls)}, vali: {np.sum(labels[val_indexs])}/{len(val_indexs)}, test: {np.sum(labels[test_indexs])}/{len(test_indexs)}.')
                self.logger.info(f'shape of user-user graph: {np.shape(graph_u2u)}, with {np.sum(graph_u2u > 0)} nnz entries ({100*np.sum(graph_u2u > 0)/(np.shape(graph_u2u)[0]**2):.4f}%).')

                if self.args.learn_method == 'rand':
                    self.logger.info('using randomized features.')
                    feat_data = torch.rand(np.shape(feat_data))

                setattr(self, dataSet+'_test', test_indexs)
                setattr(self, dataSet+'_val', val_indexs)
                setattr(self, dataSet+'_train', train_indexs)
                setattr(self, dataSet+'_train_cls', train_indexs_cls)
                setattr(self, dataSet+'_trainable', train_indexs)
                setattr(self, dataSet+'_useridmax', user_id_max)

                setattr(self, dataSet+'_feats', feat_data)
                setattr(self, dataSet+'_labels', labels)
                setattr(self, dataSet+'_adj_lists', adj_lists)
                setattr(self, dataSet+'_best_adj_lists', best_adj_lists)
                setattr(self, dataSet+'_simis', graph_simi)
                if self.args.a_loss != 'none':
                    setattr(self, dataSet+'_clusters_a', clusters_a)
                    setattr(self, dataSet+'_labels_a', labels_a)
                    setattr(self, dataSet+'_u2size_of_cluster', u2size_of_cluster)

        elif dataSet.startswith('weibo'):
            if self.args.learn_method == 'gnn':
                ds = dataSet
                graph_u2p_file = self.file_paths[ds]['graph_u2p']
                labels_file = self.file_paths[ds]['labels']
                features_bow_file = self.file_paths[ds]['features_bow']
                features_loc_file = self.file_paths[ds]['features_loc']

                graph_u2p = pickle.load(open(graph_u2p_file, 'rb'))
                labels = pickle.load(open(labels_file, 'rb'))
                feat_bow = pickle.load(open(features_bow_file, 'rb'))
                feat_loc = pickle.load(open(features_loc_file, 'rb'))

                if self.args.feature == 'all':
                    feat_data = np.concatenate((feat_loc, feat_bow), axis=1)
                elif self.args.feature == 'bow':
                    feat_data = feat_bow
                elif self.args.feature == 'loc':
                    feat_data = feat_loc

                m, n = np.shape(graph_u2p)
                adj_lists = {}
                feat_p = np.zeros((n, np.shape(feat_data)[1]))
                for i in range(m):
                    adj_lists[i] = set(m + graph_u2p[i,:].nonzero()[1])
                for j in range(n):
                    adj_lists[j+m] = set(graph_u2p[:,j].nonzero()[0])
                    feat_j = feat_data[graph_u2p[:,j].nonzero()[0]]
                    feat_j = np.mean(feat_j, 0)
                    feat_p[j] = feat_j

                feat_data = np.concatenate((feat_data, feat_p), 0)
                # feat_data = np.concatenate((feat_data, np.zeros((n, np.shape(feat_data)[1]))), 0)
                assert np.shape(feat_data)[0] == m+n

                assert len(feat_data) == len(labels)+n == len(adj_lists)
                test_indexs, val_indexs, train_indexs = self._split_data(len(labels))
                user_id_max = len(labels)
                train_indexs_cls = train_indexs
                train_indexs = np.arange(len(labels))

                user_id_max = len(labels)
                graph_simi = np.ones((10, 10))

                # get labels for anomaly losses if needed
                if self.args.a_loss != 'none':
                    self.logger.info(f'using {self.args.a_loss} anomaly loss.')
                    label_a_file = self.file_paths[ds][self.args.a_loss]["a_label"]
                    labels_a = pickle.load(open(label_a_file, 'rb')).astype(int)
                    # get clusters for anomaly losses if needed
                    if self.args.cluster_aloss:
                        clusters_file = self.file_paths[ds][self.args.a_loss]["a_cluster"]
                        clusters = pickle.load(open(clusters_file, 'rb'))
                        u2cluster = defaultdict(list)
                        for i in range(len(clusters)):
                            for u in clusters[i]:
                                u2cluster[u].append(i)
                        cluster_neighbors = defaultdict(set)
                        for u in range(np.shape(graph_u2p)[0]):
                            for clus_i in u2cluster[u]:
                                cluster_neighbors[u] |= clusters[clus_i]
                        for u in range(np.shape(graph_u2p)[0]):
                            cluster_neighbors[u] = cluster_neighbors[u] - set([u])
                            assert len(cluster_neighbors[u]) >= 1
                        u2size_of_cluster = {}
                        for u, neighbors in cluster_neighbors.items():
                            u2size_of_cluster[u] = len(neighbors)
                    else:
                        u2size_of_cluster = {}
                        for u in range(np.shape(graph_u2p)[0]):
                            u2size_of_cluster[u] = np.sum(labels_a == labels_a[u])

                setattr(self, dataSet+'_test', test_indexs)
                setattr(self, dataSet+'_val', val_indexs)
                setattr(self, dataSet+'_train', train_indexs)
                setattr(self, dataSet+'_train_cls', train_indexs_cls)
                setattr(self, dataSet+'_trainable', train_indexs)
                setattr(self, dataSet+'_useridmax', user_id_max)

                setattr(self, dataSet+'_feats', feat_data)
                setattr(self, dataSet+'_labels', labels)
                setattr(self, dataSet+'_adj_lists', adj_lists)
                setattr(self, dataSet+'_best_adj_lists', adj_lists)
                setattr(self, dataSet+'_simis', graph_simi)
                if self.args.a_loss != 'none':
                    setattr(self, dataSet+'_labels_a', labels_a)
                    setattr(self, dataSet+'_u2size_of_cluster', u2size_of_cluster)
                if self.args.cluster_aloss:
                    setattr(self, dataSet+'_cluster_neighbors', cluster_neighbors)
            else:
                ds = dataSet
                graph_u2p_file = self.file_paths[ds]['graph_u2p']
                graph_u2u_file = self.file_paths[ds]['graph_u2u']
                labels_file = self.file_paths[ds]['labels']
                features_bow_file = self.file_paths[ds]['features_bow']
                features_loc_file = self.file_paths[ds]['features_loc']

                if self.args.simi_func == 'pcc' or self.args.simi_func == 'rpcc':
                    _get_simi = getPCC
                    graph_simi_file = self.file_paths[ds]['graph_u2u_pcc']
                elif self.args.simi_func == 'acos':
                    _get_simi = getACOS
                    graph_simi_file = self.file_paths[ds]['graph_u2u_acos']
                elif self.args.simi_func == 'cos':
                    _get_simi = getCOS
                    graph_simi_file = self.file_paths[ds]['graph_u2u_cos']
                else:
                    self.logger.error('Invalid user-user similarity function.')

                graph_u2p = pickle.load(open(graph_u2p_file, 'rb'))
                graph_u2u = pickle.load(open(graph_u2u_file, 'rb'))
                labels = pickle.load(open(labels_file, 'rb'))
                feat_bow = pickle.load(open(features_bow_file, 'rb'))
                feat_loc = pickle.load(open(features_loc_file, 'rb'))

                if self.args.feature == 'all':
                    feat_data = np.concatenate((feat_loc, feat_bow), axis=1)
                elif self.args.feature == 'bow':
                    feat_data = feat_bow
                elif self.args.feature == 'loc':
                    feat_data = feat_loc
                elif self.args.feature == 'deepwalk':
                    embs_file = self.file_paths[ds]['deepwalk']
                    feat_data = pickle.load(open(embs_file, 'rb'))
                elif self.args.feature == 'bine':
                    embs_file = self.file_paths[ds]['bine']
                    feat_data = pickle.load(open(embs_file, 'rb'))
                elif self.args.feature == 'line':
                    embs_file = self.file_paths[ds]['line']
                    feat_data = pickle.load(open(embs_file, 'rb'))
                elif self.args.feature.startswith('n2v'):
                    n2v_index = self.args.feature.split('_')[-1]
                    embs_file = self.file_paths[ds]['n2v'][n2v_index]
                    feat_data = pickle.load(open(embs_file, 'rb'))
                else:
                    self.logger.error('Invalid features.')
                    sys.exit(1)

                # Compute or load the similarity graph when needed
                graph_simi = copy.deepcopy(graph_u2u)
                graph_simi[graph_simi > 0] = 1
                if self.args.learn_method not in self.args.embedding_ready_methods:
                    self.logger.info(f'Using {self.args.simi_func} user-user similarity.')
                    if pathlib.Path(graph_simi_file).is_file():
                        graph_simi = pickle.load(open(graph_simi_file, 'rb'))
                        self.logger.info('Loaded user-user similarity graph.')
                    else:
                        nz_entries = np.asarray(graph_u2u.nonzero()).T
                        self.logger.info(f'Calculating user-user {self.args.simi_func} similarity graph, {len(nz_entries)} edges to go...')
                        sz = 1000
                        n_batch = math.ceil(len(nz_entries) / sz)
                        batches = np.array_split(nz_entries, n_batch)
                        pool = Pool()
                        if self.args.simi_func == 'pcc' or self.args.simi_func == 'acos' or self.args.simi_func == 'cos':
                            # when calculating pcc or acos similarity, directly use the adjacency matrix
                            results = pool.map(get_simi_single_iter, [(entries_batch, graph_u2p, _get_simi) for entries_batch in batches])
                        elif self.args.simi_func == 'man' or self.args.simi_func == 'euc':
                            # when calculating manhattan or euclidean ditance, normalize the edge weights by power low distribution
                            _row, _col = graph_u2p.nonzero()
                            _data = np.asarray(graph_u2p[_row, _col]).squeeze()
                            pl_fit = powerlaw.Fit(_data,discrete=True,xmin=1)
                            alpha = pl_fit.power_law.alpha
                            self.logger.info(f'normalized the edge weight with log base {alpha:.2f}')
                            _data = np.log1p(_data) / np.log(alpha)
                            graph_u2p_normalized = csr_matrix((_data, (_row, _col)), shape=np.shape(graph_u2p))
                            results = pool.map(get_simi_single_iter, [(entries_batch, graph_u2p_normalized, _get_simi) for entries_batch in batches])
                        results = list(zip(*results))
                        row = np.concatenate(results[0])
                        col = np.concatenate(results[1])
                        dat = np.concatenate(results[2])
                        graph_simi = csr_matrix((dat, (row, col)), shape=np.shape(graph_u2u))
                        if np.max(graph_simi) > 1:
                            graph_simi = graph_simi / np.max(graph_simi)
                        pickle.dump(graph_simi, open(graph_simi_file, "wb"))
                        self.logger.info('Calculated user-user similarity and saved it for catch.')

                if self.args.simi_func == 'rpcc':
                    graph_simi[graph_simi<0] = 0

                # get the user-user adjacency list
                adj_lists = {}
                for i in range(np.shape(graph_u2u)[0]):
                    adj_lists[i] = set(graph_u2u[i,:].nonzero()[1])
                    assert len(adj_lists[i]) > 0

                # best neighbors list
                best_adj_lists = {}
                for i in range(np.shape(graph_u2u)[0]):
                    if len(adj_lists[i]) <= 15:
                        best_adj_lists[i] = adj_lists[i]
                    else:
                        adjs = graph_u2u[i].toarray()[0]
                        best_adj_lists[i] = set(np.argpartition(adjs, -15)[-15:])

                # get labels for anomaly losses if needed
                if self.args.a_loss != 'none':
                    self.logger.info(f'using {self.args.a_loss} anomaly loss.')
                    label_a_file = self.file_paths[ds][self.args.a_loss]["a_label"]
                    labels_a = pickle.load(open(label_a_file, 'rb')).astype(int)
                    # get clusters for anomaly losses if needed
                    if self.args.cluster_aloss:
                        clusters_file = self.file_paths[ds][self.args.a_loss]["a_cluster"]
                        clusters = pickle.load(open(clusters_file, 'rb'))
                        if self.args.noblock:
                            c = [set.union(*clusters[:-1]), clusters[-1]]
                            clusters = c
                        u2cluster = defaultdict(list)
                        for i in range(len(clusters)):
                            for u in clusters[i]:
                                u2cluster[u].append(i)
                        cluster_neighbors = defaultdict(set)
                        for u in range(np.shape(graph_u2p)[0]):
                            for clus_i in u2cluster[u]:
                                cluster_neighbors[u] |= clusters[clus_i]
                        for u in range(np.shape(graph_u2p)[0]):
                            cluster_neighbors[u] = cluster_neighbors[u] - set([u])
                            assert len(cluster_neighbors[u]) >= 1
                        u2size_of_cluster = {}
                        for u, neighbors in cluster_neighbors.items():
                            u2size_of_cluster[u] = len(neighbors)
                    else:
                        u2size_of_cluster = {}
                        for u in range(np.shape(graph_u2p)[0]):
                            u2size_of_cluster[u] = np.sum(labels_a == labels_a[u])

                assert len(feat_data) == len(labels) == len(adj_lists)
                test_indexs, val_indexs, train_indexs = self._split_data(len(labels))
                user_id_max = len(labels)
                train_indexs_cls = train_indexs
                train_indexs = np.arange(len(labels))

                user_id_max = len(labels)
                self.logger.info(f'distr: train {np.sum(labels[train_indexs_cls])}/{len(train_indexs_cls)}, vali: {np.sum(labels[val_indexs])}/{len(val_indexs)}, test: {np.sum(labels[test_indexs])}/{len(test_indexs)}.')
                self.logger.info(f'shape of user-user graph: {np.shape(graph_u2u)}, with {np.sum(graph_u2u > 0)} nnz entries ({100*np.sum(graph_u2u > 0)/(np.shape(graph_u2u)[0]**2):.4f}%).')

                if self.args.learn_method == 'rand':
                    self.logger.info('using randomized features.')
                    feat_data = torch.rand(np.shape(feat_data))

                setattr(self, dataSet+'_test', test_indexs)
                setattr(self, dataSet+'_val', val_indexs)
                setattr(self, dataSet+'_train', train_indexs)
                setattr(self, dataSet+'_train_cls', train_indexs_cls)
                setattr(self, dataSet+'_trainable', train_indexs)
                setattr(self, dataSet+'_useridmax', user_id_max)

                setattr(self, dataSet+'_feats', feat_data)
                setattr(self, dataSet+'_labels', labels)
                setattr(self, dataSet+'_adj_lists', adj_lists)
                setattr(self, dataSet+'_best_adj_lists', best_adj_lists)
                setattr(self, dataSet+'_simis', graph_simi)
                if self.args.a_loss != 'none':
                    setattr(self, dataSet+'_labels_a', labels_a)
                    setattr(self, dataSet+'_u2size_of_cluster', u2size_of_cluster)
                if self.args.cluster_aloss:
                    setattr(self, dataSet+'_cluster_neighbors', cluster_neighbors)


    def _split_data(self, num_nodes, test_split = 3, val_split = 6):
        rand_indices = np.random.permutation(num_nodes)

        test_size = num_nodes // test_split
        val_size = num_nodes // val_split
        train_size = num_nodes - (test_size + val_size)

        test_indexs = rand_indices[:test_size]
        val_indexs = rand_indices[test_size:(test_size+val_size)]
        train_indexs = rand_indices[(test_size+val_size):]

        return test_indexs, val_indexs, train_indexs

    def _split_data_limited(self, nodes, test_split = 4, val_split = 4):
        np.random.shuffle(nodes)

        test_size = len(nodes) // test_split
        val_size = len(nodes) // val_split
        train_size = len(nodes) - (test_size + val_size)

        val_indexs = nodes[:test_size]
        test_indexs_init = nodes[test_size:(test_size+val_size)]
        train_indexs_init = nodes[(test_size+val_size):]
        data_splits = []

        for _ in range(5):
            np.random.shuffle(test_indexs_init)
            np.random.shuffle(train_indexs_init)
            test_split = np.split(test_indexs_init, [test_size-10, test_size])
            train_split = np.split(train_indexs_init, [train_size-10, train_size])
            test_indexes = np.concatenate((test_split[0], train_split[1]))
            train_indexes = np.concatenate((train_split[0], test_split[1]))
            assert len(test_indexes) == test_size
            assert len(train_indexes) == train_size
            data_splits.append((test_indexes, val_indexs, train_indexes))

        return data_splits


