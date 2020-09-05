import os
import sys
import torch
import random
import numpy as np
from scipy.sparse import csr_matrix

import torch.nn as nn
import torch.nn.functional as F

class Classification(nn.Module):

    def __init__(self, emb_size, num_classes):
        super(Classification, self).__init__()

        self.fc1 = nn.Linear(emb_size, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def init_params(self):
        for param in self.parameters():
            if len(param.size()) == 2:
                nn.init.xavier_uniform_(param)
            else:
                # initialize all bias as zeros
                nn.init.constant_(param, 0.0)

    def forward(self, embeds):
        x = F.elu_(self.fc1(embeds))
        x = F.elu_(self.fc2(x))
        logists = torch.log_softmax(x, 1)
        return logists

class UnsupervisedLoss(object):
    """docstring for UnsupervisedLoss"""

    def __init__(self, Dc, ds, device, biased, a_loss, C):
        super(UnsupervisedLoss, self).__init__()
        self.Q = 10
        self.N_WALKS = 10
        self.WALK_LEN = 2
        self.N_WALK_LEN = 2
        self.MARGIN = 3
        self.C = C
        self.Dc = Dc
        self.ds = ds
        self.device = device
        self.biased = biased
        self.a_loss = a_loss
        self.cluster_aloss = Dc.args.cluster_aloss
        self.adj_lists = getattr(Dc, ds+'_adj_lists')
        self.best_adj_lists = getattr(Dc, ds+'_best_adj_lists')
        self.simi_graph = getattr(Dc, ds+'_simis')
        self.train_nodes = getattr(Dc, ds+'_train')
        self.trainable_nodes = getattr(Dc, ds+'_trainable')

        self.target_nodes = None
        self.positive_pairs = []
        self.negative_pairs = []
        self.node_positive_pairs = {}
        self.node_negative_pairs = {}
        self.positive_pairs_aloss = []
        self.negative_pairs_aloss = []
        self.node_positive_pairs_aloss = {}
        self.node_negative_pairs_aloss = {}
        self.unique_nodes_batch = []

    def get_loss_unsup(self, embeddings, nodes):
        assert len(embeddings) == len(self.unique_nodes_batch)
        assert False not in [nodes[i]==self.unique_nodes_batch[i] for i in range(len(nodes))]
        node2index = {n:i for i,n in enumerate(self.unique_nodes_batch)}

        nodes_score_mean = []
        nodes_score_maxmin = []
        assert len(self.node_positive_pairs) == len(self.node_negative_pairs)
        for node in self.node_positive_pairs:
            pps = self.node_positive_pairs[node]
            nps = self.node_negative_pairs[node]
            if len(pps) == 0 or len(nps) == 0:
                continue

            indexs = [list(x) for x in zip(*pps)]
            node_indexs = [node2index[x] for x in indexs[0]]
            neighb_indexs = [node2index[x] for x in indexs[1]]
            pos_score = F.cosine_similarity(embeddings[node_indexs], embeddings[neighb_indexs])
            pos_score = torch.log(torch.sigmoid(pos_score))
            pos_score_min, _ = torch.min(pos_score, 0)
            pos_score_mean = torch.mean(pos_score, 0)

            indexs = [list(x) for x in zip(*nps)]
            node_indexs = [node2index[x] for x in indexs[0]]
            neighb_indexs = [node2index[x] for x in indexs[1]]
            neg_score = F.cosine_similarity(embeddings[node_indexs], embeddings[neighb_indexs])
            neg_score = torch.log(torch.sigmoid(-neg_score))
            neg_score_max, _ = torch.max(neg_score, 0)
            neg_score_mean = self.Q*torch.mean(neg_score, 0)

            nodes_score_maxmin.append(torch.max(torch.tensor(0.0).to(self.device), -neg_score_max-pos_score_min+self.MARGIN).view(1,-1))
            nodes_score_mean.append(torch.mean(-neg_score_mean-pos_score_mean).view(1,-1))

        loss_maxmin = torch.mean(torch.cat(nodes_score_maxmin, 0),0)
        loss_mean = torch.mean(torch.cat(nodes_score_mean, 0))

        return loss_maxmin, loss_mean

    def get_loss_anomaly(self, embeddings, nodes):
        u2size_of_cluster = getattr(self.Dc, self.ds+'_u2size_of_cluster')
        assert len(embeddings) == len(self.unique_nodes_batch)
        assert False not in [nodes[i]==self.unique_nodes_batch[i] for i in range(len(nodes))]
        node2index = {n:i for i,n in enumerate(self.unique_nodes_batch)}

        nodes_score_mean = []
        nodes_score_maxmin = []
        assert len(self.node_positive_pairs_aloss) == len(self.node_negative_pairs_aloss)
        for node in self.node_positive_pairs_aloss:
            pps = self.node_positive_pairs_aloss[node]
            nps = self.node_negative_pairs_aloss[node]
            if len(pps) == 0 or len(nps) == 0:
                continue

            indexs = [list(x) for x in zip(*pps)]
            node_indexs = [node2index[x] for x in indexs[0]]
            neighb_indexs = [node2index[x] for x in indexs[1]]
            pos_score = F.cosine_similarity(embeddings[node_indexs], embeddings[neighb_indexs])
            pos_score = torch.log(torch.sigmoid(pos_score))
            pos_score_min, _ = torch.min(pos_score, 0)
            pos_score_mean = torch.mean(pos_score, 0)

            indexs = [list(x) for x in zip(*nps)]
            node_indexs = [node2index[x] for x in indexs[0]]
            neighb_indexs = [node2index[x] for x in indexs[1]]
            neg_score = F.cosine_similarity(embeddings[node_indexs], embeddings[neighb_indexs])
            neg_score = torch.log(torch.sigmoid(-neg_score))
            neg_score_max, _ = torch.max(neg_score, 0)
            neg_score_mean = self.Q*torch.mean(neg_score, 0)

            # nodes_score_maxmin.append(torch.max(torch.tensor(0.0).to(self.device), -neg_score_max-pos_score_min+self.MARGIN).view(1,-1))
            margin = self.C / pow(u2size_of_cluster[node], 1/4)
            nodes_score_maxmin.append(torch.max(torch.tensor(0.0).to(self.device), -neg_score_max-pos_score_min+margin).view(1,-1))
            nodes_score_mean.append(torch.mean(-neg_score_mean-pos_score_mean).view(1,-1))

        aloss_maxmin = torch.mean(torch.cat(nodes_score_maxmin, 0),0)
        aloss_mean = torch.mean(torch.cat(nodes_score_mean, 0))

        return aloss_maxmin, aloss_mean

    def extend_nodes(self, nodes, num_neg=6):
        self.positive_pairs = []
        self.node_positive_pairs = {}
        self.negative_pairs = []
        self.node_negative_pairs = {}

        self.target_nodes = nodes
        self.get_positive_nodes(nodes)
        self.get_negative_nodes(nodes, num_neg)

        self.unique_nodes_batch = list(set([i for x in self.positive_pairs for i in x]) | set([i for x in self.negative_pairs for i in x]))

        if self.a_loss != 'none':
            self.positive_pairs_aloss = []
            self.negative_pairs_aloss = []
            self.node_positive_pairs_aloss = {}
            self.node_negative_pairs_aloss = {}
            if self.cluster_aloss:
                self.get_cluster_aloss_pos_neg_nodes(nodes, 50)
            else:
                self.get_label_aloss_pos_neg_nodes(nodes, 50)
            unique_nodes_aloss = set([i for x in self.positive_pairs_aloss for i in x]) | set([i for x in self.negative_pairs_aloss for i in x])
            self.unique_nodes_batch = list(set(self.unique_nodes_batch) | unique_nodes_aloss)

        assert set(self.target_nodes) <= set(self.unique_nodes_batch)
        return self.unique_nodes_batch

    def get_cluster_aloss_pos_neg_nodes(self, nodes, num):
        cluster_neighbors = getattr(self.Dc, self.ds+'_cluster_neighbors')
        for node in nodes:
            # sampling nodes in same clusters
            pos_pairs = []
            pos_pool = list(cluster_neighbors[node])
            if len(pos_pool) <= num:
                pos_nodes = pos_pool
            else:
                pos_nodes = np.random.choice(pos_pool, num, replace=False)
            for pos_node in pos_nodes:
                if pos_node != node and pos_node in self.trainable_nodes:
                    self.positive_pairs_aloss.append((node, pos_node))
                    pos_pairs.append((node, pos_node))
            # sampling nodes in different clusters
            neg_pairs = []
            train_nodes_all = getattr(self.Dc, self.ds+'_train')
            neg_pool = list(set(train_nodes_all) - cluster_neighbors[node])
            if len(neg_pool) <= num:
                neg_nodes = neg_pool
            else:
                neg_nodes = np.random.choice(neg_pool, num, replace=False)
            for neg_node in neg_nodes:
                if neg_node != node and neg_node in self.trainable_nodes:
                    self.negative_pairs_aloss.append((node, neg_node))
                    neg_pairs.append((node, neg_node))

            self.node_positive_pairs_aloss[node] = pos_pairs
            self.node_negative_pairs_aloss[node] = neg_pairs

    def get_label_aloss_pos_neg_nodes(self, nodes, num):
        labels_a = getattr(self.Dc, self.ds+'_labels_a')
        for node in nodes:
            pos_pairs = []
            neg_pairs = []
            label_node = labels_a[node]
            # positive samples
            pos_pool = np.where(labels_a==label_node)[0]
            if len(pos_pool) <= num:
                pos_nodes = pos_pool
            else:
                pos_nodes = np.random.choice(pos_pool, num, replace=False)
            for pos_node in pos_nodes:
                if pos_node != node and pos_node in self.trainable_nodes:
                    self.positive_pairs_aloss.append((node, pos_node))
                    pos_pairs.append((node, pos_node))
            # negative samples
            neg_pool = np.where(labels_a!=label_node)[0]
            if len(neg_pool) <= num:
                neg_nodes = neg_pool
            else:
                neg_nodes = np.random.choice(neg_pool, num, replace=False)
            for neg_node in neg_nodes:
                if neg_node != node and neg_node in self.trainable_nodes:
                    self.negative_pairs_aloss.append((node, neg_node))
                    neg_pairs.append((node, neg_node))
            self.node_positive_pairs_aloss[node] = pos_pairs
            self.node_negative_pairs_aloss[node] = neg_pairs

    def get_positive_nodes(self, nodes):
        for node in nodes:
            if len(self.adj_lists[int(node)]) == 0:
                continue
            cur_pairs = []
            for i in range(self.N_WALKS):
                curr_node = node
                for j in range(self.WALK_LEN):
                    if self.Dc.args.best_rw:
                        neighs = list(self.best_adj_lists[int(curr_node)])
                    else:
                        neighs = list(self.adj_lists[int(curr_node)])
                    if self.biased:# biased random walk
                        p = self.simi_graph[int(curr_node), neighs].toarray().squeeze(0)
                        p = p - np.min(p) + 1e-12
                        p = p**3 / np.sum(p**3)
                        assert len(neighs) == len(p)
                        next_node = np.random.choice(neighs, 1, p=p)[0]
                    else:
                        # unbiased random walk
                        next_node = random.choice(neighs)

                    # self co-occurrences are useless
                    if next_node != node and next_node in self.trainable_nodes:
                        self.positive_pairs.append((node,next_node))
                        cur_pairs.append((node,next_node))
                    curr_node = next_node
            self.node_positive_pairs[node] = cur_pairs
        return self.positive_pairs

    def get_negative_nodes(self, nodes, num_neg):
        for node in nodes:
            neighbors = set([node])
            frontier = set([node])
            for i in range(self.N_WALK_LEN):
                current = set()
                for outer in frontier:
                    current |= self.adj_lists[outer]
                frontier = current - neighbors
                neighbors |= current
            far_nodes = set(self.trainable_nodes) - neighbors
            neg_samples = random.sample(far_nodes, num_neg) if num_neg < len(far_nodes) else far_nodes
            self.negative_pairs.extend([(node, neg_node) for neg_node in neg_samples])
            self.node_negative_pairs[node] = [(node, neg_node) for neg_node in neg_samples]
        return self.negative_pairs

class GNNLayer(nn.Module):
    """
    Encodes a node's using 'convolutional' GraphSage approach
    """
    def __init__(self, input_size, out_size, a_loss, gcn=False, gat=False):
        super(GNNLayer, self).__init__()

        self.input_size = input_size
        self.out_size = out_size
        self.a_loss = a_loss
        self.gcn = gcn
        self.gat = gat
        self.weight = nn.Parameter(torch.FloatTensor(self.input_size if self.gcn or self.gat else 2 * self.input_size, out_size))

        self.init_params()

    def init_params(self):
        for param in self.parameters():
            nn.init.xavier_uniform_(param)

    def forward(self, self_feats, aggregate_feats):
        """
        Generates embeddings for a batch of nodes.
        nodes    -- list of nodes
        """
        if self.gcn or self.gat:
            combined = aggregate_feats
        else:
            combined = torch.cat([self_feats, aggregate_feats], dim=1)

        if self.a_loss == 'none':
            combined = F.relu(combined.mm(self.weight))
        else:
            combined = F.elu(combined.mm(self.weight))

        combined = F.normalize(combined)
        return combined

class Attention(nn.Module):
    """Computes the self-attention between pair of nodes"""
    def __init__(self, input_size, out_size):
        super(Attention, self).__init__()

        self.input_size = input_size
        self.out_size = out_size
        self.attention_raw = nn.Linear(2*input_size, 1, bias=False)
        self.attention_emb = nn.Linear(2*out_size, 1, bias=False)

    def forward(self, row_embs, col_embs):
        if row_embs.size(1) == self.input_size:
            att = self.attention_raw
        elif row_embs.size(1) == self.out_size:
            att = self.attention_emb
        e = att(torch.cat((row_embs, col_embs), dim=1))
        return F.leaky_relu(e, negative_slope=0.2)

class GNN(nn.Module):
    """docstring for GNN"""
    def __init__(self, num_layers, input_size, out_size, raw_features, Dc, a_loss, device, gcn=False, gat=False, agg_func='MEAN'):
        super(GNN, self).__init__()

        self.input_size = input_size
        self.out_size = out_size
        self.num_layers = num_layers
        self.gcn = gcn
        self.gat = gat
        self.device = device
        self.agg_func = agg_func
        self.a_loss = a_loss

        self.raw_features = raw_features
        self.adj_lists = getattr(Dc, Dc.args.dataSet+'_adj_lists')
        self.Dc = Dc
        if Dc.args.best_rw:
            self.best_adj_lists = getattr(Dc, Dc.args.dataSet+'_best_adj_lists')

        for index in range(1, num_layers+1):
            layer_size = out_size if index != 1 else input_size
            setattr(self, 'gnn_layer'+str(index), GNNLayer(layer_size, out_size, a_loss, gcn=self.gcn, gat=self.gat))
        if self.gat:
            self.attention = Attention(input_size, out_size)

    def forward(self, nodes_batch):
        """
        Generates embeddings for a batch of nodes.
        nodes_batch -- batch of nodes to learn the embeddings
        """
        lower_layer_nodes = list(nodes_batch)
        nodes_batch_layers = [(lower_layer_nodes,)]
        for i in range(self.num_layers):
            if self.gcn or self.gat:
                lower_samp_neighs, lower_layer_nodes_dict, lower_layer_nodes= self._get_unique_neighs_list(lower_layer_nodes, num_sample=None)
            else:
                lower_samp_neighs, lower_layer_nodes_dict, lower_layer_nodes= self._get_unique_neighs_list(lower_layer_nodes)
            nodes_batch_layers.insert(0, (lower_layer_nodes, lower_samp_neighs, lower_layer_nodes_dict))

        assert len(nodes_batch_layers) == self.num_layers + 1

        pre_hidden_embs = self.raw_features
        for index in range(1, self.num_layers+1):
            nb = nodes_batch_layers[index][0]
            pre_neighs = nodes_batch_layers[index-1]
            aggregate_feats = self.aggregate(nb, pre_hidden_embs, pre_neighs)
            gnn_layer = getattr(self, 'gnn_layer'+str(index))
            if index > 1:
                nb = self._nodes_map(nb, pre_neighs)
            cur_hidden_embs = gnn_layer(self_feats=pre_hidden_embs[nb],
                                        aggregate_feats=aggregate_feats)
            pre_hidden_embs = cur_hidden_embs

        return pre_hidden_embs

    def _nodes_map(self, nodes, neighs):
        layer_nodes, samp_neighs, layer_nodes_dict = neighs
        assert len(samp_neighs) == len(nodes)
        index = [layer_nodes_dict[x] for x in nodes]
        return index

    def _get_unique_neighs_list(self, nodes, num_sample=10):
        _set = set
        if self.Dc.args.best_rw:
            to_neighs = [self.best_adj_lists[int(node)] for node in nodes]
        else:
            to_neighs = [self.adj_lists[int(node)] for node in nodes]
        if num_sample is None:
            samp_neighs = to_neighs
        else:
            _sample = random.sample
            samp_neighs = [_set(_sample(to_neigh, num_sample)) if len(to_neigh) >= num_sample else to_neigh for to_neigh in to_neighs]
        samp_neighs = [samp_neigh | set([nodes[i]]) for i, samp_neigh in enumerate(samp_neighs)]
        _unique_nodes_list = list(set.union(*samp_neighs))
        i = list(range(len(_unique_nodes_list)))
        unique_nodes = dict(list(zip(_unique_nodes_list, i)))
        return samp_neighs, unique_nodes, _unique_nodes_list

    def aggregate(self, nodes, pre_hidden_embs, pre_neighs):
        unique_nodes_list, samp_neighs, unique_nodes = pre_neighs

        assert len(nodes) == len(samp_neighs)
        indicator = [(nodes[i] in samp_neighs[i]) for i in range(len(samp_neighs))]
        assert (False not in indicator)
        if not self.gcn and not self.gat:
            samp_neighs = [(samp_neighs[i]-set([nodes[i]])) for i in range(len(samp_neighs))]

        if len(pre_hidden_embs) == len(unique_nodes):
            embed_matrix = pre_hidden_embs
        else:
            embed_matrix = pre_hidden_embs[torch.LongTensor(unique_nodes_list)]
        # get the row and column nonzero indices for the mask tensor
        column_indices = [unique_nodes[n] for samp_neigh in samp_neighs for n in samp_neigh]
        row_indices = [i for i in range(len(samp_neighs)) for j in range(len(samp_neighs[i]))]
        mask = torch.zeros(len(samp_neighs), len(unique_nodes))
        mask[row_indices, column_indices] = 1

        if self.gat:
            indices = (torch.LongTensor(row_indices), torch.LongTensor(column_indices))
            nodes_indices = torch.LongTensor([unique_nodes[nodes[n]] for n in row_indices])
            row_embs = embed_matrix[nodes_indices]
            col_embs = embed_matrix[column_indices]
            atts = self.attention(row_embs, col_embs).squeeze()
            mask = torch.zeros(len(samp_neighs), len(unique_nodes)).to(embed_matrix.device)
            mask.index_put_(indices, atts)
        # elif self.learn_method == 'bigal':
        else:
            # multiply mask with similarity
            simi_mask = getattr(self.Dc, self.Dc.args.dataSet+'_simis')[nodes, :][:, unique_nodes_list].toarray()
            mask = mask * torch.Tensor(simi_mask)

        if self.agg_func == 'MEAN':
            mask = mask.to(embed_matrix.device)
            mask = F.normalize(mask, p=1)
            aggregate_feats = mask.mm(embed_matrix)

        elif self.agg_func == 'MAX':
            indexs = [x.nonzero() for x in mask==1]
            aggregate_feats = []
            for feat in [embed_matrix[x.squeeze()] for x in indexs]:
                if len(feat.size()) == 1:
                    aggregate_feats.append(feat.view(1, -1))
                else:
                    aggregate_feats.append(torch.max(feat,0)[0].view(1, -1))
            aggregate_feats = torch.cat(aggregate_feats, 0)

        return aggregate_feats

class NoGNN(nn.Module):
    """
    This model is for testing the usefulness of the loss.
    Update features directly with the loss.
    """

    def __init__(self, features):
        super(NoGNN, self).__init__()

        self.out_size = features.size()[1]
        self.features = nn.Parameter(features)

    def forward(self, nodes_batch):
        """
        nodes_batch -- batch of nodes to learn the embeddings
        """
        return self.features[nodes_batch]
