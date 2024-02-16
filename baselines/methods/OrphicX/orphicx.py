"""
Adapted from official OrphicX code
https://github.com/WanyuGroup/CVPR2022-OrphicX

paper: https://arxiv.org/pdf/2203.15209.pdf
"""

import warnings

warnings.filterwarnings("ignore")

import argparse
import os
import random
import shutil
import sys
import time

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.nn.functional as F
import torch_geometric
from torch import nn, optim
from torch_geometric.utils import dense_to_sparse
from tqdm import tqdm

from baselines.methods.OrphicX import causaleffect
from baselines.methods.OrphicX.gae.model import VGAE2MLP
from baselines.methods.OrphicX.gae.optimizer import loss_function as gae_loss

parser = argparse.ArgumentParser()

parser.add_argument('--dataset', default='Cora', type=str)
parser.add_argument('--repeat', default=1, type=int)
parser.add_argument('--epoch', default=50, type=int)

parser.add_argument('--encoder_hidden1', type=int, default=32, help='Number of units in hidden layer 1.')
parser.add_argument('--encoder_hidden2', type=int, default=16, help='Number of units in hidden layer 2.')
parser.add_argument('--encoder_output', type=int, default=16, help='Dim of output of VGAE encoder.')
parser.add_argument('--decoder_hidden1', type=int, default=16, help='Number of units in decoder hidden layer 1.')
parser.add_argument('--decoder_hidden2', type=int, default=16, help='Number of units in decoder  hidden layer 2.')
#parser.add_argument('--n_hops', type=int, default=3, help='Number of hops.')
#parser.add_argument('-e', '--epoch', type=int, default=300, help='Number of training epochs.')
parser.add_argument('-b', '--batch_size', type=int, default=32, help='Number of samples in a minibatch.')
parser.add_argument('--lr', type=float, default=0.003, help='Initial learning rate.')
parser.add_argument('--dropout', type=float, default=0., help='Dropout rate (1 - keep probability).')
parser.add_argument('--graph_labelling', action='store_true')
parser.add_argument('--K', type=int, default=3, help='Number of alpha dims in z.')
parser.add_argument('--NX', type=int, default=2, help='Number of samples of X.')
parser.add_argument('--Nalpha', type=int, default=15, help='Number of samples of alpha.') # changed by sakkas
parser.add_argument('--Nbeta', type=int, default=50, help='Number of samples of beta.') # changed by sakkas
parser.add_argument('--coef_lambda', type=float, default=0.1, help='Coefficient of gae loss.')
parser.add_argument('--coef_kl', type=float, default=0.2, help='Coefficient of gae loss.')
parser.add_argument('--coef_causal', type=float, default=1.0, help='Coefficient of causal loss.')
parser.add_argument('--coef_size', type=float, default=0.1, help='Coefficient of size loss.')
parser.add_argument('--patient', type=int, default=100, help='Patient for early stopping.')

args = parser.parse_args()
args.retrain = True

#device = torch.device("cuda" if args.gpu else "cpu")

# sakkas: we don't need graph labeling in our experiments
# def graph_labeling(G):
#     for node in G:
#         G.nodes[node]['string'] = 1
#     old_strings = tuple([G.nodes[node]['string'] for node in G])
#     for iter_num in range(100):
#         for node in G:
#             string = sorted([G.nodes[neigh]['string'] for neigh in G.neighbors(node)])
#             G.nodes[node]['concat_string'] =  tuple([G.nodes[node]['string']] + string)
#         d = nx.get_node_attributes(G,'concat_string')
#         nodes,strings = zip(*{k: d[k] for k in sorted(d, key=d.get)}.items())
#         map_string = dict([[string, i+1] for i, string in enumerate(sorted(set(strings)))])
#         for node in nodes:
#             G.nodes[node]['string'] = map_string[G.nodes[node]['concat_string']]
#         new_strings = tuple([G.nodes[node]['string'] for node in G])
#         if old_strings == new_strings:
#             break
#         else:
#             old_strings = new_strings
#     return G

def preprocess_graph(adj):
    adj = sp.coo_matrix(adj)
    adj_ = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    return sparse_mx_to_torch_sparse_tensor(adj_normalized)

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape).to_dense()

def gaeloss(x,mu,logvar,data):
    return gae_loss(preds=x, labels=data['adj_label'],
                    mu=mu, logvar=logvar, n_nodes=data['graph_size'],
                    norm=data['norm'], pos_weight=data['pos_weight'])



class OrphicXExplainer():
    def __init__(self, data, classifier, n_hops, device):
        self.pyg_data = data
        self.classifier = classifier

        self.device = device
        self.n_hops = n_hops
        self.ceparams = {
            'Nalpha': args.Nalpha,
            'Nbeta' : args.Nbeta,
            'K'     : args.K,
            'L'     : args.encoder_output - args.K,
            'z_dim' : args.encoder_output,
            'M'     : data.y.max().item() + 1
        }
        self.model = VGAE2MLP(
            self.pyg_data.x.size(1), args.encoder_hidden1,
            args.encoder_output, args.decoder_hidden1, args.decoder_hidden2,
            args.K, args.dropout).to(device)

        self.criterion = gaeloss

    def extract_neighborhood(self, node_idx):
        """Returns the neighborhood of a given ndoe."""
        device = self.device
        mapping, edge_idxs, node_idx_new, edge_mask = torch_geometric.utils.k_hop_subgraph(int(node_idx), self.n_hops, self.pyg_data.edge_index, relabel_nodes=True)
        node_idx_new = node_idx_new.item()
        sub_adj = torch_geometric.utils.to_dense_adj(edge_idxs)[0].cpu()
        adj_norm = preprocess_graph(sub_adj)
        adj_label = sub_adj + torch.eye(sub_adj.shape[0]) # adds self loops
        pos_weight = float(sub_adj.shape[0] * sub_adj.shape[0] - sub_adj.sum()) / sub_adj.sum()
        #pos_weight = torch.from_numpy(np.array(pos_weight))
        norm = torch.tensor(sub_adj.shape[0] * sub_adj.shape[0] / float((sub_adj.shape[0] * sub_adj.shape[0] - sub_adj.sum()) * 2))
        # Calculate hop_feat:
        pow_adj = ((sub_adj @ sub_adj >=1).float() - np.eye(sub_adj.shape[0]) - sub_adj >=1).float()
        feat = self.pyg_data.x[mapping]
        sub_feat = feat
        one_hot = torch.zeros((sub_adj.shape[0], ), dtype=torch.float)
        one_hot[node_idx_new] = 1
        hop_feat = [one_hot, sub_adj[node_idx_new], pow_adj[node_idx_new]]
        if self.n_hops == 3:# how to handle this?
            pow3_adj = ((pow_adj @ pow_adj >=1).float() - np.eye(pow_adj.shape[0]) - pow_adj >=1).float()
            hop_feat += [pow3_adj[node_idx_new]]
            hop_feat = torch.stack(hop_feat).t()
            sub_feat = torch.cat((sub_feat, hop_feat), dim=1)
        # if args.graph_labelling:
        #     G = graph_labeling(nx.from_numpy_array(sub_adj.numpy()))
        #     graph_label = np.array([G.nodes[node]['string'] for node in G])
        #     graph_label_onehot = label_onehot[graph_label]
        #     sub_feat = torch.cat((sub_feat, graph_label_onehot), dim=1)
        sub_label = self.pyg_data.y[mapping] #torch.from_numpy(label[mapping])
        return {
            "node_idx_new": node_idx_new,
            "feat": feat.unsqueeze(0).to(device),
            "sub_adj": sub_adj.unsqueeze(0).to(device),
            "sub_feat": sub_feat.unsqueeze(0).to(device),
            "adj_norm": adj_norm.unsqueeze(0).to(device),
            "sub_label": sub_label.to(device),
            "mapping": mapping.to(device),
            "adj_label": adj_label.unsqueeze(0).to(device),
            "graph_size": mapping.shape[-1],
            "pos_weight": pos_weight.unsqueeze(0).to(device),
            "norm": norm.unsqueeze(0).to(device)
        }
    

    def train(self, epochs=50):
        device = self.device
        input_dim = self.pyg_data.x.size(1)
        #adj = cg_dict["adj"][0]
        label = self.pyg_data.y
        #tg_G = self.data.edge_index
        features = self.pyg_data.x
        num_classes = max(label)+1
    
        self.classifier = self.classifier.to(self.device)
        self.classifier.eval()

        feat_dim = features.shape[-1]
        # hop feature
        feat_dim += self.n_hops + 1
        # if args.graph_labelling:
        #     feat_dim += label_onehot.shape[-1]

        optimizer = optim.Adam(self.model.parameters(), lr=args.lr)

        num_nodes = label.shape[0]
        train_idxs = self.pyg_data.train_mask.nonzero().cpu().numpy()[:, 0]
        
        if train_idxs.shape[0] > 500:
            train_idxs = np.random.choice(num_nodes, 500, replace=False)
        
        # if train_idxs.shape[0] > 200:
        #     train_idxs = np.random.choice(num_nodes, train_idxs.shape[0]*0.6, replace=False)
        #test_idxs = np.array([i for i in range(num_nodes) if i not in train_idxs])

        # exclude nodes labeled as class 0 and 4
        #train_label = self.pyg_data[train_idxs]
        #test_label = self.pyg_data[test_idxs]
        # if args.dataset == 'syn2':
        #     train_idxs = train_idxs[np.where(np.logical_and(train_label != 0, train_label != 4))[0]]
        #     test_idxs = test_idxs[np.where(np.logical_and(test_label != 0, test_label != 4))[0]]
        # else:
        #     train_idxs = train_idxs[np.where(train_label != 0)[0]]
        #     test_idxs = test_idxs[np.where(test_label != 0)[0]]

        num_train = len(train_idxs)
        #val_idxs = self.pyg_data.val_mask.nonzero().cpu().numpy()[:,0].tolist()
        #num_test = len(test_idxs)
        self.dataset = dict([[node_idx,self.extract_neighborhood(node_idx)] for node_idx in train_idxs])
        #dataset.update(dict([[node_idx,extract_neighborhood(node_idx)] for node_idx in test_idxs]))
        #self.dataset.update(dict([[node_idx, self.extract_neighborhood(node_idx)] for node_idx in val_idxs]))
        #val_idxs = list(test_idxs[:num_test//2])
        #test_idxs = list(test_idxs[num_test//2:])

        start_epoch = 1
        patient = args.patient
        best_loss = 100
        self.model.train()
        start_time = time.time()
        for epoch in tqdm(range(start_epoch, epochs+1)):
            batch = 0
            perm = np.random.permutation(num_train)
            train_losses = []
            for beg_ind in range(0, num_train, args.batch_size):
                batch += 1
                end_ind = min(beg_ind+args.batch_size, num_train)
                perm_train_idxs = list(train_idxs[perm[beg_ind: end_ind]])
                optimizer.zero_grad()
                nll_loss, org_logits, alpha_logits, alpha_sparsity = zip(*map(self.train_task, perm_train_idxs))# main train is here
                causal_loss = []
                for idx in random.sample(perm_train_idxs, args.NX):
                    _causal_loss, _ = causaleffect.joint_uncond(self.ceparams, self.model.dc, self.classifier, self.dataset[idx]['sub_adj'], self.dataset[idx]['feat'], node_idx=self.dataset[idx]['node_idx_new'], act=torch.sigmoid, device=device)
                    causal_loss += [_causal_loss]
                nll_loss = torch.stack(nll_loss).mean()
                causal_loss = torch.stack(causal_loss).mean()
                alpha_logits = torch.stack(alpha_logits)
                org_logits = torch.stack(org_logits)
                org_probs = F.softmax(org_logits, dim=1)
                klloss = F.kl_div(F.log_softmax(alpha_logits, dim=1), org_probs, reduction='mean')
                alpha_sparsity = torch.stack(alpha_sparsity).mean()
                loss = args.coef_lambda * nll_loss + \
                    args.coef_causal * causal_loss + \
                    args.coef_kl * klloss + \
                    args.coef_size * alpha_sparsity
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1)
                optimizer.step()
                sys.stdout.flush()
                train_losses += [[nll_loss.item(), causal_loss.item(), klloss.item(), alpha_sparsity.item(), loss.item()]]
            nll_loss, causal_loss, klloss, size_loss, train_loss = np.mean(train_losses, axis=0)


            # val_loss = self.eval_model(val_idxs,'val')
            # patient -= 1
            # if val_loss < best_loss:
            #     best_loss = val_loss
            #     patient = 100
            # elif patient <= 0:
            #     print("Early stop.")
            #     break
        #print("Train time:", time.time() - start_time)
        # Load checkpoint with lowest val loss
        # checkpoint = torch.load('explanation/%s/model.ckpt' % args.output)
        # self.model.load_state_dict(checkpoint['model'])
        # optimizer.load_state_dict(checkpoint['optimizer'])

        del self.dataset


    def train_task(self, node_idx):
        dataset = self.dataset
        data = dataset[node_idx]
        mu, logvar = self.model.encode(data['sub_feat'], data['adj_norm'])
        sample_mu = self.model.reparameterize(mu, logvar)
        recovered = self.model.dc(sample_mu)
        nll_loss = self.criterion(recovered, mu, logvar, data).mean()
        org_logits = self.classifier(data['feat'], data['sub_adj'])[0][0, data['node_idx_new']]
        alpha_mu = torch.zeros_like(mu)
        alpha_mu[:,:,:args.K] = sample_mu[:,:,:args.K]
        alpha_adj = torch.sigmoid(self.model.dc(alpha_mu))
        alpha_size = (alpha_adj*data['sub_adj']).sum()
        org_size = data['sub_adj'].sum()
        alpha_sparsity = alpha_size/org_size
        masked_alpha_adj = alpha_adj * data['sub_adj']
        alpha_logits = self.classifier(data['feat'], masked_alpha_adj)[0][0,data['node_idx_new']]
        return nll_loss, org_logits, alpha_logits, alpha_sparsity
    
    def explain(self, node_idx):
        data = self.extract_neighborhood(node_idx)
        mu, logvar = self.model.encode(data['sub_feat'], data['adj_norm'])
        sample_mu = self.model.reparameterize(mu, logvar)
        alpha_mu = torch.zeros_like(mu)
        alpha_mu[:,:,:args.K] = sample_mu[:,:,:args.K]
        alpha_adj = torch.sigmoid(self.model.dc(alpha_mu))
        alpha_adj = alpha_adj * data['sub_adj']
        return alpha_adj[0]
    

    def eval_task(self, node_idx):
        dataset = self.dataset
        data = dataset[node_idx]
        recovered, mu, logvar = self.model(data['sub_feat'], data['adj_norm'])
        recovered_adj = torch.sigmoid(recovered)
        nll_loss = self.criterion(recovered, mu, logvar, data).mean()
        org_logits = self.classifier(data['feat'], data['sub_adj'])[0][0, data['node_idx_new']]
        masked_recovered_adj = recovered_adj * data['sub_adj']
        recovered_logits = self.classifier(data['feat'], masked_recovered_adj)[0][0,data['node_idx_new']]
        alpha_mu = torch.zeros_like(mu)
        alpha_mu[:,:,:args.K] = mu[:,:,:args.K]
        alpha_adj = torch.sigmoid(self.model.dc(alpha_mu))
        alpha_size = (alpha_adj*data['sub_adj']).sum()
        org_size = data['sub_adj'].sum()
        alpha_sparsity = alpha_size/org_size
        masked_alpha_adj = alpha_adj * data['sub_adj']
        alpha_logits = self.classifier(data['feat'], masked_alpha_adj)[0][0,data['node_idx_new']]
        beta_mu = torch.zeros_like(mu)
        beta_mu[:,:,args.K:] = mu[:,:,args.K:]
        beta_adj = torch.sigmoid(self.model.dc(beta_mu))
        masked_beta_adj = beta_adj * data['sub_adj']
        beta_logits = self.classifier(data['feat'], masked_beta_adj)[0][0,data['node_idx_new']]
        return nll_loss, org_logits, recovered_logits, alpha_logits, beta_logits, alpha_sparsity

    def eval_model(self, node_idxs, prefix=''):
        dataset = self.dataset
        device = self.device
        with torch.no_grad():
            labels = self.pyg_data.y[node_idxs]
            nll_loss, org_logits, recovered_logits, alpha_logits, beta_logits, alpha_sparsity = zip(*map(self.eval_task, node_idxs))
            causal_loss = []
            beta_info = []
            for idx in random.sample(node_idxs, args.NX):
                _causal_loss, _ = causaleffect.joint_uncond(self.ceparams, self.model.dc, self.classifier, dataset[idx]['sub_adj'], dataset[idx]['feat'], node_idx=dataset[idx]['node_idx_new'], act=torch.sigmoid, device=device)
                _beta_info, _ = causaleffect.beta_info_flow(self.ceparams, self.model.dc, self.classifier, dataset[idx]['sub_adj'], dataset[idx]['feat'], node_idx=dataset[idx]['node_idx_new'], act=torch.sigmoid, device=device)
                causal_loss += [_causal_loss]
                beta_info += [_beta_info]
            nll_loss = torch.stack(nll_loss).mean()
            causal_loss = torch.stack(causal_loss).mean()
            alpha_info = causal_loss
            beta_info = torch.stack(beta_info).mean()
            alpha_logits = torch.stack(alpha_logits)
            beta_logits = torch.stack(beta_logits)
            recovered_logits = torch.stack(recovered_logits)
            org_logits = torch.stack(org_logits)
            org_probs = F.softmax(org_logits, dim=1)
            recovered_probs = F.softmax(recovered_logits, dim=1)
            recovered_log_probs = F.log_softmax(recovered_logits, dim=1)
            klloss = F.kl_div(F.log_softmax(alpha_logits, dim=1), org_probs, reduction='mean')
            pred_labels = torch.argmax(org_probs,axis=1)
            org_acc = (torch.argmax(org_probs,axis=1) == torch.argmax(recovered_probs,axis=1)).float().mean()
            pred_acc = (torch.argmax(recovered_probs,axis=1) == labels).float().mean()
            kl_pred_org = F.kl_div(recovered_log_probs, org_probs, reduction='mean')
            alpha_probs = F.softmax(alpha_logits, dim=1)
            alpha_log_probs = F.log_softmax(alpha_logits, dim=1)
            beta_probs = F.softmax(beta_logits, dim=1)
            beta_log_probs = F.log_softmax(beta_logits, dim=1)
            alpha_gt_acc = (torch.argmax(alpha_probs,axis=1) == labels).float().mean()
            alpha_pred_acc = (torch.argmax(alpha_probs,axis=1) == pred_labels).float().mean()
            alpha_kld = F.kl_div(alpha_log_probs, org_probs, reduction='mean')
            beta_gt_acc = (torch.argmax(beta_probs,axis=1) == labels).float().mean()
            beta_pred_acc = (torch.argmax(beta_probs,axis=1) == pred_labels).float().mean()
            beta_kld = F.kl_div(beta_log_probs, org_probs, reduction='mean')
            alpha_sparsity = torch.stack(alpha_sparsity).mean()
            loss = args.coef_lambda * nll_loss + \
                args.coef_causal * causal_loss + \
                args.coef_kl * klloss + \
                args.coef_size * alpha_sparsity
        return loss.item()