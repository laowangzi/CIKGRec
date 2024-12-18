import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import pandas as pd
import time
import random
from collections import defaultdict
from functools import partial
import math
from tqdm import tqdm
from torch_geometric.nn import LGConv
from torch_geometric.utils import dropout_edge, add_self_loops
from .loss_func import _L2_loss_mean, BPRLoss, InfoNCELoss, sce_loss

class CIKGRec(torch.nn.Module):
    def __init__(self, config, edge_index):
        super(CIKGRec, self).__init__()
        self.config = config
        self.users = config['users']
        self.items = config['items']
        self.entities = config['entities']
        self.relations = config['relations']
        self.interests = config['interests']
        self.layer = config['layer']
        self.emb_dim = config['dim']
        self.weight_decay = config['l2_reg']
        self.l2_reg_kge = config['l2_reg_kge']
        self.cf_weight = config['cf_weight']
        self.edge_drop = config['edge_dropout']
        self.message_drop_rate = config['message_dropout']
        self.message_drop = nn.Dropout(p=self.message_drop_rate)

        #cl params
        self.gcl_weight = config['gcl_weight']
        self.gcl_temp = config['gcl_temp']
        self.eps = config['eps']

        #embs
        self.user_entity_emb = nn.Embedding(self.users+self.entities, self.emb_dim)
        nn.init.xavier_uniform_(self.user_entity_emb.weight)
        self.rel_emb = nn.Embedding(self.relations, self.emb_dim)
        nn.init.xavier_uniform_(self.rel_emb.weight)

        #user interest recon.
        self.encoder_layer_emb_mask = config['encoder_layer_emb_mask']
        self.decoder_layer_emb_mask = config['decoder_layer_emb_mask']
        self._replace_rate = config['replace_rate']
        self._mask_token_rate = 1.0 - self._replace_rate
        self._drop_edge_rate = config['edge_mask_drop_edge_rate']
        self.interest_recon_weight = config['interest_recon_weight']
        self.criterion_emb_mask = self.setup_loss_fn(config['emb_mask_loss'], config['emb_mask_loss_alpha'])
        self.enc_mask_token = nn.Parameter(torch.zeros(1, self.emb_dim))
        self.encoder_to_decoder = nn.Linear(self.emb_dim, self.emb_dim)
        nn.init.xavier_uniform_(self.encoder_to_decoder.weight)
        #dynamic
        self.gmae_p = None
        self.total_epoch = config['total_epoch']
        self.increase_type = config['increase_type']
        self.min_mask_rate = config['min_mask_rate']
        self.max_mask_rate = config['max_mask_rate']
        
        self.propagate = LGConv(normalize=True)

        if config['add_self_loops']:
            # print('rec graph add self loops!')
            edge_index, _ = add_self_loops(edge_index)
        self.edge_index = edge_index                  
        self.bpr = BPRLoss()

    def setup_loss_fn(self, loss_fn, alpha_l=2):
        if loss_fn == "mse":
            criterion = nn.MSELoss()
        elif loss_fn == "sce":
            criterion = partial(sce_loss, alpha=alpha_l)
        else:
            raise NotImplementedError
        return criterion

    def compute(self, x=None, edge_index=None, perturbed=False, mess_drop=False, layer_num=None):
        if x == None:
            emb = self.user_entity_emb.weight
        else:
            emb = x
        if self.message_drop_rate > 0.0 and mess_drop:
            emb = self.message_drop(emb)
        all_layer = [emb]
        if layer_num != None:
            layers = layer_num
        else:
            layers = self.layer
        for layer in range(layers):
            if edge_index == None:
                emb = self.propagate(emb, self.edge_index)
            else:
                emb = self.propagate(emb, edge_index)
            if perturbed:
                random_noise = torch.rand_like(emb).to(self.config['device'])
                emb += torch.sign(emb) * F.normalize(random_noise, dim=-1) * self.eps
            all_layer.append(emb)
        all_layer = torch.stack(all_layer, dim=1)
        all_layer = torch.mean(all_layer, dim=1)
        return all_layer                                 
        
    def forward(self, user_idx, pos_item, neg_item):
        if self.edge_drop > 0.0:
            use_edge, _ = dropout_edge(edge_index=self.edge_index, force_undirected=True, p=self.edge_drop, training=self.training)
            all_layer = self.compute(edge_index=use_edge, mess_drop=True)
        else:
            all_layer = self.compute(mess_drop=True) 
        user_emb = all_layer[user_idx]
        pos_emb = all_layer[pos_item]
        neg_emb = all_layer[neg_item]
        
        users_emb_ego = self.user_entity_emb(user_idx)
        pos_emb_ego = self.user_entity_emb(pos_item)
        neg_emb_ego = self.user_entity_emb(neg_item)

        pos_score = (user_emb * pos_emb).squeeze()
        neg_score = (user_emb * neg_emb).squeeze()
        cf_loss = self.bpr(torch.sum(pos_score, dim=-1), torch.sum(neg_score, dim=-1))
        
        reg_loss = (1/2)*(users_emb_ego.norm(p=2).pow(2)+pos_emb_ego.norm(p=2).pow(2)+neg_emb_ego.norm(p=2).pow(2))/float(len(user_idx))
        loss = self.cf_weight*cf_loss + reg_loss*self.weight_decay

        return loss

    #contrastive learning module
    def cross_domain_contrastive_loss(self, user_idx, pos_item, edge_index_interest, edge_index_kg, edge_index_cf):        
        all_layer_1 = self.compute(edge_index=edge_index_cf, perturbed=True, mess_drop=False)
        all_layer_2 = self.compute(edge_index=edge_index_interest, perturbed=False, mess_drop=False)
        all_layer_3 = self.compute(edge_index=edge_index_kg, perturbed=False, mess_drop=False)

        all_layer_1 = F.normalize(all_layer_1, dim=1)
        all_layer_2 = F.normalize(all_layer_2, dim=1)
        all_layer_3 = F.normalize(all_layer_3, dim=1)

        user_view_1, item_view_1, interest_view_1 = all_layer_1[user_idx], all_layer_1[pos_item], all_layer_1[-self.interests:]
        user_view_2, item_view_2, interest_view_2 = all_layer_2[user_idx], all_layer_2[pos_item], all_layer_2[-self.interests:]
        user_view_3, item_view_3, interest_view_3 = all_layer_3[user_idx], all_layer_3[pos_item], all_layer_3[-self.interests:]

        pos_of_user = torch.sum(user_view_1*user_view_2, dim=-1) 
        pos_of_item = torch.sum(item_view_1*item_view_3, dim=-1) 
        
        tot_of_user = torch.matmul(user_view_1, torch.transpose(all_layer_2[:self.users], 0, 1)) 
        tot_of_item = torch.matmul(item_view_1, torch.transpose(all_layer_3[self.users:self.users+self.entities], 0, 1))

        gcl_logits_user = tot_of_user - pos_of_user[:, None]               
        gcl_logits_item = tot_of_item - pos_of_item[:, None]               
        #InfoNCE Loss
        clogits_user = torch.logsumexp(gcl_logits_user / self.gcl_temp, dim=1)
        clogits_item = torch.logsumexp(gcl_logits_item / self.gcl_temp, dim=1)
        infonce_loss = torch.mean(clogits_user) + torch.mean(clogits_item)
        
        return self.gcl_weight*infonce_loss

    def encoding_mask_noise(self, x, mask_rate=0.3):
        num_nodes = x.shape[0]
        perm = torch.randperm(num_nodes, device=x.device)
        num_mask_nodes = int(mask_rate * num_nodes)

        # random masking
        mask_nodes = perm[: num_mask_nodes]
        keep_nodes = perm[num_mask_nodes: ]

        if self._replace_rate > 0:
            num_noise_nodes = int(self._replace_rate * num_mask_nodes)
            perm_mask = torch.randperm(num_mask_nodes, device=x.device)
            token_nodes = mask_nodes[perm_mask[: int(self._mask_token_rate * num_mask_nodes)]]
            noise_nodes = mask_nodes[perm_mask[-int(self._replace_rate * num_mask_nodes):]]
            noise_to_be_chosen = torch.randperm(num_nodes, device=x.device)[:num_noise_nodes]

            out_x = x.clone()
            out_x[noise_nodes] = x[noise_to_be_chosen]
        else:
            out_x = x.clone()
            token_nodes = mask_nodes

        out_x[token_nodes] = 0.0
        out_x[token_nodes] += self.enc_mask_token

        return out_x, (mask_nodes, keep_nodes)

    def interest_recon_loss(self, edge_index):
        x = self.user_entity_emb.weight
        #mask user interest nodes
        user_emb, entity_emb, interest_emb = torch.split(x, [self.users, self.entities-self.interests, self.interests])
        use_interest, (mask_nodes, keep_nodes) = self.encoding_mask_noise(interest_emb, self.gmae_p)
        mask_nodes = mask_nodes + (self.users + self.entities-self.interests)
        keep_nodes = keep_nodes + (self.users + self.entities-self.interests)
        use_x = torch.cat((user_emb, entity_emb, use_interest))

        if self._drop_edge_rate > 0.0:
            use_edge_index, masked_edges = dropout_edge(edge_index, force_undirected=True, p=self._drop_edge_rate, training=self.training)
            use_edge_index = add_self_loops(use_edge_index)[0]
        else:
            use_edge_index = edge_index

        enc_rep = self.compute(x=use_x, edge_index=use_edge_index, layer_num=self.encoder_layer_emb_mask)
        rep = self.encoder_to_decoder(enc_rep)

        rep[mask_nodes] = 0.0
        recon = self.compute(x=rep, edge_index=use_edge_index, layer_num=self.decoder_layer_emb_mask)

        x_init = x[mask_nodes]
        x_rec = recon[mask_nodes]
        loss = self.criterion_emb_mask(x_rec, x_init)
        return self.interest_recon_weight*loss

    #dynamic mask rate
    def calc_mask_rate(self, epoch):
        total_round, increase_type, init_rate, max_rate = self.total_epoch, self.increase_type, self.min_mask_rate, self.max_mask_rate
        if increase_type == "lin":
            increase_rate = (max_rate - init_rate) / total_round
            mask_rate = init_rate + epoch * increase_rate
        elif increase_type == "exp":
            alpha = (max_rate / init_rate) ** (1 / total_round)
            mask_rate = init_rate * alpha ** epoch
        mask_rate = min(mask_rate, max_rate)    
        return mask_rate

    def get_score_matrix(self):
        all_layer = self.compute()
        U_e = all_layer[:self.users].detach().cpu() 
        V_e = all_layer[self.users:self.users+self.items].detach().cpu() 
        score_matrix = torch.matmul(U_e, V_e.t())   
        return score_matrix

    def get_kg_loss(self, batch_h, batch_t_pos, batch_t_neg, batch_r):
        h = self.user_entity_emb(batch_h)
        t_pos = self.user_entity_emb(batch_t_pos)
        t_neg = self.user_entity_emb(batch_t_neg)
        r = self.rel_emb(batch_r)
        pos_score = torch.sum(torch.pow(h + r - t_pos, 2), dim=1)     
        neg_score = torch.sum(torch.pow(h + r - t_neg, 2), dim=1)     
        kg_loss = (-1.0) * F.logsigmoid(neg_score - pos_score)
        kg_loss = torch.mean(kg_loss)
        l2_loss = _L2_loss_mean(h) + _L2_loss_mean(r) + _L2_loss_mean(t_pos) + _L2_loss_mean(t_neg)
        loss = kg_loss + self.l2_reg_kge * l2_loss
        return loss