import numpy as np
import pandas as pd
import torch
import time
import json
from tqdm import tqdm
from torch_geometric.data import Data
import argparse

from models.Model import CIKGRec
from utils import MyLoader, init_seed, generate_kg_batch
from evaluate import eval_model
from prettytable import PrettyTable
from torch_geometric.utils import coalesce

def parse_args():
    parser = argparse.ArgumentParser(description="Run CIKGRec.")
    parser.add_argument('--seed', type=int, default=2024,
                        help='Random seed.')
    parser.add_argument('--dataset', nargs='?', default='dbbook2014',
                        help='Choose a dataset from {yelp2018, last-fm, amazon-book}')
    parser.add_argument("--gpu_id", type=int, default=0, help="gpu id")
    args = parser.parse_args()
    return args

args = parse_args()
dataset = args.dataset
config = json.load(open(f'./config/{dataset}.json'))
config['device'] = f'cuda:{args.gpu_id}'
seed = args.seed
Ks = config['Ks']
init_seed(seed, True)
print('-'*100)
for k, v in config.items():
    print(f'{k}: {v}')
print('-'*100)
loader = MyLoader(config)
src = loader.train.loc[:, 'userid'].to_list()+ loader.train.loc[:, 'itemid'].to_list()
tgt = loader.train.loc[:, 'itemid'].to_list() + loader.train.loc[:, 'userid'].to_list()
src_cf = loader.train.loc[:, 'userid'].to_list()+ loader.train.loc[:, 'itemid'].to_list()
tgt_cf = loader.train.loc[:, 'itemid'].to_list() + loader.train.loc[:, 'userid'].to_list()
edge_index = [src, tgt]
#kg
src_k = loader.kg_org['head'].to_list() + loader.kg_org['tail'].to_list()
tgt_k = loader.kg_org['tail'].to_list() + loader.kg_org['head'].to_list()
edge_index[0].extend(src_k)
edge_index[1].extend(tgt_k)
#user interest graph
src_in = loader.kg_interest['uid'].to_list() + loader.kg_interest['interest'].to_list()
tgt_in = loader.kg_interest['interest'].to_list() + loader.kg_interest['uid'].to_list()
edge_index[0].extend(src_in)
edge_index[1].extend(tgt_in)

edge_index_ig = [src_in+src_cf, tgt_in+src_cf]
edge_index_ig = torch.LongTensor(edge_index_ig)
edge_index_ig = coalesce(edge_index_ig)
graph_ig = Data(edge_index=edge_index_ig.contiguous())
graph_ig = graph_ig.to(config['device'])
print(f'Is ig no duplicate edge: {graph_ig.is_coalesced()}')

edge_index_kg = [src_k+src_cf, tgt_k+src_cf]
edge_index_kg = torch.LongTensor(edge_index_kg)
edge_index_kg = coalesce(edge_index_kg)
graph_kg = Data(edge_index=edge_index_kg.contiguous())
graph_kg = graph_kg.to(config['device'])
print(f'Is kg no duplicate edge: {graph_kg.is_coalesced()}')

edge_index = torch.LongTensor(edge_index)
edge_index = coalesce(edge_index)
graph = Data(edge_index=edge_index.contiguous())
graph = graph.to(config['device'])
print(f'Is cikg no duplicate edge: {graph.is_coalesced()}')
#pure cf graph
edge_index_cf = torch.LongTensor([src, tgt])
edge_index_cf = coalesce(edge_index_cf)
graph_cf = Data(edge_index=edge_index_cf.contiguous())
graph_cf = graph_cf.to(config['device'])
print(f'Is cg no duplicate edge: {graph_cf.is_coalesced()}')

model = CIKGRec(config, graph.edge_index)
model = model.to(config['device'])
optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
if config['use_kge']:
    kg_optimizer = torch.optim.Adam(model.parameters(), lr=config['lr_kg'])
train_loader, _ = loader.get_cf_loader(bs=config['batch_size'])

best_res = None
best_loss = 1e10
best_score = 0
patience = config['patience']
best_performance = []
for epoch in range(config['num_epoch']):
    loss_list = []
    print(f'epoch {epoch+1} start!')
    model.train()
    start = time.time() 
    #update mask rate (dynamic mask rate)
    model.gmae_p = model.calc_mask_rate(epoch)
    #train cf
    for data in train_loader:
        user, pos, neg = data
        user, pos, neg = user.to(config['device']), pos.to(config['device']), neg.to(config['device'])
        optimizer.zero_grad()
        loss_rec = model(user.squeeze(), pos.squeeze(), neg.squeeze())
        loss_cross_domain_contrastive = model.cross_domain_contrastive_loss(user.squeeze(), pos.squeeze(), graph_ig.edge_index, graph_kg.edge_index, graph_cf.edge_index)
        loss_interest_recon = model.interest_recon_loss(graph.edge_index)
        loss = loss_rec + loss_cross_domain_contrastive + loss_interest_recon
        loss.backward()
        optimizer.step()
        loss_list.append(loss.detach().cpu().numpy())
    sum_loss = np.sum(loss_list)/len(loss_list)
    if config['use_kge']:
        #train kg
        kg_total_loss = 0
        n_kg_batch = config['n_triplets'] // 4096
        for iter in range(1, n_kg_batch + 1):
            #entity contains interests, but at the transE stage, neg sample space shouldn't contains interests
            kg_batch_head, kg_batch_relation, kg_batch_pos_tail, kg_batch_neg_tail = generate_kg_batch(loader.kg_dict, 4096, config['entities']-config['interests'], 0)
            kg_batch_head = kg_batch_head.to(config['device'])
            kg_batch_relation = kg_batch_relation.to(config['device'])
            kg_batch_pos_tail = kg_batch_pos_tail.to(config['device'])
            kg_batch_neg_tail = kg_batch_neg_tail.to(config['device'])
    
            kg_batch_loss = model.get_kg_loss(kg_batch_head, kg_batch_pos_tail, kg_batch_neg_tail, kg_batch_relation)
    
            kg_optimizer.zero_grad()
            kg_batch_loss.backward()
            kg_optimizer.step()
            kg_total_loss += kg_batch_loss.item()
    if (epoch+1) % config['eval_interval'] == 0:            
        result = eval_model(model, loader, 'test')
        score = result['recall'][1] + result['ndcg'][1]
        if score > best_score:
            best_performance = []
            best_score = score
            patience = config['patience']
            for i, k in enumerate(Ks):
                table = PrettyTable([f'recall@{k}',f'ndcg@{k}',f'precision@{k}',f'hit_ratio@{k}'])
                table.add_row([round(result['recall'][i], 4),round(result['ndcg'][i], 4),round(result['precision'][i], 4), round(result['hit_ratio'][i], 4)])
                print(table)
                best_performance.append(table)
        else:
            patience -= 1
            print(f'patience: {patience}')

    if config['use_kge']:
        print(f'kg loss:{round(kg_total_loss / (n_kg_batch+1), 4)}')    
    print('epoch loss: ', round(sum_loss, 4))
    print('current mask rate:', round(model.gmae_p, 4))
    if patience <= 0:
        break
    end = time.time()
    print(f'train time {round(end-start)}s')
    print('-'*90)

print('Best testset performance:')
for table in best_performance:
    print(table)