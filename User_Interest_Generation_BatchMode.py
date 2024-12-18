#Generate user-side interest knowledge in json and uploaded to openAI in a batch processing mode
import os
import time
import re
import random
import jsonlines
import pandas as pd
import numpy as np
from tqdm import tqdm
from call_llm import get_system_generate, get_prompt_generate

#"gpt-3.5-turbo-0125"
llm = "gpt-3.5-turbo-0125"
version = 'v1'
max_his_num = 30
dataset = 'book-crossing'
if dataset in ['dbbook2014', 'book-crossing']:
    field = 'books'
else:
    field = 'movies'

data_path = f'./data/{dataset}'

if dataset in ['dbbook2014', 'ml1m']:
    item_list = pd.read_csv(data_path+'/item_list.txt', sep=' ')
else:
    item_list = list()    
    lines = open(data_path+'/item_list.txt', 'r').readlines()
    for l in lines[1:]:
        if l == "\n":
            continue
        tmps = l.replace("\n", "").strip()
        elems = tmps.split(' ')
        org_id = elems[0]
        remap_id = elems[1]
        if len(elems[2:]) == 0:
            continue
        title = ' '.join(elems[2:])    
        item_list.append([org_id, remap_id, title])
    item_list = pd.DataFrame(item_list, columns=['org_id', 'remap_id', 'entity_name'])
    
# kg = pd.read_csv(data_path+'/kg_final.txt', sep=' ', names=['head', 'relation', 'tail'])
# relation = pd.read_csv(data_path+'/relation_list.txt', sep=' ')
def load_ratings(file_name):
    inter_mat = list()

    lines = open(file_name, 'r').readlines()
    for l in lines:
        if l == "\n":
            continue
        tmps = l.replace("\n", "").strip()
        inters = [int(i) for i in tmps.split(' ')]

        u_id, pos_ids = inters[0], inters[1:]
        pos_ids = set(pos_ids)

        for i_id in pos_ids:
            inter_mat.append([u_id, i_id])

    df = pd.DataFrame(inter_mat, columns=['userid', 'itemid'])
    return df
ratings = load_ratings(data_path+'/train.txt')

if dataset in ['dbbook2014', 'ml1m']:
    item2entity = {int(row['remap_id']):str(row['entity_name']).split('/')[-1].strip().replace('_', ' ') for _, row in item_list.iterrows()}
elif dataset in ['book-crossing']:
    item2entity = {int(row['remap_id']):str(row['entity_name']).strip().replace('_', ' ') for _, row in item_list.iterrows()}
else:
    item2entity = {int(row['remap_id']):str(row['entity_name']) for _, row in item_list.iterrows()}
# relation2name = {int(row['remap_id']):str(row['org_id']) for _, row in relation.iterrows()}
print('user num:', len(ratings['userid'].unique()))
print('item num:', len(ratings['itemid'].unique()))
print('rating num:', len(ratings))

ratings['item_name'] = ratings['itemid'].map(item2entity)
ratings = ratings[ratings['item_name']!='nan']
ratings = ratings[ratings['item_name']!='None']
ratings = ratings[~ratings['item_name'].isna()]

user_his_dict = {}
ratings_group = ratings.groupby('userid')
his_len = []
for u, v in ratings_group:
    his = v['item_name'].to_list()
    his_len.append(len(his))
    user_his_dict[u] = his[-max_his_num:]
print(f'user num: {len(user_his_dict)}')
print(f'max his num:{max(his_len)}')
print(f'min his num:{min(his_len)}')
median = np.median(np.array(his_len))
mean = round(np.mean(np.array(his_len)), 2)
print(f'median his num:{median}')
print(f'mean his num:{mean}')

def prepare_input(uid, history):
    message = [
    {"role": "system", "content": get_system_generate(history, field)},
    {"role": "user", "content": get_prompt_generate(history, field)}
    ]
    row = {"custom_id":str(uid), "method": "POST", "url": "/v1/chat/completions", \
           "body": {"model": llm, "messages": message,"max_tokens": 1000}}
    input_content.append(row)

input_content = []
for uid, history in tqdm(user_his_dict.items(), desc='calc'):
    prepare_input(uid, history)

with jsonlines.open(f'batch_input/{dataset}_user_max_his{max_his_num}_{llm}_input.jsonl', mode='w') as writer:
    for row in input_content:
        writer.write(row)