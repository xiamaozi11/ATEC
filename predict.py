#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold,KFold
from tqdm import tqdm
import json
import os
from torch.utils import data
import torch
import torch.nn as nn
from torch.utils import data
from torch.autograd import Variable
from torch.optim.optimizer import Optimizer
import time
from torch.nn import DataParallel
import gc
from sklearn.metrics import f1_score, accuracy_score,roc_auc_score,auc
#from apex import amp
from torch.optim.lr_scheduler import ReduceLROnPlateau
# from nezha import *
from transformers import *
from transformers import BertTokenizer
import argparse
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

input_data_path = '/home/admin/workspace/job/input/test.jsonl'
# input_data_path = '/mnt/atec/train.jsonl'
# output_model_path = '/home/admin/workspace/job/output/your-model-name'
output_predictions_path = '/home/admin/workspace/job/output/predictions.jsonl'

with open(input_data_path, 'r', encoding='utf-8') as fp:
    data = fp.readlines()
texts = []
ids=[]
for d in data:
    tmp = json.loads(d)
    texts.append(str(tmp['x269']) +' '+str(tmp['x321']) +' '+str(tmp['x479']) +' '+str(tmp['x459']) +' '+ str(tmp['memo_polish']))
    ids.append(tmp['id'])
df = pd.DataFrame(columns=['id','text'])
df['text']=texts
df['id'] = ids

df_test = df
df_test['text'] = df_test['text'].astype(str)
df_test.index=range(len(df_test))

# df_train = df_train[0:1000]
# df_test = df_test[0:1000]
import random
SEED = 2020
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
SEED = 2020
PATH = './'
BERT_PATH = './'
WEIGHT_PATH = './'
MAX_SEQUENCE_LENGTH = 80
input_categories = ['text']
output_categories = 'label'
modelname='bert-base'
pretraindir = './chinese_wwm_ext_pytorch'
outputdir='./bert_wwm'

# print('train shape =', df_train.shape)
print('test shape =', df_test.shape)
try:
    os.mkdir(outputdir)
except:
    pass 
def _convert_to_transformer_inputs(question, tokenizer, max_sequence_length):
    """Converts tokenized input to ids, masks and segments for transformer (including bert)"""
    
    def return_id(str1, truncation_strategy, length):

        inputs = tokenizer.encode_plus(str1, 
            add_special_tokens=True,
            max_length=length,
            #truncation_strategy=truncation_strategy,
            truncation=True
            )
        
        input_ids =  inputs["input_ids"]
        input_masks = [1] * len(input_ids)
        input_segments = inputs["token_type_ids"]
        padding_length = length - len(input_ids)
        padding_id = tokenizer.pad_token_id
        input_ids = input_ids + ([padding_id] * padding_length)
        input_masks = input_masks + ([0] * padding_length)
        input_segments = input_segments + ([0] * padding_length)
        
        return [input_ids, input_masks, input_segments]
    
    input_ids_q, input_masks_q, input_segments_q = return_id(
        question, 'longest_first', max_sequence_length)
    

    
    return [input_ids_q, input_masks_q, input_segments_q]

def compute_input_arrays(df, columns, tokenizer, max_sequence_length):
    input_ids_q, input_masks_q, input_segments_q = [], [], []
    input_ids_a, input_masks_a, input_segments_a = [], [], []
    for _, instance in tqdm(df[columns].iterrows()):
        q = instance.text

        ids_q, masks_q, segments_q= \
        _convert_to_transformer_inputs(q, tokenizer, max_sequence_length)
        
        input_ids_q.append(ids_q)
        input_masks_q.append(masks_q)
        input_segments_q.append(segments_q)

    return [np.asarray(input_ids_q, dtype=np.int32), 
            np.asarray(input_masks_q, dtype=np.int32), 
            np.asarray(input_segments_q, dtype=np.int32)]

def compute_output_arrays(df, columns):
    return np.asarray(df[columns]).reshape(-1,1)

def search_auc(y_true, y_pred):
    best = 0
    best_t = 0
    for i in range(1,100):
        tres = i / 100
        y_pred_bin =  (y_pred > tres).astype(int)
        score = roc_auc_score(y_true, y_pred_bin)
        if score > best:
            best = score
            best_t = tres
    print('best', best)
    print('thres', best_t)
    return best, best_t

def search_f1(y_true, y_pred):
    best = 0
    best_t = 0
    for i in range(1,100):
        tres = i / 100
        y_pred_bin =  (y_pred > tres).astype(int)
        score = f1_score(y_true, y_pred_bin)
        if score > best:
            best = score
            best_t = tres
    print('best', best)
    print('thres', best_t)
    return best, best_t

if modelname == 'bert-base':
    tokenizer = BertTokenizer.from_pretrained(f"{pretraindir}/vocab.txt")
elif modelname == 'roberta-large':
    tokenizer =  BertTokenizer.from_pretrained(f"{pretraindir}/vocab.txt")
elif modelname == 'roberta-base':
    tokenizer =  BertTokenizer.from_pretrained(f"{pretraindir}/vocab.txt")
elif modelname == 'bert-base-wwm':
    tokenizer =  BertTokenizer.from_pretrained(f"{pretraindir}/vocab.txt")
elif modelname == 'bert-base-wwm-ext':
    tokenizer =  BertTokenizer.from_pretrained(f"{pretraindir}/vocab.txt")
elif modelname == 'nezha-large':
    tokenizer =  BertTokenizer.from_pretrained(f"{pretraindir}/vocab.txt")
elif modelname == 'nezha-base':
    tokenizer =  BertTokenizer.from_pretrained(f"{pretraindir}/vocab.txt")
elif modelname == 'nezha-base-wwm':
    tokenizer =  BertTokenizer.from_pretrained(f"{pretraindir}/vocab.txt")
elif modelname == 'nezha-large-wwm':
    tokenizer =  BertTokenizer.from_pretrained(f"{pretraindir}/vocab.txt")
elif modelname == 'macbert_large':
    tokenizer =  BertTokenizer.from_pretrained(f"{pretraindir}/vocab.txt")

# inputs = compute_input_arrays(df_train, input_categories, tokenizer, MAX_SEQUENCE_LENGTH)
test_inputs = compute_input_arrays(df_test, input_categories, tokenizer, MAX_SEQUENCE_LENGTH)
def compute_output_arrays(df, columns):
    return np.asarray(df[columns]).reshape(-1,1)


# outputs = compute_output_arrays(df_train, output_categories)

from torch.utils import data
class DFData(data.Dataset):
    def __init__(self,questions,labels,augument=False,training=True):
        super(DFData, self).__init__()
        self.augument=augument
        self.ids=questions[0]
        self.atts =questions[1]
        self.segs =questions[2]
        self.labels= labels
        self.len_ = len(self.ids)
        self.training=training
    def shuffle(self,d):
        return np.random.permutation(d.tolist())

    def dropout(self,d,p=0.5):
        len_ = len(d)
        index = np.random.choice(len_,int(len_*p))
        d[index]=0
        return d
    
    def __getitem__(self,index):
        if self.training:
            ids,att,seg, label =  self.ids[index],self.atts[index],self.segs[index],self.labels[index]
        else:
            ids,att,seg =  self.ids[index],self.atts[index],self.segs[index]
        
        if self.training and self.augument :
            ids= self.dropout(ids,p=0.05)
        ids=torch.tensor(ids).long()
        att=torch.tensor(att).long()
        seg=torch.tensor(seg).long()
        if self.training:
            label=torch.tensor(label,dtype=torch.float16)
            return (ids,att,seg),label
        else:
            return (ids,att,seg)

    def __len__(self):
        return self.len_
    
class BertClassificationHeadModel1(nn.Module):
    def __init__(self, weights_key, clf_dropout=0.15, n_class=1):
        super(BertClassificationHeadModel, self).__init__()
#         self.transformer = BertModel.from_pretrained(weights_key, output_hidden_states=False, torchscript=True)
#         self.dropout = nn.Dropout(clf_dropout)
#         self.linear = nn.Linear(self.transformer.config.hidden_size*3, n_class)
#         nn.init.xavier_uniform_(self.linear.weight)
#         self.linear.bias.data.fill_(0.0)
        self.transformer = BertModel.from_pretrained(weights_key,output_hidden_states=True, torchscript=True)
        self.dropout = nn.Dropout(clf_dropout)
        self.linear = nn.Linear(self.transformer.config.hidden_size*3, n_class)
        nn.init.xavier_uniform_(self.linear.weight)
        self.linear.bias.data.fill_(0.0)


    def forward(self, input_ids, position_ids=None, token_type_ids=None):
        #hidden_states, h_conc = self.transformer(input_ids = input_ids, attention_mask = position_ids, token_type_ids = token_type_ids)
        hidden_states, h_conc = self.transformer(input_ids = input_ids, attention_mask = position_ids, token_type_ids = token_type_ids)
        
        
        logits = self.linear(self.dropout(h_conc))
        return logits

class BertClassificationHeadModel(nn.Module):
    def __init__(self, weights_key, clf_dropout=0.15, n_class=1):
        super(BertClassificationHeadModel, self).__init__()
        self.transformer = BertModel.from_pretrained(weights_key, output_hidden_states=False, torchscript=True)
        self.dropout = nn.Dropout(clf_dropout)
        self.linear = nn.Linear(self.transformer.config.hidden_size, n_class)
        nn.init.xavier_uniform_(self.linear.weight)
        self.linear.bias.data.fill_(0.0)


    def forward(self, input_ids, position_ids=None, token_type_ids=None):
        hidden_states,h_conc = self.transformer(input_ids = input_ids, attention_mask = position_ids, token_type_ids = token_type_ids)

        logits = self.linear(self.dropout(h_conc))
        return logits

class RobertaClassificationModel(nn.Module):
    def __init__(self, weights_key, clf_dropout=0.15, n_class=1):
        super(RobertaClassificationModel, self).__init__()
        #self.config = 
        self.transformer = BertModel.from_pretrained(weights_key)
        self.dropout = nn.Dropout(clf_dropout)
        self.linear = nn.Linear(self.transformer.config.hidden_size, n_class)
#         nn.init.normal_(self.linear.weight, std = 0.1)
#         nn.init.normal_(self.linear.bias, std = 0.1)

    def forward(self, input_ids, position_ids=None, token_type_ids=None):
        hidden_states,  h_conc= self.transformer(input_ids = input_ids, attention_mask = position_ids, token_type_ids = token_type_ids)
        #hidden_states_cls_embeddings = [x[:, 0] for x in hidden_states[-4:]]
        #hidden_states_add = hidden_states[-1][
        #for i in [-2,-3,-4]:
        #    hidden_states_add = torch.add(hidden_states_add, hidden_states[i][:,0])
        #h_conc = torch.cat([hidden_states1, hidden_states2, hidden_states3], axis=1)
        #h_conc =  torch.add(torch.add(hidden_states1, hidden_states2) ,hidden_states3)
        #h_conc = hidden_states_add#torch.cat([hidden_states[-1][:,0], hidden_states_add], axis=1)
        logits = self.linear(self.dropout(h_conc))
        return logits
    
def create_model(modelname):
    if modelname == 'bert-base':
        model=BertClassificationHeadModel(f'{pretraindir}', n_class=1)
    elif modelname == 'roberta-large':
        model=RobertaClassificationModel(f'{pretraindir}', n_class=1)
    elif modelname == 'roberta-base':
        model=RobertaClassificationModel(f'{pretraindir}', n_class=1)
    elif modelname == 'bert-base-wwm':
        model=BertClassificationHeadModel(f'{pretraindir}', n_class=1)
    elif modelname == 'bert-base-wwm-ext':
        model=RobertaClassificationModel(f'{pretraindir}', n_class=1)
    elif modelname == 'nezha-large':
        model=RobertaClassificationModel(f'{pretraindir}', n_class=1)
    elif modelname == 'nezha-base':
        model=BertClassificationHeadModel(f'{pretraindir}', n_class=1)
    elif modelname == 'nezha-base-wwm':
        model=RobertaClassificationModel(f'{pretraindir}', n_class=1)
    elif modelname == 'nezha-large-wwm':
        model=RobertaClassificationModel(f'{pretraindir}', n_class=1)
    elif modelname == 'macbert_large':
        model =  RobertaClassificationModel(f'{pretraindir}', n_class=1)
                      
        
    model = model.to(device)
    return model

class FGM():
    def __init__(self, model):
        self.model = model
        self.backup = {}

    def attack(self, epsilon= 0.1, emb_name='word_embeddings'):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and  emb_name in name.lower():
                self.backup[name] = param.data.clone()
                try:
                    norm = torch.norm(param.grad)
                    if norm != 0:
                        r_at = epsilon * param.grad / norm
                        param.data.add_(r_at)
                    #print('good',name)
                except:
                    pass
                    #print('error',name)

    def restore(self, emb_name='word_embeddings'):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name.lower():
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}
        
from sklearn.model_selection import GroupKFold
def train(n_epochs):
#     train_preds = np.zeros((len(df_train),1))
    test_preds = np.zeros((len(df_test),1))
    batch_size = 32
    step_size = 300
    base_lr, max_lr = 2e-5, 5e-5 
    best_score_f = 0
    for fold in range(5):
        
        model = create_model(modelname)
        fgm = FGM(model)
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        
        def is_backbone(n):
            return ("transformer" in n.lower()) or ('bert' in n.lower())
        
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if is_backbone(n)],'lr': base_lr},
            {'params': [p for n, p in param_optimizer if not is_backbone(n)],'lr': base_lr * 10}
        ]

        optimizer = torch.optim.AdamW(optimizer_grouped_parameters,betas=(0.9, 0.999), lr=base_lr)
        lr_scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.2, patience=1, min_lr=1e-6, verbose=True)
        
       # model, optimizer = amp.initialize(model, optimizer, opt_level="O1",verbosity=0)
        if torch.cuda.device_count() > 1:
            model = DataParallel(model)
        loss_fn = torch.nn.BCELoss(reduction='mean')
        
        
        test_preds_fold= np.zeros((len(df_test),1))
        print(f'Fold {fold + 1}')
        with torch.no_grad():
            model = create_model(modelname)
            model.load_state_dict(torch.load(f'{outputdir}/'+ modelname+ str(fold)+'_latest3.h5'))
            if torch.cuda.device_count() > 1:
                model = DataParallel(model)
            model.eval()

            test_dataset = DFData(test_inputs,None, augument=False,training=False)
            test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

            for i, x_batch in enumerate(test_loader):
                ids, atts, segs = x_batch[0].to(device),x_batch[1].to(device),x_batch[2].to(device)
                y_pred = model(input_ids = ids,position_ids=atts, token_type_ids =segs)
                y_pred = torch.nn.functional.sigmoid(y_pred)
                test_preds_fold[i * batch_size:(i+1) * batch_size] = y_pred.cpu().numpy()
            test_preds += test_preds_fold / 5
            del model, test_dataset, test_loader
            gc.collect()
        torch.cuda.empty_cache()
#         best_score_f+=best_f1
        
#         best_score_f+=best_f1
    return test_preds

   
# 训练成功一定要以0方式退出
test_preds = train(5)

df_test['label'] = test_preds

df_test[['id','label']].to_json(output_predictions_path,orient="records", lines=True)

    

   
