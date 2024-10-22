import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
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
from sklearn.metrics import f1_score, accuracy_score,roc_auc_score,auc,precision_score,recall_score
#from apex import amp
from torch.optim.lr_scheduler import ReduceLROnPlateau
# from nezha import *
from transformers import *
from transformers import BertTokenizer
import argparse

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

input_data_path = '/home/admin/workspace/job/input/train.jsonl'
output_model_path = '/home/admin/workspace/job/output/your-model-name'
result_path = '/home/admin/workspace/job/output/result.json'

with open(input_data_path, 'r', encoding='utf-8') as fp:
    data = fp.readlines()
texts = []
ids=[]
labels=[]
for d in data:
    tmp = json.loads(d)
    texts.append(str(tmp['x269']) +' '+str(tmp['x321']) +' '+str(tmp['x479']) +' '+str(tmp['x459']) +' '+str(tmp['x117'])+' '+str(tmp['x30'])+' '+str(tmp['x379'])+' '+ str(tmp['memo_polish']))
    ids.append(tmp['id'])
    labels.append(tmp['label'])
df = pd.DataFrame(columns=['id','text','label'])
df['label'] = labels
df['text']=texts
df['id'] = ids

df_train = df
df_train = df_train[df_train.label!=-1]
df_train['text'] = df_train['text'].astype(str)
df_train.index=range(len(df_train))

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
MAX_SEQUENCE_LENGTH = 86
input_categories = ['text']
output_categories = 'label'
modelname='bert-base'
pretraindir = './bert_wwm_pretrain'
outputdir='/home/admin/workspace/job/output/bert_wwm'

print('train shape =', df_train.shape)
# print('test shape =', df_test.shape)
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
    re_score90 = 0
    re_score85 = 0
    re_score80 = 0
    pre_score90 = 0
    pre_score85= 0
    pre_score80 = 0
    for i in range(1,100):
        tres = i / 100
        y_pred_bin =  (y_pred > tres).astype(int)
        pre_score  = precision_score(y_true, y_pred_bin)
        if pre_score >=0.9:
            re_score = recall_score(y_true, y_pred_bin)
            if re_score > re_score90:
                re_score90 = re_score 
        if pre_score >=0.85:
            re_score = recall_score(y_true, y_pred_bin)
            if re_score > re_score85:
                re_score85 = re_score 
        if pre_score >=0.80:
            re_score = recall_score(y_true, y_pred_bin)
            if re_score > re_score80:
                re_score80 = re_score             
        score = f1_score(y_true, y_pred_bin)
        best = re_score90 * 0.4 + re_score85 *0.3 +re_score80*0.3
    print('best', best)
    print('thres', best_t)
    
    return best, best_t
def search_f11(y_true, y_pred):
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

inputs = compute_input_arrays(df_train, input_categories, tokenizer, MAX_SEQUENCE_LENGTH)
# test_inputs = compute_input_arrays(df_test, input_categories, tokenizer, MAX_SEQUENCE_LENGTH)
def compute_output_arrays(df, columns):
    return np.asarray(df[columns]).reshape(-1,1)


outputs = compute_output_arrays(df_train, output_categories)

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
    train_preds = np.zeros((len(df_train),1))
#     test_preds = np.zeros((len(df_test),1))
    batch_size = 32
    step_size = 300
    base_lr, max_lr = 2e-5, 5e-5 
    best_score_f = 0
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    for fold, (train_idx, valid_idx) in enumerate(kfold.split(df_train, outputs)):
        train_inputs = [inputs[i][train_idx] for i in range(len(inputs))]
        train_outputs = outputs[train_idx]
        valid_inputs = [inputs[i][valid_idx] for i in range(len(inputs))]
        valid_outputs = outputs[valid_idx]
        train_df_fold = df_train.iloc[train_idx]

        train_dataset = DFData(train_inputs , train_outputs
                                             , augument=False,training=True
                              )
        valid_dataset= DFData(valid_inputs,valid_outputs,augument=False,training=True)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
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
        
        valid_preds_fold= np.zeros((len(valid_idx),1))
#         test_preds_fold= np.zeros((len(df_test),1))
        print(f'Fold {fold + 1}')
        best_f1 = 0
        gradient_accumulation_steps =1
        for epoch in range(n_epochs):
            start_time = time.time()
            model.train(True)
            lr_scheduler.step(best_f1)
            avg_loss = 0 
            nb_tr_steps = 0
            steps_per_epoch = int(len(train_loader))
            with tqdm(total=int(steps_per_epoch), desc='Epoch %d' % (epoch + 1)) as pbar:
                for ii,(x_batch,y_batch) in enumerate(train_loader):
                    ids, atts, segs = x_batch[0].to(device),x_batch[1].to(device),x_batch[2].to(device)
                    y_batch=Variable(y_batch).to(device)
                    y_pred = model(input_ids = ids,position_ids=atts,token_type_ids =segs)

                    y_pred = torch.nn.functional.sigmoid(y_pred)
                    #print(y_pred)
                    loss = loss_fn(y_pred, y_batch.float())
                    loss.backward()
                    avg_loss += loss
                    fgm.attack()
                    y_pred = model(input_ids = ids,position_ids=atts,token_type_ids =segs)
                    y_pred = torch.nn.functional.sigmoid(y_pred)
                    loss_adv = loss_fn(y_pred, y_batch.float())
                    loss_adv.backward() 
                    fgm.restore() 

                    optimizer.step()
                    optimizer.zero_grad()
                    avg_loss += loss.item() / len(train_loader)
                    if (ii + 1) % gradient_accumulation_steps == 0:
                            nb_tr_steps += 1
                            pbar.set_postfix({'loss': '{0:1.5f}'.format(avg_loss / (nb_tr_steps))})
                            pbar.update(1)
                
            model.train(False)
            model.eval()   
            avg_val_loss = 0.
            with torch.no_grad():                
                for i, (x_batch, y_batch) in enumerate(valid_loader):
                    ids, atts, segs = x_batch[0].to(device),x_batch[1].to(device),x_batch[2].to(device)
                    y_batch=Variable(y_batch).to(device)
                    y_pred = model(input_ids = ids,position_ids=atts,token_type_ids =segs)
                    y_pred = torch.nn.functional.sigmoid(y_pred)
                    #print(y_pred)
                    avg_val_loss += loss_fn(y_pred, y_batch.float()).item() / len(valid_loader)
                    valid_preds_fold[i * batch_size:(i+1) * batch_size] = y_pred.cpu().numpy()


                elapsed_time = time.time() - start_time
                assert len(valid_preds_fold) == len(outputs[valid_idx])

                f1,t = search_f1(outputs[valid_idx],valid_preds_fold)
                acc = t
                if f1 > best_f1:
                    
                    train_preds[valid_idx] = valid_preds_fold 
                    best_f1 = f1
                    if isinstance(model, DataParallel):
                        torch.save(model.module.state_dict(),f'{outputdir}/'+ modelname+ str(fold)+'_latest3.h5')
                    else:
                        torch.save(model.state_dict(), f'{outputdir}/'+ modelname+ str(fold)+'_latest3.h5')
                print('Epoch {}/{} \t loss={:.4f} \t val_loss={:.4f} \t time={:.2f}s \t auc={:.4f} \t '.format(
                    epoch + 1, n_epochs, avg_loss, avg_val_loss,elapsed_time,f1
                ))
            lr_scheduler.step(avg_val_loss)
            
        del model,train_dataset,valid_dataset, train_loader, valid_loader,train_outputs,valid_outputs,param_optimizer,optimizer_grouped_parameters
        gc.collect()
        torch.cuda.empty_cache()
        
        best_score_f+=best_f1
    return train_preds,best_score_f/5

train_preds,best_score = train(3)
# search_f1(outputs,train_preds)

torch.cuda.empty_cache()
import gc
gc.collect()

# best_score=0.97244
# best_score_f = search_f1(outputs, train_preds)
# best_score=0.97244
# best_score_f = search_f1(outputs, train_preds)
best_score, best_t = search_f1(outputs,train_preds)
print(best_score)
print(best_t)

np.save(f'{outputdir}/oof_id_{modelname}_{best_score}.npy',train_preds)
np.save(f'{outputdir}/sub_id_{modelname}_{best_score}.npy',test_preds)
# sub = test_preds#np.average(test_preds, axis=0) 
# # sub = sub > 0.9
# # df_test['label'] = sub.astype(int)

# new_result = []

# for i in range(len(df_test)):
#     label = df_test['label'][i]
#     if label==1:        
#         new_result.append({ "label": 1})
#     else:
#         new_result.append({ "label": 0})
# new_result


# class NpEncoder(json.JSONEncoder):
#     def default(self, obj):
#         if isinstance(obj, np.integer):
#             return int(obj)
#         elif isinstance(obj, np.floating):
#             return float(obj)
#         elif isinstance(obj, np.ndarray):
#             return obj.tolist()
#         else:
#             return super(NpEncoder, self).default(obj)
# with open(f'{outputdir}/result_{best_score}.json', 'w', encoding='utf-8') as fo:      
#     for d in new_result: 
#         fo.write(json.dumps(d,ensure_ascii=False, cls=NpEncoder))
#         fo.write('\n')
# fo.close()     
