import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold,KFold
from tqdm import tqdm
import json
# import tensorflow as tf
# import tensorflow.keras.backend as K
# from tensorflow.keras.utils import to_categorical
# from tensorflow.keras.utils.np_utils import to_categorical
# from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import os
# from transformers import *
# print(tf.__version__)
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
from nezha import *
from transformers import BertTokenizer
import argparse
parser = argparse.ArgumentParser(description="Help info.")
parser.add_argument('--input', type=str, help='input path of the dataset directory.')
# parser.add_argument('--output', type=str, help='output path of the prediction file.')

# #If you need models from the server:
# huggingface = '/work/mayixiao/CAIL2021/root/big/huggingface'
# tokenizer = AutoTokenizer.from_pretrained(os.path.join(args.huggingface, "thunlp/Lawformer"))
# model = AutoModelForMaskedLM.from_pretrained(os.path.join(args.huggingface, "thunlp/Lawformer"))

args = parser.parse_args()
input_path = args.input
# output_path = args.output


# df_train = pd.read_csv('data/train.tsv',sep='\t',header=None)
# df_train.columns=['label','q1','q2']

df_test =  pd.read_csv(input_path,sep='\t',header=None)
df_test.columns=['q1','q2']
df_test['q1'] = df_test['q1'].astype(str)
df_test['q2'] = df_test['q2'].astype(str)

test_id = np.arange(0,len(df_test),1)
df_test['id'] =test_id

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
MAX_SEQUENCE_LENGTH = 48
input_categories = ['q1','q2']
output_categories = 'label'
modelname='nezha-large'
pretraindir = 'user_data/nezha-large'
outputdir='predict'

print('test shape =', df_test.shape)
try:
    os.mkdir(outputdir)
except:
    pass 
def _convert_to_transformer_inputs(question, answer, tokenizer, max_sequence_length):
    """Converts tokenized input to ids, masks and segments for transformer (including bert)"""
    
    def return_id(str1, str2, truncation_strategy, length):

        inputs = tokenizer.encode_plus(str1, str2,
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
        question, answer, 'longest_first', max_sequence_length)
    

    
    return [input_ids_q, input_masks_q, input_segments_q]

def compute_input_arrays(df, columns, tokenizer, max_sequence_length):
    input_ids_q, input_masks_q, input_segments_q = [], [], []
    input_ids_a, input_masks_a, input_segments_a = [], [], []
    for _, instance in tqdm(df[columns].iterrows()):
        q, a = instance.q1, instance.q2

        ids_q, masks_q, segments_q= \
        _convert_to_transformer_inputs(q, a, tokenizer, max_sequence_length)
        
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
    for i in range(50,51):
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

test_inputs = compute_input_arrays(df_test, input_categories, tokenizer, MAX_SEQUENCE_LENGTH)
def compute_output_arrays(df, columns):
    return np.asarray(df[columns]).reshape(-1,1)

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

class BertClassificationHeadModel(nn.Module):
    def __init__(self, weights_key, clf_dropout=0.15, n_class=1):
        super(BertClassificationHeadModel, self).__init__()
        self.transformer = NeZhaModel.from_pretrained(weights_key, output_hidden_states=False, torchscript=True)
        self.dropout = nn.Dropout(clf_dropout)
        self.linear = nn.Linear(self.transformer.config.hidden_size, n_class)
        nn.init.xavier_uniform_(self.linear.weight)
        self.linear.bias.data.fill_(0.0)


    def forward(self, input_ids, position_ids=None, token_type_ids=None):
        hidden_states,h_conc = self.transformer(input_ids = input_ids, attention_mask = position_ids, token_type_ids = token_type_ids)

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
        model=BertClassificationHeadModel(f'{pretraindir}', n_class=1)
    elif modelname == 'nezha-base':
        model=BertClassificationHeadModel(f'{pretraindir}', n_class=1)
    elif modelname == 'nezha-base-wwm':
        model=RobertaClassificationModel(f'{pretraindir}', n_class=1)
    elif modelname == 'nezha-large-wwm':
        model=RobertaClassificationModel(f'{pretraindir}', n_class=1)
    elif modelname == 'macbert_large':
        model =  RobertaClassificationModel(f'{pretraindir}', n_class=1)
                      
        
    model = model.cuda()
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
    base_lr, max_lr = 1e-5, 5e-5 
    best_score_f = 0
#     kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    for fold in range(5):
#         train_inputs = [inputs[i][train_idx] for i in range(len(inputs))]
#         train_outputs = outputs[train_idx]
#         valid_inputs = [inputs[i][valid_idx] for i in range(len(inputs))]
#         valid_outputs = outputs[valid_idx]
#         train_df_fold = df_train.iloc[train_idx]

#         train_dataset = DFData(train_inputs , train_outputs
#                                              , augument=False,training=True
#                               )
#         valid_dataset= DFData(valid_inputs,valid_outputs,augument=False,training=True)
#         train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
#         valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
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
        
#         valid_preds_fold= np.zeros((len(valid_idx),1))
        test_preds_fold= np.zeros((len(df_test),1))
        print(f'Fold {fold + 1}')
#         best_f1 = 0
#         gradient_accumulation_steps =1
#         for epoch in range(n_epochs):
#             start_time = time.time()
#             model.train(True)
#             lr_scheduler.step(best_f1)
#             avg_loss = 0 
#             nb_tr_steps = 0
#             steps_per_epoch = int(len(train_loader))
#             with tqdm(total=int(steps_per_epoch), desc='Epoch %d' % (epoch + 1)) as pbar:
#                 for ii,(x_batch,y_batch) in enumerate(train_loader):
#                     ids, atts, segs = x_batch[0].cuda(),x_batch[1].cuda(),x_batch[2].cuda()
#                     y_batch=Variable(y_batch).cuda()
#                     y_pred = model(input_ids = ids,position_ids=atts,token_type_ids =segs)

#                     y_pred = torch.nn.functional.sigmoid(y_pred)
#                     #print(y_pred)
#                     loss = loss_fn(y_pred, y_batch.float())
#                     loss.backward()
#                     avg_loss += loss
#                     fgm.attack()
#                     y_pred = model(input_ids = ids,position_ids=atts,token_type_ids =segs)
#                     y_pred = torch.nn.functional.sigmoid(y_pred)
#                     loss_adv = loss_fn(y_pred, y_batch.float())
#                     loss_adv.backward() 
#                     fgm.restore() 

#                     optimizer.step()
#                     optimizer.zero_grad()
#                     avg_loss += loss.item() / len(train_loader)
#                     if (ii + 1) % gradient_accumulation_steps == 0:
#                             nb_tr_steps += 1
#                             pbar.set_postfix({'loss': '{0:1.5f}'.format(avg_loss / (nb_tr_steps))})
#                             pbar.update(1)
                
#             model.train(False)
#             model.eval()   
#             avg_val_loss = 0.
#             with torch.no_grad():                
#                 for i, (x_batch, y_batch) in enumerate(valid_loader):
#                     ids, atts, segs = x_batch[0].cuda(),x_batch[1].cuda(),x_batch[2].cuda()
#                     y_batch=Variable(y_batch).cuda()
#                     y_pred = model(input_ids = ids,position_ids=atts,token_type_ids =segs)
#                     y_pred = torch.nn.functional.sigmoid(y_pred)
#                     #print(y_pred)
#                     avg_val_loss += loss_fn(y_pred, y_batch.float()).item() / len(valid_loader)
#                     valid_preds_fold[i * batch_size:(i+1) * batch_size] = y_pred.cpu().numpy()


#                 elapsed_time = time.time() - start_time
#                 assert len(valid_preds_fold) == len(outputs[valid_idx])
#                 #print(valid_preds_fold)
# #                 f1 = search_f1(outputs[valid_idx], valid_preds_fold)
#                 f1,t = search_f1(outputs[valid_idx],valid_preds_fold)
#                 acc = t
#                 if f1 > best_f1:
                    
#                     train_preds[valid_idx] = valid_preds_fold 
#                     best_f1 = f1
#                     if isinstance(model, DataParallel):
#                         torch.save(model.module.state_dict(),f'{outputdir}/'+ modelname+ str(fold)+'_latest3.h5')
#                     else:
#                         torch.save(model.state_dict(), f'{outputdir}/'+ modelname+ str(fold)+'_latest3.h5')
#                 print('Epoch {}/{} \t loss={:.4f} \t val_loss={:.4f} \t time={:.2f}s \t auc={:.4f} \t '.format(
#                     epoch + 1, n_epochs, avg_loss, avg_val_loss,elapsed_time,f1
#                 ))
#             lr_scheduler.step(avg_val_loss)
            
#         del model,train_dataset,valid_dataset, train_loader, valid_loader,train_outputs,valid_outputs,param_optimizer,optimizer_grouped_parameters
        gc.collect()
        torch.cuda.empty_cache()
        with torch.no_grad():
            model = create_model(modelname)
            model.load_state_dict(torch.load(f'{outputdir}/'+ modelname+ str(fold)+'_latest3.h5'))
            if torch.cuda.device_count() > 1:
                model = DataParallel(model)
            model.eval()

            test_dataset = DFData(test_inputs,None, augument=False,training=False)
            test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

            for i, x_batch in enumerate(test_loader):
                ids, atts, segs = x_batch[0].cuda(),x_batch[1].cuda(),x_batch[2].cuda()
                y_pred = model(input_ids = ids,position_ids=atts, token_type_ids =segs)
                y_pred = torch.nn.functional.sigmoid(y_pred)
                test_preds_fold[i * batch_size:(i+1) * batch_size] = y_pred.cpu().numpy()
            test_preds += test_preds_fold / 5
            del model, test_dataset, test_loader
            gc.collect()
        torch.cuda.empty_cache()
#         best_score_f+=best_f1
    return test_preds

test_preds = train(5)
# search_f1(outputs,train_preds)

torch.cuda.empty_cache()
import gc
gc.collect()

# best_score=0.97244
# best_score_f = search_f1(outputs, train_preds)
# best_score=0.97244
# best_score_f = search_f1(outputs, train_preds)
# best_score, best_t = search_f1(outputs,train_preds)
# print(best_score)
# print(best_t)
best_score = 0.897
# np.save(f'{outputdir}/oof_id_{modelname}_{best_score}.npy',train_preds)
np.save(f'{outputdir}/sub_id_{modelname}_{best_score}.npy',test_preds)
sub = test_preds#np.average(test_preds, axis=0) 
sub = sub > 0.9
df_test['label'] = sub.astype(int)

new_result = []

for i in range(len(df_test)):
    label = df_test['label'][i]
    if label==1:        
        new_result.append({ "label": 1})
    else:
        new_result.append({ "label": 0})
new_result


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)
with open(f'{outputdir}/result_{best_score}.json', 'w', encoding='utf-8') as fo:      
    for d in new_result: 
        fo.write(json.dumps(d,ensure_ascii=False, cls=NpEncoder))
        fo.write('\n')
fo.close()     
