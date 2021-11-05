#!/usr/bin/env python
# -*- coding: utf-8 -*-

import gc
import json
import random
import sys
import jieba
import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.model_selection import KFold, StratifiedKFold
from torch.utils import data
from tqdm import tqdm
from transformers import BertTokenizer, BertModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

SEED = 2021
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

DEBUG = True

PATH = './'
BERT_PATH = './'
WEIGHT_PATH = './'
MAX_SEQUENCE_LENGTH = 86
input_categories = ['text']
output_categories = 'label'
modelname = 'bert-base'
pretraindir = './chinese_wwm_ext_pytorch'
modelsdir = './bert_wwm'

if DEBUG:
    input_data_path = '/mnt/atec/train.jsonl'
    output_predictions_path = '/home/ypm4cjjhbr/atec_project/work/predictions.jsonl'
else:
    input_data_path = '/home/admin/workspace/job/input/test.jsonl'
    output_predictions_path = '/home/admin/workspace/job/output/predictions.jsonl'


def _convert_to_transformer_inputs(question, tokenizer, max_sequence_length):
    """Converts tokenized input to ids, masks and segments for transformer (including bert)"""

    def return_id(str1, truncation_strategy, length):
        inputs = tokenizer.encode_plus(str1,
                                       add_special_tokens=True,
                                       max_length=length,
                                       truncation=True
                                       )

        input_ids = inputs["input_ids"]
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

        ids_q, masks_q, segments_q = \
            _convert_to_transformer_inputs(q, tokenizer, max_sequence_length)

        input_ids_q.append(ids_q)
        input_masks_q.append(masks_q)
        input_segments_q.append(segments_q)

    return [np.asarray(input_ids_q, dtype=np.int32),
            np.asarray(input_masks_q, dtype=np.int32),
            np.asarray(input_segments_q, dtype=np.int32)]


def compute_output_arrays(df, columns):
    return np.asarray(df[columns]).reshape(-1, 1)


class DFData(data.Dataset):
    def __init__(self, questions, labels, augument=False, training=True):
        super(DFData, self).__init__()
        self.augument = augument
        self.ids = questions[0]
        self.atts = questions[1]
        self.segs = questions[2]
        self.labels = labels
        self.len_ = len(self.ids)
        self.training = training

    def shuffle(self, d):
        return np.random.permutation(d.tolist())

    def dropout(self, d, p=0.5):
        len_ = len(d)
        index = np.random.choice(len_, int(len_ * p))
        d[index] = 0
        return d

    def __getitem__(self, index):
        if self.training:
            ids, att, seg, label = self.ids[index], self.atts[index], self.segs[index], self.labels[index]
        else:
            ids, att, seg = self.ids[index], self.atts[index], self.segs[index]

        if self.training and self.augument:
            ids = self.dropout(ids, p=0.05)
        ids = torch.tensor(ids).long()
        att = torch.tensor(att).long()
        seg = torch.tensor(seg).long()
        if self.training:
            label = torch.tensor(label, dtype=torch.float16)
            return (ids, att, seg), label
        else:
            return (ids, att, seg)

    def __len__(self):
        return self.len_


class BertClassificationHeadModel(nn.Module):
    def __init__(self, weights_key, clf_dropout=0.15, n_class=1):
        super(BertClassificationHeadModel, self).__init__()
        self.transformer = BertModel.from_pretrained(weights_key, output_hidden_states=False, torchscript=True)
        self.dropout = nn.Dropout(clf_dropout)
        self.linear = nn.Linear(self.transformer.config.hidden_size, n_class)
        nn.init.xavier_uniform_(self.linear.weight)
        self.linear.bias.data.fill_(0.0)

    def forward(self, input_ids, position_ids=None, token_type_ids=None):
        hidden_states, h_conc = self.transformer(input_ids=input_ids, attention_mask=position_ids,
                                                 token_type_ids=token_type_ids)

        logits = self.linear(self.dropout(h_conc))
        return logits


def create_model(modelname):
    model = BertClassificationHeadModel(f'{pretraindir}', n_class=1)
    model = model.to(device)
    return model


def split_words(x):
    return ' '.join([i for i in jieba.cut(x)])


def lgb_f1_score(y_hat, data):
    y_true = data.get_label()
    y_hat = np.round(y_hat)  # scikits f1 doesn't like probabilities
    return 'f1', f1_score(y_true, y_hat), True


def lgb_model(train, target, test, k):
    feats = [f for f in train.columns if f not in ['request_user', 'PREV_request_label_MAX', ]]
    print('Current num of features:', len(feats))
    #     feats=import_cols
    oof_probs = np.zeros(train.shape[0])
    output_preds = 0
    offline_score = []
    feature_importance_df = pd.DataFrame()
    parameters = {
        'learning_rate': 0.03,
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': 'auc',
        'num_leaves': 31,
        'feature_fraction': 0.6,
        'bagging_fraction': 0.8,
        'min_data_in_leaf': 15,
        'verbose': -1,
        'nthread': 32,
        'max_depth': 7
    }

    seeds = [2020]
    for seed in seeds:
        folds = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
        for i, (train_index, test_index) in enumerate(folds.split(train, target)):
            train_y, test_y = target.iloc[train_index], target.iloc[test_index]
            train_X, test_X = train[feats].iloc[train_index, :], train[feats].iloc[test_index, :]

            dtrain = lgb.Dataset(train_X,
                                 label=train_y)
            dval = lgb.Dataset(test_X,
                               label=test_y)
            lgb_model = lgb.train(
                parameters,
                dtrain,
                num_boost_round=8000,
                valid_sets=[dval],
                # feval=lgb_f1_score,
                early_stopping_rounds=100,
                verbose_eval=100,
            )
            oof_probs[test_index] = lgb_model.predict(test_X[feats], num_iteration=lgb_model.best_iteration) / len(
                seeds)
            offline_score.append(lgb_model.best_score['valid_0']['auc'])
            output_preds += lgb_model.predict(test[feats],
                                              num_iteration=lgb_model.best_iteration) / folds.n_splits / len(seeds)
            print(offline_score)
            # feature importance
            fold_importance_df = pd.DataFrame()
            fold_importance_df["feature"] = feats
            fold_importance_df["importance"] = lgb_model.feature_importance(importance_type='gain')
            fold_importance_df["fold"] = i + 1
            feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
    print('OOF-MEAN-AUC:%.6f, OOF-STD-AUC:%.6f' % (np.mean(offline_score), np.std(offline_score)))
    print('feature importance:')
    print(feature_importance_df.groupby(['feature'])['importance'].mean().sort_values(ascending=False).head(50))

    return output_preds, oof_probs, np.mean(offline_score), feature_importance_df


def rank():
    ######################## reading data ########################
    # read the train data
    train = pd.read_csv('./train_data/train.csv')

    if DEBUG:
        # using label == -1 rows for debug
        # test = pd.read_csv('/home/ypm4cjjhbr/atec_project/work/unknown.csv')
        test = pd.read_csv('./train_data/train.csv')
        test = test[:1000]
        test.drop(['label'], axis=1, inplace=True)
    else:
        # real test data in inference
        with open(input_data_path, 'r', encoding='utf-8') as fp:
            data = fp.readlines()

        test_data = list()
        for line in data:
            js_data = json.loads(line)
            di = dict()
            di['id'] = js_data['id']
            for i in range(480):
                di[f'x{i}'] = js_data[f'x{i}']
            di['memo_polish'] = js_data['memo_polish']
            test_data.append(di)
        test = pd.DataFrame(test_data)

    # test texts
    texts = []
    ids = []
    for _, row in test.iterrows():
        texts.append(
            str(row['x269']) + ' ' + str(row['x321']) + ' ' + str(row['x479']) + ' ' + str(row['x459']) + ' ' + str(
                row['x117']) + ' ' + str(row['x30']) + ' ' + str(row['memo_polish']))
        ids.append(row['id'])

    df_test = pd.DataFrame({'id': ids, 'text': texts})
    df_test['text'] = df_test['text'].astype(str)
    df_test.index = range(len(df_test))

    tokenizer = BertTokenizer.from_pretrained(f"{pretraindir}/vocab.txt")
    test_inputs = compute_input_arrays(df_test, input_categories, tokenizer, MAX_SEQUENCE_LENGTH)

    # bert predict
    test_preds = np.zeros((len(df_test), 1))
    batch_size = 32
    for fold in range(5):
        test_preds_fold = np.zeros((len(df_test), 1))
        print(f'Fold {fold + 1}')
        with torch.no_grad():
            model = create_model(modelname)
            model.load_state_dict(
                torch.load(f'{modelsdir}/' + modelname + str(fold) + '_latest3.h5', map_location=device))
            model.eval()
            test_dataset = DFData(test_inputs, None, augument=False, training=False)
            test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
            for i, x_batch in enumerate(test_loader):
                ids, atts, segs = x_batch[0].to(device), x_batch[1].to(device), x_batch[2].to(device)
                y_pred = model(input_ids=ids, position_ids=atts, token_type_ids=segs)
                y_pred = torch.nn.functional.sigmoid(y_pred)
                test_preds_fold[i * batch_size:(i + 1) * batch_size] = y_pred.cpu().numpy()
            test_preds += test_preds_fold / 5
            del model, test_dataset, test_loader
        gc.collect()
        torch.cuda.empty_cache()

    oof = np.load('./train_data/bert_oof.npy')
    train['bert_pred'] = oof
    test['bert_pred'] = test_preds

    # concat train and test
    df_features = pd.concat([train, test])
    del train, test
    gc.collect()

    ######################## feature engineering  ########################
    # text stats
    df_features['memo_polish'].fillna('', inplace=True)
    df_features['memo_char_len'] = df_features['memo_polish'].apply(len)
    df_features['memo_words'] = df_features['memo_polish'].apply(lambda x: split_words(x))
    df_features['memo_word_len'] = df_features['memo_words'].apply(lambda x: len(x.split(' ')))

    # TFIDF + SVD
    n_components = 4
    text = list(df_features['memo_words'])
    tf = TfidfVectorizer(min_df=2,
                         token_pattern=r"(?u)\b\w+\b",
                         ngram_range=(1, 2),
                         max_features=10000)
    X = tf.fit_transform(text)
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    X_svd = svd.fit_transform(X)
    df_tfidf = pd.DataFrame(X_svd)
    df_tfidf.columns = [f'text_tfidf_{i}' for i in range(n_components)]
    for col in df_tfidf.columns:
        df_features[col] = df_tfidf[col].values
    del text

    # kfold label encoding
    def stat(df, df_merge, group_by, agg):
        group = df.groupby(group_by).agg(agg)
        columns = []
        for on, methods in agg.items():
            for method in methods:
                columns.append('{}_{}_{}'.format('_'.join(group_by), on, method))
        group.columns = columns
        group.reset_index(inplace=True)
        df_merge = df_merge.merge(group, on=group_by, how='left')
        del (group)
        gc.collect()
        return df_merge

    def statis_feat(df_know, df_unknow):
        df_unknow = stat(df_know, df_unknow, ['x269'], {'label': ['mean', 'std']})
        df_unknow = stat(df_know, df_unknow, ['x469'], {'label': ['mean', 'std']})
        return df_unknow

    df_train = df_features[~df_features['label'].isnull()]
    df_train = df_train.reset_index(drop=True)
    df_test = df_features[df_features['label'].isnull()]

    df_stas_feat = None
    kf = KFold(n_splits=5, random_state=2021, shuffle=True)
    for train_index, val_index in kf.split(df_train):
        df_fold_train = df_train.iloc[train_index]
        df_fold_val = df_train.iloc[val_index]
        df_fold_val = statis_feat(df_fold_train, df_fold_val)
        df_stas_feat = pd.concat([df_stas_feat, df_fold_val], axis=0)
        del (df_fold_train)
        del (df_fold_val)
        gc.collect()

    df_test = statis_feat(df_train, df_test)
    df_features = pd.concat([df_stas_feat, df_test], axis=0)

    del (df_stas_feat)
    del (df_train)
    del (df_test)
    gc.collect()

    # split train / test
    df_features.drop(['memo_words', 'memo_polish'], axis=1, inplace=True)
    train = df_features[df_features['label'].notna()].copy()
    test = df_features[df_features['label'].isna()].copy()
    del df_features
    gc.collect()

    ######################## train and predict ########################
    ycol = 'label'
    useless_cols = ['x2', 'x55', 'x91', 'x96', 'x107', 'x184',
                    'x198', 'x201', 'x204', 'x207', 'x209',
                    'x261', 'x277', 'x319', 'x351', 'x456']
    prediction = test[['id']]
    prediction[ycol] = 0
    feature_names = list(
        filter(lambda x: x not in [ycol, 'id'] + useless_cols, test.columns))
    feature_names = sorted(feature_names)

    lgb_preds, lgb_oof, lgb_score, feature_importance_df = lgb_model(train=train[feature_names],
                                                                     target=train[ycol],
                                                                     test=test[feature_names], k=7)
    prediction[ycol] = lgb_preds

    if DEBUG:
        # df_oof = pd.concat(oof)
        print('roc_auc_score:', roc_auc_score(train[ycol], lgb_oof))
        print(feature_importance_df.head(20))
        print(prediction['label'].describe())
        print(prediction[prediction['label'] > 0.9].shape)

    prediction.to_json(output_predictions_path, orient='records', lines=True)

    return True


if __name__ == '__main__':
    if rank():
        sys.exit(0)
    else:
        sys.exit(1)
