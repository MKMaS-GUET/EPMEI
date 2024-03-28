# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import absolute_import, division, print_function
""" Finetuning the library models for sequence classification on GLUE (Bert, XLM, XLNet, RoBERTa)."""
'''
由于数据集中，没有关于NONE的数据，所以在最后的结果中，对于NONE的分类一定是很差的，可以考虑数据增强(为每个样本创建1-2个错误三元组)，来增加NONE数据的数量
'''

import nltk
from nltk.tokenize import word_tokenize
import argparse
import glob
import logging
import os
import random
from collections import defaultdict
import re
import shutil
import wandb
import numpy as np
from copy import copy, deepcopy
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tensorboardX import SummaryWriter
from tqdm import tqdm, trange
import time
import linecache
from collections import Counter, defaultdict
from transformers import (WEIGHTS_NAME, BertConfig,
                                  BertTokenizer,
                                  RobertaConfig,
                                  RobertaTokenizer,
                                  get_linear_schedule_with_warmup,
                                  AdamW,
                                  BertForACEBothOneDropoutSub,
                                  AlbertForACEBothSub,
                                  AlbertConfig,
                                  AlbertTokenizer,
                                  AlbertForACEBothOneDropoutSub,
                                  BertForACEBothOneDropoutSubNoNer,
                                  )
import spacy
from transformers import AutoTokenizer
from torch.utils.data import TensorDataset, Dataset
import json
import pickle
import numpy as np
import unicodedata
import itertools
import timeit
from allennlp.modules.token_embedders.embedding import Embedding
from tqdm import tqdm

logger = logging.getLogger(__name__)
train_data = None
dev_data = None
test_data = None
ALL_MODELS = sum((tuple(conf.pretrained_config_archive_map.keys()) for conf in (BertConfig,  AlbertConfig)), ())

MODEL_CLASSES = {
    'bertsub': (BertConfig, BertForACEBothOneDropoutSub, BertTokenizer),
    'bertnonersub': (BertConfig, BertForACEBothOneDropoutSubNoNer, BertTokenizer),
    'albertsub': (AlbertConfig, AlbertForACEBothOneDropoutSub, AlbertTokenizer),
}

task_ner_labels = {
    'ace04': ['FAC', 'WEA', 'LOC', 'VEH', 'GPE', 'ORG', 'PER'],
    'ace05': ['FAC', 'WEA', 'LOC', 'VEH', 'GPE', 'ORG', 'PER'],
    'data': ['opinion','target'],
    'scierc_1_sim': ['opinion','target'],
}

task_rel_labels = {
    'ace04': ['PER-SOC', 'OTHER-AFF', 'ART', 'GPE-AFF', 'EMP-ORG', 'PHYS'],
    'ace05': ['PER-SOC', 'ART', 'ORG-AFF', 'GEN-AFF', 'PHYS', 'PART-WHOLE'],
    'data': ['POSITIVE','NEGATIVE','NEUTRAL'],
    'scierc_1_sim': ['POSITIVE','NEGATIVE','NEUTRAL'],
}
BERT_TOKEN_MAPPING = {
    "-LRB-": "(",
    "-RRB-": ")",
    "-LCB-": "{",
    "-RCB-": "}",
    "-LSB-": "[",
    "-RSB-": "]",
    "``": '"',
    "''": '"',
    "`": "'",
    '«': '"',
    '»': '"',
    '‘': "'",
    '’': "'",
    '“': '"',
    '”': '"',
    '„': '"',
    '‹': "'",
    '›': "'",
    "\u2013": "--", # en dash
    "\u2014": "--", # em dash
    }

REVERSE_TOKEN_MAPPING = dict([(value, key) for key, value in BERT_TOKEN_MAPPING.items()])
class ACEDataset(Dataset):
    def __init__(self, tokenizer,dataset_list,args=None, evaluate=False, do_test=False, max_pair_length=None,dataset=None):

        if not evaluate:
            file_path = os.path.join(args.data_dir, args.train_file)
        else:
            if do_test:
                if args.test_file.find('models')==-1:
                    file_path = os.path.join(args.data_dir, args.test_file)
                else:
                    file_path = args.test_file
            else:
                if args.dev_file.find('models')==-1:
                    file_path = os.path.join(args.data_dir, args.dev_file)
                else:
                    file_path = args.dev_file

        # assert os.path.isfile(file_path)

        self.file_path = file_path

        self.tokenizer = tokenizer
        self.max_seq_length = args.max_seq_length
        self.max_pair_length = max_pair_length
        self.max_entity_length = self.max_pair_length*2

        self.evaluate = evaluate
        self.use_typemarker = args.use_typemarker
        self.local_rank = args.local_rank
        self.args = args
        self.model_type = args.model_type
        self.no_sym = args.no_sym
        self.cont_none = 0
        self.cont_pos = 0
        self.cont_neu = 0
        self.cont_neg = 0
        if args.data_dir.find('ace05')!=-1:
            self.ner_label_list = ['NIL', 'FAC', 'WEA', 'LOC', 'VEH', 'GPE', 'ORG', 'PER']

            if args.no_sym:
                label_list = ['PER-SOC', 'ART', 'ORG-AFF', 'GEN-AFF', 'PHYS', 'PART-WHOLE']
                self.sym_labels = ['NIL']
                self.label_list = self.sym_labels + label_list
            else:
                label_list = ['ART', 'ORG-AFF', 'GEN-AFF', 'PHYS',  'PART-WHOLE']
                self.sym_labels = ['NIL', 'PER-SOC']
                self.label_list = self.sym_labels + label_list

        elif args.data_dir.find('ace04')!=-1:
            self.ner_label_list = ['NIL', 'FAC', 'WEA', 'LOC', 'VEH', 'GPE', 'ORG', 'PER']

            if args.no_sym:
                label_list = ['PER-SOC', 'OTHER-AFF', 'ART', 'GPE-AFF', 'EMP-ORG', 'PHYS']
                self.sym_labels = ['NIL']
                self.label_list = self.sym_labels + label_list
            else:
                label_list = ['OTHER-AFF', 'ART', 'GPE-AFF', 'EMP-ORG', 'PHYS']
                self.sym_labels = ['NIL', 'PER-SOC']
                self.label_list = self.sym_labels + label_list

        elif args.data_dir.find('data')!=-1:
            self.ner_label_list = ['opinion','target']

            if args.no_sym:
                label_list = ['POSITIVE','NEGATIVE','NEUTRAL']
                self.sym_labels = ['NIL']
                self.label_list = self.sym_labels + label_list
            else:
                label_list = ['POSITIVE','NEGATIVE','NEUTRAL']
                self.sym_labels = ['NIL']
                self.label_list = self.sym_labels + label_list
        elif args.data_dir.find('scierc')!=-1:
            self.ner_label_list = ['opinion','target']

            if args.no_sym:
                label_list = ['POSITIVE','NEGATIVE','NEUTRAL']
                self.sym_labels = ['NIL']
                self.label_list = self.sym_labels + label_list
            else:
                label_list = ['POSITIVE','NEGATIVE','NEUTRAL']
                self.sym_labels = ['NIL']
                self.label_list = self.sym_labels + label_list

        else:
            assert (False)
        self._span_width_embedding = Embedding(
            num_embeddings=args.num_width_embeddings, embedding_dim=args.span_width_embedding_dim
        )
        self.global_predicted_ners = {}
        self.initialize(dataset_list,dataset,do_test)

    def initialize(self,dataset_list,dataset,do_test):
        tokenizer = self.tokenizer
        vocab_size = tokenizer.vocab_size
        max_num_subwords = self.max_seq_length - 4  # for two marker
        label_map = {label: i for i, label in enumerate(self.label_list)}
        ner_label_map = {label: i for i, label in enumerate(self.ner_label_list)}

        def tokenize_word(text):
            if (
                isinstance(tokenizer, RobertaTokenizer)
                and (text[0] != "'")
                and (len(text) != 1 or not self.is_punctuation(text))
            ):
                return tokenizer.tokenize(text, add_prefix_space=True)
            return tokenizer.tokenize(text)


        # with open(self.file_path, "r", encoding='utf-8') as f:
        #     DATA = json.load(f)


        self.ner_tot_recall = 0
        self.tot_recall = 0
        self.data = []
        self.ner_golden_labels = set([])
        self.golden_labels = set([])
        self.golden_labels_withner = set([])
        self.aope_golden = set([])
        maxR = 0
        maxL = 0
        # with open(self.file_path, 'r', encoding='utf-8') as f:
        #     DATAS = json.load(f)
        all_unrolled = []
        id_vis = []

        all_temp_tokens = []
        for l_idx, line in enumerate(dataset_list):
            data = line
            # if self.args.output_dir.find('test')!=-1:
            #     if len(self.data) > 70:
            #         break
            sen_word = data['token']['sentences']

            sentences = [data['token']['sentences']]
            if 'predicted_ner' in data:       # e2e predict
                try:
                    ners = data['predicted_ner']
                except:
                    ners = data['predicted_ner']

            else:
               ners = data['token']['ner']

            std_ners = data['token']['ner']
            relations = [data['token']['relations']]

            whole_sent = " ".join(data['token']['sentences'])

            for sentence_relation in relations:
                for x in sentence_relation:
                    if x[4] in self.sym_labels[1:]:
                        print('has something sym')
                        self.tot_recall += 2
                    else:
                        self.tot_recall +=  1

            sentence_boundaries = [0]
            words = []
            L = 0
            for i in range(len(sentences)):
                L += len(sentences[i])
                sentence_boundaries.append(L)
                words += sentences[i]


            tokens = [tokenize_word(w) for w in words]
            subwords = [w for li in tokens for w in li]
            maxL = max(maxL, len(subwords))
            subword2token = list(itertools.chain(*[[i] * len(li) for i, li in enumerate(tokens)]))
            token2subword = [0] + list(itertools.accumulate(len(li) for li in tokens))
            subword_start_positions = frozenset(token2subword)
            subword_sentence_boundaries = [sum(len(li) for li in tokens[:p]) for p in sentence_boundaries]

            for n in range(len(subword_sentence_boundaries) - 1):

                sentence_ners = ners[n]
                sentence_relations = relations[n]
                std_ner = std_ners[n]

                std_entity_labels = {}
                self.ner_tot_recall += len(std_ner)

                for start, end, label in std_ner:
                    std_entity_labels[(start, end)] = label
                    self.ner_golden_labels.add( ((l_idx, n), (start, end), label) )

                self.global_predicted_ners[(l_idx, n)] = list(sentence_ners)

                doc_sent_start, doc_sent_end = subword_sentence_boundaries[n : n + 2]

                left_length = doc_sent_start
                right_length = len(subwords) - doc_sent_end
                sentence_length = doc_sent_end - doc_sent_start
                half_context_length = int((max_num_subwords - sentence_length) / 2)

                if sentence_length < max_num_subwords:

                    if left_length < right_length:
                        left_context_length = min(left_length, half_context_length)
                        right_context_length = min(right_length, max_num_subwords - left_context_length - sentence_length)
                    else:
                        right_context_length = min(right_length, half_context_length)
                        left_context_length = min(left_length, max_num_subwords - right_context_length - sentence_length)


                doc_offset = doc_sent_start - left_context_length
                target_tokens = subwords[doc_offset : doc_sent_end + right_context_length]
                target_tokens = [tokenizer.cls_token] + target_tokens[ : self.max_seq_length - 4] + [tokenizer.sep_token]
                assert(len(target_tokens) <= self.max_seq_length - 2)
                # if l_idx==199:
                #     print('================================')
                pos2label = {}
                for x in sentence_relations:
                    pos2label[(x[0],x[1],x[2],x[3])] = label_map[x[4]]
                    # if ((l_idx, n), (x[0],x[1]), (x[2],x[3]), x[4]) in self.golden_labels:
                    #     print(l_idx,n,x[0],x[1],x[2],x[3],x[4],'已经在里面了')
                    self.aope_golden.add(((l_idx, n), (x[0],x[1]), (x[2],x[3])))
                    self.golden_labels.add(((l_idx, n), (x[0],x[1]), (x[2],x[3]), x[4]))#
                    self.golden_labels_withner.add(((l_idx, n), (x[0],x[1], std_entity_labels[(x[0], x[1])]), (x[2],x[3], std_entity_labels[(x[2], x[3])]), x[4]))
                    if x[4] in self.sym_labels[1:]:
                        self.golden_labels.add(((l_idx, n),  (x[2],x[3]), (x[0],x[1]), x[4]))
                        self.golden_labels_withner.add(((l_idx, n), (x[2],x[3], std_entity_labels[(x[2], x[3])]), (x[0],x[1], std_entity_labels[(x[0], x[1])]), x[4]))

                entities = list(sentence_ners)

                for x in sentence_relations:
                    w = (x[2],x[3],x[0],x[1])# 对称关系 但是 在 ASTE任务中是否要考虑， target 和 opinion 之间的对称关系
                    if w not in pos2label:
                        if x[4] in self.sym_labels[1:]:
                            pos2label[w] = label_map[x[4]]  # bug
                        else:
                            pos2label[w] = label_map[x[4]] + len(label_map) - len(self.sym_labels)

                # if not self.evaluate:
                #     entities.append((10000, 10000, 'NIL')) # only for NER

                for sub in entities:
                    # if sub[2]!='target':
                    #     continue
                    cur_ins = []

                    if sub[0] < 10000:
                        sub_s = token2subword[sub[0]] - doc_offset + 1
                        try:
                            sub_e = token2subword[sub[1]+1] - doc_offset
                        except:
                            print(sub)
                            print(token2subword)
                        sub_label = ner_label_map[sub[2]]

                        if self.use_typemarker:
                            l_m = '<subj_start=%s>'%sub[2]
                            r_m = '<subj_end=%s>'%sub[2]
                        else:
                            l_m = '[unused0]'
                            r_m = '[unused1]'
                        #temp_tokens = words[:sub[0]] + [l_m] + words[sub[0]:sub[1] + 1] + [r_m] + words[sub[1] + 1:]
                        # all_temp_tokens.append(temp_tokens)
                        if not self.args.nosolid:
                            temp_tokens = words[:sub[0]] + [l_m] + words[sub[0]:sub[1] + 1] + [r_m] + words[sub[1] + 1:]
                            sub_tokens = target_tokens[:sub_s] + [l_m] + target_tokens[sub_s:sub_e+1] + [r_m] + target_tokens[sub_e+1: ]
                            sub_e += 2
                        else:
                            temp_tokens = words[:sub[0]]+ words[sub[0]:sub[1] + 1]+ words[sub[1] + 1:]
                            sub_tokens = target_tokens
                    else:
                        sub_s = len(target_tokens)
                        sub_e = len(target_tokens)+1
                        sub_tokens = target_tokens + ['[unused0]',  '[unused1]']
                        sub_label = -1

                    if sub_e >= self.max_seq_length-1:
                        continue
                    # assert(sub_e < self.max_seq_length)

                    for start, end, obj_label in sentence_ners:
                        if self.model_type.endswith('nersub'):
                            if start==sub[0] and end==sub[1]:
                                continue

                        doc_entity_start = token2subword[start]
                        doc_entity_end = token2subword[end+1]
                        left = doc_entity_start - doc_offset + 1
                        right = doc_entity_end - doc_offset

                        obj = (start, end)
                        if not self.args.nosolid:
                            if obj[0] >= sub[0]:
                                left += 1
                                if obj[0] > sub[1]:
                                    left += 1

                            if obj[1] >= sub[0]:
                                right += 1
                                if obj[1] > sub[1]:
                                    right += 1


                        label = pos2label.get((sub[0], sub[1], obj[0], obj[1]), 0)#判断 sub 和 obj 是否有关系

                        if right >= self.max_seq_length-1:
                            continue
                        if label == 0:
                            self.cont_none += 1
                        elif label == 1 or label == 4:
                            self.cont_pos += 1
                        elif label == 2 or label == 5:
                            self.cont_neg += 1
                        else:
                            self.cont_neu += 1
                        cur_ins.append(((left, right, ner_label_map[obj_label]), label, obj))# 这里的 都是 左闭右开  （（谓语位置，谓语label）,主谓语关系，谓语位置（没加入[s][/s]））
                        # print('sub:',sub_tokens[sub_s:sub_e+1],"obj:",sub_tokens[left:right+1])

                    maxR = max(maxR, len(cur_ins))
                    dL = self.max_pair_length
                    if self.args.shuffle:
                        np.random.shuffle(cur_ins)
                    for i in range(0, len(cur_ins), dL):
                        examples = cur_ins[i : i + dL]
                        item = {
                            'index': (l_idx, n),
                            'sentence': sub_tokens,
                            'temp_sentence': temp_tokens,
                            'examples': examples,
                            'sub': (sub, (sub_s, sub_e), sub_label), #(sub[0], sub[1], sub_label),
                            'whole_sen':whole_sent,
                            'sen_word':sen_word,
                            'origion_adj': data['graph'],
                        }

                        self.data.append(item)
        # if dataset == 'train':
        #
        #     with open('sentence_list_train.json', 'w') as json_file:
        #         # 使用json.dump将列表写入文件
        #         json.dump(all_temp_tokens, json_file)
        # elif dataset == 'dev' and do_test == False:
        #     with open('sentence_list_dev.json', 'w') as json_file:
        #         # 使用json.dump将列表写入文件
        #         json.dump(all_temp_tokens, json_file)
        # else:
        #     with open('sentence_list_test.json', 'w') as json_file:
        #         # 使用json.dump将列表写入文件
        #         json.dump(all_temp_tokens, json_file)
        print("none:",self.cont_none)
        print("pos:",self.cont_pos)
        print("neu:",self.cont_neu)
        print("neg:",self.cont_neg)
        logger.info('maxR: %s', maxR)
        logger.info('maxL: %s', maxL)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        entry = self.data[idx]
        ori_adj = entry['origion_adj']
        #===========================
        tokens_t = []
        tok2ori_map_t = []
        cont = 0
        for ori_i, w in enumerate(entry['temp_sentence']):
            if w == '[unused0]' or w == '<subj_start=opinion>' or w == '<subj_start=target>':
                tokens_t.append(w)
                tok2ori_map_t.append(-1)#实体标记和其他没有关系
                continue
            elif w == '[unused1]' or w == '<subj_end=opinion>' or w == '<subj_end=target>':
                tokens_t.append(w)
                tok2ori_map_t.append(-1)
                continue
            for t in self.args.tokenizer.tokenize(w):
                tokens_t.append(t)
                tok2ori_map_t.append(cont)
            cont += 1

        truncate_tok_len = len(tokens_t)
        tok_adj = np.zeros(
            (truncate_tok_len, truncate_tok_len), dtype='float32')
        for i in range(truncate_tok_len):
            for j in range(truncate_tok_len):
                try:
                    if tok2ori_map_t[i]!=-1 and tok2ori_map_t[j]!=-1:
                        tok_adj[i][j] = ori_adj[tok2ori_map_t[i]][tok2ori_map_t[j]]
                except:
                    tok_adj[i][j] = ori_adj[tok2ori_map_t[i]][tok2ori_map_t[j]]

        context_matrix = np.zeros(
            (self.max_seq_length + self.max_pair_length * 2, self.max_seq_length + self.max_pair_length * 2)).astype('float64')
        context_matrix[1:truncate_tok_len + 1, 1:truncate_tok_len + 1] = tok_adj
        # context_matrix[truncate_tok_len + 1,truncate_tok_len + 1] = 1
        np.fill_diagonal(context_matrix, 1)
        #===========================
        sub, sub_position, sub_label = entry['sub']
        input_ids = self.tokenizer.convert_tokens_to_ids(entry['sentence'])
        L = len(input_ids)
        input_ids += [self.tokenizer.pad_token_id] * (self.max_seq_length - len(input_ids))

        attention_mask = torch.zeros((self.max_entity_length+self.max_seq_length, self.max_entity_length+self.max_seq_length), dtype=torch.int64)
        attention_mask[:L, :L] = 1

        if self.model_type.startswith('albert'):
            input_ids = input_ids + [30002] * (len(entry['examples'])) + [self.tokenizer.pad_token_id] * (self.max_pair_length - len(entry['examples']))
            input_ids = input_ids + [30003] * (len(entry['examples'])) + [self.tokenizer.pad_token_id] * (self.max_pair_length - len(entry['examples'])) # for debug
        else:
            input_ids = input_ids + [3] * (len(entry['examples'])) + [self.tokenizer.pad_token_id] * (self.max_pair_length - len(entry['examples']))# [3] 表示obj 开头的部分
            input_ids = input_ids + [4] * (len(entry['examples'])) + [self.tokenizer.pad_token_id] * (self.max_pair_length - len(entry['examples'])) # for debug [4]表示 obj 结尾的部分

        labels = []
        ner_labels = []
        mention_pos = []
        mention_2 = []
        position_ids = list(range(self.max_seq_length)) + [0] * self.max_entity_length
        num_pair = self.max_pair_length

        for x_idx, obj in enumerate(entry['examples']):#遍历所有的 sub, obj对
            m2 = obj[0]
            label = obj[1]

            mention_pos.append((m2[0], m2[1]))
            mention_2.append(obj[2])

            w1 = x_idx
            w2 = w1 + num_pair

            w1 += self.max_seq_length
            w2 += self.max_seq_length

            position_ids[w1] = m2[0]#obj 的开头位置
            position_ids[w2] = m2[1]#obj end 的位置

            for xx in [w1, w2]:
                for yy in [w1, w2]:
                    attention_mask[xx, yy] = 1  #marker 可以互相看到
                attention_mask[xx, :L] = 1  # marker 可以看到句子

            labels.append(label)
            ner_labels.append(m2[2])

            l_m = '<subj_start=opinion>'
            r_m = '<subj_end=opinion>'
            if self.use_typemarker and not self.args.nosolid:
                if label == 0:
                    l_m = '<subj_start=opinion>'
                    r_m = '<subj_end=opinion>'
                elif label == 1:
                    l_m = '<subj_start=target>'
                    r_m = '<subj_end=target>'
                l_m = self.tokenizer.encode(l_m)
                r_m = self.tokenizer.encode(r_m)
                input_ids[w1] = l_m[1]
                input_ids[w2] = r_m[1]


        pair_L = len(entry['examples'])
        if self.args.att_left:
            attention_mask[self.max_seq_length : self.max_seq_length+pair_L, self.max_seq_length : self.max_seq_length+pair_L] = 1
        if self.args.att_right:
            attention_mask[self.max_seq_length+num_pair : self.max_seq_length+num_pair+pair_L, self.max_seq_length+num_pair : self.max_seq_length+num_pair+pair_L] = 1

        mention_pos += [(0, 0)] * (num_pair - len(mention_pos))
        labels += [-1] * (num_pair - len(labels))
        ner_labels += [-1] * (num_pair - len(ner_labels))

        t = entry['sen_word']
        t = t + ['pad'] * (self.max_seq_length - len(t))

        # te1 = np.zeros((self.max_seq_length + num_pair * 2, self.max_seq_length + num_pair * 2))#在没加[cls][sep] 以及[unused0] 和wordpiece下的 句子的邻接矩阵
        # te1[:len(entry['origion_adj'][0]), :len(entry['origion_adj'][0])] = entry['origion_adj']
        # te1 = te1.astype(float)
        # np.fill_diagonal(te1, 1)

        # te2 = np.zeros((self.max_seq_length + num_pair * 2, self.max_seq_length + num_pair * 2))#都有
        # te2[:len(adj2[0]), :len(adj2[0])] = adj2
        # te2 = te2.astype(float)
        # np.fill_diagonal(te2, 1)


        #sub_width_embedding
        sub_width = [sub[1] - sub[0]]
        sub_width = torch.tensor(sub_width)
        sub_width = torch.unsqueeze(sub_width,0)
        sub_max_value = torch.max(sub_width).item()
        if sub_max_value >= self.args.num_width_embeddings:
            sub_span_width_embedding_over = Embedding(num_embeddings=sub_max_value + 1,
                                                   embedding_dim=self.args.span_width_embedding_dim)
            sub_width_embeddings = sub_span_width_embedding_over(sub_width)
        else:
            sub_width_embeddings = self._span_width_embedding(sub_width)

        sub_width_embeddings = torch.squeeze(sub_width_embeddings, 0)
        #obj_width_embedding
        padded_x = torch.zeros((1,num_pair, 20))
        obj_width = [i[2][1] - i[2][0] for i in entry['examples']]
        obj_width = torch.tensor(obj_width)
        obj_width = torch.unsqueeze(obj_width,0)
        obj_max_value = torch.max(obj_width).item()
        if obj_max_value >= self.args.num_width_embeddings:
            _span_width_embedding_over = Embedding(num_embeddings=obj_max_value+1, embedding_dim=self.args.span_width_embedding_dim)
            obj_width_embeddings = _span_width_embedding_over(obj_width)
        else:
            obj_width_embeddings = self._span_width_embedding(obj_width)


        padded_x[:, :obj_width.size()[1], :] = obj_width_embeddings
        obj_width_embeddings = padded_x
        obj_width_embeddings = torch.squeeze(obj_width_embeddings,0)

        item = [torch.tensor(input_ids),
                attention_mask,
                torch.tensor(position_ids),
                torch.tensor(sub_position),
                torch.tensor(mention_pos),
                torch.tensor(labels, dtype=torch.int64),
                torch.tensor(ner_labels, dtype=torch.int64),
                torch.tensor(sub_label, dtype=torch.int64),
                torch.tensor(context_matrix),
                sub_width_embeddings.detach(),
                obj_width_embeddings.detach(),
                entry['whole_sen'],
                t,
        ]

        if self.evaluate:
            item.append(entry['index'])
            item.append(sub)
            item.append(mention_2)


        return item

    @staticmethod
    def collate_fn(batch):
        fields = [x for x in zip(*batch)]

        num_metadata_fields = 3
        try:
            stacked_fields = [torch.stack(field) for field in fields[:-num_metadata_fields-2]]  # don't stack metadata fields
        #stacked_fields = [torch.stack(field) for field in fields[:-num_metadata_fields]]
            stacked_fields.extend(fields[-num_metadata_fields-2:])  # add them as lists not torch tensors

        except:
            print('error')
        return stacked_fields



def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)



def _rotate_checkpoints(args, checkpoint_prefix, use_mtime=False):
    if not args.save_total_limit:
        return
    if args.save_total_limit <= 0:
        return

    # Check if we should delete older checkpoint(s)
    glob_checkpoints = glob.glob(os.path.join(args.output_dir, '{}-*'.format(checkpoint_prefix)))
    if len(glob_checkpoints) <= args.save_total_limit:
        return

    ordering_and_checkpoint_path = []
    for path in glob_checkpoints:
        if use_mtime:
            ordering_and_checkpoint_path.append((os.path.getmtime(path), path))
        else:
            regex_match = re.match('.*{}-([0-9]+)'.format(checkpoint_prefix), path)
            if regex_match and regex_match.groups():
                ordering_and_checkpoint_path.append((int(regex_match.groups()[0]), path))

    checkpoints_sorted = sorted(ordering_and_checkpoint_path)
    checkpoints_sorted = [checkpoint[1] for checkpoint in checkpoints_sorted]
    number_of_checkpoints_to_delete = max(0, len(checkpoints_sorted) - args.save_total_limit)
    checkpoints_to_be_deleted = checkpoints_sorted[:number_of_checkpoints_to_delete]
    for checkpoint in checkpoints_to_be_deleted:
        logger.info("Deleting older checkpoint [{}] due to args.save_total_limit".format(checkpoint))
        shutil.rmtree(checkpoint)



def read_sentence_depparsed(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
        return data

def get_dataset(dataset_name,args):


    train = list(read_sentence_depparsed(dataset_name + '/train.json'))
    dev = list(read_sentence_depparsed(dataset_name + '/dev.json'))
    test_file_path = args.test_file
    test = list(read_sentence_depparsed(test_file_path))
    print('测试集位置：',test_file_path)

    return train, dev, test

def train(args, model, tokenizer,dataset_list,dev_list):
    """ Train the model """
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter("logs/"+args.data_dir[max(args.data_dir.rfind('/'),0):]+"_re_logs/"+args.output_dir[args.output_dir.rfind('/'):])

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)

    train_dataset = ACEDataset(tokenizer=tokenizer,dataset_list=dataset_list,args=args, max_pair_length=args.max_pair_length,dataset='train')

    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size, num_workers=4*int(args.output_dir.find('test')==-1))

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    if args.warmup_steps==-1:
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=int(0.1*t_total), num_training_steps=t_total
        )
    else:
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
        )

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)
    # ori_model = model
    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                   args.train_batch_size * args.gradient_accumulation_steps * (torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    tr_ner_loss, logging_ner_loss = 0.0, 0.0
    tr_re_loss, logging_re_loss = 0.0, 0.0

    model.zero_grad()
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0])
    set_seed(args)  # Added here for reproductibility (even between python 2 and 3)
    best_f1 = -1



    for _ in train_iterator:
        if args.shuffle and _ > 0:
            train_dataset.initialize()
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):

            model.train()
            #sents = batch[-1]
            batch = tuple(t.to(args.device) for t in batch[:-2])
            #batch = tuple(t.to(args.device) for t in batch[:-1])

            inputs = {'input_ids':      batch[0],
                      'attention_mask': batch[1],
                      'position_ids':   batch[2],
                      'labels':         batch[5],
                      'ner_labels':     batch[6],
                      'adj':   batch[8],
                      'sub_width_embedding': batch[9],
                      'obj_width_embedding': batch[10],
                      }


            inputs['sub_positions'] = batch[3]
            if args.model_type.find('span')!=-1:
                inputs['mention_pos'] = batch[4]
            if args.model_type.endswith('bertonedropoutnersub'):
                inputs['sub_ner_labels'] = batch[7]
            outputs = model(**inputs)
            loss = outputs[0]  # model outputs are always tuple in pytorch-transformers (see doc)
            re_loss = outputs[1]
            ner_loss = outputs[2]

            if args.n_gpu > 1:
                loss = loss.mean() # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
                re_loss = re_loss / args.gradient_accumulation_steps
                ner_loss = ner_loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            tr_loss += loss.item()
            if re_loss > 0:
                tr_re_loss += re_loss.item()
            if ner_loss > 0:
                tr_ner_loss += ner_loss.item()


            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.max_grad_norm > 0:
                    if args.fp16:
                        torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                # if args.model_type.endswith('rel') :
                #     ori_model.bert.encoder.layer[args.add_coref_layer].attention.self.relative_attention_bias.weight.data[0].zero_() # 可以手动乘个mask

                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    # Log metrics
                    tb_writer.add_scalar('lr', scheduler.get_lr()[0], global_step)
                    tb_writer.add_scalar('loss', (tr_loss - logging_loss)/args.logging_steps, global_step)
                    logging_loss = tr_loss

                    tb_writer.add_scalar('RE_loss', (tr_re_loss - logging_re_loss)/args.logging_steps, global_step)
                    logging_re_loss = tr_re_loss

                    tb_writer.add_scalar('NER_loss', (tr_ner_loss - logging_ner_loss)/args.logging_steps, global_step)
                    logging_ner_loss = tr_ner_loss


                if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0: # valid for bert/spanbert
                    update = True
                    # Save model checkpoint
                    if args.evaluate_during_training:  # Only evaluate when single GPU otherwise metrics may not average well
                        results = evaluate(args,dev_list, model, tokenizer)
                        f1 = results['f1_with_ner']
                        tb_writer.add_scalar('f1_with_ner', f1, global_step)

                        if f1 > best_f1:
                            best_f1 = f1
                            print ('Best F1', best_f1)
                        else:
                            update = False

                    if update:
                        checkpoint_prefix = 'checkpoint'
                        output_dir = os.path.join(args.output_dir, '{}-{}'.format(checkpoint_prefix, global_step))
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)
                        model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training

                        model_to_save.save_pretrained(output_dir)

                        torch.save(args, os.path.join(output_dir, 'training_args.bin'))
                        logger.info("Saving model checkpoint to %s", output_dir)

                        _rotate_checkpoints(args, checkpoint_prefix)

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break
        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    if args.local_rank in [-1, 0]:
        tb_writer.close()


    return global_step, tr_loss / global_step, best_f1

def to_list(tensor):
    return tensor.detach().cpu().tolist()

def evaluate(args,dataset_list,model, tokenizer, prefix="", do_test=False):

    eval_output_dir = args.output_dir

    eval_dataset = ACEDataset(tokenizer=tokenizer,dataset_list=dataset_list, args=args, evaluate=True, do_test=do_test, max_pair_length=args.max_pair_length,dataset='dev')
    aope_golden = set(eval_dataset.aope_golden)
    golden_labels = set(eval_dataset.golden_labels)# 正确的 三元组关系
    golden_labels_withner = set(eval_dataset.golden_labels_withner)# 正确的带有NER标签的 三元组关系
    label_list = list(eval_dataset.label_list)
    sym_labels = list(eval_dataset.sym_labels)
    tot_recall = eval_dataset.tot_recall#有一个重复了

    if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(eval_output_dir)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)


    scores = defaultdict(dict)
    # ner_pred = not args.model_type.endswith('noner')
    example_subs = set([])
    num_label = len(label_list)

    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Batch size = %d", args.eval_batch_size)

    model.eval()

    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size,  collate_fn=ACEDataset.collate_fn, num_workers=4*int(args.output_dir.find('test')==-1))

    # Eval!
    logger.info("  Num examples = %d", len(eval_dataset))

    start_time = timeit.default_timer()

    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        indexs = batch[-3]
        subs = batch[-2]
        batch_m2s = batch[-1]
        ner_labels = batch[6]
        SEN = batch[-5]#整个句子
        sen_word = batch[-4]#单个单词
        batch = tuple(t.to(args.device) for t in batch[:11])

        with torch.no_grad():
            inputs = {'input_ids':      batch[0],
                    'attention_mask': batch[1],
                    'position_ids':   batch[2],
                    'adj':   batch[8],
                    #   'labels':         batch[4],
                    #   'ner_labels':     batch[5],
                    'sub_width_embedding':batch[9],
                    'obj_width_embedding':batch[10],
                    }

            inputs['sub_positions'] = batch[3]
            if args.model_type.find('span')!=-1:
                inputs['mention_pos'] = batch[4]

            outputs = model(**inputs)

            logits = outputs[0]

            if args.eval_logsoftmax:  # perform a bit better
                logits = torch.nn.functional.log_softmax(logits, dim=-1)

            elif args.eval_softmax:
                logits = torch.nn.functional.softmax(logits, dim=-1)

            if args.use_ner_results or args.model_type.endswith('nonersub'):
                ner_preds = ner_labels# 判断是否使用 ner 结果 ，即是否使用真正的NER
            else:
                ner_preds = torch.argmax(outputs[1], dim=-1)#否则 用预测的结果
            logits = logits.cpu().numpy()
            ner_preds = ner_preds.cpu().numpy()# 这里长度 指的是每个标签 每个值对应该标签的ner_label
            for i in range(len(indexs)):#第i 个样例（1batch中有多个样例）
                index = indexs[i]
                sub = subs[i]
                m2s = batch_m2s[i]
                example_subs.add(((index[0], index[1]), (sub[0], sub[1])))
                for j in range(len(m2s)):#该样例中的一个 term
                    obj = m2s[j]
                    ner_label = eval_dataset.ner_label_list[ner_preds[i,j]]#obj是 target or opinion
                    scores[(index[0], index[1])][( (sub[0], sub[1]), (obj[0], obj[1]))] = (logits[i, j].tolist(), ner_label,SEN[i],sen_word[i])

    aope_cor = 0
    aope_tot_pred = 0
    cor = 0
    tot_pred = 0
    cor_with_ner = 0
    global_predicted_ners = eval_dataset.global_predicted_ners#预测的 NER label
    ner_golden_labels = eval_dataset.ner_golden_labels# 正确的 NER label
    ner_cor = 0
    ner_tot_pred = 0
    ner_ori_cor = 0
    tot_output_results = defaultdict(list)
    columns_name = ['sentence','True', 'Predict']
    run = wandb.init(
       project=f'pl_marker_project_{start_time}',
       config={
            'epoch': args.num_train_epochs,
            'train_batch_size': args.per_gpu_train_batch_size,
            'lr': args.learning_rate,
            'model_name': args.model_name_or_path
       }
    )
    TRUE_L = []
    PRE_L = []
    SEN_LIST = []
    if not args.eval_unidirect:     # eval_unidrect is for ablation study
        # print (len(scores))

        for example_index, pair_dict in sorted(scores.items(), key=lambda x:x[0]):
            visited  = set([])
            sentence_results = []
            exam_sen = pair_dict[next(iter(pair_dict))][3]
            example_pre = []#用来记录 这个样本的预测结果
            example_t = list(filter(lambda x: x[0] == example_index, golden_labels))  # 记录这个样本 正确的结果
            example_true = []
            first_key,first_val = next(iter(pair_dict.items()))
            SEN_LIST.append(first_val[2])
            for i in example_t:
                w1 = first_val[3][i[1][0]:i[1][1] + 1]
                w2 = first_val[3][i[2][0]:i[2][1] + 1]
                ww1 = ' '.join(w1)
                ww2 = ' '.join(w2)
                t = (ww1, ww2, i[3])
                example_true.append(t)

            no_overlap = []
            output_preds = []
            if args.other_method != True:

                for k1, (v1, v2_ner_label, sen_,
                         sen_word) in pair_dict.items():  # k1: sub 和 obj v1:7个标签的概率  v2_ner_label:Obj的label

                    if k1 in visited:
                        continue
                    visited.add(k1)

                    if v2_ner_label == 'NIL':
                        continue
                    v1 = list(v1)
                    m1 = k1[0]  # sub
                    m2 = k1[1]  # obj
                    if m1 == m2:
                        continue
                    k2 = (m2, m1)  # 判断对应的对称关系
                    v2s = pair_dict.get(k2, None)  # 判断是否有关系
                    if v2s is not None:
                        visited.add(k2)
                        v2, v1_ner_label = v2s[0], v2s[1]
                        v2 = v2[: len(sym_labels)] + v2[num_label:] + v2[len(sym_labels): num_label]  # 将 sub 和 obj 为对称关系的预测score 加在一起

                        for j in range(len(v2)):
                            v1[j] += v2[j]
                    else:
                        assert (False)

                    if v1_ner_label == 'NIL':
                        continue

                    pred_label = np.argmax(v1)

                    if pred_label > 0:
                        if pred_label >= num_label:  # 真的有分到这边的吗
                            pred_label = pred_label - num_label + len(sym_labels)

                            m1, m2 = m2, m1
                            v1_ner_label, v2_ner_label = v2_ner_label, v1_ner_label

                        pred_score = v1[pred_label]

                        # sentence_results.append( (pred_score, m1, m2, pred_label, v1_ner_label, v2_ner_label,sen_,sen_word) )
                        sentence_results.append((pred_score, m1, m2, pred_label, v1_ner_label, v2_ner_label))
                    # sentence_results.append(
                    #     (pred_score, m1, m2, pred_label, 'target', v2_ner_label, sen_, sen_word))
                sentence_results.sort(key=lambda x: -x[0])

                def is_overlap(m1, m2):
                    if m2[0] <= m1[0] and m1[0] <= m2[1]:
                        return True
                    if m1[0] <= m2[0] and m2[0] <= m1[1]:
                        return True
                    return False

                output_preds = []

                for item in sentence_results:
                    m1 = item[1]
                    m2 = item[2]
                    overlap = False
                    for x in no_overlap:
                        _m1 = x[1]
                        _m2 = x[2]
                        # same relation type & overlap subject & overlap object --> delete
                        if item[3] == x[3] and (is_overlap(m1, _m1) and is_overlap(m2, _m2)):
                            overlap = True
                            break

                    pred_label = label_list[item[3]]

                    if not overlap:
                        no_overlap.append(item)
            else:
                pred_dicts = {}
                for k1, (v1, v2_ner_label, sen_, sen_word) in pair_dict.items():
                    m1 = k1[0]
                    m2 = k1[1]
                    sco = torch.max(torch.tensor(v1))
                    pred_label = np.argmax(v1)
                    beh_term_label = v2_ner_label
                    t1 = exam_sen[m1[0]:m1[1] + 1]
                    t2 = exam_sen[m2[0]:m2[1] + 1]
                    tt1 = ' '.join(t1)
                    tt2 = ' '.join(t2)
                    if m1 == m2:
                        continue
                    if (m2, m1) in pair_dict:
                        fron_tern_label = pair_dict[(m2, m1)][1]
                    if pred_label >= num_label:
                        pred_label = pred_label - num_label + len(sym_labels)
                    if m1 == m2:
                        continue
                    if (m1, m2) in pred_dicts:
                        score_dict = pred_dicts[(m1, m2)][0]
                        score_new = sco
                        if score_dict > score_new:
                            continue
                        else:
                            pred_dicts[(m1, m2)] = (sco, pred_label, fron_tern_label, beh_term_label, tt1, tt2)
                    elif (m2, m1) in pred_dicts:
                        score_dict = pred_dicts[(m2, m1)][0]
                        score_new = sco
                        if score_dict > score_new:
                            continue
                        else:
                            pred_dicts[(m2, m1)] = (sco, pred_label, beh_term_label, fron_tern_label, tt2, tt1)
                    else:
                        pred_dicts[(m1, m2)] = (sco, pred_label, fron_tern_label, beh_term_label, tt1, tt2)
                history = []
                for k1, v1 in pred_dicts.items():
                    aspect_span1 = k1[0]
                    opinion_span1 = k1[1]
                    for k2, v2 in pred_dicts.items():
                        if (k1, k2) in history: \
                                continue
                        history.append((k1, k2))
                        history.append((k2, k1))
                        if k1 == k2:
                            continue
                        aspect_span2 = k2[0]
                        opinion_span2 = k2[1]
                        repeat_a_span = list(set(aspect_span1) & set(aspect_span2))
                        repeat_o_span = list(set(opinion_span1) & set(opinion_span2))
                        if len(repeat_a_span) == 0 or len(repeat_o_span) == 0:
                            continue
                        elif len(repeat_a_span) <= min(len(aspect_span1), len(aspect_span2)) and \
                                len(repeat_o_span) <= min(len(opinion_span1), len(opinion_span2)):
                            i_score = v1[0]
                            j_score = v2[0]
                            if i_score >= j_score:
                                pred_dicts[k2] = (0, 0, 0, 0)
                            else:
                                pred_dicts[k1] = (0, 0, 0, 0)
                        else:
                            raise (KeyboardInterrupt)

                for k, v in pred_dicts.items():
                    # try:
                    if v[0] != 0 and v[1] != 0:
                        no_overlap.append((v[0], k[0], k[1], v[1], v[2], v[3], v[4], v[5]))
                    # except:
                    #     print(example_index)
                    #     print(v)
                    #     print(k)

            pos2ner = {}

            for item in no_overlap:
                m1 = item[1]
                m2 = item[2]
                label1 = item[4]
                label2 = item[5]

                pred_label = label_list[item[3]]
                tot_pred += 1  # 总共预测了多少个
                aope_tot_pred += 1
                t1 = exam_sen[m1[0]:m1[1] + 1]
                t2 = exam_sen[m2[0]:m2[1] + 1]
                tt1 = ' '.join(t1)
                tt2 = ' '.join(t2)
                if pred_label in sym_labels:
                    tot_pred += 1  # duplicate
                    if (example_index, m1, m2, pred_label) in golden_labels or (
                    example_index, m2, m1, pred_label) in golden_labels:
                        cor += 2
                else:
                    # if (example_index, m1, m2, pred_label) in golden_labels or (
                    # example_index, m2, m1, pred_label) in golden_labels:
                    if (example_index, m1, m2, pred_label) in golden_labels:
                        cor += 1  # 预测正确的数量
                        example_pre.append((tt1, tt2, pred_label, '\u221a'))
                    else:
                        example_pre.append((tt1, tt2, pred_label, 'X'))

                    if (example_index, m1, m2) in aope_golden or (
                            example_index, m2, m1) in aope_golden:
                        aope_cor += 1

                if m1 not in pos2ner:
                    pos2ner[m1] = item[4]
                if m2 not in pos2ner:
                    pos2ner[m2] = item[5]

                output_preds.append((m1, m2, pred_label))
                if pred_label in sym_labels:
                    if (example_index, (m1[0], m1[1], pos2ner[m1]), (m2[0], m2[1], pos2ner[m2]),
                        pred_label) in golden_labels_withner \
                            or (example_index, (m2[0], m2[1], pos2ner[m2]), (m1[0], m1[1], pos2ner[m1]),
                                pred_label) in golden_labels_withner:
                        cor_with_ner += 2
                else:
                    if (example_index, (m1[0], m1[1], pos2ner[m1]), (m2[0], m2[1], pos2ner[m2]),
                        pred_label) in golden_labels_withner \
                            or (example_index, (m2[0], m2[1], pos2ner[m2]), (m1[0], m1[1], pos2ner[m1]),
                                pred_label) in golden_labels_withner:
                        cor_with_ner += 1

            if do_test:
                # output_w.write(json.dumps(output_preds) + '\n')
                tot_output_results[example_index[0]].append((example_index[1], output_preds))

            sentence = []
            # refine NER results
            ner_results = list(global_predicted_ners[example_index])  #
            for i in range(len(ner_results)):
                start, end, label = ner_results[i]
                if (example_index, (start, end), label) in ner_golden_labels:
                    ner_ori_cor += 1
                if (start, end) in pos2ner:
                    label = pos2ner[(start, end)]  # 这里的label 可以是来自于这个模型的中的预测ner label(out[1]) 或者是来自于golden_ner_label
                if (example_index, (start, end), label) in ner_golden_labels:
                    ner_cor += 1
                ner_tot_pred += 1  # 记录总共预测了多少ner

            TRUE_L.append(example_true)
            PRE_L.append(example_pre)

    else:

        for example_index, pair_dict in sorted(scores.items(), key=lambda x:x[0]):
            sentence_results = []
            for k1, (v1, v2_ner_label) in pair_dict.items():

                if v2_ner_label=='NIL':
                    continue
                v1 = list(v1)
                m1 = k1[0]
                m2 = k1[1]
                if m1 == m2:
                    continue

                pred_label = np.argmax(v1)
                if pred_label>0 and pred_label < num_label:

                    pred_score = v1[pred_label]

                    sentence_results.append( (pred_score, m1, m2, pred_label, None, v2_ner_label) )

            sentence_results.sort(key=lambda x: -x[0])
            no_overlap = []
            def is_overlap(m1, m2):
                if m2[0]<=m1[0] and m1[0]<=m2[1]:
                    return True
                if m1[0]<=m2[0] and m2[0]<=m1[1]:
                    return True
                return False

            output_preds = []

            for item in sentence_results:
                m1 = item[1]
                m2 = item[2]
                overlap = False
                for x in no_overlap:
                    _m1 = x[1]
                    _m2 = x[2]
                    if item[3]==x[3] and (is_overlap(m1, _m1) and is_overlap(m2, _m2)):
                        overlap = True
                        break

                pred_label = label_list[item[3]]

                output_preds.append((m1, m2, pred_label))

                if not overlap:
                    no_overlap.append(item)

            pos2ner = {}
            predpos2ner = {}
            ner_results = list(global_predicted_ners[example_index])
            for start, end, label in ner_results:
                predpos2ner[(start, end)] = label

            for item in no_overlap:
                m1 = item[1]
                m2 = item[2]
                pred_label = label_list[item[3]]
                tot_pred += 1

                if (example_index, m1, m2, pred_label) in golden_labels:
                    cor += 1

                if m1 not in pos2ner:
                    pos2ner[m1] = predpos2ner[m1]#item[4]

                if m2 not in pos2ner:
                    pos2ner[m2] = item[5]

                # if pred_label in sym_labels:
                #     if (example_index, (m1[0], m1[1], pos2ner[m1]), (m2[0], m2[1], pos2ner[m2]), pred_label) in golden_labels_withner \
                #         or (example_index,  (m2[0], m2[1], pos2ner[m2]), (m1[0], m1[1], pos2ner[m1]), pred_label) in golden_labels_withner:
                #         cor_with_ner += 2
                # else:
                if (example_index, (m1[0], m1[1], pos2ner[m1]), (m2[0], m2[1], pos2ner[m2]), pred_label) in golden_labels_withner:
                    cor_with_ner += 1

            # refine NER results
            ner_results = list(global_predicted_ners[example_index])
            for i in range(len(ner_results)):
                start, end, label = ner_results[i]
                if (example_index, (start, end), label) in ner_golden_labels:
                    ner_ori_cor += 1
                if (start, end) in pos2ner:
                    label = pos2ner[(start, end)]
                if (example_index, (start, end), label) in ner_golden_labels:
                    ner_cor += 1
                ner_tot_pred += 1
    ttrue_L = []
    ppre_L = []
    for i in range(len(SEN_LIST)):
        t = "".join(str(x) for x in TRUE_L[i])
        t2 = "".join(str(x) for x in PRE_L[i])
        ttrue_L.append(t)
        ppre_L.append(t2)
    # for i in range(len(SEN_LIST)):
    #     print(SEN_LIST[i], ttrue_L[i], ppre_L[i])
    my_table = wandb.Table(
        columns=columns_name,
        data=[(SEN_LIST[i], ttrue_L[i], ppre_L[i]) for i in range(len(SEN_LIST))]
    )
    run.log({f'final_result-{args.data_dir}_{args.seed}': my_table})
    evalTime = timeit.default_timer() - start_time
    logger.info("  Evaluation done in total %f secs (%f example per second)", evalTime,  len(global_predicted_ners) / evalTime)

    if do_test:
        output_w = open(os.path.join(args.output_dir, 'pred_results.json'), 'w')
        json.dump(tot_output_results, output_w)

    ner_p = ner_cor / ner_tot_pred if ner_tot_pred > 0 else 0
    ner_r = ner_cor / len(ner_golden_labels)
    ner_f1 = 2 * (ner_p * ner_r) / (ner_p + ner_r) if ner_cor > 0 else 0.0

    p = cor / tot_pred if tot_pred > 0 else 0
    r = cor / tot_recall
    f1 = 2 * (p * r) / (p + r) if cor > 0 else 0.0
    # assert(tot_recall==len(golden_labels))

    aope_p = aope_cor/aope_tot_pred if aope_tot_pred > 0 else 0
    aope_r = aope_cor/len(aope_golden)
    aope_f1 = 2 * (aope_p * aope_r) / (aope_p + aope_r) if aope_cor > 0 else 0.0
    p_with_ner = cor_with_ner / tot_pred if tot_pred > 0 else 0
    r_with_ner = cor_with_ner / tot_recall
    # assert(tot_recall==len(golden_labels_withner))
    f1_with_ner = 2 * (p_with_ner * r_with_ner) / (p_with_ner + r_with_ner) if cor_with_ner > 0 else 0.0

    results = {'f1':  f1,'p':p,'r':r,  'f1_with_ner': f1_with_ner, 'ner_f1': ner_f1,'aope_p':aope_p,'aope_r':aope_r,'aope_f1':aope_f1}

    logger.info("Result: %s", json.dumps(results))

    return results



def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir", default='ace_data', type=str, required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--model_type", default=None, type=str, required=True,
                        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                        help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(ALL_MODELS))
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--max_seq_length", default=384, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")

    parser.add_argument("--evaluate_during_training", action='store_true',
                        help="Rul evaluation during training at each logging step.")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")

    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=2e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=10.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=-1, type=int,
                        help="Linear warmup over warmup_steps.")

    parser.add_argument('--logging_steps', type=int, default=5,
                        help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=1000,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument("--eval_all_checkpoints", action='store_true',
                        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument('--overwrite_output_dir', action='store_true',
                        help="Overwrite the content of the output directory")
    parser.add_argument('--overwrite_cache', action='store_true',
                        help="Overwrite the cached training and evaluation sets")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")

    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument('--server_ip', type=str, default='', help="For distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="For distant debugging.")
    parser.add_argument('--save_total_limit', type=int, default=2,
                        help='Limit the total amount of checkpoints, delete the older checkpoints in the output_dir, does not delete by default')

    parser.add_argument("--train_file",  default="train.json", type=str)
    parser.add_argument("--dev_file",  default="dev.json", type=str)
    parser.add_argument("--test_file",  default="test.json", type=str)
    parser.add_argument('--max_pair_length', type=int, default=64,  help="")
    parser.add_argument("--alpha", default=1.0, type=float)
    parser.add_argument('--save_results', action='store_true')
    parser.add_argument('--no_test', action='store_true')
    parser.add_argument('--eval_logsoftmax', action='store_true')
    parser.add_argument('--eval_softmax', action='store_true')
    parser.add_argument('--shuffle', action='store_true')
    parser.add_argument('--lminit', action='store_true')
    parser.add_argument('--no_sym', action='store_true')
    parser.add_argument('--att_left', action='store_true')
    parser.add_argument('--att_right', action='store_true')
    parser.add_argument('--use_ner_results', action='store_true')
    parser.add_argument('--use_typemarker', action='store_true')
    parser.add_argument('--eval_unidirect', action='store_true')

    parser.add_argument('--n_gcn', type=int, default=1,
                        help='Number of GCN layers.')
    parser.add_argument('--embedding_dim', type=int, default=768,
                        help='Dimension of glove embeddings')
    parser.add_argument('--span_width_embedding_dim', type=int, default=20,
                        help='')
    parser.add_argument('--num_width_embeddings', type=int, default=8,
                        help='')
    parser.add_argument('--use_span_width', action='store_true',
                        help='')
    parser.add_argument('--gcn', action='store_true',
                        help='')
    parser.add_argument('--nosolid', action='store_true',
                        help='')
    parser.add_argument('--other_method', action='store_true',
                        help='')
    parser.add_argument('--beta', type=float, default=1.0,
                        help='')
    parser.add_argument('--focalloss', action='store_true',
                        help='')
    args = parser.parse_args()
    args.highway = False
    wandb.login()

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train and not args.overwrite_output_dir:
        raise ValueError("Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(args.output_dir))

    def create_exp_dir(path, scripts_to_save=None):
        if args.output_dir.endswith("test"):
            return
        if not os.path.exists(path):
            os.mkdir(path)

        print('Experiment dir : {}'.format(path))
        if scripts_to_save is not None:
            if not os.path.exists(os.path.join(path, 'scripts')):
                os.mkdir(os.path.join(path, 'scripts'))
            for script in scripts_to_save:
                dst_file = os.path.join(path, 'scripts', os.path.basename(script))
                shutil.copyfile(script, dst_file)

    if args.do_train and args.local_rank in [-1, 0] and args.output_dir.find('test')==-1:
        current_file_path = __file__
        file_name = os.path.basename(current_file_path)
        create_exp_dir(args.output_dir, scripts_to_save=[file_name, 'transformers/src/transformers/modeling_bert.py', 'transformers/src/transformers/modeling_albert.py'])


    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt = '%m/%d/%Y %H:%M:%S',
                        level = logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                    args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16)

    # Set seed
    set_seed(args)

    if args.data_dir.find('ace')!=-1:
        num_ner_labels = 8

        if args.no_sym:
            num_labels = 7 + 7 - 1
        else:
            num_labels = 7 + 7 - 2
    elif args.data_dir.find('data')!=-1:
        num_ner_labels = 3

        if args.no_sym:
            num_labels = 4 + 4 - 1
        else:
            num_labels = 4 + 4 - 1
    elif args.data_dir.find('scierc')!=-1:
        num_ner_labels = 3

        if args.no_sym:
            num_labels = 4 + 4 - 1
        else:
            num_labels = 4 + 4 - 1
    else:
        assert (False)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab
    #处理数据集
    global train_data,dev_data,test_data
    train_, dev_, test_ = get_dataset(args.data_dir,args)
    args.model_type = args.model_type.lower()

    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]

    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path, num_labels=num_labels)
    tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path,  do_lower_case=args.do_lower_case)
    if args.use_typemarker:
        new_tokens = []
        for label in task_ner_labels['data']:
            new_tokens.append('<SUBJ_START=%s>' % label)
            new_tokens.append('<SUBJ_END=%s>' % label)
        tokenizer.add_tokens(new_tokens)

    args.tokenizer = tokenizer
    config.max_seq_length = args.max_seq_length
    config.alpha = args.alpha
    config.num_ner_labels = num_ner_labels

    model = model_class.from_pretrained(args.model_name_or_path, from_tf=bool('.ckpt' in args.model_name_or_path), config=config,args=args)
    model.resize_token_embeddings(len(tokenizer))

    if args.model_type.startswith('albert'):
        if args.use_typemarker:
            special_tokens_dict = {'additional_special_tokens': ['[unused' + str(x) + ']' for x in range(num_ner_labels*4+2)]}
        else:
            special_tokens_dict = {'additional_special_tokens': ['[unused' + str(x) + ']' for x in range(4)]}
        tokenizer.add_special_tokens(special_tokens_dict)
        # print ('add tokens:', tokenizer.additional_special_tokens)
        # print ('add ids:', tokenizer.additional_special_tokens_ids)
        model.albert.resize_token_embeddings(len(tokenizer))

    if args.do_train:
        subject_id = tokenizer.encode('subject', add_special_tokens=False)
        assert(len(subject_id)==1)
        subject_id = subject_id[0]
        object_id = tokenizer.encode('object', add_special_tokens=False)
        assert(len(object_id)==1)
        object_id = object_id[0]

        mask_id = tokenizer.encode('[MASK]', add_special_tokens=False)
        assert(len(mask_id)==1)
        mask_id = mask_id[0]

        logger.info(" subject_id = %s, object_id = %s, mask_id = %s", subject_id, object_id, mask_id)

        if args.lminit:
            if args.model_type.startswith('albert'):
                word_embeddings = model.albert.embeddings.word_embeddings.weight.data
                subs = 30000
                sube = 30001
                objs = 30002
                obje = 30003
            else:
                word_embeddings = model.bert.embeddings.word_embeddings.weight.data
                subs = 1
                sube = 2
                objs = 3
                obje = 4

            word_embeddings[subs].copy_(word_embeddings[mask_id])
            word_embeddings[sube].copy_(word_embeddings[subject_id])

            word_embeddings[objs].copy_(word_embeddings[mask_id])
            word_embeddings[obje].copy_(word_embeddings[object_id])

    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    model.to(args.device)

    logger.info("Training/evaluation parameters %s", args)
    best_f1 = 0
    # Training
    if args.do_train:
        # train_dataset = load_and_cache_examples(args,  tokenizer, evaluate=False)
        global_step, tr_loss, best_f1 = train(args, model, tokenizer,train_,dev_)
        # global_step, tr_loss, best_f1 = train(args, model, tokenizer,train_,test_)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    # Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()
    if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        # Create output directory if needed
        if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(args.output_dir)
        update = True
        if args.evaluate_during_training:
            results = evaluate(args,dev_, model, tokenizer)
            # results = evaluate(args,test_, model, tokenizer)
            f1 = results['f1_with_ner']
            if f1 > best_f1:
                best_f1 = f1
                print ('Best F1', best_f1)
            else:
                update = False

        if update:
            checkpoint_prefix = 'checkpoint'
            output_dir = os.path.join(args.output_dir, '{}-{}'.format(checkpoint_prefix, global_step))
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training

            model_to_save.save_pretrained(output_dir)

            torch.save(args, os.path.join(output_dir, 'training_args.bin'))
            logger.info("Saving model checkpoint to %s", output_dir)
            _rotate_checkpoints(args, checkpoint_prefix)

        tokenizer.save_pretrained(args.output_dir)

        torch.save(args, os.path.join(args.output_dir, 'training_args.bin'))


    # Evaluation
    results = {'dev_best_f1': best_f1}
    if args.do_eval and args.local_rank in [-1, 0]:

        checkpoints = [args.output_dir]

        WEIGHTS_NAME = 'pytorch_model.bin'

        if args.eval_all_checkpoints:
            checkpoints = list(os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + '/**/' + WEIGHTS_NAME, recursive=True)))

        logger.info("Evaluate the following checkpoints: %s", checkpoints)
        for checkpoint in checkpoints:
            global_step = checkpoint.split('-')[-1] if len(checkpoints) > 1 else ""

            model = model_class.from_pretrained(checkpoint, config=config,args=args)

            model.to(args.device)
            result = evaluate(args, test_,model, tokenizer, prefix=global_step, do_test=not args.no_test)
            result = dict((k + '_{}'.format(global_step), v) for k, v in result.items())
            results.update(result)
        print (results)

        if args.no_test:  # choose best resutls on dev set
            bestv = 0
            k = 0
            for k, v in results.items():
                if v > bestv:
                    bestk = k
            print (bestk)

        output_eval_file = os.path.join(args.output_dir, "results.json")
        json.dump(results, open(output_eval_file, "w"))


if __name__ == "__main__":
    os.environ["WANDB_API_KEY"] = 'fd6485f6c00464e97c16eb1243fdd48710f7fd93'  # 将引号内的+替换成自己在wandb上的一串值
    #os.environ["WANDB_MODE"] = "offline"  # 离线  （此行代码不用修改）
    # os.environ['WANDB_TIMEOUT'] = '300'
    main()
    wandb.finish()



