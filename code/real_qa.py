import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
import os
import torch
import random
import logging
import warnings
import sys

sys.path.append('..')
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# os.environ['CUDA_LAUNCH_BLOCKING'] = '5'
import numpy as np
from tqdm import tqdm
from torch import nn
from d2l import torch as d2l
from model_args import BoundaryExtractionArgs
from transformers import AutoModel, AutoConfig, AutoTokenizer, \
    BertModel, BertConfig, BertTokenizer, BertTokenizerFast, \
    RobertaModel, RobertaConfig, RobertaTokenizer, PreTrainedTokenizerFast, \
    BertPreTrainedModel, AdamW, BertForTokenClassification, AutoModelForTokenClassification
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from torch.nn.functional import cross_entropy, relu
import torch.nn.functional as F
from models import *
from losses.R_drop import MRCRDrop as rloss
from analyses import recover_input

warnings.filterwarnings("ignore", category=DeprecationWarning)

MODEL_CLASSES = {
    "auto": (AutoConfig, AutoModel, AutoTokenizer),
    "bert": (BertConfig, BertModel, BertTokenizer),
    "roberta": (RobertaConfig, RobertaModel, RobertaTokenizer)
}
ENTITY_QUERIES = {
    'disease': 'disease',
    'drug': 'drug',
    'gene': 'gene',
    'species': 'species'
}
# MIX_QUERY = {
#     'disease': 'v1 ',
#     'drug': 'v2 ',
#     'gene': 'v3 ',
#     'species': 'v4 '
# }
ENTITY_DESC = {
    "disease": 'A disease is an unhealthy state where something bad happens to the body or mind. ',
    'species': 'a species is the largest group of organisms.',
    'gene': 'gene is a basic unit of heredity or a sequence of nucleotides in DNA.',
    'drug': "A drug is any chemical substance that causes a change in an organism's physiology or psychology when consumed."
}
SOFT_QUERIES_1 = {
    'disease': 'v1',
    'drug': 'v2',
    'gene': 'v3',
    'species': 'v4'
}
SOFT_QUERIES_3 = {
    'disease': 'v1 v2 v3',
    'drug': 'v4 v5 v6',
    'gene': 'v7 v8 v9',
    'species': 'v10 v11 v12'
}
SOFT_QUERIES_5 = {
    'disease': 'v1 v2 v3 v4 v5',
    'drug': 'v6 v7 v8 v9 v10',
    'gene': 'v11 v12 v13 v14 v15',
    'species': 'v16 v17 v18 v19 v20'
}
SOFT_QUERIES_7 = {
    'disease': 'v1 v2 v3 v4 v5 v6 v7',
    'drug': 'v8 v9 v10 v11 v12 v13 v14',
    'gene': 'v15 v16 v17 v18 v19 v20 v21',
    'species': 'v22 v23 v24 v25 v26 v27 v28'
}
SOFT_QUERIES_9 = {
    'disease': 'v1 v2 v3 v4 v5 v6 v7 v8 v9',
    'drug': 'v10 v11 v12 v13 v14 v15 v16 v17 v18',
    'gene': 'v19 v20 v21 v22 v23 v24 v25 v26 v27',
    'species': 'v28 v29 v30 v31 v32 v33 v34 v35 v36'
}

QUERY_LEN = {
    '1': SOFT_QUERIES_1,
    '3': SOFT_QUERIES_3,
    '5': SOFT_QUERIES_5,
    '7': SOFT_QUERIES_7,
    '9': SOFT_QUERIES_9,
}
SENT_QUERY = "find {} entities in the next sentence ."
PRE_QUERY = "find entities, such as "
LAST_QUERY = ", in the next sentence."

LOCAL_PATH = {
    'bert': '/gemini/code/cache/bert',
    'biobert': '/gemini/pretrain',
    'pubmed': '/gemini/code/cache/pubmed',
}

LOSS_FUNC = {
    'cross_entropy': nn.CrossEntropyLoss(),
    'rdrop': RDrop(),
    'focal_loss': FocalLoss()
}

CORPORA_CLASS2NAME = {
    "disease": ['NCBI', 'BC5CDR-disease'],
    "drug": ['BC5CDR-chem', 'BC4CHEMD'],
    "gene": ['JNLPBA', 'BC2GM'],
    "species": ['LINNAEUS', 'S800']
}

CORPORA_NAME2CLASS = {
    'NCBI': "disease",
    'BC5CDR-disease': "disease",
    'BC5CDR-chem': "drug",
    'BC4CHEMD': "drug",
    'JNLPBA': "gene",
    'BC2GM': "gene",
    'LINNAEUS': "species",
    'S800': "species",
}

CORPORA_CLASS2LABEL = {
    'NCBI': "tumour",
    'BC5CDR-disease': "tumour",
    'BC5CDR-chem': "selegiline",
    'BC4CHEMD': "aflatoxin",
    'JNLPBA': "E1A gene",
    'BC2GM': "E2 protein",
    'LINNAEUS': "yeast",
    'S800': "human",
}

SIM_FUNC = {
    "kl": kl_sim,
    "cos": cos_sim,
    "euc": euc_sim
}

logger = logging.getLogger(__name__)

# TODO:setting abstract dataset base class
tokenizer = AutoTokenizer.from_pretrained('/gemini/pretrain', use_fast=True)


class QADatasets(Dataset):
    """
    一个通用的实体边界检测数据集类
    要求输入文件格式如conll2003所示，即单词-标签一一对应
    """

    def __init__(self, mode='train', split='train', max_length=512, padding='max_length', args=None):
        """
        :param mode: train is source domain, test is target domain,用于区分源领域和目标领域
        :param split: 数据集划分
        :param max_length: 数据集最大长度
        :param padding:
        :param args:
        """
        self.args = args
        self.split = split
        try:
            self.tokenizer = AutoTokenizer.from_pretrained('/gemini/pretrain')
            self.max_length = self.args.max_length
            # 基于通用预训练的方法
            if mode == 'target':
                if self.split == 'train':
                    # 随机k-shot
                    self.filename = '/gemini/code/data/NERdata/' + self.args.target_dataset + '/' + str( \
                        self.args.few_shot) + '-shot/train' + str(self.args.manual_seed) + '.tsv'
                elif self.split == 'test':
                    self.filename = '/gemini/code/data/NERdata/' + self.args.target_dataset + '/' + 'test.tsv'
            if mode == 'source':
                self.filename = '/gemini/code/data/NERdata/' + self.args.source_dataset + '/train.tsv'
            self.mode = mode
        except Exception as e:
            print("CommonDatasetsForBoundaryDetection object has no attribute 'args'!")
            os._exit()
        print("filename:", self.filename)
        self.corpus = self.read_corpus()
        self.real_corpus = self.ensemble_query_and_sent()
        self.examples = self.get_input_tensor_label()

    def __getitem__(self, item):
        ins = self.examples[item]
        token, label, token_ids = ins['token'], ins['label'], ins['ids']
        attention_mask = [1 for _ in range(len(token_ids))]
        token_type_ids = [0 for _ in range(len(token_ids))]

        assert len(attention_mask) == len(token) == len(label) == len(token_ids)
        input_ids = torch.tensor(token_ids)
        token_type_ids = torch.tensor(token_type_ids)
        attention_mask = torch.tensor(attention_mask)

        pre, cur, last = 0, 0, 0
        start_index, end_index = [0], [0]
        for i in range(1, len(label) - 1):
            pre = label[i - 1]
            cur = label[i]
            last = label[i + 1]
            if cur == 1:
                if last == 2:
                    start_index.append(1)
                    end_index.append(0)
                else:
                    start_index.append(1)
                    end_index.append(1)
            elif cur == 2 and last != 2:
                start_index.append(0)
                end_index.append(1)
            else:
                start_index.append(0)
                end_index.append(0)

        start_index.append(0)
        end_index.append(0)

        assert len(start_index) == len(end_index) == len(label)
        start_index = torch.tensor(start_index)
        end_index = torch.tensor(end_index)

        if attention_mask.shape[0] >= self.max_length:
            input_ids = input_ids[:self.max_length]
            token_type_ids = token_type_ids[:self.max_length]
            attention_mask = attention_mask[:self.max_length]
            start_index = start_index[:self.max_length]
            end_index = end_index[:self.max_length]
        else:
            need_pad = self.max_length - attention_mask.shape[0]
            input_ids_pad = torch.full([need_pad], self.tokenizer.pad_token_id, dtype=torch.long)
            pad = torch.full([need_pad], 0, dtype=torch.long)

            input_ids = torch.cat((input_ids, input_ids_pad), dim=0)
            attention_mask = torch.cat((attention_mask, pad), dim=0)
            token_type_ids = torch.cat((token_type_ids, pad), dim=0)
            start_index = torch.cat((start_index, pad), dim=0)
            end_index = torch.cat((end_index, pad), dim=0)

        inputs = {
            'input_ids': input_ids,
            'token_type_ids': token_type_ids,
            'attention_mask': attention_mask,
            'start_index': start_index,
            'end_index': end_index,
        }
        return inputs

    def __len__(self):
        return len(self.examples)

    def ensemble_query_and_sent(self):
        real_corpus = []
        # ds_domain = ""
        dsname = ''

        # 判断当前数据集所属的类别
        if self.mode == 'source':
            dsname = self.args.source_dataset
        else:
            dsname = self.args.target_dataset

        ds_domain = CORPORA_NAME2CLASS[dsname]
        # count = 0
        # 对不同了领域的数据集使用不同的label
        for ins in self.corpus:
            sent, tag = ins
            # for domain in ['disease', 'drug', 'gene', 'species']:
            query = "1 2 3 4 "
            if self.args.query_type == 'hard':
                query = SENT_QUERY.format(ENTITY_QUERIES[ds_domain])
                query = query.split()
                query_tag = [0 for i in range(len(query))]
            elif self.args.query_type == 'mix':
                soft_query = QUERY_LEN[self.args.soft_len]
                query = SENT_QUERY.format(soft_query[ds_domain])
                query = query.split()
                query_tag = [0 for i in range(len(query))]
            elif self.args.query_type == 'soft':
                SOFT_QUERIES = QUERY_LEN[self.args.soft_len]
                query = SOFT_QUERIES[ds_domain]
                query = query.split()
                query_tag = [0 for i in range(len(query))]

            elif self.args.query_type == 'proto':
                pre_query = PRE_QUERY.split()
                last_query = LAST_QUERY.split()
                label = CORPORA_CLASS2LABEL[dsname].split()
                query = pre_query + label + last_query
                query_tag = [0 for _ in range(len(pre_query))] + \
                            [1 for _ in range(len(label))] + \
                            [0 for _ in range(len(last_query))]
            if self.args.query_type != 'None':
                new_sent = query + sent
                new_tag = query_tag + tag
            else:
                new_sent = sent
                new_tag = tag

            real_corpus.append((new_sent, new_tag))
        return real_corpus

    def read_corpus(self):
        with open(self.filename, 'r') as f:
            lines = f.readlines()
        sentence = []
        tag = []
        doc = []
        for line in lines:
            if len(line) > 2:
                line = line.strip()
                line = line.split()
                if line[-1] != 'O':
                    if line[-1].startswith('B'):
                        tag.append(1)
                    elif line[-1].startswith('I'):
                        if self.args.num_labels == 3:  # 这里考虑是io标注还是bio标注方式
                            tag.append(2)
                        elif self.args.num_labels == 2:
                            tag.append(1)
                else:
                    tag.append(0)
                sentence.append(line[0])
            else:
                doc.append((sentence, tag))
                sentence, tag = [], []
        return doc

    def get_input_tensor_label(self):
        examples = []
        count = 0
        for instance in self.real_corpus:
            if instance[0]:
                example = {}
                real_token, real_label, real_span = [], [], []
                start_index, end_index = 0, 0
                instance[0].insert(0, '[CLS]')
                instance[1].insert(0, 0)
                instance[0].append('[SEP]')
                instance[1].append(0)

                for word, word_label in zip(instance[0], instance[1]):
                    token = self.tokenizer.tokenize(word)
                    if len(token) > 1 and word_label == 1:
                        real_label.append(word_label)
                        real_label = real_label + [2 for _ in range(len(token) - 1)]
                    elif len(token) == 1:
                        real_label.append(word_label)
                    else:
                        real_label = real_label + [word_label for _ in range(len(token))]
                    real_token = real_token + token

                real_ids = tokenizer.convert_tokens_to_ids(real_token)
                example['token'] = real_token
                example['label'] = real_label
                example['ids'] = real_ids
            examples.append(example)
        return examples


class Packer(Dataset):
    """
    将多个数据集合并起来
    """

    def __init__(self, args):
        self.ds = []
        for arg in args:
            for i in range(len(arg)):
                self.ds.append(arg[i])

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, item):
        return self.ds[item]


class QABIODatasets(Dataset):
    """
    一个通用的实体边界检测数据集类
    要求输入文件格式如conll2003所示，即单词-标签一一对应
    """

    def __init__(self, mode='train', split='train', max_length=512, padding='max_length', args=None):
        """
        :param mode: train is source domain, test is target domain,用于区分源领域和目标领域
        :param split: 数据集划分
        :param max_length: 数据集最大长度
        :param padding:
        :param args:
        """
        self.args = args
        self.split = split
        try:
            self.tokenizer = AutoTokenizer.from_pretrained('/gemini/code/cache/biobert')
            self.max_length = self.args.max_length
            # 基于通用预训练的方法
            if mode == 'target':
                if self.split == 'train':
                    # 随机k-shot
                    self.filename = '/gemini/code/data/NERdata/' + self.args.target_dataset + '/' + str( \
                        self.args.few_shot) + '-shot/' + 'train.tsv'
                elif self.split == 'test':
                    self.filename = '/gemini/code/data/NERdata/' + self.args.target_dataset + '/' + 'test.tsv'
            if mode == 'source':
                self.filename = '/gemini/code/data/NERdata/' + self.args.source_dataset + '/train.tsv'
            self.mode = mode
        except Exception as e:
            self.args.logger.info("CommonDatasetsForBoundaryDetection object has no attribute 'args'!")
            os._exit()
        print("filename:", self.filename)
        self.corpus = self.read_corpus()
        self.real_corpus = self.ensemble_query_and_sent()
        self.examples = self.get_input_tensor_label()

    def __getitem__(self, item):
        ins = self.examples[item]
        word, token, label, token_ids, span = ins['origin_token'], ins['token'], ins['label'], ins['ids'], ins['span']
        attention_mask = [1 for _ in range(len(token_ids))]
        token_type_ids = [0 for _ in range(len(token_ids))]
        # print(len(attention_mask), len(token), len(label), len(token_ids))
        assert len(attention_mask) == len(token) == len(label) == len(token_ids)
        assert len(word) == len(span)
        input_ids = torch.tensor(token_ids)
        token_type_ids = torch.tensor(token_type_ids)
        attention_mask = torch.tensor(attention_mask)
        label = torch.tensor(label)

        if attention_mask.shape[0] >= self.max_length:
            input_ids = input_ids[:self.max_length]
            token_type_ids = token_type_ids[:self.max_length]
            attention_mask = attention_mask[:self.max_length]
            label = label[:self.max_length]
        else:
            need_pad = self.max_length - attention_mask.shape[0]
            input_ids_pad = torch.full([need_pad], self.tokenizer.pad_token_id, dtype=torch.long)
            pad = torch.full([need_pad], 0, dtype=torch.long)

            input_ids = torch.cat((input_ids, input_ids_pad), dim=0)
            attention_mask = torch.cat((attention_mask, pad), dim=0)
            token_type_ids = torch.cat((token_type_ids, pad), dim=0)
            label = torch.cat((label, pad), dim=0)

        inputs = {
            'input_ids': input_ids,
            'token_type_ids': token_type_ids,
            'attention_mask': attention_mask,
            'labels': label
        }
        return inputs

    def __len__(self):
        return len(self.examples)

    def ensemble_query_and_sent(self):
        real_corpus = []
        # ds_domain = ""
        dsname = ''

        # 判断当前数据集所属的类别
        if self.mode == 'source':
            dsname = self.args.source_dataset
        else:
            dsname = self.args.target_dataset

        ds_domain = CORPORA_NAME2CLASS[dsname]
        # count = 0
        # 对不同了领域的数据集使用不同的label
        for ins in self.corpus:
            sent, tag = ins
            # for domain in ['disease', 'drug', 'gene', 'species']:
            query = "1 2 3 4 "
            if self.args.query_type == 'hard':
                query = SENT_QUERY.format(ENTITY_QUERIES[ds_domain])
                query = query.split()
                query_tag = [0 for i in range(len(query))]
            elif self.args.query_type == 'mix':
                soft_query = QUERY_LEN[self.args.soft_len]
                query = SENT_QUERY.format(soft_query[ds_domain])
                query = query.split()
                query_tag = [0 for i in range(len(query))]
            elif self.args.query_type == 'soft':
                SOFT_QUERIES = QUERY_LEN[self.args.soft_len]
                query = SOFT_QUERIES[ds_domain]
                query = query.split()
                query_tag = [0 for i in range(len(query))]

            elif self.args.query_type == 'proto':
                pre_query = PRE_QUERY.split()
                last_query = LAST_QUERY.split()
                label = CORPORA_CLASS2LABEL[dsname].split()
                query = pre_query + label + last_query
                query_tag = [0 for _ in range(len(pre_query))] + \
                            [1 for _ in range(len(label))] + \
                            [0 for _ in range(len(last_query))]
            if self.args.query_type != 'None':
                new_sent = query + sent
                new_tag = query_tag + tag
            else:
                new_sent = sent
                new_tag = tag

            real_corpus.append((new_sent, new_tag))
        return real_corpus

    def read_corpus(self):
        with open(self.filename, 'r') as f:
            lines = f.readlines()
        sentence = []
        tag = []
        doc = []
        for line in lines:
            if len(line) > 2:
                line = line.strip()
                line = line.split()
                if line[-1] != 'O':
                    if line[-1].startswith('B'):
                        tag.append(1)
                    elif line[-1].startswith('I'):
                        if self.args.num_labels == 3:  # 这里考虑是io标注还是bio标注方式
                            tag.append(2)
                        elif self.args.num_labels == 2:
                            tag.append(1)
                else:
                    tag.append(0)
                sentence.append(line[0])
            else:
                doc.append((sentence, tag))
                sentence, tag = [], []
        return doc

    def get_input_tensor_label(self):
        examples = {}
        count = 0
        for instance in self.real_corpus:
            if instance[0]:
                example = {}
                real_token, real_label, real_span = [], [], []
                start_index, end_index = 0, 0
                instance[0].insert(0, '[CLS]')
                instance[1].insert(0, 0)
                instance[0].append('[SEP]')
                instance[1].append(0)
                for word, word_label in zip(instance[0], instance[1]):
                    start_index = end_index
                    token = self.tokenizer.tokenize(word)
                    real_token = real_token + token
                    end_index = start_index + len(token)
                    real_span.append((start_index, end_index))
                    if self.args.num_labels == 2:  # 对bio和io进行处理
                        real_label = real_label + [word_label for _ in range(len(token))]
                    elif self.args.num_labels == 3:
                        if len(token) > 1:  # 对B-Disease的单词进行token标签的划分
                            if word_label == 1:
                                real_label.append(1)
                                real_label += [2 for _ in range(len(token) - 1)]
                            else:
                                real_label += [word_label for _ in range(len(token))]
                        else:
                            real_label.append(word_label)
                real_ids = tokenizer.convert_tokens_to_ids(real_token)
                example['origin_token'], example['token'], example['label'], example['ids'], example[
                    'span'] = instance[0], real_token, real_label, real_ids, real_span
                examples[count] = example
                count += 1
        return examples


"""
模型架构：
1、BERT+CRF
2、BERT+BiLSTM+CRF
3、BERT+FNN
# TODO：BERT+BiLSTM+FNN
        BERT+CNN+FNN
"""


class FGM():
    '''
    Example
    # 初始化
    fgm = FGM(model,epsilon=1,emb_name='word_embeddings.')
    for batch_input, batch_label in data:
        # 正常训练
        loss = model(batch_input, batch_label)
        loss.backward() # 反向传播，得到正常的grad
        # 对抗训练
        fgm.attack() # 在embedding上添加对抗扰动
        loss_adv = model(batch_input, batch_label)
        loss_adv.backward() # 反向传播，并在正常的grad基础上，累加对抗训练的梯度
        fgm.restore() # 恢复embedding参数
        # 梯度下降，更新参数
        optimizer.step()
        model.zero_grad()
    '''

    def __init__(self, model, emb_name, epsilon=1.0):
        # emb_name这个参数要换成你模型中embedding的参数名
        self.model = model
        self.epsilon = epsilon
        self.emb_name = emb_name
        self.backup = {}

    def attack(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = self.epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


class PGD():
    '''
    Example
    pgd = PGD(model,emb_name='word_embeddings.',epsilon=1.0,alpha=0.3)
    K = 3
    for batch_input, batch_label in data:
        # 正常训练
        loss = model(batch_input, batch_label)
        loss.backward() # 反向传播，得到正常的grad
        pgd.backup_grad()
        # 对抗训练
        for t in range(K):
            pgd.attack(is_first_attack=(t==0)) # 在embedding上添加对抗扰动, first attack时备份param.data
            if t != K-1:
                model.zero_grad()
            else:
                pgd.restore_grad()
            loss_adv = model(batch_input, batch_label)
            loss_adv.backward() # 反向传播，并在正常的grad基础上，累加对抗训练的梯度
        pgd.restore() # 恢复embedding参数
        # 梯度下降，更新参数
        optimizer.step()
        model.zero_grad()
    '''

    def __init__(self, model, emb_name, epsilon=1., alpha=0.3):
        # emb_name这个参数要换成你模型中embedding的参数名
        self.model = model
        self.emb_name = emb_name
        self.epsilon = epsilon
        self.alpha = alpha
        self.emb_backup = {}
        self.grad_backup = {}

    def attack(self, is_first_attack=False):
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name:
                if is_first_attack:
                    self.emb_backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0:
                    r_at = self.alpha * param.grad / norm
                    param.data.add_(r_at)
                    param.data = self.project(name, param.data, self.epsilon)

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name:
                assert name in self.emb_backup
                param.data = self.emb_backup[name]
        self.emb_backup = {}

    def project(self, param_name, param_data, epsilon):
        r = param_data - self.emb_backup[param_name]
        if torch.norm(r) > epsilon:
            r = epsilon * r / torch.norm(r)
        return self.emb_backup[param_name] + r

    def backup_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                self.grad_backup[name] = param.grad.clone()

    def restore_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                param.grad = self.grad_backup[name]


class SpanExtraction:
    """
    A Universal BoundaryExtractor.
    model_architecture: Bert_series + CRF
    """

    def __init__(self, encoder_type=None, encoder_name=None, config=None, args=None, **kwargs):
        logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                            datefmt='%m/%d/%Y %H:%M:%S',
                            level=logging.INFO)
        self.logger = logging.getLogger('yhdong')

        self.args = BoundaryExtractionArgs()
        self.args.load('bert-base-uncased')  # 从本地文件中加载参数
        if isinstance(args, dict):
            self.args.update_from_dict(args)
            self.logger.info("Update model_args from dict finished!")
        for key, value in args.items():
            self.logger.info("%s: %s" % (key, str(value)))

        if self.args.manual_seed:
            torch.manual_seed(self.args.manual_seed)
            np.random.seed(self.args.manual_seed)
            random.seed(self.args.manual_seed)
            torch.cuda.manual_seed_all(self.args.manual_seed)

        if self.args.use_cache:
            self.logger.info("Using cache!")
        if self.args.model == 'QA':
            self.model = BertQA(encoder_type=encoder_type, encoder_name=encoder_name, num_labels=self.args.num_labels,
                                args=self.args)

        if self.args.model == "FNN":
            self.model = BertFFN(encoder_type=encoder_type, encoder_name=encoder_name, num_labels=self.args.num_labels,
                                 args=self.args)

        *_, self.tokenizer = MODEL_CLASSES[encoder_type]
        self.loss_func = LOSS_FUNC[self.args.loss]  # 损失函数
        self.encoder_type = encoder_type
        self.encoder_name = encoder_name
        self.f1 = 0.0
        self.p = 0.0
        self.r = 0.0

        self.len_test = 0

    def train_model(self):
        # # TODO: 参数在训练阶段获取，以便更好的使用学习率调度
        # # TODO: 把训练函数整合为一个
        # # TODO: 目前得到数据迭代器的方法不太优雅
        specify = str(self.args.model) + "_" + CORPORA_NAME2CLASS[self.args.target_dataset] + '_' + str(
            self.args.query_type) + "_" + str(self.args.soft_len)

        def check_and_mkdir(specify, st):
            filename = specify + st
            filepath = os.path.join(self.args.cache_path + filename)
            if not os.path.isfile(filepath):
                self.args.use_cache = False
                with open(filepath, 'w') as f:
                    f.write('123')
            else:
                self.args.use_cache = True
            return filepath

        source_path = check_and_mkdir(specify, self.args.source_output_dir)
        target_path = check_and_mkdir(specify, self.args.target_output_dir)
        self.args.source_output_dir = source_path
        self.args.target_output_dir = target_path

        if not self.args.use_cache:
            source_dataset = self.args.source_dataset

        if not self.args.source_dataset:
            self.logger.info("In domain setting.")
            pre_train, pre_test = self.make_dataset(self.args.target_dataset, domain='target')
            target_train = DataLoader(pre_train, shuffle=True, batch_size=self.args.test_batch_size, drop_last=False)
            target_test = DataLoader(pre_test, shuffle=False, batch_size=self.args.test_batch_size, drop_last=False)
        else:
            self.logger.info("Domain adaption setting.")
            if not self.args.use_cache:
                source_train = self.make_dataset(self.args.source_dataset, domain='source')
                source_train = DataLoader(source_train, shuffle=True, batch_size=self.args.train_batch_size,
                                          drop_last=False)

            target_train, target_test = self.make_dataset(self.args.target_dataset, domain='target')
            target_train = DataLoader(target_train, shuffle=True, batch_size=self.args.test_batch_size, drop_last=False)
            self.target_test = DataLoader(target_test, shuffle=False, batch_size=self.args.test_batch_size,
                                          drop_last=False)

        ##################################Pretraining#############################
        if self.args.source_dataset:
            self.logger.info("即将开始training")

            if self.args.use_cache != True:
                if self.args.model == 'QA':
                    self.pre_train_with_QA(source_train)
                if self.args.model == 'FNN':
                    self.pre_train_with_FNN(source_train)
            if self.args.model == 'QA':
                self.train_with_QA(target_train, self.target_test)
            if self.args.model == 'FNN':
                self.train_with_FNN(target_train, self.target_test)
        else:
            self.train_with_QA(target_train, self.target_test)

    def pre_train_with_FNN(self, train):
        model = self.model
        model.to(self.args.device)
        model.train()
        best_result = 0.0
        best_epoch = 0
        ffn_params = list(model.ffn.named_parameters())
        bert_params = list(model.model.named_parameters())
        optimizer_grouped_parameters = [{'params': [p for n, p in bert_params], 'lr': self.args.train_bert_lr,
                                         'weight_decay': self.args.weight_decay},
                                        {'params': [p for n, p in ffn_params]}]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.args.train_lr)
        for i in range(self.args.train_epoch):
            model.train()
            self.logger.info("---------------------------Training: epoch_%d--------------------" % (i + 1))
            for index, batch in enumerate(tqdm(train)):
                optimizer.zero_grad()
                inputs = {
                    'input_ids': batch['input_ids'],
                    'attention_mask': batch['attention_mask'],
                }
                inputs = {k: v.to(self.args.device) for k, v in inputs.items()}
                labels = batch['labels'].to(torch.int64)
                labels = labels.to(self.args.device)
                pred = model(inputs)
                pred = pred.permute(0, 2, 1)
                loss = self.loss_func(pred, labels)

                if index % 500 == 499:
                    self.logger.info("epoch: %d, index: %d, loss: %.5f" % (i + 1, index, loss))
                # if self.args.few_shot != -1:
                #     self.logger.info("index: %d, loss: %.5f" % (index, loss))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        self.save_model(model, self.args.source_output_dir)

    def train_with_FNN(self, train, test):
        if self.args.source_dataset:
            model = self.load_model(self.args.source_output_dir)
        else:
            model = self.model
        model.to(self.args.device)
        model.train()
        best_result = 0.0
        best_epoch = 0
        ffn_params = list(model.ffn.named_parameters())
        bert_params = list(model.model.named_parameters())
        optimizer_grouped_parameters = [{'params': [p for n, p in bert_params], 'lr': self.args.test_bert_lr,
                                         'weight_decay': self.args.weight_decay},
                                        {'params': [p for n, p in ffn_params]}]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.args.train_lr)
        for i in range(self.args.test_epoch):
            model.train()
            self.logger.info("---------------------------Training: epoch_%d--------------------" % (i + 1))
            train_loss = 0
            for index, batch in enumerate(tqdm(train)):
                optimizer.zero_grad()
                inputs = {
                    'input_ids': batch['input_ids'],
                    'attention_mask': batch['attention_mask'],
                }
                inputs = {k: v.to(self.args.device) for k, v in inputs.items()}
                labels = batch['labels'].to(torch.int64)
                labels = labels.to(self.args.device)
                pred = model(inputs)
                pred = pred.permute(0, 2, 1)
                loss = self.loss_func(pred, labels)
                train_loss += loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            self.save_model(model, self.args.target_output_dir)
            self.logger.info("train loss: %s" % train_loss)
        f1, p, r = self.evaluate_with_FNN(test)
        self.f1, self.p, self.r = f1, p, r
        self.logger.info("Epoch: %d, F1: %.5f, P: %.5f ,R: %.5f" % (i + 1, f1, p, r))

    def evaluate_with_FNN(self, test):
        self.logger.info("-----------------------validating:------------------------")
        model = self.load_model(self.args.target_output_dir)
        model.to(self.args.device)
        model.eval()
        correct = 0
        total_gold = 0
        total_pred = 0

        for index, batch in enumerate(tqdm(test)):
            inputs = {
                'input_ids': batch['input_ids'],
                'attention_mask': batch['attention_mask'],
            }
            inputs = {k: v.to(self.args.device) for k, v in inputs.items()}
            pred = model(inputs)
            pred = torch.argmax(pred, dim=-1)
            attention_mask, pred, labels = batch['attention_mask'].tolist(), pred.tolist(), batch['labels'].tolist()
            result = self.cal_correct_gold_pred(attention_mask, labels, pred)
            correct += result[0]
            total_gold += result[1]
            total_pred += result[2]

        self.logger.info("CORRECT:%d, total_gold:%d, total_pred:%d" % (correct, total_gold, total_pred))
        f1, p, r = self.cal_prf(correct, total_gold, total_pred)
        self.logger.info("F1:%.5f, P:%.5f, R:%.5f" % (f1, p, r))
        return f1, p, r

    def cal_correct_gold_pred(self, attention_mask, labels, pred):
        correct, total_gold, total_pred = 0, 0, 0
        for i in range(len(attention_mask)):
            p = pred[i]
            label = labels[i]
            length = sum(attention_mask[i])
            pred_span = self.result2span_list(p[:length])
            label_span = self.result2span_list(label[:length])
            total_pred += len(pred_span)
            total_gold += len(label_span)
            for m in pred_span:
                for n in label_span:
                    if m[0] == n[0] and m[1] == n[1]:
                        correct += 1
                        break
        return correct, total_gold, total_pred

    def result2span_list(self, label_list):
        span_list = []
        start_index = 0
        end_index = 0
        if self.args.model == 'tempalte':
            for l in range(1, len(label_list)):
                if label_list[l] == tokenizer.convert_token_to_ids('entity'):
                    if label_list[l - 1] != tokenizer.convert_token_to_ids('entity'):
                        start_index = l
                    else:
                        end_index = l

                if label_list[l] != tokenizer.convert_token_to_ids('entity'):
                    if label_list[l - 1] == tokenizer.convert_token_to_ids('entity'):
                        end_index = l
                        if end_index > start_index:
                            span_list.append((start_index, end_index))

        if self.args.num_labels == 3:
            for l in range(1, len(label_list)):
                if label_list[l] == 1:
                    if label_list[l - 1] == 0:
                        start_index = l
                    elif label_list[l - 1] == 2:
                        end_index = l
                        span_list.append((start_index, end_index))
                        start_index = l
                    else:
                        pass
                if label_list[l] == 2:
                    pass
                if label_list[l] == 0:
                    if label_list[l - 1] != 0:
                        end_index = l
                        span_list.append((start_index, end_index))
                    else:
                        pass

        elif self.args.num_labels == 2:
            for l in range(1, len(label_list)):
                if label_list[l] == 1:
                    if label_list[l - 1] == 0:
                        start_index = l
                    else:
                        end_index = l
                if label_list[l] == 0:
                    if label_list[l - 1] == 1:
                        end_index = l
                        if end_index > start_index:
                            span_list.append((start_index, end_index))

        return span_list

    def pre_train_with_QA(self, train):
        model = self.model
        model.to(self.args.device)
        model.train()
        test_result = 0
        best_epoch = 0
        ffn1_params = list(model.ffn1.named_parameters())
        ffn2_params = list(model.ffn2.named_parameters())
        bert_params = list(model.model.named_parameters())

        optimizer_grouped_parameters = [{'params': [p for n, p in bert_params], 'lr': self.args.train_bert_lr,
                                         'weight_decay': self.args.weight_decay},
                                        {'params': [p for n, p in ffn1_params]},
                                        {'params': [p for n, p in ffn2_params]}]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.args.train_lr)

        for i in range(self.args.train_epoch):
            self.logger.info("--------------------Pretraining:%d------------------" % (i + 1))
            for index, batch in enumerate(tqdm(train)):
                inputs = {
                    'input_ids': batch['input_ids'],
                    'attention_mask': batch['attention_mask'],
                }

                start_index = batch['start_index'].cuda()
                end_index = batch['end_index'].cuda()

                inputs = {k: v.to(self.args.device) for k, v in inputs.items()}
                start_logits, end_logits = self.model(inputs)

                start_logits = start_logits.view(-1, 2)
                end_logits = end_logits.view(-1, 2)

                shape = start_logits.shape
                start_index = start_index.view(shape[0])
                end_index = end_index.view(shape[0])

                start_loss = self.loss_func(start_logits, start_index)
                end_loss = self.loss_func(end_logits, end_index)

                loss = start_loss + end_loss

                if index % 500 == 499:
                    self.logger.info("index: %d, loss: %.9f" % (index, loss))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        self.save_model(self.model, self.args.source_output_dir)

    def train_with_QA(self, train, test):
        if self.args.source_dataset:
            model = self.load_model(self.args.source_output_dir)
        else:
            model = self.model
        model.to(self.args.device)
        model.train()
        # for name, params in model.named_parameters():
        #     print(name, params.shape)
        test_result = 0
        best_epoch = 0
        ffn1_params = list(model.ffn1.named_parameters())
        ffn2_params = list(model.ffn2.named_parameters())
        bert_params = list(model.model.named_parameters())
        optimizer_grouped_parameters = [{'params': [p for n, p in bert_params], 'lr': self.args.test_bert_lr,
                                         'weight_decay': self.args.weight_decay},
                                        {'params': [p for n, p in ffn1_params]},
                                        {'params': [p for n, p in ffn2_params]}]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.args.test_lr)

        for i in range(self.args.test_epoch):
            self.logger.info("--------------------fine-tune:%d------------------" % (i + 1))
            train_loss = 0
            if self.args.attack:
                if self.args.attack == "FGM":
                    fgm = FGM(model, epsilon=1, emb_name='word_embeddings')
                elif self.args.attack == "PGD":
                    pgd = PGD(model, emb_name='word_embeddings', epsilon=1.0, alpha=0.3)
                    K = 3
            for index, batch in enumerate(tqdm(train)):
                inputs = {
                    'input_ids': batch['input_ids'],
                    'attention_mask': batch['attention_mask'],
                }

                start_index = batch['start_index'].cuda()
                end_index = batch['end_index'].cuda()

                inputs = {k: v.to(self.args.device) for k, v in inputs.items()}
                start_logits, end_logits = model(inputs)
                if self.args.span_loss:
                    # print("span loss.")
                    span_loss = self.get_span_loss(start_index, end_index, start_logits, end_logits)
                if self.args.kl:
                    start_logits1, end_logits1 = model(inputs)
                    loss = self.get_kl_loss(start_logits, start_logits1, end_logits, \
                                            end_logits1, start_index, end_index, kl_weight=self.args.kl)
                    train_loss += loss.mean()
                    loss.mean().backward()
                else:
                    start_logits = start_logits.view(-1, 2)
                    end_logits = end_logits.view(-1, 2)

                    shape = start_logits.shape
                    start_index = start_index.view(shape[0])
                    end_index = end_index.view(shape[0])

                    start_loss = self.loss_func(start_logits, start_index)
                    end_loss = self.loss_func(end_logits, end_index)

                    loss = start_loss + end_loss
                    train_loss += loss
                    if self.args.span_loss:
                        loss += span_loss * 0.01
                    loss.backward()

                if self.args.attack:
                    if self.args.attack == 'FGM':
                        fgm.attack()
                        ################ 对抗损失 ################
                        start_logits, end_logits = model(inputs)
                        if self.args.span_loss:
                            start_index = batch['start_index'].cuda()
                            end_index = batch['end_index'].cuda()

                            adv_span_loss = self.get_span_loss(start_index, end_index, start_logits, end_logits)

                        start_logits = start_logits.view(-1, 2)
                        end_logits = end_logits.view(-1, 2)

                        shape = start_logits.shape
                        start_index = start_index.view(shape[0])
                        end_index = end_index.view(shape[0])

                        adv_start_loss = self.loss_func(start_logits, start_index)
                        adv_end_loss = self.loss_func(end_logits, end_index)
                        adv_loss = adv_start_loss + adv_end_loss
                        if self.args.span_loss:
                            adv_loss += adv_span_loss * 0.01

                        adv_loss.backward()
                        fgm.restore()
                    elif self.args.attack == "PGD":
                        pgd.backup_grad()
                        for t in range(K):
                            pgd.attack(is_first_attack=(t == 0))
                            if t != K - 1:
                                model.zero_grad()
                            else:
                                pgd.restore_grad()
                            ################ 对抗损失 ################
                            start_logits, end_logits = model(inputs)
                            if self.args.span_loss:
                                start_index = batch['start_index'].cuda()
                                end_index = batch['end_index'].cuda()

                                adv_span_loss = self.get_span_loss(start_index, end_index, start_logits, end_logits)

                            start_logits = start_logits.view(-1, 2)
                            end_logits = end_logits.view(-1, 2)

                            shape = start_logits.shape
                            start_index = start_index.view(shape[0])
                            end_index = end_index.view(shape[0])

                            adv_start_loss = self.loss_func(start_logits, start_index)
                            adv_end_loss = self.loss_func(end_logits, end_index)
                            adv_loss = adv_start_loss + adv_end_loss

                            if self.args.span_loss:
                                adv_loss += adv_span_loss * 0.01
                            adv_loss.backward()
                        pgd.restore()
                optimizer.step()
                model.zero_grad()
            self.save_model(model, self.args.target_output_dir)
            self.logger.info("train loss: %.9f" % train_loss)

        f1, p, r = self.evaluate_with_QA(test)

        self.f1, self.p, self.r = f1, p, r
        if self.args.test_epoch == 0:
            pass
        else:
            self.logger.info("Epoch: %d, F1: %.5f, P: %.5f ,R: %.5f" % (i + 1, f1, p, r))

    def evaluate_with_QA(self, test):
        self.logger.info("-----------------------validating:------------------------")
        if self.args.test_epoch == 0:
            model = self.load_model(self.args.source_output_dir)
        else:
            model = self.load_model(self.args.target_output_dir)
        model.to(self.args.device)
        model.eval()
        correct = 0
        total_gold = 0
        total_pred = 0
        for index, batch in enumerate(tqdm(test)):
            inputs = {
                'input_ids': batch['input_ids'],
                'attention_mask': batch['attention_mask'],
            }
            inputs = {k: v.to(self.args.device) for k, v in inputs.items()}
            start_logits, end_logits = model(inputs)
            start_index_pred = torch.argmax(start_logits, dim=-1).cpu().tolist()
            end_index_pred = torch.argmax(end_logits, dim=-1).cpu().tolist()
            pred_span = self.start_end_to_span(start_index_pred, end_index_pred)

            start_index = batch['start_index'].tolist()
            end_index = batch['end_index'].tolist()

            real_span = self.start_end_to_span(start_index, end_index)
            cor, gold, pred = self.cal_cor_gold_pred(pred_span, real_span)

            correct += cor
            total_gold += gold
            total_pred += pred

        if self.args.query_type == 'proto':
            correct -= self.len_test
            total_pred -= self.len_test
            total_gold -= self.len_test
        self.logger.info("CORRECT:%d, total_gold:%d, total_pred:%d" % (correct, total_gold, total_pred))
        f1, p, r = self.cal_prf(correct, total_gold, total_pred)
        self.logger.info("F1:%.5f, P:%.5f, R:%.5f" % (f1, p, r))
        return f1, p, r

    def get_kl_loss(self, start_logits, start_logits1, end_logits, \
                    end_logits1, start_index, end_index, kl_weight=1):
        rdrop = rloss()
        kl_loss = rdrop(start_logits, start_logits1, end_logits, \
                        end_logits1, start_index, end_index, kl_weight=1)
        return kl_loss

    def kl_loss(self, logits1, logits2):
        kld = nn.KLDivLoss(reduction='none')
        kl_loss1 = kld(F.log_softmax(logits1, dim=-1), F.softmax(logits2, dim=-1)).sum(-1)
        kl_loss2 = kld(F.log_softmax(logits2, dim=-1), F.softmax(logits1, dim=-1)).sum(-1)
        kl_loss = (kl_loss1 + kl_loss2) / 2
        return kl_loss

    def get_span_loss(self, start_index, end_index, start_logits, end_logits):
        start_index_ls = start_index.tolist()
        end_index_ls = end_index.tolist()
        batch_span = self.start_end_to_span(start_index_ls, end_index_ls)
        count = -1
        train_loss = 0
        num_tokens = 0
        for sent_span in batch_span:
            count += 1
            for span in sent_span:
                start_idx, end_idx = span
                if end_idx == None or start_idx == None:
                    continue
                num_tokens += (end_idx - start_idx + 1) * 2

                start_logit = start_logits[count, start_idx:end_idx + 1, :]
                end_logit = end_logits[count, start_idx:end_idx + 1, :]

                start_gold = start_index[count, start_idx:end_idx + 1]
                end_gold = end_index[count, start_idx:end_idx + 1]

                span_loss = self.loss_func(start_logit, start_gold) + self.loss_func(end_logit, end_gold)
                train_loss += span_loss

        return train_loss / num_tokens

    def cal_cor_gold_pred(self, pred_span, real_span):
        cor, gold, pred = 0, 0, 0
        for rs in real_span:
            gold += len(rs)

        for i in range(len(pred_span)):
            ps = pred_span[i]
            rs = real_span[i]
            for p in ps:
                for r in rs:
                    if p[0] == r[0] and p[1] == r[1]:
                        cor += 1
                        break
                pred += 1
        return cor, gold, pred

    def start_end_to_span(self, start, end):
        # print("start: ", start)
        # print("end: ", end)
        batch_span = []
        for i in range(len(start)):
            start_sent = start[i]
            end_sent = end[i]
            span = []
            start_index = []
            end_index = []
            # print("start sent: ", start_sent)
            for index in range(len(start_sent)):
                if start_sent[index] == 1:
                    start_index.append(index)

            for index in start_index:
                flag = 0
                temp = 0
                for j in range(index, len(end_sent)):
                    if end_sent[j] == 1:
                        flag = 1
                        temp = j
                        break
                if flag == 1:
                    end_index.append(temp)
                else:
                    end_index.append(None)

            for i, j in zip(start_index, end_index):
                span.append((i, j))
            batch_span.append(span)
        return batch_span

    def desc_case_study(self):
        target_train, target_test = self.make_dataset(self.args.target_dataset, domain='target')
        # target_train = DataLoader(target_train, shuffle=True, batch_size=self.args.test_batch_size, drop_last=False)
        target_test = DataLoader(target_test, shuffle=False, batch_size=1,
                                      drop_last=False)
        path = "/gemini/code/cache/"
        hard_name = "QA_disease_hard_9_target_best_model.pth"
        mix_name = "QA_disease_mix_5_target_best_model.pth"
        soft_name = "QA_disease_soft_5_target_best_model.pth"

        hard_path = path + hard_name
        mix_path = path + mix_name
        soft_path = path + soft_name

        hard_model = self.load_model(hard_path)
        mix_model = self.load_model(mix_path)
        soft_model = self.load_model(soft_path)

        hard_model.to(self.args.device)
        mix_model.to(self.args.device)
        soft_model.to(self.args.device)

        hard_model.eval()
        mix_model.eval()
        soft_model.eval()

        correct = 0
        total_gold = 0
        total_pred = 0
        for index, batch in enumerate(tqdm(target_test)):
            inputs = {
                'input_ids': batch['input_ids'],
                'attention_mask': batch['attention_mask'],
            }
            start_index = batch['start_index'].tolist()
            end_index = batch['end_index'].tolist()

            inputs = {k: v.to(self.args.device) for k, v in inputs.items()}
            mix_start_logits, mix_end_logits = mix_model(inputs)
            hard_start_logits, hard_end_logits = hard_model(inputs)
            soft_start_logits, soft_end_logits = soft_model(inputs)

            hard_start_index_pred = torch.argmax(hard_start_logits, dim=-1).cpu().tolist()
            hard_end_index_pred = torch.argmax(hard_end_logits, dim=-1).cpu().tolist()

            mix_start_index_pred = torch.argmax(mix_start_logits, dim=-1).cpu().tolist()
            mix_end_index_pred = torch.argmax(mix_end_logits, dim=-1).cpu().tolist()

            soft_start_index_pred = torch.argmax(soft_start_logits, dim=-1).cpu().tolist()
            soft_end_index_pred = torch.argmax(soft_end_logits, dim=-1).cpu().tolist()

            # hard_pred_span = self.start_end_to_span(hard_start_index_pred, hard_end_index_pred)
            # soft_pred_span = self.start_end_to_span(soft_start_index_pred, soft_end_index_pred)
            # mix_pred_span = self.start_end_to_span(mix_start_index_pred, mix_end_index_pred)
            recover_input(batch['input_ids'], batch['attention_mask'], start_index, end_index, type="correct")
            recover_input(batch['input_ids'], batch['attention_mask'], hard_start_index_pred, hard_end_index_pred,
                          type="hard")
            recover_input(batch['input_ids'], batch['attention_mask'], soft_start_index_pred, soft_end_index_pred,
                          type="soft")
            recover_input(batch['input_ids'], batch['attention_mask'], mix_start_index_pred, mix_end_index_pred,
                          type="mix")

    def make_dataset(self, datasets, domain='source'):
        if domain == 'target':
            mode = domain
            if self.args.model == "QA":
                train = QADatasets(mode=mode, split='train', max_length=self.args.max_length, padding='max_length',
                                   args=self.args)
                test = QADatasets(mode=mode, split='test', max_length=self.args.max_length, padding='max_length',
                                  args=self.args)
                self.len_test = len(test)
            elif self.args.model == 'FNN':
                train = QABIODatasets(mode=mode, split='train', max_length=self.args.max_length, padding='max_length',
                                      args=self.args)
                test = QABIODatasets(mode=mode, split='test', max_length=self.args.max_length, padding='max_length',
                                     args=self.args)
            return train, test

        source_datasets = []
        for dsname in datasets:
            self.args.source_dataset = dsname
            mode = 'source'
            if self.args.use_cache != True:
                if self.args.model == "QA":
                    train = QADatasets(mode=mode, split='train', max_length=self.args.max_length, padding='max_length',
                                       args=self.args)
                elif self.args.model == 'FNN':
                    train = QABIODatasets(mode=mode, split='train', max_length=self.args.max_length,
                                          padding='max_length',
                                          args=self.args)
                source_datasets.append(train)

        return Packer(source_datasets)

    def cal_prf(self, correct, total_gold, total_pred):
        p = correct / total_pred if correct > 0 else 0.0
        r = correct / total_gold if correct > 0 else 0.0
        f1 = 2 * p * r / (p + r) * 100 if correct else 0.0
        return f1, p * 100, r * 100

    def save_model(self, model, dir):
        torch.save(model, dir)
        # torch.cuda.empty_cache()  # 清空显卡缓存区
        self.logger.info("Model saved to %s!" % (str(dir)))

    def load_model(self, dir):
        model = torch.load(dir)
        model.to(self.args.device)
        self.logger.info("Model loaded from %s" % (str(dir)))
        return model


if __name__ == '__main__':
    """
    encoder_type:transformers库的模型类型，例如auto，bert，roberta
    encoder_name:transformer库的模型名称或者是本地模型，biobert,bert,....
    datasets:BC2GM,BC4,BC5CDR-chem,BC5CDR-disease,JNLPBA,linnaeus,NCBI-disease,s800
    corpora_class2name = {
        "disease":['NCBI', 'BC5CDR-disease'],
        "drug":['BC5CDR-chem', 'BC4CHEMD'],
        "gene":['JNLPBA', 'BC2GM'],
        "species":['LINNAEUS', 'S800']
    }
    """
    import sys

    from utils.random_sample import *

    # from utils.cal_distribution_match import *
    # from utils.corpus_filter import *

    # def return_top5(result):
    #     min_f1 = 100
    #     t = 0
    #     for i in range(len(result)):
    #         j = result[i]
    #         f1 = j[1]
    #         if f1 < min_f1:
    #             min_f1 = f1
    #             t = i
    #     del result[t]
    #     return result

    result = []
    model_args = {
        'manual_seed': 43,  # 41,42,43,44,45
        'max_length': 256,
        'num_labels': 3,  # 3：bio，2：io
        'time_stamp': 3,
        'train_epoch': 1,  #

        'test_lr': 1e-4,
        'test_bert_lr': 1e-4,  # 这里的学习率不能太大，最好时3e-5这个量级的
        'test_crf_lr': 1e-2,

        'train_lr': 1e-4,
        'train_bert_lr': 1e-5,  # 这里的学习率不能太大，最好时3e-5这个量级的
        'train_crf_lr': 1e-2,
        'weight_decay': 0.0,
        'train_batch_size': 8,
        'test_batch_size': 2,
        'few_shot': 50,  # 为-1时自动使用全监督设置
        'loss': 'cross_entropy',  # rdrop
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        # 'device': d2l.try_gpu(7),
        'tokenizer': AutoTokenizer.from_pretrained('/gemini/pretrain'),
        'dropout': 0.1,
        'source_dataset': [],
        # 'source_dataset': ["LINNAEUS", 'S800'],
        'source_dataset': ["BC5CDR-chem", 'BC4CHEMD', 'JNLPBA', 'BC2GM', "LINNAEUS", 'S800'],
        # 'source_dataset': ["BC5CDR-disease", 'NCBI', 'JNLPBA', 'BC2GM', "LINNAEUS", 'S800'],
        # 'source_dataset': ["BC5CDR-disease", 'NCBI', 'BC5CDR-chem', 'BC4CHEMD', "LINNAEUS", 'S800'],
        # 'source_dataset': ["BC5CDR-disease", 'NCBI', 'BC5CDR-chem', 'BC4CHEMD', "JNLPBA", 'BC2GM'],

        # disease_source, chem_source, gene_source, species_source
        # 'target_dataset': "BC5CDR-chem",
        # 'target_dataset': "BC2GM",
        # 'target_dataset': "BC5CDR-disease",
        'target_dataset': "NCBI",
        # 'target_dataset': "BC2GM",
        'cache_path': '/gemini/code/cache/',
        'source_output_dir': '_source_best_model.pth',
        'target_output_dir': '_target_best_model.pth',
        'use_cache': False,
        "model": "QA",  # FNN QA
        'span_loss': False,
        "query_type": "hard",  # hard,soft,proto,None
        "soft_len": '9',
        "attack": "",  # PGD,FGM
        "sampling": True,
        'test_epoch': 10,
        'kl': False,
        # 少样本会导致结果的精确率很高，召回率很低，
        # 原因是少样本的情况下，找不到对应的实例，但是找到的实例一般都是正确的
    }
    model = SpanExtraction(encoder_type='auto', encoder_name='biobert', args=model_args)
    model.desc_case_study()
    # if model_args['sampling']:
    #     for few in [5]:
    #         model_args['few_shot'] = few
    #         # for seed in [0, 1, 2, 3, 4]:
    #         for seed in [40]:
    #             model_args['manual_seed'] = seed
    #             # precise_random_sampling(model_args['target_dataset'], 'train', model_args['few_shot'],
    #             #                         model_args['manual_seed'])
    #             # standard_N_way_K_shot_sampling(model_args['target_dataset'], 'train', model_args['few_shot'],
    #             #                                model_args['manual_seed'])
    #             N_sentences_K_shot_sampling(model_args['target_dataset'], 'train', model_args['few_shot'],
    #                                         model_args['manual_seed'])
    #
    # # 基于通用的方法
    # for few in [5]:
    #     model_args['few_shot'] = few
    #     result = []
    #     for seed in [40]:
    #         model_args['manual_seed'] = seed
    #         model = SpanExtraction(encoder_type='auto', encoder_name='biobert', args=model_args)
    #         model.train_model()
    #
    #         f1, p, r = model.f1, model.p, model.r
    #         result.append((seed, f1, p, r))
    #         print("This round of training is over. We're about to clear the cache and start the next round")
    #         for i in range(10):
    #             torch.cuda.empty_cache()  # 释放显存
    #
    #     avg_f1, avg_p, avg_r = 0, 0, 0
    #
    #     for i in result:
    #         avg_f1 += i[1]
    #         avg_p += i[2]
    #         avg_r += i[3]
    #     avg_p /= len(result)
    #     avg_r /= len(result)
    #     square_bias = 0.0
    #     avg_f1 = 2 * (avg_p * avg_r) / (avg_p + avg_r)
    #
    #     for i in result:
    #         square_bias += pow((avg_f1 - i[1]), 2)
    #     square_bias /= len(result)
    #     square_bias = pow(square_bias, 0.5)
    #
    #     with open('result.txt', 'a') as f:
    #         f.write(str(model_args['few_shot']) + ": " + str(round(avg_f1, 2)) + ' ' + str(round(avg_p, 2)) +
    #                 " " + str(round(avg_r, 2)) + ' ' + str(round(square_bias, 2)) + '\n')
    #     print(round(avg_f1, 2), round(avg_p, 2), round(avg_r, 2), round(square_bias, 2))
