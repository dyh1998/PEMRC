import os
import torch
from transformers import AutoTokenizer, AutoModel
from d2l import torch as d2l
import numpy as np
import sys

sys.path.append('/gemini/code/utils')
from cal_distribution_match import kl_sim, euc_sim, cos_sim
from datasets import ASimplerDataset

"""
句子级别和实例级别
三种尺度标准：
1.欧式
2.余弦
3.KL散度
"""


# corpus = 'BC2GM'
# corpus = 'BC4CHEMD'
# corpus = 'BC5CDR-chem'
# corpus = 'BC5CDR-disease'
# corpus = 'JNLPBA'
# corpus = 'LINNAEUS'
# corpus = 'NCBI'
# # corpus = 'S800'
# # split = 'train'
# split = 'devel'
# np.seterr(divide='ignore', invalid='ignore')


def list_to_entity_span(token_index):
    index = []
    for i in token_index:
        index.append(i[0])

    entity_span = []
    span = []
    if len(index) == 1:
        entity_span.append([index[0]])
    else:
        for j in range(len(index) - 1):
            span.append(index[j])
            if index[j] == index[j + 1] - 1:
                # span.append(index[j])
                pass
            else:
                # span.append()
                entity_span.append(span)
                span = []
        if not span:
            entity_span.append([index[-1]])
        else:
            span.append(index[-1])
            entity_span.append(span)
    return entity_span


def writes(filename, tensor):
    count = 0
    with open(filename, 'w') as f:
        for i, j in tensor.items():
            count += 1
            if j is None:
                f.write(str(i) + ':' + ' None' + '\n')
            else:
                f.write(str(i) + ':' + ' ' + str(j) + '\n')
    print("源文件一共写入%d次数" % count)


def writes_(filename, tensor):
    with open(filename, 'w') as f:
        f.write(str(tensor))


class CorpusFilter:
    """
    计算语料中实例的平均向量，这个向量可以式entity平均，也可以是entity-token平均，也可以全部token平均，也可以是cls嵌入
    """

    def __init__(self, corpus, split, type, k_shot):
        self.device = d2l.try_gpu(0)

        self.type = type
        self.k_shot = k_shot
        self.model = AutoModel.from_pretrained('/gemini/pretrain')
        self.model.to(self.device)
        self.model.eval()
        # print(self.device)
        self.split = split
        self.corpus_name = corpus
        self.corpus = ASimplerDataset(corpus=corpus, split=split, type=self.type, k_shot=self.k_shot)
        print("len of corpus:", len(self.corpus))
        self.doc = self.corpus.corpus

        self.avg_cls_dict = {}
        self.avg_sent_dict = {}
        self.avg_entity_dict = {}
        self.avg_entity_token_dict = {}

        self.target_cls_proto = torch.zeros(1, 768)
        self.target_sent_proto = torch.zeros(1, 768)
        self.target_entity_proto = torch.zeros(1, 768)
        self.target_entity_token_proto = torch.zeros(1, 768)

    def return_avg_tensor_dict(self):
        all_entity_token_num = 0
        sent_num = 0
        all_entity_num = 0
        all_token_num = 0
        count = 0
        for idx in range(len(self.corpus)):
            count += 1
            inputs = self.corpus[idx]
            labels = inputs['labels']
            inputs = {
                'input_ids': torch.unsqueeze(inputs['input_ids'], 0),
                'attention_mask': torch.unsqueeze(inputs['attention_mask'], 0)
            }
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            self.model.train()
            self.model.config.hidden_dropout_prob = 0.1
            output = self.model(**inputs)
            logits = output[0]
            if self.k_shot != 0:
                self.model.config.hidden_dropout_prob = 0.3
                output1 = self.model(**inputs)
                logits1 = output1[0]
                self.model.config.hidden_dropout_prob = 0.5
                output2 = self.model(**inputs)
                logits2 = output2[0]
                self.model.config.hidden_dropout_prob = 0.7
                output3 = self.model(**inputs)
                logits3 = output3[0]
                self.model.config.hidden_dropout_prob = 0.9
                output4 = self.model(**inputs)
                logits4 = output4[0]
                logits = (logits + logits1 + logits2 + logits3 + logits4) / 5
            entity_token_index = torch.nonzero(labels).tolist()  # token_index = [[26], [27], [34], [35], [36], [37]]
            token_index = torch.nonzero(inputs['attention_mask']).tolist()
            if entity_token_index:
                entity_span = list_to_entity_span(
                    entity_token_index)  # [[26, 27], [34, 35, 36, 37]]                    c
            else:
                entity_span = []
            entity_num = len(entity_span)  # 句子中实体的span——token位置
            entity_token_num = len(entity_token_index)  # 句子中实体token的个数
            token_num = token_index[-1][-1]  # 句子中所有token的个数

            sent_num += 1
            all_entity_num += entity_num
            all_entity_token_num += entity_token_num
            all_token_num += (token_num - 2)

            # cls embedding
            cls_embedding = logits[:, 0, :].detach().cpu()
            self.target_cls_proto += cls_embedding

            # sent embedding
            sent_proto = torch.zeros(1, 768)
            for i in range(1, token_num - 1):
                token_embedding = logits[:, i, :].detach().cpu()
                sent_proto += token_embedding
                self.target_sent_proto += token_embedding
            sent_proto /= token_num

            if entity_token_index:
                # entity embedding
                sent_entity_proto = torch.zeros(1, 768)
                for span in entity_span:
                    span_len = len(span)
                    entity_avg_embedding = torch.zeros(1, 768)
                    for i in span:
                        token_embedding = logits[:, i, :].detach().cpu()
                        entity_avg_embedding += token_embedding
                    entity_avg_embedding /= span_len

                    sent_entity_proto += entity_avg_embedding
                    self.target_entity_proto += entity_avg_embedding
                sent_entity_proto /= entity_num

                # entity_toke n_embedding
                entity_token_proto = torch.zeros(1, 768)
                for i in entity_token_index:
                    token_embedding = logits[:, i[0], :].detach().cpu()
                    entity_token_proto += token_embedding
                    self.target_entity_token_proto += token_embedding
                entity_token_proto /= entity_token_num
            else:
                sent_entity_proto = None
                entity_token_proto = None

            self.avg_cls_dict[idx] = cls_embedding
            self.avg_sent_dict[idx] = sent_proto
            self.avg_entity_dict[idx] = sent_entity_proto
            self.avg_entity_token_dict[idx] = entity_token_proto
        print("原型计算次数:", count)
        if self.type == 'source':
            return self.avg_cls_dict, self.avg_sent_dict, \
                   self.avg_entity_token_dict, self.avg_entity_dict
        elif self.type == 'target':
            self.target_entity_proto = self.target_entity_proto / all_entity_num
            self.target_cls_proto = self.target_cls_proto / sent_num
            self.target_sent_proto = self.target_sent_proto / all_token_num
            self.target_entity_token_proto = self.target_entity_token_proto / all_entity_token_num

            return self.target_cls_proto, self.target_sent_proto, \
                   self.target_entity_token_proto, self.target_entity_proto

    def write_to_file(self):
        if self.type == "source":
            dir = '../data/NER_tensor/' + self.corpus_name + '/' + self.split
            cls, sent, entity_token, entity = self.return_avg_tensor_dict()
            print("嵌入长度%d" % len(cls))
        else:
            dir = '../data/NER_tensor/' + self.corpus_name + '/' + str(self.k_shot) + '-shot'
            cls_proto, sent_proto, entity_token_proto, entity_proto = self.return_avg_tensor_dict()

        if not os.path.exists(dir):
            os.makedirs(dir)

        if self.type == 'source':
            cls_filename = dir + '/cls' + '.tsv'
            sent_filename = dir + '/sent' + '.tsv'
            entity_token_filename = dir + '/entity_token' + '.tsv'
            entity_filename = dir + '/entity' + '.tsv'

            writes(cls_filename, cls)
            writes(sent_filename, sent)
            writes(entity_token_filename, entity_token)
            writes(entity_filename, entity)
        else:
            cls_filename = dir + '/cls_proto' + '.tsv'
            sent_filename = dir + '/sent_proto' + '.tsv'
            entity_token_filename = dir + '/entity_token_proto' + '.tsv'
            entity_filename = dir + '/entity_proto' + '.tsv'

            writes_(cls_filename, cls_proto)
            writes_(sent_filename, sent_proto)
            writes_(entity_token_filename, entity_token_proto)
            writes_(entity_filename, entity_proto)


'''
获取每个数据集少样本的原型表征并写回文件中
'''
