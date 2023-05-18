import os
import torch
import random
import logging
import warnings

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
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
from TorchCRF import CRF
from torch.nn.functional import cross_entropy, relu
import torch.nn.functional as F
from utils.datasets import CommonDatasetsForNERBoundaryDetection as ds
from utils.datasets import TemplateFreeDataset as tfd
from utils.datasets import QADatasets as qa
# from utils.datasets import Packer as P
from losses.R_drop import RDrop
from losses.focal_loss import FocalLoss
from utils.random_sample import *
from utils.cal_distribution_match import *
from utils.corpus_filter import *
from models import *

warnings.filterwarnings("ignore", category=DeprecationWarning)

MODEL_CLASSES = {
    "auto": (AutoConfig, AutoModel, AutoTokenizer),
    "bert": (BertConfig, BertModel, BertTokenizer),
    "roberta": (RobertaConfig, RobertaModel, RobertaTokenizer)
}
LOCAL_PATH = {
    'bert': '/gemini/pretrain/bert-base-uncased',
    'biobert': '/gemini/pretrain',
    'roberta': '../predata/roberta-base',
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

SIM_FUNC = {
    "kl": kl_sim,
    "cos": cos_sim,
    "euc": euc_sim
}

logger = logging.getLogger(__name__)

"""
模型架构：
1、BERT+CRF
2、BERT+BiLSTM+CRF
3、BERT+FNN
# TODO：BERT+BiLSTM+FNN
        BERT+CNN+FNN
"""


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
        if self.args.with_bilstm and self.args.with_CRF:
            self.model = BertBiLSTMCRF(encoder_type=encoder_type, encoder_name=encoder_name,
                                       num_labels=self.args.num_labels, args=self.args)
        elif self.args.with_CRF:
            self.model = BertCRF(encoder_type=encoder_type, encoder_name=encoder_name, num_labels=self.args.num_labels,
                                 args=self.args, dropout=self.args.dropout)
        elif self.args.model == 'pre_mlp':
            self.model = PTuning(encoder_type=encoder_type, encoder_name=encoder_name, num_labels=self.args.num_labels,
                                 args=self.args)
        elif self.args.model == 'template':
            self.model = MaskedLM(encoder_type=encoder_type, encoder_name=encoder_name, num_labels=self.args.num_labels,
                                  args=self.args)
        elif self.args.model == 'QA':
            self.model = BertQA(encoder_type=encoder_type, encoder_name=encoder_name, num_labels=self.args.num_labels,
                                args=self.args)
        *_, self.tokenizer = MODEL_CLASSES[encoder_type]
        self.loss_func = LOSS_FUNC[self.args.loss]  # 损失函数
        self.encoder_type = encoder_type
        self.encoder_name = encoder_name
        self.f1 = 0.0
        self.p = 0.0
        self.r = 0.0

    def train_model(self):
        source_train = []
        source_dev = []
        target_train = []
        target_dev = []
        target_test = []

        # # TODO: 参数在训练阶段获取，以便更好的使用学习率调度
        # # TODO: 把训练函数整合为一个
        # # TODO: 目前得到数据迭代器的方法不太优雅
        if not self.args.use_cache:
            source_dataset = self.args.source_dataset

        target_dataset = self.args.target_dataset
        if not self.args.source_dataset:
            self.logger.info("In domain setting.")
            pre_train, pre_valid, pre_test = self.make_dataset(self.args.target_dataset, domain='target')
            target_train = DataLoader(pre_train, shuffle=True, batch_size=self.args.test_batch_size, drop_last=False)
            target_test = DataLoader(pre_test, shuffle=False, batch_size=self.args.test_batch_size, drop_last=False)

        else:
            self.logger.info("Domain adaption setting.")
            if not self.args.use_cache:
                source_train, source_dev = self.make_dataset(self.args.source_dataset, domain='source')
                source_train = DataLoader(source_train, shuffle=True, batch_size=self.args.train_batch_size,
                                          drop_last=False)
                source_dev = DataLoader(source_dev, shuffle=False, batch_size=self.args.train_batch_size,
                                        drop_last=False)

            target_train, target_dev, target_test = self.make_dataset(self.args.target_dataset, domain='target')

            target_train = DataLoader(target_train, shuffle=True, batch_size=self.args.test_batch_size,
                                      drop_last=False)
            target_dev = DataLoader(target_dev, shuffle=False, batch_size=self.args.test_batch_size, drop_last=False)
            target_test = DataLoader(target_test, shuffle=False, batch_size=self.args.test_batch_size, drop_last=False)

        ##################################Pretraining#############################
        if self.args.source_dataset:
            self.logger.info("即将开始training")
            if self.args.with_CRF == True:
                if not self.args.use_cache:
                    self.pre_train_with_CRF(source_train, source_dev)
                self.train_with_CRF(target_train, source_train, target_test)
            elif self.args.model == 'pre_mlp':
                self.pre_train_with_PTuning(source_train, source_dev)
                self.train_with_PTuning(target_train, target_dev, target_test)
            elif self.args.model == 'template':
                if self.args.use_cache != True:
                    self.pre_train_with_MaskedLM(source_train, source_dev)
                self.train_with_MaskedLM(target_train, target_dev, target_test)
            elif self.args.model == "QA":
                if self.args.use_cache != True:
                    self.pre_train_with_QA(source_train, source_dev)
                self.train_with_QA(target_train, target_dev, target_test)

    def pre_train_with_QA(self, train, dev):
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

                if index % 20 == 19:
                    self.logger.info("index: %d, loss: %.5f" % (index, loss))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        self.save_model(self.model, self.args.source_output_dir)

    def train_with_QA(self, train, dev, test):
        model = self.load_model(self.args.source_output_dir)
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

        for i in range(self.args.test_epoch):
            self.logger.info("--------------------fine-tune:%d------------------" % (i + 1))
            train_loss = 0
            for index, batch in enumerate(tqdm(train)):
                inputs = {
                    'input_ids': batch['input_ids'],
                    'attention_mask': batch['attention_mask'],
                }
                start_index = batch['start_index'].cuda()
                end_index = batch['end_index'].cuda()

                inputs = {k: v.to(self.args.device) for k, v in inputs.items()}
                start_logits, end_logits = model(inputs)

                start_logits = start_logits.view(-1, 2)
                end_logits = end_logits.view(-1, 2)

                shape = start_logits.shape
                # print("shape:", shape)
                start_index = start_index.view(shape[0])
                end_index = end_index.view(shape[0])

                start_loss = self.loss_func(start_logits, start_index)
                end_loss = self.loss_func(end_logits, end_index)

                loss = start_loss + end_loss
                train_loss += loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            self.save_model(model, self.args.target_output_dir)
            self.logger.info("loss: %.5f" % loss)
            f1, p, r = self.evaluate_with_QA(test)
            self.logger.info("Epoch: %d, F1: %.5f, P: %.5f ,R: %.5f" % (i + 1, f1, p, r))

    def evaluate_with_QA(self, test):
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

        self.logger.info("CORRECT:%d, total_gold:%d, total_pred:%d" % (correct, total_gold, total_pred))
        f1, p, r = self.cal_prf(correct, total_gold, total_pred)
        self.logger.info("F1:%.5f, P:%.5f, R:%.5f" % (f1, p, r))
        return f1, p, r

    def cal_cor_gold_pred(self, pred_span, real_span):
        cor, gold, pred = 0, 0, 0
        for ps, rs in (pred_span, real_span):
            for p in ps:
                start = p[0]
                end = p[1]
                flag = 0
                for r in rs:
                    if start == r[0] and end == r[1]:
                        flag = 1
                        break
                if flag == 1:
                    cor += 1
                pred += 1
            gold += len(rs)
        return cor, gold, pred

    def start_end_to_span(self, start, end):
        batch_span = []

        for start_sent, end_sent in (start, end):
            count = 0
            span = []
            start_index = []
            end_index = []

            for index in range(len(start_sent)):
                if start_sent[index] == 1:
                    start_index.append(index)

            len_end_index = 0
            for index in start_index:
                len_end_index += 1
                for j in range(index, len(end_sent)):
                    if end_sent[j] == 1:
                        end_index.append(j)
                        break
                if len(end_index) == (len_end_index - 1):
                    end_index.append(None)

            # print("start_index:", start_index)
            # print("end_index:", end_index)

            for i, j in zip(start_index, end_index):
                span.append((i, j))
            batch_span.append(span)

        return batch_span

    def pre_train_with_MaskedLM(self, train, dev):
        model = self.model
        model.to(self.args.device)
        model.train()
        test_result = 0
        best_epoch = 0
        ffn_params = list(model.ffn.named_parameters())
        bert_params = list(model.model.named_parameters())
        optimizer_grouped_parameters = [{'params': [p for n, p in bert_params], 'lr': self.args.train_bert_lr,
                                         'weight_decay': self.args.weight_decay},
                                        {'params': [p for n, p in ffn_params]}]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.args.train_lr)

        for i in range(self.args.train_epoch):
            self.logger.info("--------------------Pretraining:%d------------------" % (i + 1))
            for index, batch in enumerate(tqdm(train)):
                inputs = {
                    'input_ids': batch['input_ids'],
                    'attention_mask': batch['attention_mask'],
                }

                inputs = {k: v.to(self.args.device) for k, v in inputs.items()}
                tags = batch['labels'].to(self.args.device)
                pred = self.model(inputs)
                pred = pred.view(-1, self.model.config.vocab_size)
                shape = pred.shape
                tags = tags.view(shape[0])
                loss = self.loss_func(pred, tags)

                if index % 20 == 19:
                    self.logger.info("index: %d, loss: %.5f" % (index, loss))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        self.save_model(self.model, self.args.source_output_dir)

    def train_with_MaskedLM(self, train_loader, valid_loader, test_loader):
        if self.args.fine_tune:
            model = self.load_model(self.args.source_output_dir)
        else:
            model = self.model
        model.cuda()
        model.train()
        test_result = 0
        best_epoch = 0
        bert_params = list(model.model.named_parameters())
        ffn_params = list(model.ffn.named_parameters())
        optimizer_grouped_parameters = [{'params': [p for n, p in bert_params], 'lr': self.args.test_bert_lr},
                                        {'params': [p for n, p in ffn_params]}]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.args.test_lr)
        labels_tensor = []
        non_labels_tensor = []
        for i in range(self.args.test_epoch):
            self.logger.info("--------------------fine-tuning:%d------------------" % (i + 1))
            train_loss = 0
            labels_tensor = []
            non_labels_tensor = []
            for index, batch in enumerate(tqdm(train_loader)):
                inputs = {
                    'input_ids': batch['input_ids'],
                    'attention_mask': batch['attention_mask'],
                }
                inputs = {k: v.to(self.args.device) for k, v in inputs.items()}
                tags = batch['labels'].to(self.args.device)
                pred = model(inputs)
                pred = pred.view(-1, model.config.vocab_size)
                shape = pred.shape
                tags = tags.view(shape[0])
                loss = self.loss_func(pred, tags)
                train_loss += loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            self.logger.info("train loss: %s" % train_loss)
            self.save_model(model, self.args.target_output_dir)
            f1, p, r = self.evaluate_with_MaskedLM(test_loader)
            self.logger.info("Epoch: %d, F1: %.5f, P: %.5f ,R: %.5f" % (i + 1, f1, p, r))
        self.save_model(model, self.args.target_output_dir)
        f1, p, r = self.evaluate_with_MaskedLM(test_loader)
        self.logger.info("F1: %.5f, P: %.5f ,R: %.5f" % (f1, p, r))
        self.p, self.r, self.f1 = p, r, f1

    def evaluate_with_MaskedLM(self, test_loader):
        self.logger.info("-----------------------validating:------------------------")
        model = self.load_model(self.args.target_output_dir)
        model.cuda()
        model.eval()
        correct = 0
        total_gold = 0
        total_pred = 0
        for index, batch in enumerate(tqdm(test_loader)):
            inputs = {
                'input_ids': batch['input_ids'],
                'attention_mask': batch['attention_mask'],
            }
            inputs = {k: v.to(self.args.device) for k, v in inputs.items()}
            tags = batch['labels'].to(self.args.device)
            pred = model(inputs)
            pred = torch.argmax(pred, dim=-1)
            attention_mask, labels = batch['attention_mask'].tolist(), batch['labels'].tolist()
            pred = pred.tolist()
            result = self.cal_correct_gold_pred(attention_mask, labels, pred)
            correct += result[0]
            total_gold += result[1]
            total_pred += result[2]
        self.logger.info("CORRECT:%d, total_gold:%d, total_pred:%d" % (correct, total_gold, total_pred))
        f1, p, r = self.cal_prf(correct, total_gold, total_pred)
        self.logger.info("F1:%.5f, P:%.5f, R:%.5f" % (f1, p, r))
        return f1, p, r

    def train_with_CL(self, train_loader, valid_loader, test_loader):
        model = BERTForContrastiveLearningForTokenMetric(encoder_type='auto', encoder_name='biobert',
                                                         num_labels=self.args.num_labels, args=self.args)
        model.cuda()
        model.train()
        test_result = 0
        best_epoch = 0
        bert_params = list(model.model.named_parameters())
        optimizer_grouped_parameters = [{'params': [p for n, p in bert_params]}]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.args.train_lr)
        labels_tensor = []
        non_labels_tensor = []
        for i in range(self.args.test_epoch):
            self.logger.info("--------------------fine-tuning:%d------------------" % (i + 1))
            train_loss = 0
            labels_tensor = []
            non_labels_tensor = []
            for index, batch in enumerate(tqdm(train_loader)):
                inputs = {
                    'input_ids': batch['input_ids'],
                    'attention_mask': batch['attention_mask'],
                }
                inputs = {k: v.to(self.args.device) for k, v in inputs.items()}
                tags = batch['labels'].to(self.args.device)
                logits, loss, label_tensor, non_label_tensor = model(inputs, tags)
                labels_tensor += label_tensor
                non_labels_tensor += non_label_tensor
                train_loss += loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            self.logger.info("train loss: %s" % train_loss)
            # self.save_model(model, self.args.target_output_dir)
            # f1, p, r = self.evaluate_with_PTuning(test_loader)
            # self.logger.info("Epoch: %d, F1: %.5f, P: %.5f ,R: %.5f" % (i + 1, f1, p, r))
        self.save_model(model, self.args.target_output_dir)
        f1, p, r = self.evaluate_with_MaskedLM(test_loader, labels_tensor, non_labels_tensor)
        self.logger.info("F1: %.5f, P: %.5f ,R: %.5f" % (f1, p, r))

    def evaluate_with_CL(self, test_loader, label_tensor, non_label_tensor):

        self.logger.info("-----------------------validating:------------------------")
        model = self.load_model(self.args.target_output_dir)
        model.cuda()
        model.eval()
        print(len(label_tensor))
        print(len(non_label_tensor))
        correct = 0
        total_gold = 0
        total_pred = 0
        for index, batch in enumerate(tqdm(test_loader)):
            inputs = {
                'input_ids': batch['input_ids'],
                'attention_mask': batch['attention_mask'],
            }
            inputs = {k: v.to(self.args.device) for k, v in inputs.items()}
            tags = batch['labels'].to(self.args.device)
            logits = model(inputs)
            shape = logits.shape
            pred = torch.zeros(shape[0], shape[1])
            # print("开始计算标签...")
            for i in range(shape[0]):
                sent_embed = logits[i, :, :]
                pred[i, :] = self.return_label(sent_embed, label_tensor, non_label_tensor)
            # print("标签计算结束!")
            pred = pred.cpu().tolist()
            tags = tags.cpu().tolist()
            attention_mask = batch['attention_mask']
            attention_mask = attention_mask.tolist()
            result = self.cal_correct_gold_pred(attention_mask, tags, pred)
            correct += result[0]
            total_gold += result[1]
            total_pred += result[2]
        self.logger.info("CORRECT:%d, total_gold:%d, total_pred:%d" % (correct, total_gold, total_pred))
        f1, p, r = self.cal_prf(correct, total_gold, total_pred)
        self.logger.info("F1:%.5f, P:%.5f, R:%.5f" % (f1, p, r))
        return f1, p, r

    def return_label(self, sent_embed, label_tensor, non_label_tensor):
        pred = []

        def list_to_matrix(label):
            labels = torch.zeros(len(label), 768)
            for i in range(len(label)):
                labels[i, :] = label[i]
            return labels.cuda()

        def euclidean_dist(x, y):
            m, n = x.size(0), y.size(0)
            xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
            yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
            dist = xx + yy
            dist.addmm_(1, -2, x, y.t())
            dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
            return dist

        label_tensor = list_to_matrix(label_tensor)
        non_label_tensor = list_to_matrix(non_label_tensor)

        label_id = euclidean_dist(sent_embed, label_tensor)
        label_id = torch.min(label_id, dim=-1)[0]

        non_label_id = euclidean_dist(sent_embed, non_label_tensor)
        non_label_id = torch.min(non_label_id, dim=-1)[0]
        pred = torch.where(label_id > non_label_id, 0, 1)
        # pred = pred.view(-1, self.args.max_length)
        # print(pred.shape)
        return pred

    def pre_train_with_PTuning(self, train, dev):
        model = self.model
        model.to(self.args.device)
        model.train()
        test_result = 0
        best_epoch = 0
        ffn_params = list(model.ffn.named_parameters())
        bert_params = list(model.model.named_parameters())
        optimizer_grouped_parameters = [{'params': [p for n, p in bert_params], 'lr': self.args.train_bert_lr,
                                         'weight_decay': self.args.weight_decay},
                                        {'params': [p for n, p in ffn_params]}]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.args.train_lr)

        for i in range(self.args.train_epoch):
            self.logger.info("--------------------Pretraining:%d------------------" % (i + 1))
            for index, batch in enumerate(tqdm(train)):
                inputs = {
                    'input_ids': batch['input_ids'],
                    'attention_mask': batch['attention_mask'],
                }

                inputs = {k: v.to(self.args.device) for k, v in inputs.items()}
                tags = batch['labels'].to(self.args.device)
                pred = model(inputs)
                pred = pred.permute(0, 2, 1)
                loss = self.loss_func(pred, tags)

                if index % 20 == 19:
                    self.logger.info("index: %d, loss: %.5f" % (index, loss))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        self.save_model(self.model, self.args.source_output_dir)

    def train_with_PTuning(self, train_loader, valid_loader, test_loader):
        self.args.is_pretraining = False
        model = PTuning(encoder_type=self.encoder_type, encoder_name=self.encoder_name, num_labels=self.args.num_labels,
                        args=self.args).to(self.args.device)
        model.train()
        test_result = 0
        best_epoch = 0
        ffn_params = list(model.ffn.named_parameters())
        mlp_params = list(model.pre_mlp.named_parameters())
        optimizer_grouped_parameters = [{'params': [p for n, p in ffn_params]},
                                        {'params': [p for n, p in mlp_params]}]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.args.train_lr)

        for i in range(self.args.test_epoch):
            self.logger.info("--------------------fine-tuning:%d------------------" % (i + 1))
            for index, batch in enumerate(tqdm(train_loader)):
                inputs = {
                    'input_ids': batch['input_ids'],
                    'attention_mask': batch['attention_mask'],
                }

                inputs = {k: v.to(self.args.device) for k, v in inputs.items()}
                shape = inputs['input_ids'].shape
                tags = batch['labels'].to(self.args.device)
                prefix_input = torch.zeros(shape[0], shape[1], dtype=torch.long).cuda()  # 必须转化为long才能作为Embedding的输入
                print("tag shape:", tags.shape)
                pred = model(inputs, prefix=prefix_input)
                pred = pred.permute(0, 2, 1)
                loss = self.loss_func(pred, tags)

                self.logger.info("index: %d, loss: %.5f" % (index, loss))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            # self.save_model(model, self.args.target_output_dir)
            # f1, p, r = self.evaluate_with_PTuning(test_loader)
            # self.logger.info("Epoch: %d, F1: %.5f, P: %.5f ,R: %.5f" % (i + 1, f1, p, r))
        self.save_model(model, self.args.target_output_dir)
        f1, p, r = self.evaluate_with_PTuning(test_loader)
        self.logger.info("F1: %.5f, P: %.5f ,R: %.5f" % (f1, p, r))

    def evaluate_with_PTuning(self, valid_loader, model=None):
        torch.cuda.empty_cache()
        torch.cuda.empty_cache()
        torch.cuda.empty_cache()
        if model is not None:
            self.logger.info("------------------------------Testing---------------------------")
        else:
            self.logger.info("------------------------------Validating------------------------")
            model = self.load_model(self.args.target_output_dir)
            # for name, params in model.named_parameters():
            #     print(name, params.shape)
        model.eval()
        correct, total_gold, total_pred = 0, 0, 0
        with torch.no_grad():
            for index, batch in enumerate(valid_loader):
                inputs = {
                    'input_ids': batch['input_ids'],
                    'attention_mask': batch['attention_mask'],
                }
                inputs = {k: v.to(self.args.device) for k, v in inputs.items()}
                shape = inputs['input_ids'].shape
                prefix_input = torch.ones(shape[0], shape[1], dtype=torch.long).cuda()
                pred = model(inputs, prefix_input)
                pred = torch.argmax(pred, dim=-1)
                attention_mask, labels = batch['attention_mask'].tolist(), batch['labels'].tolist()
                pred = pred.tolist()
                result = self.cal_correct_gold_pred(attention_mask, labels, pred)
                correct += result[0]
                total_gold += result[1]
                total_pred += result[2]
        self.logger.info("CORRECT:%d, total_gold:%d, total_pred:%d" % (correct, total_gold, total_pred))
        f1, p, r = self.cal_prf(correct, total_gold, total_pred)
        self.logger.info("F1:%.5f, P:%.5f, R:%.5f" % (f1, p, r))
        return f1, p, r

    def pre_train_with_CRF(self, train, dev):
        model = self.model
        model.to(self.args.device)
        model.train()
        test_result = 0
        best_epoch = 0

        crf_params = list(model.crf.named_parameters())
        bert_params = list(model.model.named_parameters())
        ffn_params = list(model.position_wise_ffn.named_parameters())
        optimizer_grouped_parameters = [{'params': [p for n, p in bert_params], 'lr': self.args.train_bert_lr,
                                         'weight_decay': self.args.weight_decay},
                                        {'params': [p for n, p in ffn_params]},
                                        {'params': [p for n, p in crf_params], 'lr': self.args.train_crf_lr}]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.args.train_lr)

        for i in range(self.args.train_epoch):
            self.logger.info("--------------------Pretraining:%d------------------" % (i + 1))
            for index, batch in enumerate(tqdm(train)):
                inputs = {
                    'input_ids': batch['input_ids'],
                    'attention_mask': batch['attention_mask'],
                }

                inputs = {k: v.to(self.args.device) for k, v in inputs.items()}
                tags = batch['labels'].to(self.args.device)

                pred1 = model(inputs, tags=tags)
                loss1, logits = pred1
                loss1 = -torch.sum(loss1)
                if index % 20 == 19:
                    self.logger.info("index: %d, loss: %.5f" % (index, loss1))
                optimizer.zero_grad()
                loss1.backward()
                optimizer.step()

        self.save_model(self.model, self.args.source_output_dir)

    def train_with_CRF(self, train_loader, valid_loader, test_loader):  # todo:增加保存的地址参数
        best_result = 0.0
        best_train_f1 = 0.0
        best_valid_f1 = 0.0
        min_valid_loss = sys.maxsize  #
        min_train_loss = sys.maxsize  #
        best_epoch = 0
        train_loss_list = []
        train_loss = 0
        flag = 0
        source_result = 0.0
        pre_loss = 0.0
        if self.args.source_dataset:
            model = self.load_model(self.args.source_output_dir)
            self.logger.info("Using source domain model!")
        elif self.args.use_cache:
            model = self.load_model(self.args.source_output_dir)
            self.logger.info("Using source domain model!")
        else:
            self.logger.info("Using in domain model!")
            model = self.model
            model.to(self.args.device)
        model.train()

        crf_params = list(model.crf.named_parameters())
        bert_params = list(model.model.named_parameters())
        ffn_params = list(model.position_wise_ffn.named_parameters())
        optimizer_grouped_parameters = [{'params': [p for n, p in bert_params], 'lr': self.args.test_bert_lr,
                                         'weight_decay': self.args.weight_decay},
                                        {'params': [p for n, p in ffn_params]},
                                        {'params': [p for n, p in crf_params], 'lr': self.args.test_crf_lr}]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.args.test_lr)

        for i in range(self.args.test_epoch):
            self.logger.info("---------------------------Training: epoch_%d--------------------" % (i + 1))
            train_loss = 0.0
            for index, batch in enumerate(tqdm(train_loader)):
                optimizer.zero_grad()
                inputs = {
                    'input_ids': batch['input_ids'],
                    'attention_mask': batch['attention_mask'],
                }
                inputs = {k: v.to(self.args.device) for k, v in inputs.items()}
                tags = batch['labels'].to(self.args.device)
                pred1 = model(inputs, tags=tags)
                loss1, logits = pred1
                loss1 = -torch.sum(loss1)
                # print("train loss:", loss1)
                train_loss += loss1
                optimizer.zero_grad()
                loss1.backward()
                optimizer.step()

            self.logger.info("train loss: %s " % train_loss)
            """
            收敛策略
            """
            # 一旦训练损失不再减小就停止训练
            # pre_loss = train_loss
            if pre_loss > train_loss and i > 20 and source_result < 50.0:
                flag == 1
                break
            else:
                self.save_model(model, self.args.target_output_dir)
            pre_loss = train_loss

        if flag == 1:
            self.logger.info(
                "The convergence state determined by human has been reached, the model is saved for inference!")
            # # self.save_model(model, self.args.target_output_dir)
            # self.logger.info(train_loss_list)
        else:
            best_epoch = self.args.test_epoch
            self.logger.info("The last epoch finished! Model is saved for inference!")
            # self.logger.info("best epoch in %d." % best_epoch)
        test_model = self.load_model(self.args.target_output_dir)
        test_f1, p, r = self.evaluate_with_CRF(test_loader, test_model)
        self.f1, self.p, self.r = test_f1, p, r
        return self.f1, self.p, self.r

    def evaluate_with_CRF(self, valid_loader, model=None):
        if model is not None:
            self.logger.info("------------------------------Testing---------------------------")
            model = model
            model.eval()
            # model.train()
        else:
            self.logger.info("------------------------------Validating------------------------")
            model = self.load_model(self.args.target_output_dir)
            model.eval()
        correct = 0
        total_pred = 0
        total_gold = 0
        for index, batch in enumerate(valid_loader):
            inputs = {
                'input_ids': batch['input_ids'],
                'attention_mask': batch['attention_mask'],
            }
            inputs = {k: v.to(self.args.device) for k, v in inputs.items()}
            pred = model(inputs)
            attention_mask, pred, labels = batch['attention_mask'].tolist(), pred, batch['labels'].tolist()
            result = self.cal_correct_gold_pred(attention_mask, labels, pred)
            correct += result[0]
            total_gold += result[1]
            total_pred += result[2]
        self.logger.info("CORRECT:%d, total_gold:%d, total_pred:%d" % (correct, total_gold, total_pred))
        f1, p, r = self.cal_prf(correct, total_gold, total_pred)
        self.logger.info("F1:%.5f, P:%.5f, R:%.5f" % (f1, p, r))
        return f1, p, r

    def evaluate_with_FFN(self, valid_loader, model=None):
        if model is not None:
            self.logger.info("------------------------------Testing---------------------------")
        else:
            self.logger.info("------------------------------Validating------------------------")
            model = self.load_model(self.args.target_output_dir)
        model.eval()
        correct, total_gold, total_pred = 0, 0, 0
        for index, batch in enumerate(valid_loader):
            inputs = {
                'input_ids': batch['input_ids'],
                'attention_mask': batch['attention_mask'],
            }
            inputs = {k: v.to(self.args.device) for k, v in inputs.items()}
            pred = model(inputs, )
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

    def make_dataset(self, datasets, domain='source'):
        # source_train = []
        # source_dev = []
        # source_test = []
        # target_train = []
        # target_dev = []
        # target_test = []
        if domain == 'target':
            mode = domain
            if self.args.model == 'cl':
                train = tfd(mode=mode, split='train', max_length=self.args.max_length, padding='max_length',
                            args=self.args)
            elif self.args.model == 'QA':
                train = qa(mode=mode, split='train', max_length=self.args.max_length, padding='max_length',
                           args=self.args)
            else:
                train = ds(mode=mode, split='train', max_length=self.args.max_length, padding='max_length',
                           args=self.args)
            if self.args.model == 'QA':
                valid = qa(mode=mode, split='devel', max_length=self.args.max_length, padding='max_length',
                           args=self.args)
                test = qa(mode=mode, split='test', max_length=self.args.max_length, padding='max_length',
                          args=self.args)
            else:
                valid = ds(mode=mode, split='devel', max_length=self.args.max_length, padding='max_length',
                           args=self.args)
                test = ds(mode=mode, split='test', max_length=self.args.max_length, padding='max_length',
                          args=self.args)

            return train, valid, test
        dsname = datasets[0]
        if domain == 'source':
            self.args.source_dataset = dsname
            mode = 'source'
        if self.args.use_cache != True:
            if self.args.model == 'QA':
                train = qa(mode=mode, split='train', max_length=self.args.max_length, padding='max_length',
                           args=self.args)
                valid = qa(mode=mode, split='devel', max_length=self.args.max_length, padding='max_length',
                           args=self.args)
                return train, valid
            else:
                train = ds(mode=mode, split='train', max_length=self.args.max_length, padding='max_length',
                           args=self.args)
                valid = ds(mode=mode, split='devel', max_length=self.args.max_length, padding='max_length',
                           args=self.args)
                return train, valid

    def train_with_FFN(self, train_loader, valid_loader, test_loader, optimizer):
        best_result = 0.0
        best_epoch = 0
        for i in range(self.args.test_epoch):
            self.model.train()
            self.logger.info("---------------------------Training: epoch_%d--------------------" % (i + 1))
            for index, batch in enumerate(train_loader):
                optimizer.zero_grad()
                inputs = {
                    'input_ids': batch['input_ids'],
                    'attention_mask': batch['attention_mask'],
                }
                inputs = {k: v.to(self.args.device) for k, v in inputs.items()}
                labels = batch['labels'].to(torch.int64)
                labels = labels.to(self.args.device)
                if self.args.loss == 'rdrop':
                    pred1 = self.model(inputs)
                    pred2 = self.model(inputs)
                    pred1 = pred1.permute(0, 2, 1)
                    pred2 = pred2.permute(0, 2, 1)
                    loss = self.loss_func(pred1, pred2, labels)
                else:
                    pred = self.model(inputs)
                    pred = pred.permute(0, 2, 1)
                    loss = self.loss_func(pred, labels)

                if index % 50 == 1:
                    self.logger.info("epoch: %d, index: %d, loss: %.5f" % (i + 1, index, loss))
                if self.args.few_shot != -1:
                    self.logger.info("index: %d, loss: %.5f" % (index, loss))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            f1, p, r = self.evaluate_with_FFN(valid_loader)
            self.logger.info("Epoch: %d, F1: %.5f, P: %.5f ,R: %.5f" % (i + 1, f1, p, r))
            f1, p, r = self.evaluate_with_FFN(test_loader)
            self.logger.info("Epoch: %d, F1: %.5f, P: %.5f ,R: %.5f" % (i + 1, f1, p, r))

    def train_with_contrastive_learning(self, train_loader, valid_loader, test_loader):  # todo:增加保存的地址参数
        model = AutoModel.from_pretrained('/gemini/pretrain')

        # *************** 自定义取出需要共享的参数 *******************
        from collections import OrderedDict
        temp = OrderedDict()

        ide_state_dict = model.state_dict(destination=None)
        for name, parameter in self.model.named_parameters():
            if name in ide_state_dict:
                parameter.requires_grad = True
                temp[name] = parameter

        # ************** 将共享的参数更新到需训练的模型中 ****************
        ide_state_dict.update(temp)  # 更新参数值
        model.load_state_dict(ide_state_dict)
        model.train()

        best_result = 0.0
        best_train_f1 = 0.0
        best_valid_f1 = 0.0
        min_valid_loss = sys.maxsize  #
        min_train_loss = sys.maxsize  #
        best_epoch = 0
        train_loss_list = []
        train_loss = 0
        flag = 0

        model.train()
        optimizer = AdamW(model.parameters(), lr=self.args.test_lr)

        for i in range(self.args.test_epoch):
            self.logger.info("---------------------------Training: epoch_%d--------------------" % (i + 1))
            train_loss = 0.0
            for index, batch in enumerate(tqdm(train_loader)):
                optimizer.zero_grad()
                inputs = {
                    'input_ids': batch['input_ids'],
                    'attention_mask': batch['attention_mask'],
                }
                inputs = {k: v.to(self.args.device) for k, v in inputs.items()}
                tags = batch['labels'].to(self.args.device)
                label_token_index = torch.nonzero(tags)
                token_index = torch.nonzero(batch['attention_mask'])
                print("label_token_index:", label_token_index.shape)  # (31,2)
                print(label_token_index)
                print("token_index:", token_index.shape)  # 111,2
                print(token_index)

                output = model(inputs)
                logits = output.last_hidden_state
                print("logits:", logits.shape)
                print(logits)
                break

                optimizer.zero_grad()
                loss1.backward()
                optimizer.step()
            print("train loss: ", train_loss)
            break
        # train_loss = self.train_loss(train_loader, model)
        """
        收敛策略
        """

        if flag == 1:
            self.logger.info(
                "The convergence state determined by human has been reached, the model is saved for inference!")
            # self.save_model(model, self.args.target_output_dir)
            self.logger.info(train_loss_list)
        else:
            best_epoch = self.args.test_epoch
            self.logger.info("The last epoch finished! Model is saved for inference!")
            # self.save_model(model, self.args.target_output_dir)
        # self.logger.info("best epoch in %d." % best_epoch)
        test_model = self.load_model(self.args.target_output_dir)
        test_f1, p, r = self.evaluate_with_CRF(test_loader, test_model)
        self.args.results.append((test_f1, p, r))
        self.logger.info("test set: %.5f, P: %.5f, R: %.5f" % (test_f1, p, r))

    def train_loss(self, dl, model):
        loss_sum = 0.0
        model = model
        model.eval()
        self.logger.info("-----------------------Calculating DevSet Loss------------------")
        # self.logger.info("------------------------------Testing---------------------------")
        for index, batch in enumerate(dl):
            inputs = {
                'input_ids': batch['input_ids'],
                'attention_mask': batch['attention_mask'],
            }
            inputs = {k: v.to(self.args.device) for k, v in inputs.items()}
            tags = batch['labels'].to(self.args.device)
            pred = model(inputs, tags=tags)
            loss = torch.sum(pred[0])
            loss_sum -= loss
        self.logger.info("loss: %.5f" % loss_sum)
        return loss_sum

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

    def cal_prf(self, correct, total_gold, total_pred):
        p = correct / total_pred if correct > 0 else 0.0
        r = correct / total_gold if correct > 0 else 0.0
        f1 = 2 * p * r / (p + r) * 100 if correct else 0.0
        return f1, p * 100, r * 100

    def cal_correct_gold_pred(self, attention_mask, labels, pred):
        correct, total_gold, total_pred = 0, 0, 0
        for i in range(len(attention_mask)):
            # print(i)
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

    sys.path.append('../')
    from random_sample import *
    from cal_distribution_match import *
    from corpus_filter import *


    def return_top5(result):
        min_f1 = 100
        t = 0
        for i in range(len(result)):
            j = result[i]
            f1 = j[1]
            if f1 < min_f1:
                min_f1 = f1
                t = i
        del result[t]
        return result


    result = []
    model_args = {
        # 'break_times': 5,  # 训练损失连续break_times不下降，即停止训练
        'manual_seed': 43,  # 41,42,43,44,45
        'max_length': 256,
        'num_labels': 3,  # 3：bio，2：io
        'time_stamp': 3,
        'train_epoch': 1,  #
        'test_epoch': 30,

        'test_lr': 1e-4,
        'test_bert_lr': 1e-4,  # 这里的学习率不能太大，最好时3e-5这个量级的
        'test_crf_lr': 1e-2,

        'train_lr': 1e-4,
        'train_bert_lr': 1e-5,  # 这里的学习率不能太大，最好时3e-5这个量级的
        'train_crf_lr': 1e-2,

        "use_cache": False,  # 使用缓存则直接使用上一步保存的source训练模型
        'weight_decay': 0.0,
        'train_batch_size': 32,
        'test_batch_size': 2,
        'few_shot': 20,  # 为-1时自动使用全监督设置
        'loss': 'cross_entropy',  # rdrop
        'with_bilstm': False,  # 是否使用BertBiLSTMCRF
        'with_CRF': False,  # 是否使用CRF
        # CORPORA_CLASS2NAME = {
        #     "disease": ['NCBI', 'BC5CDR-disease'],
        #     "drug": ['BC5CDR-chem', 'BC4CHEMD'],
        #     "gene": ['JNLPBA', 'BC2GM'],
        #     "species": ['LINNAEUS', 'S800']
        # }
        'source_dataset': ['disease_source'],  # disease_source,chem_source,gene_source,species_source
        'target_dataset': "NCBI",
        'source_output_dir': '/gemini/code/cache/source_best_model.pth',
        'target_output_dir': '/gemini/code/cache/target_best_model.pth',
        'device': 'cuda:0' if torch.cuda.is_available() else 'cpu',
        'tokenizer': AutoTokenizer.from_pretrained('/gemini/pretrain'),

        'topk': 5000,  # topk表示保留相似度前k个实例
        'level': 'entity_token',  # cls, sent, entity-token, entity
        'sim_func': "euc",  # kl, cos, euc
        'results': [],
        'dropout': 0.1,
        'method': 'other',
        'is_pretraining': True,
        "prefix_dim": 256,
        'metric': 'euc',
        'model': 'QA',
        'fine_tune': False,
        # 少样本会导致结果的精确率很高，召回率很低，
        # 原因是少样本的情况下，找不到对应的实例，但是找到的实例一般都是正确的
    }

    # 基于搜索的方法
    # for metric in ['euc', 'kl']:
    #     model_args['sim_func'] = metric
    #     for level in ['entity_token', 'entity', 'cls', 'sent']:
    #         model_args['level'] = level
    #         for topk in [5000]:
    #             model_args['topk'] = topk
    #             for seed in [40, 41, 42, 43, 45, 46]:
    #                 model_args['manual_seed'] = seed
    #                 standard_N_way_K_shot_sampling(model_args['target_dataset'], 'train', model_args['few_shot'],
    #                                                model_args['manual_seed'])
    #                 cf = CorpusFilter(model_args['target_dataset'], 'train', type='target',
    #                                   k_shot=model_args['few_shot'])
    #                 cf.write_to_file()
    #                 for i in model_args['source_dataset']:
    #                     ins = get_topk_instance_id(source=i, target=model_args['target_dataset'],
    #                                                level=model_args['level'],
    #                                                sim_func=SIM_FUNC[model_args['sim_func']],
    #                                                k_shot=model_args['few_shot'],
    #                                                topk=model_args['topk'])
    #                     write_topk_ins_to_file(source=i, target=model_args['target_dataset'], topk=ins,
    #                                            level=model_args['level'],
    #                                            k_shot=model_args['few_shot'])
    #
    #                     model = SpanExtraction(encoder_type='auto', encoder_name='biobert', args=model_args)
    #                     model.train_model()
    #                     f1, p, r = model.f1, model.p, model.r
    #                     result.append((seed, f1, p, r))
    #                     print("This round of training is over. We're about to clear the cache and start the next round")
    #                     for i in range(10):
    #                         torch.cuda.empty_cache()  # 释放显存
    #
    #             avg_f1, avg_p, avg_r = 0, 0, 0
    #
    #             for i in result:
    #                 avg_f1 += i[1]
    #                 avg_p += i[2]
    #                 avg_r += i[3]
    #             avg_f1 /= len(result)
    #             avg_p /= len(result)
    #             avg_r /= len(result)
    #             square_bias = 0.0
    #             for i in result:
    #                 square_bias += pow((avg_f1 - i[1]), 2)
    #             square_bias /= len(result)
    #             square_bias = pow(square_bias, 0.5)
    #             with open('result.txt', 'a') as f:
    #                 f.write(metric + ": " + level + ': ' + str(topk) + ': ' + str(round(avg_f1, 2)) + ' ' + \
    #                         str(round(avg_p, 2)) + " " + str(round(avg_r, 2)) + ' ' + str(round(square_bias, 2)) + '\n')
    #             print(round(avg_f1, 2), round(avg_p, 2), round(avg_r, 2), round(square_bias, 2))

    # 基于通用的方法
    for seed in [40, 41, 42, 43, 44]:
        model_args['manual_seed'] = seed
        N_sentences_K_shot_sampling(model_args['target_dataset'], 'train', model_args['few_shot'],
                                    model_args['manual_seed'])
        model = SpanExtraction(encoder_type='auto', encoder_name='biobert', args=model_args)
        model.train_model()
        f1, p, r = model.f1, model.p, model.r
        result.append((seed, f1, p, r))
        print("This round of training is over. We're about to clear the cache and start the next round")
        for i in range(10):
            torch.cuda.empty_cache()  # 释放显存

    avg_f1, avg_p, avg_r = 0, 0, 0

    for i in result:
        avg_f1 += i[1]
        avg_p += i[2]
        avg_r += i[3]
    avg_f1 /= len(result)
    avg_p /= len(result)
    avg_r /= len(result)
    square_bias = 0.0
    for i in result:
        square_bias += pow((avg_f1 - i[1]), 2)
    square_bias /= len(result)
    square_bias = pow(square_bias, 0.5)
    with open('result.txt', 'a') as f:
        f.write(str(model_args['test_epoch']) + ': ' + str(round(avg_f1, 2)) + ' ' + str(round(avg_p, 2)) + " " + str(
            round(avg_r, 2)) + ' ' + str(round(square_bias, 2)) + '\n')
    print(round(avg_f1, 2), round(avg_p, 2), round(avg_r, 2), round(square_bias, 2))
