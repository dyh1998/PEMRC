import os
import torch
import random
import logging
import warnings

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
# from utils.datasets import CommonDatasetsForNERBoundaryDetection as ds
# from utils.datasets import Packer as P
from losses.R_drop import RDrop
from losses.focal_loss import FocalLoss
# from utils.random_sample import *
from utils.cal_distribution_match import *
# from utils.corpus_filter import *
from losses.clloss import *

warnings.filterwarnings("ignore", category=DeprecationWarning)

MODEL_CLASSES = {
    "auto": (AutoConfig, AutoModel, AutoTokenizer),
    "bert": (BertConfig, BertModel, BertTokenizer),
    "roberta": (RobertaConfig, RobertaModel, RobertaTokenizer)
}
# LOCAL_PATH = {
#     'bert': '/home/yhdong/BioFSNER/code/cache/bert',
#     'biobert': '/home/yhdong/BioFSNER/code/cache/biobert',
#     'pubmed': '/home/yhdong/BioFSNER/code/cache/pubmed',
# }

LOCAL_PATH = {
    # 'bert': '/gemini/code/cache/bert',
    'biobert': '/gemini/pretrain',
    # 'pubmed': '/gemini/code/cache/pubmed',
}
LOSS_FUNC = {
    'cross_entropy': nn.CrossEntropyLoss(),
    'rdrop': RDrop(),
    'focal_loss': FocalLoss()
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

# tokenizer = AutoTokenizer.from_pretrained('/home/yhdong/BioFSNER/code/cache/biobert')
tokenizer = AutoTokenizer.from_pretrained('/gemini/pretrain')


# TODO: 模型之间的耦合性还不够高，计划写一个BERTs_for_token_cls的模型囊括大部分类型的token级别的模型类，
#  这是可以做到的，且不会造成使用上的不方面
class FeedForwardNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout_rate=0.1):
        super(FeedForwardNetwork, self).__init__()
        self.dropout_rate = dropout_rate
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x_proj = F.dropout(relu(self.linear1(x)), p=self.dropout_rate)
        x_proj = self.linear2(x_proj)
        return x_proj


class BertCRF(nn.Module):
    """
    几种不同的标注方式：
    1、BIO: num_labels = 3  # BIO会减少目标领域中少样本的实例数量
    2、IO: num_labels = 2  # IO无法很好的分清边界，但是在少样本简单的NER任务中还是有优势的
    3、首尾标记  # 这个待考虑
    """

    def __init__(self, encoder_type: str = None, encoder_name: str = None, num_labels: int = 0,
                 local_path: bool = True, args=None, dropout=0.1) -> None:
        super(BertCRF, self).__init__()
        self.args = args
        config_class, model_class, tokenizer_class = MODEL_CLASSES[encoder_type]
        self.encoder_name = encoder_name
        if local_path:
            self.encoder_name = LOCAL_PATH[self.encoder_name]
        self.model = model_class.from_pretrained(self.encoder_name)
        self.model.config.hidden_dropout_prob = dropout
        self.config = self.model.config
        self.num_labels = num_labels

        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)
        self.crf = CRF(num_labels=num_labels)
        self.position_wise_ffn = nn.Linear(self.config.hidden_size, num_labels)

    def forward(self, x, tags=None):
        outputs = self.model(**x)
        last_encoder_layer = outputs[0]
        last_encoder_layer = self.dropout(last_encoder_layer)
        emissions = self.position_wise_ffn(last_encoder_layer)
        if tags is not None:
            log_likelihood, sequence_of_tags = self.crf(emissions, tags, x['attention_mask']), self.crf.viterbi_decode(
                emissions, x['attention_mask'])
            return log_likelihood, sequence_of_tags
        else:
            sequence_of_tags = self.crf.viterbi_decode(emissions, x['attention_mask'])
            return sequence_of_tags


class BertBiLSTMCRF(nn.Module):
    """
    BertBiLSTMCRF
    """

    def __init__(self, encoder_type: str = None, encoder_name: str = None, num_labels: int = 0,
                 local_path: bool = True, args=None) -> None:
        super(BertBiLSTMCRF, self).__init__()
        if args is not None:
            self.args = args
        config_class, model_class, tokenizer_class = MODEL_CLASSES[encoder_type]
        self.encoder_name = encoder_name
        if local_path:
            self.encoder_name = LOCAL_PATH[self.encoder_name]
        self.model = model_class.from_pretrained(self.encoder_name)
        self.config = self.model.config
        self.num_labels = num_labels
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)
        self.crf = CRF(num_labels=num_labels)
        self.bilstm = nn.LSTM(input_size=self.config.hidden_size, hidden_size=(self.config.hidden_size) // 2,
                              num_layers=2, dropout=self.config.hidden_dropout_prob, batch_first=True,
                              bidirectional=True)

        self.position_wise_ffn = nn.Linear(self.config.hidden_size, num_labels)

    def forward(self, x, tags=None):
        """
        :param x:
        :param tags: torch.tensor(batch,length_attention_mask)
                if tags==0:training else:evaluating
        :return:
        """

        outputs = self.model(**x)
        last_encoder_layer = outputs[0]
        last_encoder_layer = self.dropout(last_encoder_layer)
        outputs, hc = self.bilstm(last_encoder_layer)
        emissions = self.position_wise_ffn(outputs)

        if tags is not None:
            log_likelihood, sequence_of_tags = self.crf(emissions, tags, x['attention_mask']), self.crf.viterbi_decode(
                emissions, x['attention_mask'])
            return log_likelihood, sequence_of_tags
        else:
            sequence_of_tags = self.crf.viterbi_decode(emissions, x['attention_mask'])
            return sequence_of_tags


class BertBiLSTMCRF_(BertPreTrainedModel):
    """
    使用transformers库的模型结构
    """

    def __init__(self, config, encoder_type: str = None, encoder_name: str = None, num_labels: int = 2,
                 local_path: bool = True, args=None) -> None:
        super().__init__(config)
        if args is not None:
            self.args = args
        config_class, model_class, tokenizer_class = MODEL_CLASSES[encoder_type]
        self.model = model_class(config)
        self.config = self.model.config

        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)
        print(num_labels)
        self.crf = CRF(num_labels=num_labels)
        self.bilstm = nn.LSTM(input_size=self.config.hidden_size, hidden_size=(self.config.hidden_size) // 2,
                              num_layers=2, dropout=self.config.hidden_dropout_prob, batch_first=True,
                              bidirectional=True)

        self.position_wise_ffn = nn.Linear(self.config.hidden_size, num_labels)

    def forward(self, x, tags=None):
        """
        :param x:
        :param tags: torch.tensor(batch,length_attention_mask)
        :return:
        """

        outputs = self.model(**x)
        last_encoder_layer = outputs[0]
        last_encoder_layer = self.dropout(last_encoder_layer)
        outputs, hc = self.bilstm(last_encoder_layer)

        emissions = self.position_wise_ffn(outputs)

        if tags is not None:
            log_likelihood, sequence_of_tags = self.crf(emissions, tags, emissions, x['attention_mask'])
            return log_likelihood, sequence_of_tags
        else:  # tag inference
            sequence_of_tags = self.crf.viterbi_decode(emissions, x['attention_mask'])
            return sequence_of_tags


class PTuning(nn.Module):
    """
    P-tuning v2
    prefix: mlp
    注意要冻结编码器参数，并且在优化器中只设置更新prefix参数
    """

    def __init__(self, encoder_type: str = None, encoder_name: str = None, num_labels: int = 0,
                 local_path: bool = True, args=None) -> None:
        super(PTuning, self).__init__()
        self.args = args
        prefix_dim = self.args.prefix_dim
        config_class, model_class, tokenizer_class = MODEL_CLASSES[encoder_type]
        self.encoder_name = encoder_name

        self.num_labels = num_labels

        def load_model(dir):
            model = torch.load(dir)
            model.to(self.args.device)
            return model

        if local_path:
            self.encoder_name = LOCAL_PATH[self.encoder_name]

        if self.args.is_pretraining:
            self.model = model_class.from_pretrained(self.encoder_name)
            config = self.model.config
            self.dropout = nn.Dropout(config.hidden_dropout_prob)
            self.ffn = nn.Linear(config.hidden_size, num_labels)
        else:
            source_model = load_model(self.args.source_output_dir)
            # *************** 自定义取出需要共享的参数 *******************
            self.model = model_class.from_pretrained(self.encoder_name)
            config = self.model.config

            from collections import OrderedDict
            temp = OrderedDict()

            ide_state_dict = self.model.state_dict(destination=None)
            for name, parameter in source_model.named_parameters():
                if name in ide_state_dict:
                    parameter.requires_grad = False
                    temp[name] = parameter

            # ************** 将共享的参数更新到需训练的模型中 ****************
            ide_state_dict.update(temp)  # 更新参数值
            self.model.load_state_dict(ide_state_dict)

            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(config.hidden_dropout_prob)
            self.embedder = nn.Embedding(config.num_attention_heads, config.hidden_size / config.num_attention_heads)
            self.pre_mlp = nn.Linear(config.hidden_size / config.num_attention_heads,
                                     config.hidden_size / config.num_attention_heads)
            self.ffn = nn.Linear(config.hidden_size + config.hidden_size / config.num_attention_heads, num_labels)

    def forward(self, x, prefix=None):
        if self.args.is_pretraining:
            outputs = self.model(**x)
            logits = self.dropout(outputs[0])
            output = self.ffn(logits)
            output = nn.functional.log_softmax(output, dim=-1)
        else:
            outputs = self.model(**x)
            logits_size = outputs[0].shape
            prefix = self.embedder(prefix)
            prefix = self.pre_mlp(prefix)
            prefix = self.relu(prefix)
            output = torch.cat((prefix, outputs[0]), dim=-1)

            output = self.dropout(output)
            output = self.ffn(output)
            output = nn.functional.log_softmax(output, dim=-1)
            print("output_shape:", output.shape)
        return output


class BertFFN(nn.Module):
    def __init__(self, encoder_type: str = None, encoder_name: str = None, num_labels: int = 0,
                 local_path: bool = True, args=None) -> None:
        super(BertFFN, self).__init__()
        self.args = args
        config_class, model_class, tokenizer_class = MODEL_CLASSES[encoder_type]
        self.encoder_name = encoder_name
        if local_path:
            self.encoder_name = LOCAL_PATH[self.encoder_name]
        self.model = model_class.from_pretrained(self.encoder_name)
        self.config = self.model.config
        self.num_labels = num_labels

        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)
        self.ffn = nn.Linear(self.config.hidden_size, num_labels)

    def forward(self, x):
        outputs = self.model(**x)
        outputs = outputs[0]  # last_encoder_layer
        outputs = self.dropout(outputs)
        outputs = self.ffn(outputs)
        return outputs


class BertQA(nn.Module):
    def __init__(self, encoder_type: str = None, encoder_name: str = None, num_labels: int = 0,
                 local_path: bool = True, args=None) -> None:
        super(BertQA, self).__init__()
        self.args = args
        config_class, model_class, tokenizer_class = MODEL_CLASSES[encoder_type]
        self.encoder_name = encoder_name
        if local_path:
            self.encoder_name = LOCAL_PATH[self.encoder_name]
        self.model = model_class.from_pretrained(self.encoder_name)
        self.config = self.model.config
        self.num_labels = num_labels

        self.dropout1 = nn.Dropout(self.config.hidden_dropout_prob)
        self.dropout2 = nn.Dropout(self.config.hidden_dropout_prob)

        self.ffn1 = nn.Linear(self.config.hidden_size, 2)
        self.ffn2 = nn.Linear(self.config.hidden_size, 2)

    def forward(self, x):
        x = {k: v.to(self.args.device) for k, v in x.items()}
        outputs = self.model(**x)
        outputs = outputs[0]  # last_encoder_layer
        start_logits = self.dropout1(outputs)
        end_logits = self.dropout2(outputs)

        start_logits = self.ffn1(start_logits)
        end_logits = self.ffn2(end_logits)
        return start_logits, end_logits


class MaskedLM(nn.Module):
    def __init__(self, encoder_type: str = None, encoder_name: str = None, num_labels: int = 0,
                 local_path: bool = True, args=None) -> None:
        super(MaskedLM, self).__init__()
        self.args = args
        config_class, model_class, tokenizer_class = MODEL_CLASSES[encoder_type]
        self.encoder_name = encoder_name
        if local_path:
            self.encoder_name = LOCAL_PATH[self.encoder_name]
        self.model = model_class.from_pretrained(self.encoder_name)
        if self.args.fine_tune:
            self.model = load_model(self.args.source_output_dir)
        self.config = self.model.config
        self.ffn = nn.Linear(self.config.hidden_size, self.config.vocab_size)
        self.bias = nn.Parameter(torch.zeros(self.config.vocab_size))

    def forward(self, x):
        output = self.model(**x)
        return self.ffn(output[0]) + self.bias


SIM_FUNC = {
    "kl": kl_sim,
    "cos": cos_sim,
    "euc": euc_sim
}


def CLLoss(logits, labels, metric, proto):
    loss = 0
    shape = logits.shape
    metric = SIM_FUNC[metric]
    num_token = 0
    for i in range(shape[0]):
        sentence_vector = logits[i, :, :]
        sentence_label = labels[i, :]
        token_idx = torch.nonzero(sentence_label).squeeze().tolist()
        for j in token_idx:
            proto = proto.view(1, -1)
            sent_vec = sentence_vector[j, :].view(1, -1)
            loss += metric(proto, sent_vec)
            num_token += 1
    return loss / num_token


def get_proto(logits, labels):
    entity_ids = tokenizer.convert_tokens_to_ids('entity')
    proto = torch.zeros(768).cuda()
    shape = logits.shape
    num_token = 0
    label_tensor = []
    non_label_tensor = []
    for i in range(shape[0]):
        sentence_vector = logits[i, :, :]
        sentence_label = labels[i, :]

        label_idx = torch.eq(sentence_label, entity_ids)
        label_idx = torch.nonzero(label_idx)
        label_idx = label_idx.squeeze().tolist()
        label_idx.remove(0)
        non_label_idx = []
        pad_idx = torch.nonzero(sentence_label).squeeze().tolist()

        for p in pad_idx:
            if p not in label_idx:
                non_label_idx.append(p)
        num_token += len(label_idx)

        for j in label_idx:
            proto += sentence_vector[j, :]
            label_tensor.append(sentence_vector[j, :])
        for j in non_label_idx:
            non_label_tensor.append(sentence_vector[j, :])
    proto = proto / num_token

    return proto, label_tensor, non_label_tensor


def load_model(dir):
    model = torch.load(dir)
    model.to('cuda:0')
    # self.logger.info("Model loaded from %s" % (str(dir)))
    print("Model loaded from %s" % (str(dir)))
    return model


class BERTForContrastiveLearningForTokenMetric(nn.Module):
    def __init__(self, encoder_type: str = None, encoder_name: str = None, num_labels: int = 0,
                 local_path: bool = True, args=None) -> None:
        super(BERTForContrastiveLearningForTokenMetric, self).__init__()
        self.args = args
        config_class, model_class, tokenizer_class = MODEL_CLASSES[encoder_type]
        self.encoder_name = encoder_name
        if local_path:
            self.encoder_name = LOCAL_PATH[self.encoder_name]

        source_model = load_model(self.args.source_output_dir)

        # *************** 自定义取出需要共享的参数 *******************
        self.model = model_class.from_pretrained(self.encoder_name)
        config = self.model.config

        from collections import OrderedDict
        temp = OrderedDict()

        ide_state_dict = self.model.state_dict(destination=None)
        for name, parameter in source_model.named_parameters():
            if name in ide_state_dict:
                parameter.requires_grad = False
                temp[name] = parameter

        # ************** 将共享的参数更新到需训练的模型中 ****************
        ide_state_dict.update(temp)  # 更新参数值
        self.model.load_state_dict(ide_state_dict)

        # for name, params in self.model.named_parameters():
        #     print(name, params.shape)
        self.config = self.model.config

    def forward(self, x, labels=None):
        output = self.model(**x)
        logits = output[0]
        if labels is None:
            return logits
        else:
            proto, label_tensor, non_label_tensor = get_proto(logits, labels)
            loss = CLLoss(logits, labels, self.args.metric, proto)
            return logits, loss, label_tensor, non_label_tensor


class BertCNNCRF(nn.Module):
    """
    BertCNNCRF模型
    """

    def __init__(self):
        None
