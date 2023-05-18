from transformers import AutoModel, AutoConfig, AutoTokenizer, BertModel, BertConfig, BertTokenizer, RobertaConfig, \
    RobertaTokenizer, RobertaModel
from torch import nn
import torch

LOCAL_PATH = {
    'bert': '/gemini/pretrain/bert-base-uncased',
    'biobert': '/gemini/pretrain',
    'roberta': '../predata/roberta-base',
}
MODEL_CLASSES = {
    "auto": (AutoConfig, AutoModel, AutoTokenizer),
    "bert": (BertConfig, BertModel, BertTokenizer),
    "roberta": (RobertaConfig, RobertaModel, RobertaTokenizer)
}

model_args = {
    # 'break_times': 5,  # 训练损失连续break_times不下降，即停止训练
    'manual_seed': 43,  # 41,42,43,44,45
    'max_length': 256,
    'num_labels': 3,  # 3：bio，2：io
    'time_stamp': 3,
    'train_epoch': 3,  #
    'test_epoch': 50,

    'test_lr': 1e-4,
    'test_bert_lr': 1e-4,  # 这里的学习率不能太大，最好时3e-5这个量级的
    'test_crf_lr': 1e-2,

    'train_lr': 3e-5,
    'train_bert_lr': 3e-5,  # 这里的学习率不能太大，最好时3e-5这个量级的
    'train_crf_lr': 1e-2,

    "use_cache": False,  # 使用缓存则直接使用上一步保存的source训练模型
    'weight_decay': 0.0,
    'train_batch_size': 32,
    'test_batch_size': 2,
    'few_shot': 5,  # 为-1时自动使用全监督设置
    'loss': 'cross_entropy',  # rdrop
    'with_bilstm': False,  # 是否使用BertBiLSTMCRF
    'with_CRF': True,  # 是否使用CRF

    'source_dataset': ['disease_source'],  # disease_source,chem_source,gene_source,species_source
    'target_dataset': "NCBI",
    'source_output_dir': '/gemini/code/cache/source_best_model.pth',
    'target_output_dir': '/gemini/code/cache/target_best_model.pth',
    'device': 'cuda:0' if torch.cuda.is_available() else 'cpu',
    'tokenizer': AutoTokenizer.from_pretrained('/gemini/pretrain'),

    'topk': 20000,  # topk表示保留相似度前k个实例
    'level': 'entity_token',  # cls, sent, entity-token, entity
    'sim_func': "cos",  # kl, cos, euc
    'results': [],
    'dropout': 0.1,
    'method': 'other',
    'is_pretraining': False,
    "prefix_dim": 256,
    # 少样本会导致结果的精确率很高，召回率很低，
    # 原因是少样本的情况下，找不到对应的实例，但是找到的实例一般都是正确的
}


# class BertCRF(nn.Module):
#     """
#     几种不同的标注方式：
#     1、BIO: num_labels = 3  # BIO会减少目标领域中少样本的实例数量
#     2、IO: num_labels = 2  # IO无法很好的分清边界，但是在少样本简单的NER任务中还是有优势的
#     3、首尾标记  # 这个待考虑
#     """
#
#     def __init__(self, encoder_type: str = None, encoder_name: str = None, num_labels: int = 0,
#                  local_path: bool = True, args=None, dropout=0.1) -> None:
#         super(BertCRF, self).__init__()
#         self.args = args
#         config_class, model_class, tokenizer_class = MODEL_CLASSES[encoder_type]
#         self.encoder_name = encoder_name
#         if local_path:
#             self.encoder_name = LOCAL_PATH[self.encoder_name]
#         self.model = model_class.from_pretrained(self.encoder_name)
#         self.model.config.hidden_dropout_prob = dropout
#         self.config = self.model.config
#         self.num_labels = num_labels
#
#         self.dropout = nn.Dropout(self.config.hidden_dropout_prob)
#         self.crf = CRF(num_labels=num_labels)
#         self.position_wise_ffn = nn.Linear(self.config.hidden_size, num_labels)
#
#     def forward(self, x, tags=None):
#         outputs = self.model(**x)
#         last_encoder_layer = outputs[0]
#         last_encoder_layer = self.dropout(last_encoder_layer)
#         emissions = self.position_wise_ffn(last_encoder_layer)
#         if tags is not None:
#             log_likelihood, sequence_of_tags = self.crf(emissions, tags, x['attention_mask']), self.crf.viterbi_decode(
#                 emissions, x['attention_mask'])
#             return log_likelihood, sequence_of_tags
#         else:
#             sequence_of_tags = self.crf.viterbi_decode(emissions, x['attention_mask'])
#             return sequence_of_tags


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
        prefix_dim = self.args['prefix_dim']
        config_class, model_class, tokenizer_class = MODEL_CLASSES[encoder_type]
        self.encoder_name = encoder_name
        self.num_labels = num_labels
        self.softmax = nn.Softmax(dim=-1)

        def load_model(dir):
            model = torch.load(dir)
            model.to(self.args['device'])
            self.logger.info("Model loaded from %s" % (str(dir)))
            return model

        if local_path:
            self.encoder_name = LOCAL_PATH[self.encoder_name]

        if self.args['is_pretraining']:
            self.model = model_class.from_pretrained(self.encoder_name)
            self.dropout = nn.Dropout(0.1)
            self.ffn = nn.Linear(768, num_labels)
        else:
            source_model = load_model(self.args['source_output_dir'])
            for name, param in source_model.named_parameters():
                print(name, param.shape)
            # *************** 自定义取出需要共享的参数 *******************
            self.model = model_class.from_pretrained(self.encoder_name)
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
            self.dropout = nn.Dropout(0.1)
            self.embedder = nn.Embedding(1, prefix_dim)
            self.pre_mlp = nn.Linear(prefix_dim, prefix_dim)
            self.ffn = nn.Linear(768 + prefix_dim, num_labels)

    def forward(self, x):
        if self.args.is_pretraining:
            logits, *_ = self.model(**x)
            logits = self.dropout(logits)
            output = self.ffn(logits)
            output = self.softmax(output)
        else:
            logits, *_ = self.model(**x)
            logits_size = logits.shape

            prefix_input = torch.ones(logits_size[0], logits_size[1], 1)
            prefix = self.embedder(prefix_input)
            prefix = self.pre_mlp(prefix)
            prefix = self.relu(prefix)

            output = torch.cat((prefix, logits), dim=-1)
            output = self.dropout(output)
            output = self.ffn(output)

        return output


model = PTuning(encoder_type='auto', encoder_name='biobert', num_labels=2, local_path=True, args=model_args)
for name, param in model.named_parameters():
    print(name, param.shape)
