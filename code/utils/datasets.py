import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

# TODO:setting abstract dataset base class
tokenizer = AutoTokenizer.from_pretrained('/gemini/pretrain')

"""
corpora source:
https://drive.google.com/u/0/uc?id=1OletxmPYNkz2ltOr9pyT0b0iBtUWxslh&export=download
"""


class CommonDatasetsForNERBoundaryDetection:
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
            # 基于搜索的文件路径
            # if mode == 'target':
            #     if self.args.few_shot != -1 and self.split == 'train':
            #         # 随机k-shot
            #         self.filename = '/gemini/code/data/NERdata/' + self.args.target_dataset + '/' + str( \
            #             self.args.few_shot) + '-shot/' + self.split + '.tsv'
            #     elif self.split == 'test':
            #         self.filename = '/gemini/code/data/NERdata/' + self.args.target_dataset + '/' + self.split + '.tsv'
            #     else:
            #         self.filename = '/gemini/code/data/NERdata/' + self.args.target_dataset + '/devel.tsv'

            # 基于通用预训练的方法
            if mode == 'target':
                if self.args.few_shot != -1 and self.split == 'train':
                    # 随机k-shot
                    self.filename = '/gemini/code/data/NERdata/' + self.args.target_dataset + '/' + str( \
                        self.args.few_shot) + '-shot/' + 'train.tsv'
                elif self.split == 'test':
                    self.filename = '/gemini/code/data/NERdata/' + self.args.target_dataset + '/' + 'test.tsv'
                else:
                    self.filename = '/gemini/code/data/NERdata/' + self.args.target_dataset + '/' + str( \
                        self.args.few_shot) + '-shot/' + 'train.tsv'
            if mode == 'source':
                self.filename = '/gemini/code/data/NERdata/' + self.args.source_dataset + '/train.tsv'

        except Exception as e:
            self.args.logger.info("CommonDatasetsForBoundaryDetection object has no attribute 'args'!")
            os._exit()
        print("filename:", self.filename)
        self.corpus = self.read_corpus()
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
        for instance in self.corpus:
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
                if self.args.few_shot != -1 and self.split == 'train':
                    # 随机k-shot
                    self.filename = '/gemini/code/data/NERdata/' + self.args.target_dataset + '/' + str( \
                        self.args.few_shot) + '-shot/' + 'train.tsv'
                elif self.split == 'test':
                    self.filename = '/gemini/code/data/NERdata/' + self.args.target_dataset + '/' + 'test.tsv'
                else:
                    self.filename = '/gemini/code/data/NERdata/' + self.args.target_dataset + '/' + str( \
                        self.args.few_shot) + '-shot/' + 'train.tsv'
            if mode == 'source':
                self.filename = '/gemini/code/data/NERdata/' + self.args.source_dataset + '/train.tsv'

        except Exception as e:
            self.args.logger.info("CommonDatasetsForBoundaryDetection object has no attribute 'args'!")
            os._exit()
        print("filename:", self.filename)
        self.corpus = self.read_corpus()
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

        pre, cur, last = 0, 0, 0
        start_index, end_index = [], []
        if self.args.num_labels == 2:
            for i in range(len(label) - 1):
                cur = label[i]
                last = label[i + 1]
                if cur == 1 and pre == 0:
                    start_index.append(1)
                    end_index.append(0)
                elif last == 0 and cur == 1:
                    end_index.append(1)
                    start_index.append(0)
                else:
                    end_index.append(0)
                    start_index.append(0)
                pre = cur
        elif self.args.num_labels == 3:
            for i in range(len(label) - 1):
                cur = label[i]
                last = label[i + 1]
                if cur == 1:
                    start_index.append(1)
                    end_index.append(0)
                elif cur == 2 and last != 2:
                    end_index.append(1)
                    start_index.append(0)
                else:
                    start_index.append(0)
                    end_index.append(0)
                pre = cur
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
        # print("inputs:", inputs)
        return inputs

    def __len__(self):
        return len(self.examples)

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
        for instance in self.corpus:
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


# class Packer(Dataset):
#     def __init__(self, args):
#         self.ds = []
#         for arg in args:
#             for i in range(len(arg)):
#                 # print("t:", t)
#                 self.ds.append(arg[i])

# def __len__(self):
#     return len(self.ds)
#
# def __getitem__(self, item):
#     return self.ds[item]


class ASimplerDataset(Dataset):
    def __init__(self, split='train', max_length=512, padding='max_length', corpus=None, type=None, k_shot=None):
        # self.args = args
        self.split = split
        self.tokenizer = tokenizer
        self.max_length = max_length
        if type == 'source':
            self.filename = '/gemini/code/data/NERdata/' + corpus + '/' + self.split + '.tsv'
        else:
            self.filename = '/gemini/code/data/NERdata/' + corpus + '/' + str(k_shot) + '-shot/' + self.split + '.tsv'
        self.corpus = self.read_corpus()
        print("len of corpus:", len(self.corpus))
        self.examples = self.get_input_tensor_label()
        print("len of examples:", len(self.examples))

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

    def read_corpus(self):
        with open(self.filename, 'r') as f:
            lines = f.readlines()
        sentence = []
        tag = []
        doc = []
        num_ins = 0
        for line in lines:
            if len(line) > 2:
                line = line.strip()
                line = line.split()
                if line[-1] != 'O':
                    if line[-1].startswith('B'):
                        tag.append(1)
                        num_ins += 1
                    elif line[-1].startswith('I'):
                        # if self.args.num_labels == 3:  # 这里考虑是io标注还是bio标注方式
                        tag.append(2)
                else:
                    tag.append(0)
                sentence.append(line[0])
            else:
                doc.append(((sentence, tag), num_ins))
                sentence, tag = [], []
                num_ins = 0
        first_sent = doc[0]
        last_sent = doc[-1]
        print(first_sent)
        print(last_sent)
        # print("当前数据集的第一句为：%s" % first_sent[0])
        # print("当前数据集的最后一句为：%s" % last_sent[0])
        print("当前数据集的句子数为：%d" % len(doc))
        return doc

    def get_input_tensor_label(self):
        examples = {}
        count = 0
        for instance, _ in self.corpus:
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
                    # if self.args.num_labels == 2:  # 对bio和io进行处理
                    #     real_label = real_label + [word_label for _ in range(len(token))]
                    # elif self.args.num_labels == 3:
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
        print("当前数据集共有%d个输入实例。" % len(examples))
        return examples


class TemplateFreeDataset(Dataset):
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
                if self.args.few_shot != -1 and self.split == 'train':
                    # 随机k-shot
                    self.filename = '/gemini/code/data/NERdata/' + self.args.target_dataset + '/' + str( \
                        self.args.few_shot) + '-shot/' + 'train.tsv'
                elif self.split == 'test':
                    self.filename = '/gemini/code/data/NERdata/' + self.args.target_dataset + '/' + 'test.tsv'
                else:
                    self.filename = '/gemini/code/data/NERdata/' + self.args.target_dataset + '/' + str( \
                        self.args.few_shot) + '-shot/' + 'train.tsv'
            if mode == 'source':
                self.filename = '/gemini/code/data/NERdata/' + self.args.source_dataset + '/train.tsv'

        except Exception as e:
            self.args.logger.info("TemplateFreeDataset object has no attribute 'args'!")
            os._exit()
        print("filename:", self.filename)
        self.corpus = self.read_corpus()
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
        for instance in self.corpus:
            if instance[0]:
                example = {}
                real_token, real_label, real_span = [], [], []
                start_index, end_index = 0, 0
                instance[0].insert(0, '[CLS]')
                instance[1].insert(0, self.tokenizer.convert_tokens_to_ids('[CLS]'))
                instance[0].append('[SEP]')
                instance[1].append(self.tokenizer.convert_tokens_to_ids('[SEP]'))
                for word, word_label in zip(instance[0], instance[1]):
                    start_index = end_index
                    token = self.tokenizer.tokenize(word)
                    real_token = real_token + token
                    end_index = start_index + len(token)
                    real_span.append((start_index, end_index))
                    if self.args.num_labels == 2:  # 对bio和io进行处理
                        if word_label == 0:
                            real_label = real_label + [self.tokenizer.convert_tokens_to_ids(_) for _ in token]
                        else:
                            real_label = real_label + [self.tokenizer.convert_tokens_to_ids('entity') for _ in
                                                       range(len(token))]
                    elif self.args.num_labels == 3:
                        if len(token) > 1:  # 对B-Disease的单词进行token标签的划分
                            if word_label == 1:
                                real_label.append(self.tokenizer.convert_tokens_to_ids('entity'))
                                real_label += [self.tokenizer.convert_tokens_to_ids('entity') for _ in
                                               range(len(token) - 1)]
                            else:
                                real_label += [self.tokenizer.convert_tokens_to_ids(_) for _ in token]
                        else:
                            real_label.append(self.tokenizer.convert_tokens_to_ids(token))
                real_ids = tokenizer.convert_tokens_to_ids(real_token)
                example['origin_token'], example['token'], example['label'], example['ids'], example[
                    'span'] = instance[0], real_token, real_label, real_ids, real_span
                examples[count] = example
                count += 1

        return examples
# if __name__ == '__main__':
#     ds = ASimplerDataset(split='train', corpus='BC2GM')
#     print(ds[len(ds) - 1])
