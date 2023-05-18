"""
根据目标域数据和源领域数据之间的分布来从目标域中选取分布较为相似的分布进行训练
句子级别分布相似度：使用目标域中的句子相似度来进行选择
实例级别：根据目标域中的实例相似度来选择

相似度尺度：余弦，欧式，KL散度
"""
import sys

sys.path.append('..')
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import CosineSimilarity, PairwiseDistance


class KLSim(nn.Module):
    def __init__(self):
        super(KLSim, self).__init__()
        self.kld = nn.KLDivLoss(reduction='none')

    def forward(self, ins1, ins2):
        kl_loss1 = self.kld(F.log_softmax(ins1, dim=-1), F.softmax(ins2, dim=-1)).sum(-1)
        kl_loss2 = self.kld(F.log_softmax(ins2, dim=-1), F.softmax(ins1, dim=-1)).sum(-1)
        kl_loss = (kl_loss1 + kl_loss2) / 2

        return kl_loss


def kl_sim(ins1, ins2):
    kld = nn.KLDivLoss(reduction='none')  # 这里不能使用mean，会报维度不匹配的错误，得用batchmean
    kl_loss1 = kld(F.log_softmax(ins1, dim=-1), F.softmax(ins2, dim=-1)).sum(-1)
    kl_loss2 = kld(F.log_softmax(ins2, dim=-1), F.softmax(ins1, dim=-1)).sum(-1)
    kl_loss = (kl_loss1 + kl_loss2) / 2
    return 0.1 / kl_loss


def cos_sim(ins1, ins2):
    sim = CosineSimilarity(dim=-1, eps=1e-6)
    return sim(ins1, ins2)


def euc_sim(ins1, ins2):
    sim = PairwiseDistance(p=2)
    return sim(ins1, ins2)


def get_level_proto(corpus, level, k_shot):
    dir = '../data/NER_tensor/' + corpus + '/' + str(k_shot) + '-shot/' + level + '_proto.tsv'
    with open(dir, 'r') as f:
        # entity_proto = torch.tensor(f.read())
        entity_proto = f.read()
    entity_proto = entity_proto[7:-1]
    entity_proto = eval(entity_proto)
    entity_proto = torch.tensor(entity_proto)
    return entity_proto


def source_file_tensor_to_tensor(corpus, level):
    """
    :param corpus: NCBI
    :param level: cls,sent,entity_token,entity
    :return: level_proto_dict
    """
    entity_dc = {}
    dir = '/gemini/code/data/NER_tensor/' + corpus + '/train/' + level + '.tsv'
    with open(dir, 'r') as f:
        lines = f.read()

    lines = lines.split(':')
    # print("tensor_file 0:", lines[0])
    # print("tensor_file -1:", lines[-1])
    del lines[0]  # 删去不包含张量信息的piece
    for i in range(0, len(lines)):
        tensor = lines[i]
        if len(tensor) < 100:  # 100表示None所在分割处的最大长度，为了方便设置为零100
            entity_dc[i] = None
        else:
            tensor = tensor.split('(')[1]
            tensor = tensor.split(')')[0]
            tensor = eval(tensor)
            tensor = torch.tensor(tensor)
            entity_dc[i] = tensor
    return entity_dc


def get_topk_instance_id(source, target, level, sim_func, k_shot, topk):
    """
    :param source: 源领域语料名称
    :param target: 目标领域语料名称
    :param level: 距离尺度的级别，cls,sent,entity-token,entity
    :param sim_func: kl_sim, cos_sim, euc_sim
    :param k_shot: 5, 10, 20, 50
    :param topk: 100, 200
    :return:
    """
    entity_dic = source_file_tensor_to_tensor(corpus=source, level=level)
    entity_proto = get_level_proto(corpus=target, level=level, k_shot=k_shot)
    final_dist_dict = {}
    print("len_entity_dict:", len(entity_dic))
    for id, tensor in entity_dic.items():
        if tensor is not None:
            dist = sim_func(tensor, entity_proto)
            final_dist_dict[id] = float(dist)
    print("final_dist_dict:", len(final_dist_dict))
    t = sorted(final_dist_dict.items(), key=lambda x: x[1], reverse=True)
    ins = t[:topk]
    # ins = t[-topk:]
    print("len of ins:", len(ins))
    return ins


def write_topk_ins_to_file(source, target, topk, level, k_shot):
    dir = "/gemini/code/data/NERdata/" + source + "/train.tsv"
    with open(dir, 'r') as f:
        lines = f.readlines()
    real_sent = []
    sent = []
    for line in lines:
        if len(line) < 2:
            real_sent.append(sent)
            sent = []
        else:
            sent.append(line)
    print("len_of_sent:", len(real_sent))
    # print(real_sent[0])
    # print(real_sent[-1])
    file_dir = '/gemini/code/data/NERdata/' + target + '/' + str(k_shot) + '-shot/' + level + '_source.txt'
    print("筛选文件写回位置%s" % file_dir)
    with open(file_dir, 'w') as f:
        for id, score in topk:
            line = real_sent[id]
            # print("筛选示例：")
            for i in line:
                f.write(i)
            f.write('\n')

# ins = get_topk_instance_id(source='NCBI', target="BC2GM", level='entity',
#                            sim_func=kl_sim, k_shot=5, topk=100)
# write_topk_ins_to_file(source='NCBI', target="BC2GM", topk=ins, level='entity', k_shot=5)
