path = '../shared/ncbi/fs/train.txt'


def statistic_sentence_ins(path):
    """
    :param path:原始语料的存放路径
    :param k_shot: 对每个类别进行采样的数量
    :return: None
    """
    # 对每个句子中的实例数量进行统计
    with open(path, 'r') as f:
        lines = f.readlines()
    doc = []
    sentence = []
    tag = []
    count = 0
    for line in lines:
        if len(line) > 2:
            line = line.strip()
            line = line.split()
            sentence.append(line[0])
            tag.append(line[-1])
            if line[-1].startswith('B'):
                count += 1
        else:
            doc.append((sentence, tag, count))
            sentence, tag, count = [], [], 0
    return doc
    # TODO：对实例进行随机采样


def sampler():
    # TODO:
    return None
