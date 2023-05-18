"""
N-way K-shot设置：
N-way K-shot learning is conducted by iterativelyconstructing episodes.
For each episode in training, N classes (N-way) and K examples (K-shot)
for each class are sampled to build a support set S_train = {x(i), y(i)}^(N∗K')_(j=1) ,
and K0 examples foreach of N classes are sampled to construct a query set Q_train = {x(j), y(j)}^(N∗K')_(j=1) , and S^Q = ∅.
Few-shot learning systems are trained by predicting labels of query set Q_train with the information
of support set S_train. The supervision of S_train and Q_train are available in training. In the testing procedure, all the classes are unseen in the training
phase, and by using few labeled examples of support set S_test, few-shot learning systems need to make predictions of the unlabeled query set Q_test (S^Q = ∅).
"""
import random
import os


def precise_random_sampling(corpus, mode, k, seed):
    """
    本函数由于是针对 1 way k shot进行设置的，所以使用了精确采样
    这使得我们区别于N way K shot，最终采样得到的语料是精确的K个
    :param corpus: corpus_name
    :param mode: train, devel or test
    :param k: k-instance number
    :param seed: random seed
    :return: None
    """
    print("-------------------------------Precise Sampling %s %d-shot------------------------- " % (mode, k))
    random.seed(seed)
    filelist_numins = []
    sent = []
    num_instances = []
    count = 0
    filename = '/gemini/code/data/NERdata/' + corpus + '/' + mode + '.tsv'
    with open(filename, 'r') as f:
        lines = f.readlines()
    for line in lines:
        if len(line) > 2:
            line = line.strip()
            line = line.split()
            if line[-1] == 'B':
                count += 1
            sent.append(line[0] + ' ' + line[-1])
        else:
            filelist_numins.append(sent)
            num_instances.append(count)
            sent = []
            count = 0

    count = 0
    all_sent = []
    random_index_ls = []
    while True:
        random_index = random.randint(0, len(filelist_numins) - 1)
        if random_index in random_index_ls:
            continue
        else:
            random_index_ls.append(random_index)
        num_ins, sent = num_instances[random_index], filelist_numins[random_index]
        count += num_ins
        if num_ins != 0:
            if count > k:
                count -= num_ins
            elif count < k:
                all_sent.append(sent)
                pass
            else:
                all_sent.append(sent)
                break
        else:
            pass
    print("sampled instances number:", count)
    path = '/gemini/code/data/NERdata/' + corpus + '/' + str(k) + '-shot'
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)
    filename = path + mode + str(seed) + '.tsv'
    with open(filename, 'w') as f:
        for sent in all_sent:
            for line in sent:
                f.write(line)
                f.write('\n')
            f.write('\n')


def standard_N_way_K_shot_sampling(corpus, mode, k, seed):
    """
    这种方法最后实际采样到的实例数目是>=K个的，使用这种方法是因为在有多类型的实体语料中，
    采样会随着迭代次数增加而逐渐有限制，例如对于5way 5shot设置，如果支持集中已经有了4个
    类中有5个实例，1个类中有4个实例，这导致采样变得困难，必须得有仅包含一个特定类型实体的
    句子，而这对于标注稠密的数据集来说是不合适的
    cite:Ning Ding, Guangwei Xu, Yulin Chen, Xiaobin Wang, Xu Han, Pengjun
    Xie, Haitao Zheng, and Zhiyuan Liu. 2021. Few-nerd: A few-shot named
    entity recognition dataset. In ACL, pages 3198–3213.

    :param corpus: corpus_name
    :param mode: train, devel or test
    :param k: k-instance number
    :param seed: random seed
    :return: None
    """
    print("-------------------------------N way K shot Sampling %s %d-shot------------------------- " % (mode, k))
    random.seed(seed)
    filelist_numins = []
    sent = []
    num_instances = []
    sent_entity_count = 0  # 用于记录句子中的实体数量
    filename = '/gemini/code/data/NERdata/' + corpus + '/' + mode + '.tsv'
    with open(filename, 'r') as f:
        lines = f.readlines()
    for line in lines:
        if len(line) > 2:
            line = line.strip()
            line = line.split()
            if line[-1] == 'B':
                sent_entity_count += 1
            sent.append(line[0] + ' ' + line[-1])
        else:
            filelist_numins.append(sent)
            num_instances.append(sent_entity_count)
            sent = []
            sent_entity_count = 0

    sampled_count = 0  # 用于统计随机采样的实体数量
    all_sent = []
    random_index_ls = []
    while True:
        random_index = random.randint(0, len(filelist_numins) - 1)
        if random_index in random_index_ls:
            continue
        else:
            random_index_ls.append(random_index)
        num_ins, sent = num_instances[random_index], filelist_numins[random_index]
        if num_ins != 0:
            sampled_count += num_ins
            all_sent.append(sent)
        if sampled_count >= k:
            break
    print("sampled instances number:", sampled_count)
    path = '/gemini/code/data/NERdata/' + corpus + '/' + str(k) + '-shot'
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)

    filename = path + '/' + mode + str(seed) + '.tsv'
    with open(filename, 'w') as f:
        for sent in all_sent:
            for line in sent:
                f.write(line)
                f.write('\n')
            f.write('\n')
    print("sampling %d instances from %s %s." % (sampled_count, corpus, mode))


def N_sentences_K_shot_sampling(corpus, mode, k, seed):
    """
    这种方法最后实际采样到的实例数目是>=K个的，使用这种方法是因为在有多类型的实体语料中，
    采样会随着迭代次数增加而逐渐有限制，例如对于5way 5shot设置，如果支持集中已经有了4个
    类中有5个实例，1个类中有4个实例，这导致采样变得困难，必须得有仅包含一个特定类型实体的
    句子，而这对于标注稠密的数据集来说是不合适的
    cite:Ning Ding, Guangwei Xu, Yulin Chen, Xiaobin Wang, Xu Han, Pengjun
    Xie, Haitao Zheng, and Zhiyuan Liu. 2021. Few-nerd: A few-shot named
    entity recognition dataset. In ACL, pages 3198–3213.

    :param corpus: corpus_name
    :param mode: train, devel or test
    :param k: k-instance number
    :param seed: random seed
    :return: None
    """
    print("-------------------------------N way K shot Sampling %s %d-shot------------------------- " % (mode, k))
    random.seed(seed)
    filelist_numins = []
    sent = []
    num_instances = []
    sent_entity_count = 0  # 用于记录句子中的实体数量
    filename = '/gemini/code/data/NERdata/' + corpus + '/' + mode + '.tsv'
    with open(filename, 'r') as f:
        lines = f.readlines()
    for line in lines:
        if len(line) > 2:
            line = line.strip()
            line = line.split()
            if line[-1] == 'B':
                sent_entity_count += 1
            sent.append(line[0] + ' ' + line[-1])
        else:
            filelist_numins.append(sent)
            num_instances.append(sent_entity_count)
            sent = []
            sent_entity_count = 0

    sampled_count = 0  # 用于统计随机采样的实体数量
    all_sent = []
    random_index_ls = []
    while True:
        random_index = random.randint(0, len(filelist_numins) - 1)
        if random_index in random_index_ls:
            continue
        else:
            random_index_ls.append(random_index)
        num_ins, sent = num_instances[random_index], filelist_numins[random_index]
        if num_ins != 0:
            sampled_count += 1
            all_sent.append(sent)
        if sampled_count >= k:
            break
    print("sampled instances number:", sampled_count)
    path = '/gemini/code/data/NERdata/' + corpus + '/' + str(k) + '-shot'
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)

    filename = path + '/' + mode + str(seed) + '.tsv'
    with open(filename, 'w') as f:
        for sent in all_sent:
            for line in sent:
                f.write(line)
                f.write('\n')
            f.write('\n')
    print("sampling %d instances from %s %s." % (sampled_count, corpus, mode))
