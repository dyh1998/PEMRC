import os
import torch
from transformers import AutoTokenizer, AutoModel, AutoConfig
from sampler import statistic_sentence_ins
from datasets import CommonDatasetsForNERBoundaryDetection, ASimplerDataset
from d2l import torch as d2l
import numpy as np

np.seterr(divide='ignore', invalid='ignore')
device = d2l.try_gpu(i=1)
model = AutoModel.from_pretrained('../cache/biobert-v1.1')
model = model.to(device)
model.eval()


def writes(top_k, t):
    gap = 10
    top_k_file = []
    for i in t[:(top_k + 5) * gap:gap]:
        index = i[0]
        top_k_file.append(doc[index])
    post_pre_sentence = []
    count = 0
    for sent_ins in top_k_file:
        sent, num_ins = sent_ins
        count += num_ins
        if num_ins != 0:
            if count > top_k:
                count -= num_ins
            elif count < top_k:
                post_pre_sentence.append(sent)
            else:
                post_pre_sentence.append(sent)
                break
    path = '../shared/NERdata-test/' + corpus + '/' + str(top_k) + '-shot'
    if not os.path.exists(path):
        os.makedirs(path)
    path = '../shared/NERdata-test/' + corpus + '/' + str(top_k) + '-shot/new_' + split + '.tsv'
    with open(path, 'w') as f:
        for sent in post_pre_sentence:
            i = sent[0]
            j = sent[1]
            i = i[1:-1]
            j = j[1:-1]
            for ins, tag in zip(i, j):
                if tag == 0:
                    tag = 'O'
                elif tag == 1:
                    tag = 'B'
                else:
                    tag = 'I'
                # print(ins, tag)
                f.write(ins + ' ' + tag + '\n')
            f.write('\n')
    print("write finished!")


def cos_func(w, x):
    t = x.resize_(1, 768)
    cos = torch.mv(t, w) / (torch.sqrt(torch.sum(w * w, axis=-1) + 1e-9) * torch.sqrt((x * x).sum()))
    return cos


def biobert_sampling(corpus, split):
    ncbi = ASimplerDataset(corpus=corpus, split=split)
    doc = ncbi.corpus
    tensor_dict = {}
    for ind in range(len(ncbi)):
        inputs = ncbi[ind]
        # if ind % 100 == 99:
        #     print('99 finished!')
        label = inputs['labels'].cpu()
        label = torch.squeeze(label)
        index = torch.nonzero(label).tolist()
        inputs = {
            'input_ids': torch.unsqueeze(inputs['input_ids'], 0),
            'attention_mask': torch.unsqueeze(inputs['attention_mask'], 0)
        }

        inputs = {k: v.to(device) for k, v in inputs.items()}
        output = model(**inputs)
        logit = output[0].cpu()
        this_logits = np.zeros(768)

        for i in index:
            t = logit[:, i[0], :].detach()
            t = torch.squeeze(t).numpy()
            this_logits = np.sum([t, this_logits], axis=0)

        this_logits = this_logits / len(index)
        tensor_dict[ind] = this_logits.tolist()
    avg_tensor = np.zeros(768)
    sum_tensor = np.zeros(768)
    cos_dict = {}
    count = 0
    for key, value in tensor_dict.items():
        value = torch.tensor(value, dtype=torch.double)
        cos_sim = cos_func(avg_tensor, value)
        cos_dict[count] = float(cos_sim)
        count += 1

    t = sorted(cos_dict.items(), key=lambda x: x[1], reverse=True)
    writes(5, t)
    writes(10, t)
    writes(20, t)
    writes(50, t)


if __name__ == '__main__':
    corpora = ['BC2GM', 'BC4CHEMD', 'BC5CDR-chem', 'BC5CDR-disease', 'JNLPBA', 'LINNAEUS', 'NCBI', 'S800']
    modes = ['train', 'devel']
    for corpus in corpora:
        for mode in modes:
            print("corpus:", corpus)
            print("mode:", mode)
            biobert_sampling(corpus, mode)
    # biobert_sampling()
