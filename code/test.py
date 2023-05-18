"""
MaskedLM模型测试
"""
# from transformers import AutoTokenizer, BertForMaskedLM
# import torch
#
# tokenizer = AutoTokenizer.from_pretrained("/gemini/pretrain")
# model = BertForMaskedLM.from_pretrained("/gemini/pretrain")
#
# inputs = tokenizer(" is [MASK].", return_tensors="pt")
#
# with torch.no_grad():
#     logits = model(**inputs).logits
#
# # retrieve index of [MASK]
# mask_token_index = (inputs.input_ids == tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]
#
# predicted_token_id = logits[0, mask_token_index].argmax(axis=-1)
# print(tokenizer.decode(predicted_token_id))
#
# # labels = tokenizer("The capital of France is Paris.", return_tensors="pt")["input_ids"]
# # # mask labels of non-[MASK] tokens
# # labels = torch.where(inputs.input_ids == tokenizer.mask_token_id, labels, -100)
# #
# # outputs = model(**inputs, labels=labels)
# # round(outputs.loss.item(), 2)

"""
语料统计
"""
# def corpus_sta(name):
#     path = '/gemini/code/data/NERdata/'
#     train = path + name + '/train.tsv'
#     dev = path + name + '/devel.tsv'
#     test = path + name + '/test.tsv'
#     result = []
#     for filename in [train, dev, test]:
#         with open(filename, 'r') as f:
#             lines = f.readlines()
#         count = 0
#         for line in lines:
#             if len(line) > 1:
#                 line = line.split()
#                 if line[-1].startswith('B'):
#                     count += 1
#         result.append(count)
#     return result
#
#
# for name in ['NCBI', 'BC5CDR-disease', 'BC5CDR-chem', 'BC4CHEMD', 'BC2GM', "JNLPBA", "LINNAEUS", "S800"]:
#     result = corpus_sta(name)
#     print(name, result)

"""
依存句法辅助命名实体识别
"""
# from stanfordcorenlp import StanfordCoreNLP
#
# path = '/gemini/code/cache/stanford-parser-full-2020-11-17'
# nlp = StanfordCoreNLP(path, lang='en')
#
# s = "Selegiline - induced postural hypotension in Parkinson's disease: a longitudinal study on the effects of drug withdrawal."
# token = nlp.word_tokenize(s)
# print("token:", token)
# dependencyParse = nlp.dependency_parse(s)
#
# for i, begin, end in dependencyParse:
#     print(i, '-'.join([str(begin), token[begin - 1]]), '-'.join([str(end), token[end - 1]]))

from transformers import AutoModel, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('/gemini/pretrain')
sent = tokenizer("v1 v36")
print(sent)
