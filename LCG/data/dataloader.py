import nltk.tokenize as tokenize
import torch.utils.data as datautils
import torch
import torch.random
import nltk
from IPython import embed
import transformers
from transformers import BasicTokenizer
import numpy as np
import tqdm
import pickle
import random


class NaiveTokenizer(object):
    def __init__(self, from_pretrained=None):
        fin = None
        self.str2idx = {" " : 0, "<s>" : 1, "</s>" : 2, "<sep>": 3, "<mask>": 4, "<pad>": 5}
        self.idx2str = ["", "<s>", "</s>", "<sep>", "<mask>", "<pad>"]
        if from_pretrained is not None:
            try:
                fin = open(from_pretrained, "rb")
            except:
                fin = None

            if fin is not None:
                self.str2idx, self.idx2str = pickle.load(fin)
                fin.close()
            else:
                print("Warning: pretrained file at location \"%s\" not found." % from_pretrained)

        self.basic_tokenizer = BasicTokenizer(do_lower_case=False, never_split=["<mask>", "</s>", "<EOT>", "<EOL>", "<sep>", "<mask>", "<pad>"])
        self.bos_token_id = 1
        self.eos_token_id = 2
        self.sep_token_id = 3
        self.mask_token_id = 4
        self.pad_token_id = 5
        self.vocab_closed = False

    def get_vocab(self):
        return self.str2idx

    def dump(self, path):
        with open(path, "wb") as fout:
            pickle.dump((self.str2idx, self.idx2str), fout)

    def close_vocab(self):
        self.vocab_closed = True

    def tokenize(self, text, max_len=None):
        tokens = self.basic_tokenizer.tokenize(text)
        token_ids = []
        for token in tokens:
            if token in self.str2idx:
                token_ids.append(self.str2idx[token])
            elif not self.vocab_closed:
                self.str2idx[token] = len(self.idx2str)
                self.idx2str.append(token)
                token_ids.append(self.str2idx[token])
        if max_len is not None:
            while len(token_ids) < max_len:
                token_ids.append(1)
        return token_ids

    def encode(self, text, max_len=None, add_special_tokens=False):
        tokens = self.basic_tokenizer.tokenize(text)
        token_ids = []
        for token in tokens:
            if token in self.str2idx:
                token_ids.append(self.str2idx[token])
            elif not self.vocab_closed:
                self.str2idx[token] = len(self.idx2str)
                self.idx2str.append(token)
                token_ids.append(self.str2idx[token])
        if max_len is not None:
            while len(token_ids) < max_len:
                token_ids.append(1)
        return token_ids

    def decode(self, list_of_idx):
        if type(list_of_idx) is int:
            list_of_idx = [list_of_idx]
        ret = []
        for idx in list_of_idx:
            if idx < self.size():
                ret.append(self.idx2str[idx])
            else:
                ret.append("UNKTOKEN")
        return " ".join(ret)

    def size(self):
        return len(self.idx2str)

    def __len__(self):
        return len(self.idx2str)


class RandomKeywordSequentialDataset(datautils.Dataset):
    def __init__(self, tokenizer=None, max_len=300, keyword_num=4):
        # <s> src <sep> tgt </s>
        if tokenizer is not None:
            self.tokenizer = tokenizer
        else:
            self.tokenizer = NaiveTokenizer

        self.capacity = 16
        self.max_len = max_len
        self.size = 0
        self.sequence_buffer = torch.zeros(16, self.max_len, dtype=torch.long)
        self.clean_buffer = dict()
        self.length_buffer = torch.zeros(16, 2, dtype=torch.long)
        self.keyword_num = keyword_num

    def __len__(self):
        return self.size

    def add(self, dataset_name):
        fin = open("./data/%s/%s-train.txt" % (dataset_name, dataset_name), "r")
        for line in tqdm.tqdm(fin.readlines()):
            line = line.strip()
            line_ids = self.tokenizer.encode(text=line, add_special_tokens=False)
            clean_ids = []
            for line_tok in line.split():
                if line_tok.isalpha():
                    clean_ids.append(line_tok)
            self.sequence_buffer[self.size][0:len(line_ids)] = torch.tensor(line_ids)
            self.clean_buffer[self.size] = clean_ids
            self.length_buffer[self.size][0] = len(line_ids)
            self.length_buffer[self.size][1] = len(clean_ids)
            self.size += 1
            if self.size == self.capacity:
                self.expand_capacity()
        fin.close()

    def expand_capacity(self):
        self.sequence_buffer = torch.cat((self.sequence_buffer, torch.zeros_like(self.sequence_buffer)), dim=0)
        # self.clean_buffer = torch.cat((self.clean_buffer, torch.zeros_like(self.clean_buffer)), dim=0)
        self.length_buffer = torch.cat((self.length_buffer, torch.zeros_like(self.length_buffer)), dim=0)
        self.capacity *= 2

    def __getitem__(self, item):
        bos_id = self.tokenizer.bos_token_id
        sep_id = self.tokenizer.sep_token_id
        eos_id = self.tokenizer.eos_token_id
        seq_len = self.length_buffer[item][0]
        raw_seq = self.sequence_buffer[item][0:seq_len]
        valid_len = self.length_buffer[item][1]
        selected_id = torch.sort(torch.randperm(valid_len)[0:self.keyword_num]).values
        selected = [self.clean_buffer[item][i] for i in selected_id]
        selected = " ".join(selected)
        selected = torch.tensor(self.tokenizer.encode(text=selected, add_special_tokens=False))
        keyword_num = len(selected)
        return_ids = torch.zeros((self.max_len,), dtype=torch.long)
        return_ids[0] = bos_id
        return_ids[1:1+keyword_num] = selected
        return_ids[1+keyword_num] = sep_id
        return_ids[2+keyword_num:2+keyword_num+seq_len] = raw_seq
        return_ids[2+keyword_num+seq_len] = eos_id
        mask = torch.zeros((self.max_len-1,), dtype=torch.float)
        mask[keyword_num+1:keyword_num+1+seq_len+1] = 1.0
        return return_ids, mask, 1 + keyword_num + 1 + seq_len + 1


    def produce_keys(self, file):
        with open(file, "w") as fout:
            check_existence = set()
            for item in np.random.permutation(self.size):
                seq_len = self.length_buffer[item][0]
                valid_len = self.length_buffer[item][1]
                selected_id = torch.sort(torch.randperm(valid_len)[0:self.keyword_num]).values
                selected = [self.clean_buffer[item][i] for i in selected_id]
                selected = " ".join(selected)
                if selected not in check_existence:
                    check_existence.add(selected)
                    print(selected, file=fout)

class SplittedRandomKeywordSequentialDataset(RandomKeywordSequentialDataset):
    def __getitem__(self, item):
        bos_id = self.tokenizer.bos_token_id
        sep_id = self.tokenizer.sep_token_id
        eos_id = self.tokenizer.eos_token_id
        seq_len = self.length_buffer[item][0]
        raw_seq = self.sequence_buffer[item][0:seq_len]
        valid_len = self.length_buffer[item][1]
        selected_id = torch.sort(torch.randperm(valid_len)[0:self.keyword_num]).values
        selected = [self.clean_buffer[item][i] for i in selected_id]
        selected = " ".join(selected)
        selected = torch.tensor(self.tokenizer.encode(text=selected, add_special_tokens=False))
        keyword_num = len(selected)
        return_ids = torch.zeros((self.max_len,), dtype=torch.long)
        return_ids[0] = sep_id
        return_ids[1:1+seq_len] = raw_seq
        return_ids[1+seq_len] = eos_id
        cond_ids = torch.ones((self.max_len // 10,), dtype=torch.long) * sep_id
        cond_ids[0] = bos_id
        cond_ids[1:1 + keyword_num] = selected
        mask = torch.zeros((self.max_len - 1,), dtype=torch.float)
        mask[1:1 + seq_len + 1] = 1.0
        return cond_ids, return_ids, mask, 1 + keyword_num + 1 + seq_len + 1

class GivenKeywordSequentialDataset(datautils.Dataset):
    def __init__(self, tokenizer=None, max_len=300):
        # <s> src <sep> tgt </s>
        if tokenizer is not None:
            self.tokenizer = tokenizer
        else:
            self.tokenizer = NaiveTokenizer

        self.capacity = 16
        self.max_len = max_len
        self.size = 0
        self.sequence_buffer = torch.zeros(16, self.max_len, dtype=torch.long)
        self.length_buffer = torch.zeros(16, 2, dtype=torch.long)

    def __len__(self):
        return self.size

    def add(self, huggingface_obj, field_keywords, field_sequence):
        fin = huggingface_obj
        bos_id = self.tokenizer.bos_token_id
        sep_id = self.tokenizer.sep_token_id
        eos_id = self.tokenizer.eos_token_id
        for instance in tqdm.tqdm(huggingface_obj):
            line = instance[field_sequence]
            keywords = " ".join(instance[field_keywords])
            line_ids = self.tokenizer.encode(text=line, add_special_tokens=False)
            keyword_ids = self.tokenizer.encode(text=keywords, add_special_tokens=False)
            clean_ids = []
            for line_tok in line.split():
                if line_tok.isalpha():
                    clean_ids.append(line_tok)
            self.sequence_buffer[self.size][0] = bos_id
            self.sequence_buffer[self.size][1:len(keyword_ids)+1] = torch.tensor(keyword_ids)
            self.sequence_buffer[self.size][len(keyword_ids)+1] = sep_id
            self.sequence_buffer[self.size][len(keyword_ids)+2:len(keyword_ids)+2+len(line_ids)] = torch.tensor(line_ids)
            self.sequence_buffer[self.size][len(keyword_ids)+2+len(line_ids)] = eos_id
            self.length_buffer[self.size][0] = len(keyword_ids)
            self.length_buffer[self.size][1] = len(line_ids)
            self.size += 1
            if self.size == self.capacity:
                self.expand_capacity()

    def expand_capacity(self):
        self.sequence_buffer = torch.cat((self.sequence_buffer, torch.zeros_like(self.sequence_buffer)), dim=0)
        # self.clean_buffer = torch.cat((self.clean_buffer, torch.zeros_like(self.clean_buffer)), dim=0)
        self.length_buffer = torch.cat((self.length_buffer, torch.zeros_like(self.length_buffer)), dim=0)
        self.capacity *= 2

    def __getitem__(self, item):
        return_ids = self.sequence_buffer[item]
        keyword_num, seq_len = self.length_buffer[item][0], self.length_buffer[item][1]
        mask = torch.zeros((self.max_len-1,), dtype=torch.float)
        mask[keyword_num+1:keyword_num+1+seq_len+1] = 1.0
        return return_ids, mask, 1 + keyword_num + 1 + seq_len + 1

    def produce_keys(self, file):
        with open(file, "w") as fout:
            for i in np.random.permutation(self.size):
                key_len = self.length_buffer[i][0]
                print(self.tokenizer.decode(self.sequence_buffer[i][1:key_len+1]), file=fout)

class DomainAdaptationSequentialDataset(datautils.Dataset):
    def __init__(self, tokenizer=None, max_len=300):
        # <s> src <sep> tgt </s>
        if tokenizer is not None:
            self.tokenizer = tokenizer
        else:
            self.tokenizer = NaiveTokenizer

        self.capacity = 16
        self.max_len = max_len
        self.size = 0
        self.sequence_buffer = torch.zeros(16, self.max_len, dtype=torch.long)
        self.length_buffer = torch.zeros(16, 2, dtype=torch.long)

    def __len__(self):
        return self.size

    def add(self, dataset_name):
        fin = open("./data/%s/%s-train.txt" % (dataset_name, dataset_name), "r")
        bos_id = self.tokenizer.bos_token_id
        sep_id = self.tokenizer.sep_token_id
        eos_id = self.tokenizer.eos_token_id
        for line in tqdm.tqdm(fin.readlines()):
            line = line.strip()
            line_ids = self.tokenizer.encode(text=line, add_special_tokens=False)
            self.sequence_buffer[self.size][0] = sep_id
            self.sequence_buffer[self.size][1:1+len(line_ids)] = torch.tensor(line_ids)
            self.sequence_buffer[self.size][1+len(line_ids)] = eos_id
            self.length_buffer[self.size][0] = len(line_ids)
            self.length_buffer[self.size][1] = len(line_ids)
            self.size += 1
            if self.size == self.capacity:
                self.expand_capacity()
        fin.close()

    def add_huggingface(self, huggingface_obj, field_keywords, field_sequence):
        fin = huggingface_obj
        bos_id = self.tokenizer.bos_token_id
        sep_id = self.tokenizer.sep_token_id
        eos_id = self.tokenizer.eos_token_id
        for instance in tqdm.tqdm(huggingface_obj):
            line = instance[field_sequence]
            keywords = " ".join(instance[field_keywords])
            line_ids = self.tokenizer.encode(text=line, add_special_tokens=False)
            keyword_ids = self.tokenizer.encode(text=keywords, add_special_tokens=False)
            clean_ids = []
            for line_tok in line.split():
                if line_tok.isalpha():
                    clean_ids.append(line_tok)
            self.sequence_buffer[self.size][0] = sep_id
            self.sequence_buffer[self.size][1:1+len(line_ids)] = torch.tensor(line_ids)
            self.sequence_buffer[self.size][1+len(line_ids)] = eos_id
            self.length_buffer[self.size][0] = len(keyword_ids)
            self.length_buffer[self.size][1] = len(line_ids)
            self.size += 1
            if self.size == self.capacity:
                self.expand_capacity()

    def expand_capacity(self):
        self.sequence_buffer = torch.cat((self.sequence_buffer, torch.zeros_like(self.sequence_buffer)), dim=0)
        # self.clean_buffer = torch.cat((self.clean_buffer, torch.zeros_like(self.clean_buffer)), dim=0)
        self.length_buffer = torch.cat((self.length_buffer, torch.zeros_like(self.length_buffer)), dim=0)
        self.capacity *= 2

    def __getitem__(self, item):
        return_ids = self.sequence_buffer[item]
        keyword_num, seq_len = self.length_buffer[item][0], self.length_buffer[item][1]
        mask = torch.zeros((self.max_len-1,), dtype=torch.float)
        mask[0:seq_len+1] = 1.0
        return return_ids, mask, 1 + seq_len + 1

import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet



def nltk_pos_tagger(nltk_tag):
    if nltk_tag.startswith('J'):
        return wordnet.ADJ
    elif nltk_tag.startswith('V'):
        return wordnet.VERB
    elif nltk_tag.startswith('N'):
        return wordnet.NOUN
    elif nltk_tag.startswith('R'):
        return wordnet.ADV
    else:
        return None


def lemmatize_sentence(lemmatizer, sentence):
    nltk_tagged = nltk.pos_tag(nltk.word_tokenize(sentence))
    wordnet_tagged = map(lambda x: (x[0], nltk_pos_tagger(x[1])), nltk_tagged)
    lemmatized_sentence = []

    for word, tag in wordnet_tagged:
        if tag is None:
            lemmatized_sentence.append(word)
        else:
            lemmatized_sentence.append(lemmatizer.lemmatize(word, tag))
    return " ".join(lemmatized_sentence)

import torch.utils.data as datautils

import torch.multiprocessing as mp
from torch.multiprocessing import Pool
import torch.cuda as cuda
import time

def getitem(args):
    node_id, lines, tokenizer, base_model, expansion_num, lemmatizer, max_len = args
    N_cuda = cuda.device_count()
    if N_cuda >= 1:
        device = "cuda:%d" % (node_id % N_cuda)
    else:
        device = "cpu"
    N = len(lines)
    N_expansion_num = N * expansion_num
    generated_ = torch.ones(N_expansion_num, max_len, dtype=torch.long, device="cpu") * tokenizer.eos_token_id
    mask_ = torch.zeros(N_expansion_num, max_len, dtype=torch.float, device="cpu")
    label_ = torch.zeros(N_expansion_num, dtype=torch.float, device="cpu")
    length = torch.zeros(N_expansion_num, dtype=torch.long, device="cpu")
    base_model = base_model.to(device)
    iterator = tqdm.tqdm(lines)
    iterator.write("Node %d launched on GPU %s" % (node_id, device))
    for i, line in enumerate(iterator):
        input_ids = torch.tensor(
            [tokenizer.bos_token_id] + tokenizer.encode(line.strip()) + [tokenizer.sep_token_id], device="cpu")
        ids_t = input_ids.unsqueeze(dim=0).expand(expansion_num, input_ids.size(0)).to(device)
        generated_all = torch.ones((expansion_num, max_len), dtype=torch.long, device="cpu") * tokenizer.eos_token_id
        generated = base_model.generate(input_ids=ids_t, max_length=max_len, do_sample=True, top_p=0.80, repetitive_penalty=1.2,
                                        pad_token_id=tokenizer.eos_token_id)
        generated_all[0:expansion_num, 0:generated.shape[1]] = generated.cpu()
        del ids_t
        del generated
        seq = [None] * expansion_num
        keywords = line.strip().split()
        mask = torch.ones_like(generated_all[:, 1:]).to(torch.float)
        mask -= (generated_all[:, 0:-1] == tokenizer.eos_token_id).to(torch.float)
        mask[:, 0:input_ids.shape[0] - 1] = 0.
        label = torch.ones(size=(expansion_num,), dtype=torch.float, device="cpu")
        for ix in range(expansion_num):
            seq[ix] = lemmatize_sentence(lemmatizer,
                                        tokenizer.decode(generated_all[ix, input_ids.shape[0]:],
                                                         skip_special_tokens=True))
            for key in keywords:
                if seq[ix].count(key) == 0:
                    label[ix] = 0.0
                    break
        mask_all = torch.zeros((expansion_num, max_len), dtype=torch.float, device="cpu")
        mask_all[:, 0:mask.shape[1]] = mask.cpu()
        generated_[i * expansion_num: (i + 1) * expansion_num, :] = generated_all.cpu()
        mask_[i * expansion_num: (i + 1) * expansion_num, :] = mask_all
        label_[i * expansion_num: (i + 1) * expansion_num] = label
        length[i * expansion_num: (i + 1) * expansion_num] = (generated_all != tokenizer.eos_token_id).to(torch.long).sum(
            dim=-1).cpu()
    return generated_.cpu(), mask_.cpu(), label_.cpu(), length.cpu()

class LexicalCheckingDataset(datautils.Dataset):
    def __init__(self, tokenizer, expansion_num=32):
        self.capacity = expansion_num
        self.max_len = 300
        self.size = 0
        self.tokenizer = tokenizer
        self.sequence_buffer = torch.ones(size=(expansion_num, self.max_len), dtype=torch.long) * tokenizer.eos_token_id
        self.mask_buffer = torch.zeros(size=(expansion_num, self.max_len), dtype=torch.float)
        self.label_buffer = torch.zeros(size=(expansion_num,), dtype=torch.float)
        self.length_buffer = torch.zeros(size=(expansion_num,), dtype=torch.long)
        self.expansion_num = expansion_num


    def add(self, keys, base_model,):
        expansion_num = self.expansion_num
        tokenizer = self.tokenizer
        lemmatizer = WordNetLemmatizer()
        mp.set_start_method("spawn", force=True)
        with keys as fin:
            lines = fin.readlines()
            N_cuda = cuda.device_count()
            if N_cuda >= 1:
                N = N_cuda
            else:
                N = 1
            def split_set(lines, N):
                size = (len(lines) - 1) // N + 1
                ret_ = []
                for i in range(N):
                    if i * size < len(lines):
                        ret_.append(lines[i * size: (i + 1) * size])
                return ret_
            lines_split = split_set(lines, N)
            tick = time.time()

            with Pool(processes=N) as p:
                results = p.map(getitem, [(i, line, tokenizer, base_model, expansion_num, lemmatizer, self.max_len) for i, line in enumerate(lines_split)])

        tock = time.time()
        print("Sampling ended in", tock - tick, "s")
        self.sequence_buffer = torch.cat([generated_.cpu() for (generated_, _, _, _) in results], dim=0)
        self.mask_buffer = torch.cat([mask.cpu() for (_, mask, _, _) in results], dim=0)
        self.label_buffer = torch.cat([label.cpu() for (_, _, label, _) in results], dim=0)
        self.length_buffer = torch.cat([length.cpu() for (_, _, _, length) in results], dim=0)
        self.size = self.sequence_buffer.shape[0]
        self.capacity = self.size

    def expand_capacity(self):
        self.sequence_buffer = torch.cat((self.sequence_buffer, torch.ones_like(self.sequence_buffer) * self.tokenizer.eos_token_id), dim=0)
        self.mask_buffer = torch.cat((self.mask_buffer, torch.zeros_like(self.mask_buffer)), dim=0)
        self.label_buffer = torch.cat((self.label_buffer, torch.zeros_like(self.label_buffer)), dim=0)
        self.length_buffer = torch.cat((self.length_buffer, torch.zeros_like(self.length_buffer)), dim=0)
        self.capacity *= 2
        print("expanded")

    def __len__(self):
        return self.size

    def __getitem__(self, item):
        return self.sequence_buffer[item], self.mask_buffer[item], self.label_buffer[item], self.length_buffer[item]

def getitem_exactform(args):
    node_id, lines, tokenizer, base_model, expansion_num, lemmatizer, max_len = args
    N_cuda = cuda.device_count()
    if N_cuda >= 1:
        device = "cuda:%d" % (node_id % N_cuda)
    else:
        device = "cpu"
    base_model = base_model.to(device)
    N = len(lines)
    N_expansion_num = N * expansion_num
    generated_ = torch.ones(N_expansion_num, max_len, dtype=torch.long) * tokenizer.eos_token_id
    mask_ = torch.zeros(N_expansion_num, max_len, dtype=torch.float)
    label_ = torch.zeros(N_expansion_num, dtype=torch.float)
    length = torch.zeros(N_expansion_num, dtype=torch.long)
    for i, line in enumerate(tqdm.tqdm(lines)):
        input_ids = torch.tensor(
            [tokenizer.bos_token_id] + tokenizer.encode(line.strip()) + [tokenizer.sep_token_id])
        ids_t = input_ids.to(device).unsqueeze(dim=0).expand(expansion_num, input_ids.size(0))
        generated = base_model.generate(input_ids=ids_t, max_length=max_len, do_sample=True, top_p=0.90,
                                        pad_token_id=tokenizer.eos_token_id, output_scores=True).cpu()
        seq = [None] * expansion_num
        keywords = line.strip().split()
        mask = torch.ones_like(generated[:, 1:]).to(torch.float)
        mask -= (generated[:, 0:-1] == tokenizer.eos_token_id).to(torch.float)
        mask[:, 0:input_ids.shape[0] - 1] = 0.
        label = torch.ones(size=(expansion_num,), dtype=torch.float)
        for ix in range(expansion_num):
            seq[ix] = tokenizer.decode(generated[ix, input_ids.shape[0]:], skip_special_tokens=True)
            for key in keywords:
                if seq[ix].count(key) == 0:
                    label[ix] = 0.0
                    break
        generated_all = torch.ones((expansion_num, max_len), dtype=torch.long) * tokenizer.eos_token_id
        generated_all[:, 0:generated.shape[1]] = generated.cpu()
        mask_all = torch.zeros((expansion_num, max_len), dtype=torch.float)
        mask_all[:, 0:mask.shape[1]] = mask.cpu()
        generated_[i * expansion_num: (i + 1) * expansion_num, :] = generated_all
        mask_[i * expansion_num: (i + 1) * expansion_num, :] = mask_all
        label_[i * expansion_num: (i + 1) * expansion_num] = label
        length[i * expansion_num: (i + 1) * expansion_num] = (generated != tokenizer.eos_token_id).to(torch.long).sum(
            dim=-1).cpu()
    return generated_, mask_, label_, length

class ExactFormLexicalCheckingDataset(LexicalCheckingDataset):
    def __init__(self, tokenizer, expansion_num=32):
        super(LexicalCheckingDataset, self).__init__(tokenizer, expansion_num)
        self.capacity = expansion_num
        self.max_len = 300
        self.size = 0
        self.tokenizer = tokenizer
        self.sequence_buffer = torch.ones(size=(expansion_num, self.max_len), dtype=torch.long) * tokenizer.eos_token_id
        self.mask_buffer = torch.zeros(size=(expansion_num, self.max_len), dtype=torch.float)
        self.label_buffer = torch.zeros(size=(expansion_num,), dtype=torch.float)
        self.denominator_buffer = torch.ones(size=(expansion_num,), dtype=torch.float)
        self.length_buffer = torch.zeros(size=(expansion_num,), dtype=torch.long)
        self.expansion_num = expansion_num


    def add(self, keys, base_model,):
        expansion_num = self.expansion_num
        tokenizer = self.tokenizer
        lemmatizer = WordNetLemmatizer()
        base_model.share_memory()
        mp.set_start_method("spawn", force=True)
        with keys as fin:
            lines = fin.readlines()
            N_cuda = cuda.device_count()
            if N_cuda >= 1:
                N = N_cuda * 2
            else:
                N = 1
            def split_set(lines, N):
                size = len(lines) // N
                ret_ = []
                for i in range(N + 1):
                    if i * size < len(lines):
                        ret_.append(lines[i * size: (i + 1) * size])
                return ret_
            lines_split = split_set(lines, N)
            tick = time.time()

            with Pool(processes=N) as p:
                results = p.map(getitem_exactform, [(i, line, tokenizer, base_model, expansion_num, lemmatizer) for i, line in enumerate(lines_split)])

            tock = time.time()
            print("Sampling ended in", tock - tick, "s")
            self.sequence_buffer = torch.cat([generated_ for (generated_, _, _, _) in results], dim=0)
            self.mask_buffer = torch.cat([mask for (_, mask, _, _) in results], dim=0)
            self.label_buffer = torch.cat([label for (_, _, label, _) in results], dim=0)
            self.length_buffer = torch.cat([length for (_, _, _, length) in results], dim=0)
            self.size = self.sequence_buffer.shape[0]
            self.capacity = self.size

    def expand_capacity(self):
        self.sequence_buffer = torch.cat((self.sequence_buffer, torch.ones_like(self.sequence_buffer) * self.tokenizer.eos_token_id), dim=0)
        self.mask_buffer = torch.cat((self.mask_buffer, torch.zeros_like(self.mask_buffer)), dim=0)
        self.label_buffer = torch.cat((self.label_buffer, torch.zeros_like(self.label_buffer)), dim=0)
        self.length_buffer = torch.cat((self.length_buffer, torch.zeros_like(self.length_buffer)), dim=0)
        self.capacity *= 2
        print("expanded")

    def __len__(self):
        return self.size

    def __getitem__(self, item):
        return self.sequence_buffer[item], self.mask_buffer[item], self.label_buffer[item], self.length_buffer[item]