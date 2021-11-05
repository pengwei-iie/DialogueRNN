import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import pickle
import os
import re
import pandas as pd
from tqdm import tqdm

MAX_VOCAB_SIZE = 10000  # 词表长度限制
UNK, PAD = '<UNK>', '<PAD>'  # 未知字，padding符号


class IEMOCAPDataset(Dataset):

    def __init__(self, path, vocab_path, train=True):
        # self.videoIDs, self.videoSpeakers, self.videoLabels, _,\
        # self.videoAudio, self.videoVisual, self.videoSentence, self.trainVid,\
        # self.testVid = pickle.load(open(path, 'rb'), encoding='latin1')
        _, self.videoSpeakers, self.videoLabels, _, \
        _, _, self.videoSentence, self.trainVid, \
        self.testVid = pickle.load(open(path, 'rb'), encoding='latin1')
        self.label_index_mapping = {'happy': 0, 'sad': 1, 'neutral': 2, 'angry': 3, 'excioted': 4, 'frustrated': 5}

        # 以空格隔开，word-level
        tokenizer = lambda x: x.split(' ')
        pad_size = 26

        if os.path.exists(vocab_path):
            vocab = pickle.load(open(vocab_path, 'rb'))
        else:
            vocab = self.build_vocab(tokenizer=tokenizer, max_size=MAX_VOCAB_SIZE, min_freq=1)
            pickle.dump(vocab, open(vocab_path, 'wb'))

        self.keys = [x for x in (self.trainVid if train else self.testVid)]
        self.videoText = {}
        # word to id
        for key, sentences in self.videoSentence.items():
            contents = []
            for sen in sentences:
                words_line = []
                string = re.sub("[\s+\.\!\/_,$%^*(+\"\']+|[+——!,.?~@#￥%……&*（）]+", " ", sen)
                token = list(filter(lambda x: len(x) != 0, tokenizer(string)))
                seq_len = len(token)
                if pad_size:
                    if seq_len < pad_size:
                        token.extend([PAD] * (pad_size - len(token)))
                    else:
                        token = token[:pad_size]
                        seq_len = pad_size
                for word in token:
                    # if word == '':
                    #     continue
                    words_line.append(vocab.get(word, vocab.get(UNK)))
                contents.append(words_line)
            self.videoText[key] = contents
        self.len = len(self.keys)

    def __getitem__(self, index):
        vid = self.keys[index]
        return torch.LongTensor(self.videoText[vid]), \
               torch.LongTensor(self.videoText[vid]), \
               torch.LongTensor(self.videoText[vid]), \
               torch.FloatTensor([[1, 0] if x == 'M' else [0, 1] for x in \
                                  self.videoSpeakers[vid]]), \
               torch.FloatTensor([1] * len(self.videoLabels[vid])), \
               torch.LongTensor(self.videoLabels[vid]), \
               vid

    def __len__(self):
        return self.len

    def collate_fn(self, data):
        dat = pd.DataFrame(data)
        return [pad_sequence(dat[i], True) if i < 4 else pad_sequence(dat[i], True) if i < 6 else dat[i].tolist() for i
                in dat]

    def build_vocab(self, tokenizer, max_size, min_freq):
        vocab_dic = {}
        # with open(path, 'r', encoding='UTF-8') as f:
        for key, lines in tqdm(self.videoSentence.items()):
            for line in lines:
                string = re.sub("[\s+\.\!\/_,$%^*(+\"\']+|[+——!,.?~@#￥%……&*（）]+", " ", line)
                if not string:
                    continue
                for word in tokenizer(string):
                    if word == '':
                        continue
                    vocab_dic[word] = vocab_dic.get(word, 0) + 1
        vocab_list = sorted([_ for _ in vocab_dic.items() if _[1] >= min_freq], key=lambda x: x[1], reverse=True)[
                     :max_size]
        vocab_dic = {word_count[0]: idx for idx, word_count in enumerate(vocab_list)}
        vocab_dic.update({UNK: len(vocab_dic), PAD: len(vocab_dic) + 1})
        return vocab_dic


class AVECDataset(Dataset):

    def __init__(self, path, train=True):
        self.videoIDs, self.videoSpeakers, self.videoLabels, self.videoText, \
        self.videoAudio, self.videoVisual, self.videoSentence, \
        self.trainVid, self.testVid = pickle.load(open(path, 'rb'), encoding='latin1')

        self.keys = [x for x in (self.trainVid if train else self.testVid)]

        self.len = len(self.keys)

    def __getitem__(self, index):
        vid = self.keys[index]
        return torch.FloatTensor(self.videoText[vid]), \
               torch.FloatTensor(self.videoVisual[vid]), \
               torch.FloatTensor(self.videoAudio[vid]), \
               torch.FloatTensor([[1, 0] if x == 'user' else [0, 1] for x in \
                                  self.videoSpeakers[vid]]), \
               torch.FloatTensor([1] * len(self.videoLabels[vid])), \
               torch.FloatTensor(self.videoLabels[vid])

    def __len__(self):
        return self.len

    def collate_fn(self, data):
        dat = pd.DataFrame(data)
        return [pad_sequence(dat[i]) if i < 4 else pad_sequence(dat[i], True) for i in dat]


class MELDDataset(Dataset):

    def __init__(self, path, n_classes, train=True):
        if n_classes == 3:
            self.videoIDs, self.videoSpeakers, _, self.videoText, \
            self.videoAudio, self.videoSentence, self.trainVid, \
            self.testVid, self.videoLabels = pickle.load(open(path, 'rb'))
        elif n_classes == 7:
            self.videoIDs, self.videoSpeakers, self.videoLabels, self.videoText, \
            self.videoAudio, self.videoSentence, self.trainVid, \
            self.testVid, _ = pickle.load(open(path, 'rb'))
        '''
        label index mapping = {'neutral': 0, 'surprise': 1, 'fear': 2, 'sadness': 3, 'joy': 4, 'disgust': 5, 'anger':6}
        '''
        self.keys = [x for x in (self.trainVid if train else self.testVid)]

        self.len = len(self.keys)

    def __getitem__(self, index):
        vid = self.keys[index]
        return torch.FloatTensor(self.videoText[vid]), \
               torch.FloatTensor(self.videoAudio[vid]), \
               torch.FloatTensor(self.videoSpeakers[vid]), \
               torch.FloatTensor([1] * len(self.videoLabels[vid])), \
               torch.LongTensor(self.videoLabels[vid]), \
               vid

    def __len__(self):
        return self.len

    def collate_fn(self, data):
        dat = pd.DataFrame(data)
        return [pad_sequence(dat[i]) if i < 3 else pad_sequence(dat[i], True) if i < 5 else dat[i].tolist() for i in
                dat]


class DailyDialogueDataset(Dataset):

    def __init__(self, split, path):

        self.Speakers, self.InputSequence, self.InputMaxSequenceLength, \
        self.ActLabels, self.EmotionLabels, self.trainId, self.testId, self.validId = pickle.load(open(path, 'rb'))

        if split == 'train':
            self.keys = [x for x in self.trainId]
        elif split == 'test':
            self.keys = [x for x in self.testId]
        elif split == 'valid':
            self.keys = [x for x in self.validId]

        self.len = len(self.keys)

    def __getitem__(self, index):
        conv = self.keys[index]

        return torch.LongTensor(self.InputSequence[conv]), \
               torch.FloatTensor([[1, 0] if x == '0' else [0, 1] for x in self.Speakers[conv]]), \
               torch.FloatTensor([1] * len(self.ActLabels[conv])), \
               torch.LongTensor(self.ActLabels[conv]), \
               torch.LongTensor(self.EmotionLabels[conv]), \
               self.InputMaxSequenceLength[conv], \
               conv

    def __len__(self):
        return self.len


class DailyDialoguePadCollate:

    def __init__(self, dim=0):
        self.dim = dim

    def pad_tensor(self, vec, pad, dim):
        pad_size = list(vec.shape)
        pad_size[dim] = pad - vec.size(dim)
        return torch.cat([vec, torch.zeros(*pad_size).type(torch.LongTensor)], dim=dim)

    def pad_collate(self, batch):
        # find longest sequence
        max_len = max(map(lambda x: x.shape[self.dim], batch))

        # pad according to max_len
        batch = [self.pad_tensor(x, pad=max_len, dim=self.dim) for x in batch]

        # stack all
        return torch.stack(batch, dim=0)

    def __call__(self, batch):
        dat = pd.DataFrame(batch)

        return [self.pad_collate(dat[i]).transpose(1, 0).contiguous() if i == 0 else \
                    pad_sequence(dat[i]) if i == 1 else \
                        pad_sequence(dat[i], True) if i < 5 else \
                            dat[i].tolist() for i in dat]
