import os
import numpy as np
import pandas as pd
import torch
import pickle
from torch.utils.data import Dataset

def save_object(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)

def load_object(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data

# data path
train_data_path = "./data/snli_1.0/snli_1.0_train.txt"
dev_data_path = "./data/snli_1.0/snli_1.0_dev.txt"
test_data_path = "./data/snli_1.0/snli_1.0_test.txt"

class SNLI(object):
    def __init__(self, word_vector):
        self.word_vector = word_vector
        self.label2idx = {'entailment':0, 'contradiction':1, 'neutral':2} 
        self.word2idx = dict()
        self.idx2word = dict()

        self.word2idx['<PAD>'] = 0
        self.word2idx['<NULL>'] = 1
        self.idx2word[0] = '<PAD>'
        self.idx2word[1] = '<NULL>'

        self.get_vocab()
        print('data vocab size', len(self.word2idx))

        self.word_vector['<PAD>'] = torch.zeros([300])
        self.word_vector['<NULL>'] = torch.ones([300])
        print('pretrained word embedding size', len(word_vector))

        self.unseen_word_embed = dict()
        self.unseen_word_count = dict()
        
        self.train_data, self.dev_data, self.test_data = self.get_dataset()
        
        print('train', len(self.train_data))
        print('dev', len(self.dev_data))
        print('test', len(self.test_data))

        # glove --> word2vec embedding matching
        self.word_embedding = torch.zeros([len(self.word_vector), 300])
        for idx in self.idx2word:
            self.word_embedding[idx] = self.word_vector[self.idx2word[idx]]

    def get_dataset(self):
        self.train_path = train_data_path
        self.dev_path = dev_data_path
        self.test_path = test_data_path

        train_data = self.load_data(self.train_path)
        dev_data = self.load_data(self.dev_path)
        test_data = self.load_data(self.test_path)
        
        # unseen words
        for w in self.unseen_word_embed:
            if w in self.unseen_word_count:
                self.unseen_word_embed[w] /= self.unseen_word_count[w]
            self.word_vector[w] = self.unseen_word_embed[w]
    
        return train_data, dev_data, test_data

    def get_vocab(self):
        '''
        dataset_name: {'train', 'dev', 'test'}
        updates word2idx, idx2word
        '''
        print('Get vocabulary.....')
        def update_dict(path):
            with open(path, 'r') as f:
                tokens = 0
                for i, line in enumerate(f):
                    if i == 0:
                        continue
                    cols = line.rstrip().split('\t')
                    if cols[0] == '-':
                        continue
                    premise = [w for w in cols[1].split(' ') if w not in ('(', ')')]
                    hypothesis = [w for w in cols[2].split(' ') if w not in ('(', ')')]
                    tokens += len(premise + hypothesis)
                    for word in premise + hypothesis:
                        if word not in self.word2idx:
                            idx = len(self.word2idx)
                            self.word2idx[word] = idx
                            self.idx2word[idx] = word
        update_dict(train_data_path)
        update_dict(dev_data_path)
        update_dict(test_data_path)
       
    def load_data(self, path):
        data = list()
        null_idx = self.word2idx['<NULL>']

        def unseen2dict(sentence):
            window_size = 4
            for i, w in enumerate(sentence):
                if w not in self.word_vector:
                    if w not in self.unseen_word_embed:
                        self.unseen_word_embed[w] = torch.zeros([300])
                    for r in range(-window_size, window_size+1):
                        if r != 0 and 0 <= i + r < len(sentence) and sentence[i+r] in self.word_vector: 
                            self.unseen_word_embed[w] += self.word_vector[sentence[i+r]]
                            if w in self.unseen_word_count:
                                self.unseen_word_count[w] += 1
                            else:
                                self.unseen_word_count[w] = 1

#                    context_words = sentence[i-window_size:i] + sentence[i+1:i+1+window_size]
#                    context_words = [x for x in context_words if x in self.word_vector] # ignore another unseen word within window size
#                    print 'context words list', context_words
#                    unseen_embed = [self.word_vector[c] for c in context_words]
#                    if len(context_words) == 0:
#                        unseen_embed = torch.zeros([300])
#                    else:
#                        unseen_embed = sum(unseen_embed)/len(context_words)
#    
#                    self.unseen_word_embed[w] += unseen_embed
#                    self.unseen_word_count[w] += 1

#            self.unseen_word_embed[w] /= self.unseen_word_count
#            self.word_vector[w] = self.unseen_word_embed[w]
        
        with open(path, 'r') as f:
            for idx, line in enumerate(f):
                if idx == 0:
                    continue
                
                cols = line.rstrip().split('\t')
                if cols[0] == '-':
                    continue

                premise = [w for w in cols[1].split(' ') if w not in ('(', ')')]
                hypothesis = [w for w in cols[2].split(' ') if w not in ('(', ')')]
                label = self.label2idx[cols[0]]
                
                # unseen word
                unseen2dict(premise)
                unseen2dict(hypothesis)
                
                # sentence to index
                premise_idx = [self.word2idx[w] for w in premise]
                premise_idx.append(null_idx)
                hypothesis_idx = [self.word2idx[w] for w in hypothesis]

                premise_len = len(premise_idx)
                hypothesis_len = len(hypothesis_idx)
                 
                # to data
                data.append([premise_idx, premise_len, hypothesis_idx, hypothesis_len, label])
        
        return data

    def get_padded(self, data):
        """ returns mini-batch tensors from data.
        Args:
            data: list of [premise_idx, premise_len, hypothesis_idx, hypothesis_len, label]
        Returns:
            prem_idx, hypo_idx: (len(data), max_len)
            prme_len, hypo_len, label
        """
        def merge(sequences):
            lengths = [len(seq) for seq in sequences]
            lengths = torch.Tensor(lengths).long()
            padded_seqs = torch.zeros(len(sequences), max(lengths)).long()
            for i, seq in enumerate(sequences):
                end = lengths[i]
                padded_seqs[i, :end] = torch.Tensor(seq[:end]).long()
            return padded_seqs, lengths
        
        # sort data by length (descending order)
        data.sort(key=lambda x: len(x[0]), reverse=True)
        
        # seperate
        prem_idx, prem_len, hypo_idx, hypo_len, label = zip(*data)

        # merge
        prem_idx, prem_len = merge(prem_idx)
        hypo_idx, hypo_len = merge(hypo_idx)
        label = torch.Tensor(label).long()

        return prem_idx, prem_len, hypo_idx, hypo_len, label

    def get_dataloader(self, batch_size=32, shuffle=True, num_workers=2, pin_memory=True):
        train_loader = torch.utils.data.DataLoader(
                SNLIDataset(self.train_data),
                shuffle=shuffle,
                batch_size = batch_size,
                num_workers = num_workers,
                collate_fn = self.get_padded,
                pin_memory = pin_memory)

        dev_loader = torch.utils.data.DataLoader(
                SNLIDataset(self.dev_data),
                batch_size = batch_size,
                num_workers = num_workers,
                collate_fn = self.get_padded,
                pin_memory = pin_memory)

        test_loader = torch.utils.data.DataLoader(
                SNLIDataset(self.test_data),
                batch_size = batch_size,
                num_workers = num_workers,
                collate_fn = self.get_padded,
                pin_memory = pin_memory)

        return train_loader, dev_loader, test_loader

class SNLIDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

def create_dataset():

    word_vector = load_object('word_vector.pkl')
    snli_dataset = SNLI(word_vector)

    # save data as .pkl
    with open('./data/snli_dataset.pkl', 'wb') as f:
        pickle.dump(snli_dataset, f)

#    train_loader, dev_loader, test_loader = snli_dataset.get_dataloader(batch_size=32, num_workers=2, pin_memory=torch.cuda.is_available())

