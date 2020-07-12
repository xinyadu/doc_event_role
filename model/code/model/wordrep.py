# -*- coding: utf-8 -*-
# @Author: Jie Yang
# @Date:   2017-10-17 16:47:32
# @Last Modified by:   Jie Yang,     Contact: jieynlp@gmail.com
# @Last Modified time: 2019-02-01 15:52:01
from __future__ import print_function
from __future__ import absolute_import
import torch
import torch.nn as nn
import numpy as np


seed_num = 42
torch.manual_seed(seed_num)
np.random.seed(seed_num)

torch.cuda.manual_seed_all(seed_num)
torch.backends.cudnn.deterministic=True

from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM, WordpieceTokenizer

class WordRep(nn.Module):
    def __init__(self, data):
        super(WordRep, self).__init__()
        print("build word representation...")
        self.gpu = data.HP_gpu
        self.batch_size = data.HP_batch_size

        self.embedding_dim = data.word_emb_dim
        self.drop = nn.Dropout(data.HP_dropout)
        self.word_embedding = nn.Embedding(data.word_alphabet.size(), self.embedding_dim)
        if data.pretrain_word_embedding is not None:
            self.word_embedding.weight.data.copy_(torch.from_numpy(data.pretrain_word_embedding))
        else:
            self.word_embedding.weight.data.copy_(torch.from_numpy(self.random_embedding(data.word_alphabet.size(), self.embedding_dim)))

        # self.hiddentoEmbdim = nn.Linear(768, self.embedding_dim)


        # bert feature
        self.word_alphabet = data.word_alphabet
        self.use_bert = data.use_bert

        if self.use_bert:
            # Load pre-trained model (weights)
            self.bert_model = BertModel.from_pretrained('bert-base-uncased')
            self.bert_model.eval()
            # Load pre-trained model tokenizer (vocabulary)
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            self.wpiecetokenizer = WordpieceTokenizer(self.tokenizer.vocab)

        if self.gpu:
            self.drop = self.drop.cuda()
            # self.hiddentoEmbdim = self.hiddentoEmbdim.cuda()
            self.word_embedding = self.word_embedding.cuda()
            if self.use_bert:
                self.bert_model = self.bert_model.cuda()

    def random_embedding(self, vocab_size, embedding_dim):
        pretrain_emb = np.empty([vocab_size, embedding_dim])
        scale = np.sqrt(3.0 / embedding_dim)
        for index in range(vocab_size):
            pretrain_emb[index,:] = np.random.uniform(-scale, scale, [1, embedding_dim])
        return pretrain_emb


    def bert_fea(self, ids_batch):
        tokens_tensor_batch = []
        context_tokens_uncased_batch = []

        for ids in ids_batch:
            context_tokens_uncased = []
            for i in ids: 
                token = self.word_alphabet.get_instance(i)
                if token == "</unk>" or not token:
                    context_tokens_uncased.append("[UNK]")
                elif token == "<PAD>":
                    context_tokens_uncased.append("[PAD]")
                else:
                    context_tokens_uncased.append(token)

            # for i, token in enumerate(context_tokens_uncased):
            #     if len(tokenizer.tokenize(token)) == 0:
            #         print(token)
            #         context_tokens_uncased[i] = token
            #     else:
            #         if token != tokenizer.tokenize(token)[0]:
            #             # context_tokens_uncased[i] = wpiecetokenizer.tokenize(token)[0]
            #             context_tokens_uncased[i] = tokenizer.tokenize(token)[0]
            context_tokens_uncased_batch.append(context_tokens_uncased)

            # Tokenized input
            # text = "[CLS] Who was Jim [UNK] ?" # tokenized_text = ["[CLS]", "who", "was", "jim", "unk", "?"] 
            # text = "[CLS] " + " ".join(context_tokens_uncased)
            # tokenized_text = tokenizer.tokenize(text)

            # Convert token to vocabulary indices
            # print(context_tokens_uncased)
            indexed_tokens = self.tokenizer.convert_tokens_to_ids(context_tokens_uncased)

            # bert_longer_cnt = len(indexed_tokens) - (len(context_tokens_uncased) + 1)
            # if bert_longer_cnt > 0:
            #     indexed_tokens = indexed_tokens[:len(context_tokens_uncased) + 1]
            # elif bert_longer_cnt < 0:
            #     indexed_tokens.extend([0] * (bert_longer_cnt * (-1)))

            # indexed_tokens = [0] * (len(ids) + 1)
            # Convert inputs to PyTorch tensors
            # tokens_tensor = torch.tensor([indexed_tokens]).to('cuda')
            tokens_tensor_batch.append(indexed_tokens)
            # Define sentence A and B indices associated to 1st and 2nd sentences (see paper)

        tokens_tensor_batch = torch.tensor(tokens_tensor_batch)
        if self.gpu:
            tokens_tensor_batch = tokens_tensor_batch.to('cuda')    

        # Predict hidden states features for each layer
        with torch.no_grad():
            encoded_layers, _ = self.bert_model(tokens_tensor_batch)

             
        # get the avg of last 4 layers hidden states (for each token)
        # batchsize * doc len * 768 (bert hidden size)
        # import ipdb; ipdb.set_trace()
        # avg = (encoded_layers[-1] + encoded_layers[-2] + encoded_layers[-3] + encoded_layers[-4])/4
        avg = sum(encoded_layers)/len(encoded_layers)

        # we do not use [CLS] fea and only use the first 100 of avg4
        context_bert_feature_batch = avg[:, :, :]

        # context_bert_feature_batch = encoded_layers[-1][:, :, :100]

        # return self.hiddentoEmbdim(context_bert_feature_batch)
        return context_bert_feature_batch

    def forward(self, word_inputs, word_seq_lengths):
        """
            input:
                word_inputs: (batch_size, sent_len)
                word_seq_lengths: list of batch_size, (batch_size,1)
            output:
                Variable(batch_size, sent_len, hidden_dim)
        """
        # batch_size = word_inputs.size(0)
        # sent_len = word_inputs.size(1)

        word_embs =  self.word_embedding(word_inputs)

        word_list = [word_embs]

        if self.use_bert:
            context_bert_feature_batch = self.bert_fea(word_inputs)
            word_list.append(context_bert_feature_batch)

        word_embs = torch.cat(word_list, 2)
        word_represent = self.drop(word_embs)

        return word_represent
