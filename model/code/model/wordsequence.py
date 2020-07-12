# -*- coding: utf-8 -*-
# @Author: Jie Yang
# @Date:   2017-10-17 16:47:32
# @Last Modified by:   Jie Yang,     Contact: jieynlp@gmail.com
# @Last Modified time: 2019-02-01 15:59:26
from __future__ import print_function
from __future__ import absolute_import
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from .wordrep import WordRep
import numpy as np

seed_num = 42
torch.manual_seed(seed_num)

torch.cuda.manual_seed_all(seed_num)
torch.backends.cudnn.deterministic=True

class WordSequence(nn.Module):
    def __init__(self, data):
        super(WordSequence, self).__init__()
        print("build word sequence feature extractor: %s..."%(data.word_feature_extractor))
        self.gpu = data.HP_gpu
        # self.batch_size = data.HP_batch_size
        # self.hidden_dim = data.HP_hidden_dim
        self.droplstm = nn.Dropout(data.HP_dropout)
        self.droplstm_sent = nn.Dropout(data.HP_dropout-0.1)
        self.bilstm_flag = data.HP_bilstm
        self.lstm_layer = data.HP_lstm_layer
        self.wordrep = WordRep(data)
        self.input_size = data.word_emb_dim

        # bert fea size
        if data.use_bert:
            self.input_size += 768
        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        if self.bilstm_flag:
            lstm_hidden = data.HP_hidden_dim // 2
        else:
            lstm_hidden = data.HP_hidden_dim
        self.lstm_hidden = lstm_hidden

        # self.word_feature_extractor = data.word_feature_extractor
        # if self.word_feature_extractor == "GRU":
            # self.lstm = nn.GRU(self.input_size, lstm_hidden, num_layers=self.lstm_layer, batch_first=True, bidirectional=self.bilstm_flag)
        # elif self.word_feature_extractor == "LSTM":
        self.lstm = nn.LSTM(self.input_size, lstm_hidden, num_layers=self.lstm_layer, batch_first=True, bidirectional=self.bilstm_flag)

        self.sent_lstm = nn.LSTM(self.input_size, lstm_hidden, num_layers=self.lstm_layer, batch_first=True, bidirectional=self.bilstm_flag)

        self.lstm2 = nn.LSTM(lstm_hidden*2, lstm_hidden, num_layers=self.lstm_layer, batch_first=True, bidirectional=self.bilstm_flag)

        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(data.HP_hidden_dim, data.label_alphabet_size)
        # self.hidden2tag = nn.Linear(data.HP_hidden_dim * 2, data.label_alphabet_size)
        self.hidden2tag_sent_level = nn.Linear(data.HP_hidden_dim, data.label_alphabet_size)

        self.gate = nn.Linear(data.HP_hidden_dim * 2, data.HP_hidden_dim)

        self.sigmoid = nn.Sigmoid()

        if self.gpu:
            self.droplstm = self.droplstm.cuda()
            self.droplstm_sent = self.droplstm_sent.cuda()
            self.hidden2tag = self.hidden2tag.cuda()
            self.hidden2tag_sent_level = self.hidden2tag_sent_level.cuda()
            self.lstm = self.lstm.cuda()
            self.sent_lstm = self.sent_lstm.cuda()

            self.gate = self.gate.cuda()
            self.sigmoid = self.sigmoid.cuda()


    def get_sent_rep(self, sent, sent_length):
        word_represent = self.wordrep(sent, sent_length)

        ## word_embs (batch_size, seq_len, embed_size)
        packed_words = pack_padded_sequence(word_represent, sent_length, True)
        hidden = None
        lstm_out, hidden = self.sent_lstm(packed_words, hidden)
        lstm_out, _ = pad_packed_sequence(lstm_out)
        ## lstm_out (seq_len, seq_len, hidden_size)
        feature_out_sent = self.droplstm_sent(lstm_out.transpose(1,0))
        ## feature_out (batch_size, seq_len, hidden_size)
        return feature_out_sent

    def forward(self, word_inputs, list_sent_words_tensor, word_seq_lengths):
        """
            input:
                word_inputs: (batch_size, sent_len)
                feature_inputs: [(batch_size, sent_len), ...] list of variables
                word_seq_lengths: list of batch_size, (batch_size,1)
            output:
                Variable(batch_size, sent_len, hidden_dim)
        """
        word_represent = self.wordrep(word_inputs, word_seq_lengths)

        # #!!! sent-level reps
        # word_represent_sents = torch.zeros((word_represent.size()[0], word_represent.size()[1], word_represent.size()[2]), requires_grad =  False).float()
        # if self.gpu:
        #     word_represent_sents.cuda()

        # for idx, seq in enumerate(list_sent_words_tensor):
        #     feature_out_seq = []
        #     for sent in seq:
        #         feature_out_sent = self.wordrep(sent, np.array([len(sent[0])]))
        #         feature_out_seq.append(feature_out_sent.squeeze(0))

        #     feature_out_seq = torch.cat(feature_out_seq, 0)
        #     # feature_out[idx][:len(feature_out_seq)][:] += 0*feature_out_seq
        #     # feature_out[idx][:len(feature_out_seq)][:] += feature_out_seq
        #     word_represent_sents[idx][:len(feature_out_seq)][:] = feature_out_seq
        # # import ipdb; ipdb.set_trace()

        # word_represent_concat = torch.cat((word_represent, word_represent_sents[:100]), 2)

        ## word_embs (batch_size, seq_len, embed_size)
        packed_words = pack_padded_sequence(word_represent, word_seq_lengths.cpu().numpy(), True)
        hidden = None
        lstm_out, hidden = self.lstm(packed_words, hidden)
        lstm_out, _ = pad_packed_sequence(lstm_out)
        ## lstm_out (seq_len, seq_len, hidden_size)
        feature_out = self.droplstm(lstm_out.transpose(1,0))
        # feature_out = lstm_out.transpose(1,0)
        ## feature_out (batch_size, seq_len, hidden_size)


        
        #!!! sent-level reps
        feature_out_sents = torch.zeros((feature_out.size()[0], feature_out.size()[1], feature_out.size()[2]), requires_grad =  False).float()
        if self.gpu:
            feature_out_sents = feature_out_sents.cuda()

        for idx, seq in enumerate(list_sent_words_tensor):
            feature_out_seq = []
            for sent in seq:
                feature_out_sent = self.get_sent_rep(sent, np.array([len(sent[0])]))
                feature_out_seq.append(feature_out_sent.squeeze(0))

            # if idx > 1:
                # import ipdb; ipdb.set_trace()
            feature_out_seq = torch.cat(feature_out_seq, 0)
            
            if self.gpu:
                feature_out_seq.cuda()
            # feature_out[idx][:len(feature_out_seq)][:] += 0*feature_out_seq
            # feature_out[idx][:len(feature_out_seq)][:] += feature_out_seq
            feature_out_sents[idx][:len(feature_out_seq)][:] = feature_out_seq

        # feature_out_concat = torch.cat((feature_out, feature_out_sents), 2)
        # if self.gpu:
            # feature_out_sents.cuda()

        # outputs = self.hidden2tag(feature_out)
        # outputs_sent_level = self.hidden2tag_sent_level(feature_out_sents)
        # outputs = self.hidden2tag(feature_out_concat)

        # outputs_final = outputs + outputs_sent_level
        # outputs_final = outputs
        gamma = self.sigmoid(self.gate(torch.cat((feature_out, feature_out_sents), 2)))
        outputs_final = self.hidden2tag(gamma * feature_out + (1-gamma) * feature_out_sents)
        return outputs_final
