
from __future__ import print_function
import time
import sys
import argparse
import random
import torch
import gc
import torch.nn as nn
import torch.optim as optim
import numpy as np
from utils.metric import get_ner_fmeasure
from model.seqlabel import SeqLabel
# from model.sentclassifier import SentClassifier
from utils.data import Data
# from eval import get_macro_avg
# from eval_no_duplicate import get_macro_avg

try:
    import cPickle as pickle
except ImportError:
    import pickle


seed_num = 42
random.seed(seed_num)
torch.manual_seed(seed_num)
np.random.seed(seed_num)

torch.cuda.manual_seed_all(seed_num)
torch.backends.cudnn.deterministic=True

def data_initialization(data):
    data.initial_feature_alphabets()
    data.build_alphabet(data.train_dir)
    data.build_alphabet(data.dev_dir)
    data.build_alphabet(data.test_dir)
    data.fix_alphabet()


def predict_check(pred_variable, gold_variable, mask_variable):
    """
        input:
            pred_variable (batch_size, sent_len): pred tag result, in numpy format
            gold_variable (batch_size, sent_len): gold result variable
            mask_variable (batch_size, sent_len): mask variable
    """
    pred = pred_variable.cpu().data.numpy()
    gold = gold_variable.cpu().data.numpy()
    mask = mask_variable.cpu().data.numpy()
    overlaped = (pred == gold)

    right_token = np.sum(overlaped * mask)
    total_token = mask.sum()
    # print("right: %s, total: %s"%(right_token, total_token))
    return right_token, total_token


def recover_label(pred_variable, gold_variable, mask_variable, label_alphabet, word_recover, sentence_classification=False):
    """
        input:
            pred_variable (batch_size, sent_len): pred tag result
            gold_variable (batch_size, sent_len): gold result variable
            mask_variable (batch_size, sent_len): mask variable
    """
    pred_variable = pred_variable[word_recover]
    gold_variable = gold_variable[word_recover]
    mask_variable = mask_variable[word_recover]
    batch_size = gold_variable.size(0)
    if sentence_classification:
        pred_tag = pred_variable.cpu().data.numpy().tolist()
        gold_tag = gold_variable.cpu().data.numpy().tolist()
        pred_label = [label_alphabet.get_instance(pred) for pred in pred_tag]
        gold_label = [label_alphabet.get_instance(gold) for gold in gold_tag]
    else:
        seq_len = gold_variable.size(1)
        mask = mask_variable.cpu().data.numpy()
        pred_tag = pred_variable.cpu().data.numpy()
        gold_tag = gold_variable.cpu().data.numpy()
        batch_size = mask.shape[0]
        pred_label = []
        gold_label = []
        for idx in range(batch_size):
            pred = [label_alphabet.get_instance(pred_tag[idx][idy]) for idy in range(seq_len) if mask[idx][idy] != 0]
            gold = [label_alphabet.get_instance(gold_tag[idx][idy]) for idy in range(seq_len) if mask[idx][idy] != 0]
            assert(len(pred)==len(gold))
            pred_label.append(pred)
            gold_label.append(gold)
    return pred_label, gold_label


def recover_nbest_label(pred_variable, mask_variable, label_alphabet, word_recover):
    """
        input:
            pred_variable (batch_size, sent_len, nbest): pred tag result
            mask_variable (batch_size, sent_len): mask variable
            word_recover (batch_size)
        output:
            nbest_pred_label list: [batch_size, nbest, each_seq_len]
    """
    # exit(0)
    pred_variable = pred_variable[word_recover]
    mask_variable = mask_variable[word_recover]
    batch_size = pred_variable.size(0)
    seq_len = pred_variable.size(1)
    nbest = pred_variable.size(2)
    mask = mask_variable.cpu().data.numpy()
    pred_tag = pred_variable.cpu().data.numpy()
    batch_size = mask.shape[0]
    pred_label = []
    for idx in range(batch_size):
        pred = []
        for idz in range(nbest):
            each_pred = [label_alphabet.get_instance(pred_tag[idx][idy][idz]) for idy in range(seq_len) if mask[idx][idy] != 0]
            pred.append(each_pred)
        pred_label.append(pred)
    return pred_label


def lr_decay(optimizer, epoch, decay_rate, init_lr):
    lr = init_lr/(1+decay_rate*epoch)
    print(" Learning rate is set as:", lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer



def evaluate(data, model, name, nbest=None):
    if name == "train":
        instances = data.train_Ids
    elif name == "dev":
        instance_texts, instances = data.dev_texts, data.dev_Ids
    elif name == 'test':
        instance_texts, instances = data.test_texts, data.test_Ids
    elif name == 'raw':
        instance_texts, instances = data.raw_texts, data.raw_Ids
    else:
        print("Error: wrong evaluate name,", name)
        exit(1)
    right_token = 0
    whole_token = 0
    nbest_pred_results = []
    pred_results = []
    gold_results = []

    sequences, doc_ids = [], []
    ## set model in eval model
    model.eval()
    batch_size = data.HP_batch_size
    start_time = time.time()
    train_num = len(instances)
    total_batch = train_num//batch_size+1
    for batch_id in range(total_batch):
        start = batch_id*batch_size
        end = (batch_id+1)*batch_size
        if end > train_num:
            end =  train_num
        instance = instances[start:end]
        instance_text = instance_texts[start:end]
        if not instance:
            continue

        batch_word, batch_wordlen, batch_wordrecover, list_sent_words_tensor, batch_label, mask  = batchify_sequence_labeling_with_label(instance, data.HP_gpu, False)
        tag_seq = model(batch_word, batch_wordlen, list_sent_words_tensor, mask)

        pred_label, gold_label = recover_label(tag_seq, batch_label, mask, data.label_alphabet, batch_wordrecover, data.sentence_classification)
        pred_results += pred_label
        gold_results += gold_label

        sequences += [item[0] for item in instance_text]
        doc_ids += [item[-1] for item in instance_text]

    # import ipdb; ipdb.set_trace()
    decode_time = time.time() - start_time
    speed = len(instances)/decode_time
    # acc, p, r, f = get_ner_fmeasure(gold_results, pred_results, data.tagScheme)
    p, r, f = get_macro_avg(sequences, pred_results, doc_ids)
    return speed, p, r, f, pred_results


def batchify_sequence_labeling_with_label(input_batch_list, gpu, if_train=True):
    """
        input: list of words, chars and labels, various length. [[words, features, chars, labels],[words, features, chars,labels],...]
            words: word ids for one sentence. (batch_size, sent_len)
            labels: label ids for one sentence. (batch_size, sent_len)

        output:
            zero padding for word and char, with their batch length
            word_seq_tensor: (batch_size, max_sent_len) Variable
            word_seq_lengths: (batch_size,1) Tensor
            label_seq_tensor: (batch_size, max_sent_len)
            mask: (batch_size, max_sent_len)
    """
    batch_size = len(input_batch_list)
    words = [sent[0] for sent in input_batch_list]
    sent_words = [sent[1] for sent in input_batch_list]
    labels = [sent[2] for sent in input_batch_list]

    word_seq_lengths = torch.LongTensor(list(map(len, words)))
    max_seq_len = word_seq_lengths.max().item()
    word_seq_tensor = torch.zeros((batch_size, max_seq_len), requires_grad =  if_train).long()
    label_seq_tensor = torch.zeros((batch_size, max_seq_len), requires_grad =  if_train).long()

    mask = torch.zeros((batch_size, max_seq_len), requires_grad =  if_train).byte()
    for idx, (seq, label, seqlen) in enumerate(zip(words, labels, word_seq_lengths)):
        seqlen = seqlen.item()
        word_seq_tensor[idx, :seqlen] = torch.LongTensor(seq)
        label_seq_tensor[idx, :seqlen] = torch.LongTensor(label)
        mask[idx, :seqlen] = torch.Tensor([1]*seqlen)
    word_seq_lengths, word_perm_idx = word_seq_lengths.sort(0, descending=True)
    word_seq_tensor = word_seq_tensor[word_perm_idx]

    label_seq_tensor = label_seq_tensor[word_perm_idx]
    mask = mask[word_perm_idx]

    _, word_seq_recover = word_perm_idx.sort(0, descending=False)


    list_sent_words_tensor = []
    for sent_words_one_example in sent_words:
        one_example_list = []
        for sent in sent_words_one_example:
            sent_tensor = torch.zeros((1, len(sent)), requires_grad =  if_train).long()
            sent_tensor[0, :len(sent)] = torch.LongTensor(sent)
            if gpu:
                one_example_list.append(sent_tensor.cuda())
            else:
                one_example_list.append(sent_tensor)
        list_sent_words_tensor.append(one_example_list)

    word_perm_idx = word_perm_idx.data.numpy().tolist()
    list_sent_words_tensor_perm = []
    for idx in word_perm_idx:
        list_sent_words_tensor_perm.append(list_sent_words_tensor[idx])

    if gpu:
        word_seq_tensor = word_seq_tensor.cuda()
        word_seq_lengths = word_seq_lengths.cuda()
        word_seq_recover = word_seq_recover.cuda()
        label_seq_tensor = label_seq_tensor.cuda()
        mask = mask.cuda()
    return word_seq_tensor, word_seq_lengths, word_seq_recover, list_sent_words_tensor_perm, label_seq_tensor, mask


def train(data):
    print("Training model...")
    data.show_data_summary()
    save_data_name = data.model_dir +".dset"
    data.save(save_data_name)
    if data.sentence_classification:
        model = SentClassifier(data)
    else:
        model = SeqLabel(data)

    if data.optimizer.lower() == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=data.HP_lr, momentum=data.HP_momentum,weight_decay=data.HP_l2)
    elif data.optimizer.lower() == "adagrad":
        optimizer = optim.Adagrad(model.parameters(), lr=data.HP_lr, weight_decay=data.HP_l2)
    elif data.optimizer.lower() == "adadelta":
        optimizer = optim.Adadelta(model.parameters(), lr=data.HP_lr, weight_decay=data.HP_l2)
    elif data.optimizer.lower() == "rmsprop":
        optimizer = optim.RMSprop(model.parameters(), lr=data.HP_lr, weight_decay=data.HP_l2)
    elif data.optimizer.lower() == "adam":
        optimizer = optim.Adam(model.parameters(), lr=data.HP_lr, weight_decay=data.HP_l2)
    else:
        print("Optimizer illegal: %s"%(data.optimizer))
        exit(1)
    best_dev = -10
    # data.HP_iteration = 1
    ## start training
    for idx in range(data.HP_iteration):
        epoch_start = time.time()
        temp_start = epoch_start
        print("\nEpoch: %s/%s" %(idx,data.HP_iteration))
        if data.optimizer == "SGD":
            optimizer = lr_decay(optimizer, idx, data.HP_lr_decay, data.HP_lr)
        instance_count = 0
        sample_id = 0
        sample_loss = 0
        total_loss = 0
        right_token = 0
        whole_token = 0
        random.shuffle(data.train_Ids)
        print("Shuffle: first input word list:", data.train_Ids[0][0])
        ## set model in train model
        model.train()
        model.zero_grad()
        batch_size = data.HP_batch_size
        batch_id = 0
        train_num = len(data.train_Ids)
        total_batch = train_num//batch_size+1
        for batch_id in range(total_batch):
            start = batch_id*batch_size
            end = (batch_id+1)*batch_size
            if end >train_num:
                end = train_num
            instance = data.train_Ids[start:end]
            if not instance:
                continue
            batch_word, batch_wordlen, batch_wordrecover, list_sent_words_tensor, batch_label, mask  = batchify_sequence_labeling_with_label(instance, data.HP_gpu, True)    
            instance_count += 1
            loss, tag_seq = model.calculate_loss(batch_word, batch_wordlen, list_sent_words_tensor, batch_label, mask)
            right, whole = predict_check(tag_seq, batch_label, mask)
            right_token += right
            whole_token += whole
            # print("loss:",loss.item())
            sample_loss += loss.item()
            total_loss += loss.item()
            if end%500 == 0:
                temp_time = time.time()
                temp_cost = temp_time - temp_start
                temp_start = temp_time
                print("     Instance: %s; Time: %.2fs; loss: %.4f; acc: %s/%s=%.4f"%(end, temp_cost, sample_loss, right_token, whole_token,(right_token+0.)/whole_token))
                if sample_loss > 1e8 or str(sample_loss) == "nan":
                    print("ERROR: LOSS EXPLOSION (>1e8) ! PLEASE SET PROPER PARAMETERS AND STRUCTURE! EXIT....")
                    exit(1)
                sys.stdout.flush()
                sample_loss = 0
            loss.backward()
            optimizer.step()
            model.zero_grad()
        temp_time = time.time()
        temp_cost = temp_time - temp_start
        print("     Instance: %s; Time: %.2fs; loss: %.4f; acc: %s/%s=%.4f"%(end, temp_cost, sample_loss, right_token, whole_token,(right_token+0.)/whole_token))

        epoch_finish = time.time()
        epoch_cost = epoch_finish - epoch_start
        print("Epoch: %s training finished. Time: %.2fs, speed: %.2fst/s,  total loss: %s"%(idx, epoch_cost, train_num/epoch_cost, total_loss))
        print("totalloss:", total_loss)
        if total_loss > 1e8 or str(total_loss) == "nan":
            print("ERROR: LOSS EXPLOSION (>1e8) ! PLEASE SET PROPER PARAMETERS AND STRUCTURE! EXIT....")
            exit(1)

        # continue
        speed, p, r, f, _ = evaluate(data, model, "dev")
        dev_finish = time.time()
        dev_cost = dev_finish - epoch_finish

        # if data.seg:
        current_score = f
        print("Dev: time: %.2fs, speed: %.2fst/s; p: %.4f, r: %.4f, f: %.4f"%(dev_cost, speed, p, r, f))
        # else:
        #     current_score = acc
        #     print("Dev: time: %.2fs speed: %.2fst/s; acc: %.4f"%(dev_cost, speed, acc))

        if current_score > best_dev:
            # if data.seg:
            print("!!!Exceed previous best f score:", best_dev)
            # else:
                # print("!!!Exceed previous best acc score:", best_dev)
            model_name = data.model_dir +'.'+ str(idx) + ".model"
            print("Save current best model in file:", model_name)
            torch.save(model.state_dict(), data.model_dir + '.' + 'best' + ".model")
            best_dev = current_score
        model_name = data.model_dir +'.'+ str(idx) + ".model"
        torch.save(model.state_dict(), model_name)

        # ## decode test
        # speed, acc, p, r, f, _,_ = evaluate(data, model, "test")
        # test_finish = time.time()
        # test_cost = test_finish - dev_finish
        # if data.seg:
        #     print("Test: time: %.2fs, speed: %.2fst/s; acc: %.4f, p: %.4f, r: %.4f, f: %.4f"%(test_cost, speed, acc, p, r, f))
        # else:
        #     print("Test: time: %.2fs, speed: %.2fst/s; acc: %.4f"%(test_cost, speed, acc))
        gc.collect()


def load_model_decode(data, name):
    print("Load Model from file: ", data.model_dir)
    if data.sentence_classification:
        model = SentClassifier(data)
    else:
        model = SeqLabel(data)
    # model = SeqModel(data)
    ## load model need consider if the model trained in GPU and load in CPU, or vice versa
    # if not gpu:
    #     model.load_state_dict(torch.load(model_dir))
    #     # model.load_state_dict(torch.load(model_dir), map_location=lambda storage, loc: storage)
    #     # model = torch.load(model_dir, map_location=lambda storage, loc: storage)
    # else:
    #     model.load_state_dict(torch.load(model_dir))
    #     # model = torch.load(model_dir)
    model.load_state_dict(torch.load(data.load_model_dir))

    # print("Decode %s data, nbest: %s ..."%(name, data.nbest))
    start_time = time.time()
    speed, p, r, f, pred_results = evaluate(data, model, name, data.nbest)
    end_time = time.time()
    time_cost = end_time - start_time
    # if data.seg:
    #     print("%s: time:%.2fs, speed:%.2fst/s; acc: %.4f, p: %.4f, r: %.4f, f: %.4f"%(name, time_cost, speed, acc, p, r, f))
    # else:
    #     print("%s: time:%.2fs, speed:%.2fst/s; acc: %.4f"%(name, time_cost, speed, acc))
    return pred_results




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Tuning with NCRF++')
    # parser.add_argument('--status', choices=['train', 'decode'], help='update algorithm', default='train')
    parser.add_argument('--config',  help='Configuration File', default='None')
    parser.add_argument('--wordemb',  help='Embedding for words', default='None')
    parser.add_argument('--charemb',  help='Embedding for chars', default='None')
    parser.add_argument('--status', choices=['train', 'decode'], help='update algorithm', default='train')
    parser.add_argument('--savemodel', default="data/model/saved_model.lstmcrf.")
    parser.add_argument('--savedset', help='Dir of saved data setting')
    parser.add_argument('--train', default="data/conll03/train.bmes") 
    parser.add_argument('--dev', default="data/conll03/dev.bmes" )  
    parser.add_argument('--test', default="data/conll03/test.bmes") 
    parser.add_argument('--seg', default="True") 
    parser.add_argument('--raw') 
    parser.add_argument('--loadmodel')
    parser.add_argument('--output') 

    args = parser.parse_args()
    data = Data()
    data.HP_gpu = torch.cuda.is_available()
    if args.config == 'None':
        data.train_dir = args.train 
        data.dev_dir = args.dev 
        data.test_dir = args.test
        data.model_dir = args.savemodel
        data.dset_dir = args.savedset
        print("Save dset directory:",data.dset_dir)
        save_model_dir = args.savemodel
        data.word_emb_dir = args.wordemb
        data.char_emb_dir = args.charemb
        if args.seg.lower() == 'true':
            data.seg = True
        else:
            data.seg = False
        print("Seed num:",seed_num)
    else:
        data.read_config(args.config)
    # data.show_data_summary()
    status = data.status.lower()
    print("Seed num:",seed_num)

    if status == 'train':
        print("MODEL: train")
        data_initialization(data)
        data.generate_instance('train')
        data.generate_instance('dev')
        data.generate_instance('test')
        data.build_pretrain_emb()
        train(data)

        print("\n\n\nMODEL: decode")
        data.load(data.dset_dir)
        decode_results = load_model_decode(data, 'test')
        data.write_decoded_results(decode_results, 'test')

    elif status == 'decode':
        print("MODEL: decode")
        data.load(data.dset_dir)
        data.read_config(args.config)
        print(data.raw_dir)

        data.generate_instance('raw')
        print("nbest: %s"%(data.nbest))
        decode_results = load_model_decode(data, 'raw')
        data.write_decoded_results(decode_results, 'raw')
    else:
        print("Invalid argument! Please use valid arguments! (train/test/decode)")

