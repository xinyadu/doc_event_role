"""
process the train/dev/test file to sent tagging format files
"""

import argparse
import json
import random
from collections import OrderedDict
import spacy
from pytorch_pretrained_bert import BertTokenizer
# random.seed(0) # seed will affect data prop
tags_to_extract =['incident_instrument_id', 'perp_individual_id', 'perp_organization_id', 'phys_tgt_id', 'hum_tgt_name']

def create_sent_tagging(doc_keys):
# def create_sent_tagging(doc_dict, keys_dict_none_empty):
    
    # turn doc_keys (entity) into doc_keys (mentions)
    for docid in doc_keys:
        for role in doc_keys[docid]["roles"]:
            mentions = list()
            for entity in doc_keys[docid]["roles"][role]:
                for mention in entity:
                    if mention not in mentions:
                        mentions.append(mention)
            doc_keys[docid]["roles"][role] = mentions

    #
    seqs_all_o = []
    seqs_not_all_o = []
    all_examples = []
    para_lens = []
    # summ = 0
    for key in doc_keys:
        if int(key.split('-')[-1]) % 100 == 0:
            print(key)
        # if key not in keys_dict_none_empty: continue

        # get and sort doc-level spans to extract from doc key
        doc = doc_keys[key]["doc"]
        tags_values = doc_keys[key]["roles"]
        for tag in tags_values:
            values = tags_values[tag]
            # values.sort(key=lambda x: len(x) *(-1))
            values.sort(key=lambda x: len(x))


        # get all the sentences from this doc key
        paragraphs = doc.split("\n\n")
        doc_sents = []
        for para in paragraphs:
            para2 = " ".join(para.split("\n")).lower()
            para2 = nlp(para2)
            cnt = 0
            for sent in para2.sents:
                cnt += 1
                doc_sents.append(sent.text)
            para_lens.append(cnt)


        # get seqs and annotate
        num_sent_to_include = 3
        for idx in range(len(doc_sents)):
            if args.div == 'test':
                start = idx * num_sent_to_include
            else:
                start = idx
            end = start + num_sent_to_include
            if start >= len(doc_sents): break

            if end > len(doc_sents): end = len(doc_sents)

            sequence = " ".join(doc_sents[start: end])

            
            all_o = True
            seq_tokenized = tokenizer.tokenize(sequence)
            seq_tag_pair = [[token, 'O'] for token in seq_tokenized]
            for tag_anno in tags_values:
                values = tags_values[tag_anno]
                for value in values:
                    value_tokenized = tokenizer.tokenize(value)
                    for idx, token_tag in enumerate(seq_tag_pair):
                        token, tag = token_tag[0], token_tag[1]
                        if token == value_tokenized[0]:
                            start, end = idx, idx + len(value_tokenized)
                            if end <= len(seq_tag_pair):
                                candidate = [x[0] for x in seq_tag_pair[start: end]]

                                tags = [x[1] for x in seq_tag_pair[start: end]] 

                                already_annoted = False
                                for tag in tags: 
                                    if tag != 'O': already_annoted = True
                                if already_annoted: continue

                                if " ".join(candidate) == " ".join(value_tokenized):
                                    all_o = False
                                    seq_tag_pair[start][1] = "B-" + tag_anno
                                    for i in range(start + 1, end):
                                        seq_tag_pair[i][1] = "I-" + tag_anno


            all_examples.append([key, seq_tag_pair])
            if not all_o:
                seqs_not_all_o.append([key, seq_tag_pair])
            else:
                seqs_all_o.append([key, seq_tag_pair])

    seqs_all_o_sample = random.sample(seqs_all_o, len(seqs_not_all_o))
    all_examples_sample_neg = seqs_not_all_o + seqs_all_o_sample
    print("Average paragraph sent # :", sum(para_lens)/len(para_lens))

    return all_examples_sample_neg, all_examples
    

if __name__=='__main__':

    p = argparse.ArgumentParser()
    p.add_argument('--div', default='train', choices=['train', 'dev', 'test'])
    args = p.parse_args()

    # Load pre-trained model tokenizer (vocabulary)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # spacy
    nlp = spacy.load("en_core_web_sm")

    # doc_dict, keys_dict = read_files(doc_file, keys_file)
    with open("../data/processed/" + args.div + "_full.json", "r") as doc_keys_file:
        doc_keys = json.load(doc_keys_file, object_pairs_hook=OrderedDict)

    # print(all_values_list)
    
    # for key in keys_dict_none_empty:
        # print(key, keys_dict_none_empty[key])
        # import ipdb; ipdb.set_trace()

    # check
    # for key in doc_dict:
        # assert key in keys_dict

    # all_examples_sample_neg, all_examples = create_sent_tagging(doc_dict, keys_dict)
    all_examples_sample_neg, all_examples = create_sent_tagging(doc_keys)

    # write to files
    if args.div == 'test':
        output_examples = all_examples
        f = open("./data_seq_tag_pairs/" + args.div, "w+") 
    else:
        output_examples = all_examples_sample_neg
        f = open("./data_seq_tag_pairs/" + args.div + "_full", "w+") 

    for example in output_examples:
        f.write(example[0] + "\n")
        seq_tag_pair = example[1]
        for pair in seq_tag_pair:
            f.write(pair[0] + " " + pair[1] + "\n")
        f.write("\n")