"""
script for transforming seq-tag pairs into extractions (json format)
"""

import argparse
import sys
import json
from collections import OrderedDict

tags_to_extract =['perp_individual_id', 'perp_organization_id', 'phys_tgt_id', 'hum_tgt_name', 'incident_instrument_id']
tag2category = {'perp_individual_id': "PerpInd", 'perp_organization_id':"PerpOrg", 'phys_tgt_id':"Target", 'hum_tgt_name':"Victim", 'incident_instrument_id':"Weapon"}
## input as sentence level labels

def reverse_style(input_string):
    target_position = input_string.index('[')
    input_len = len(input_string)
    output_string = input_string[target_position:input_len] + input_string[0:target_position]
    return output_string


def get_ner_BIO(label_list):
    # list_len = len(word_list)
    # assert(list_len == len(label_list)), "word list size unmatch with label list"
    list_len = len(label_list)
    begin_label = 'B-'
    inside_label = 'I-'
    whole_tag = ''
    index_tag = ''
    tag_list = []
    stand_matrix = []
    for i in range(0, list_len):
        # wordlabel = word_list[i]
        current_label = label_list[i].upper()
        if begin_label in current_label:
            if index_tag == '':
                whole_tag = current_label.replace(begin_label,"",1) +'[' +str(i)
                index_tag = current_label.replace(begin_label,"",1)
            else:
                tag_list.append(whole_tag + ',' + str(i-1))
                whole_tag = current_label.replace(begin_label,"",1)  + '[' + str(i)
                index_tag = current_label.replace(begin_label,"",1)

        elif inside_label in current_label:
            if current_label.replace(inside_label,"",1) == index_tag:
                whole_tag = whole_tag
            else:
                if (whole_tag != '')&(index_tag != ''):
                    tag_list.append(whole_tag +',' + str(i-1))
                whole_tag = ''
                index_tag = ''
        else:
            if (whole_tag != '')&(index_tag != ''):
                tag_list.append(whole_tag +',' + str(i-1))
            whole_tag = ''
            index_tag = ''

    if (whole_tag != '')&(index_tag != ''):
        tag_list.append(whole_tag)
    tag_list_len = len(tag_list)

    for i in range(0, tag_list_len):
        if  len(tag_list[i]) > 0:
            tag_list[i] = tag_list[i]+ ']'
            insert_list = reverse_style(tag_list[i])
            stand_matrix.append(insert_list)
    return stand_matrix


def read_instance(input_file, max_sent_length=250):
    in_lines = open(input_file,'r', encoding="utf8").readlines()
    instance_texts = []
    words = []
    labels = []
    doc_id = ""

    ### for sequence labeling data format i.e. CoNLL 2003
    for line in in_lines:
        if not doc_id: 
            doc_id = line.strip()
            continue
        if len(line) > 2:
            pairs = line.strip().split()
            word = pairs[0]
            # if sys.version_info[0] < 3: word = word.decode('utf-8')
            words.append(word)
            # if number_normalized:
                # word = normalize_word(word)
            label = pairs[-1]
            labels.append(label)
        else:
            if (len(words) > 0) and ((max_sent_length < 0) or (len(words) < max_sent_length)) :
                instance_texts.append([words, labels, doc_id])
                # instance_Ids.append([word_Ids, feature_Ids, char_Ids,label_Ids])
            words = []
            labels = []
            doc_id = ""
    if (len(words) > 0) and ((max_sent_length < 0) or (len(words) < max_sent_length)) :
        instance_texts.append([words, labels, doc_id])
        # instance_Ids.append([word_Ids, feature_Ids, char_Ids,label_Ids])
        words = []
        labels = []
        doc_id = ""
    return instance_texts



if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--seqfile')
    args = p.parse_args()

    # read pred spans
    instance_texts = read_instance(args.seqfile)
    doc_pred_spans = OrderedDict()
    for instance in instance_texts:
        instance_sent, pred_label_list, doc_id = instance[0], instance[1], instance[2]
        if doc_id not in doc_pred_spans:
            doc_pred_spans[doc_id] = dict()
            for tag in tags_to_extract:
                doc_pred_spans[doc_id][tag] = list()
        stand_matrix =  get_ner_BIO(pred_label_list)
        if stand_matrix: 
            for item in stand_matrix:
                target_position = item.index(']')
                offsets_str = item[:target_position + 1]
                label_str = item[target_position + 1:].lower()
                offsets = json.loads(offsets_str)
                if len(offsets) == 2:
                    span = instance_sent[offsets[0]: offsets[1] + 1]
                else:
                    span = instance_sent[offsets[0]: offsets[0] + 1]
                span = " ".join(span).replace(' ##', '')
                doc_pred_spans[doc_id][label_str].append(span)

    with open("pred.json", "w+") as predfile:
        predfile.write(json.dumps(doc_pred_spans, indent=4))