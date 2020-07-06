"""
eval script
"""

import argparse
import sys
import json
import re
import string
import copy
from collections import OrderedDict
import spacy
key2role = OrderedDict([('perp_individual_id', "PerpInd"), ('perp_organization_id',"PerpOrg"), ('phys_tgt_id',"Target"), ('hum_tgt_name',"Victim"), ('incident_instrument_id',"Weapon")])

def normalize_string(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        regex = re.compile(r'\b(a|an|the)\b', re.UNICODE)
        return re.sub(regex, ' ', text)
    def white_space_fix(text):
        return ' '.join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))

def match_exact(preds, golds):
    for pred_mention in preds:
        for gold_mention in golds:
            if pred_mention == gold_mention:
                return True
    return False

def remove_duplicate(items):
    items_no_dup = list()
    for item in items:
        if item not in items_no_dup:
            items_no_dup.append(item)
    return items_no_dup

def match_head_noun(preds, golds):
    for pred in preds:
        for gold in golds:
            if pred['span'] == gold['span']: # must have this line (no head noun in 'fmln')
                return True
            if pred['hn'] and gold['hn']:
                for n1 in pred['hn']:
                    for n2 in gold['hn']:
                        if n1 == n2:
                            return True
    return False

def remove_duplicate_head_noun(items):
    items_no_dup = list()
    items_no_dup_hn = list()
    for item in items:
        if item["hn"] not in items_no_dup_hn or not item["hn"]:
            items_no_dup.append(item)
            items_no_dup_hn.append(item["hn"])
    return items_no_dup


### eval methods

def eval_exact(doc_pred_mentions, doc_gold_entitys):
    ## normalize pred mentions 
    for doc_id in doc_pred_mentions:
        for key in key2role:
            mentions_norm = list()
            for span in doc_pred_mentions[doc_id][key]:
                mentions_norm.append(normalize_string(span))
            doc_pred_mentions[doc_id][key] = remove_duplicate(mentions_norm)

    ## normalize gold
    for doc_id in doc_gold_entitys:
        for key in key2role:
            entitys_norm = list()
            for entity in doc_gold_entitys[doc_id]["roles"][key]:
                mentions_norm = list()
                for span in entity:
                    mentions_norm.append(normalize_string(span))
                entitys_norm.append(mentions_norm)
            doc_gold_entitys[doc_id]["roles"][key] = entitys_norm


    print("================Exact Match=================") 
    print("Prec, Recall, F-1")
    prec_macro, recall_macro = 0, 0
    for key in key2role:
        gold_entity_num, right_entity_num, pred_mention_num, right_mention_num = 0, 0, 0, 0
        for doc_id in doc_gold_entitys:
            gold_entitys = doc_gold_entitys[doc_id]["roles"][key]
            if doc_id not in doc_pred_mentions: continue
            pred_mentions = doc_pred_mentions[doc_id][key]
            # for recall
            for entity in gold_entitys:
                gold_entity_num += 1
                if match_exact(pred_mentions, entity):
                    right_entity_num += 1
            # for prec
            for mention in pred_mentions:
                pred_mention_num += 1
                all_entitys = list()
                for entity in gold_entitys:
                    all_entitys += entity
                if match_exact([mention], all_entitys):
                    right_mention_num += 1

        if gold_entity_num: 
            recall = (right_entity_num + 0.0) / gold_entity_num
        else:
            recall = -1
        if pred_mention_num: 
            prec = (right_mention_num + 0.0) / pred_mention_num
        else:
            prec = -1
        if prec <= 0 or recall <= 0:
            f_measure = -1
        else:
            prec *= 100
            recall *= 100
            f_measure = 2*prec*recall/(prec+recall)

        print("%s\n%.4f %.4f %.4f"%(key2role[key], prec, recall, f_measure))
        prec_macro += prec
        recall_macro += recall

    prec_macro = prec_macro / len(key2role)
    recall_macro = recall_macro / len(key2role)
    f_measure_macro = 2*prec_macro*recall_macro/(prec_macro+recall_macro)
    print("MACRO average:")
    print("%.4f %.4f %.4f"%(prec_macro, recall_macro, f_measure_macro))

def eval_head_noun(doc_pred_mentions, doc_gold_entitys):
    nlp = spacy.load("en_core_web_sm") # for finding head noun
    ## pred
    doc_pred_mentions_head_noun = dict()
    for doc_id in doc_pred_mentions:
        doc_pred_mentions_head_noun[doc_id] = dict()
        for key in key2role:
            doc_pred_mentions_head_noun[doc_id][key] = list()
            for mention in doc_pred_mentions[doc_id][key]:
                mention_norm = normalize_string(mention)
                head_noun = list()
                noun_chunks = list(nlp(mention_norm).noun_chunks)
                for noun_chunk in noun_chunks: 
                    head_noun.append(noun_chunk.root.text)
                doc_pred_mentions_head_noun[doc_id][key].append({"span": mention_norm, "hn": head_noun})
            doc_pred_mentions_head_noun[doc_id][key] = remove_duplicate_head_noun(doc_pred_mentions_head_noun[doc_id][key])


    ## gold 
    doc_gold_entitys_head_noun = dict()
    for doc_id in doc_gold_entitys:
        doc_gold_entitys_head_noun[doc_id] = dict()
        for key in key2role:
            doc_gold_entitys_head_noun[doc_id][key] = list()
            for entity in doc_gold_entitys[doc_id]["roles"][key]:
                entity_head_noun = list()
                for mention in entity:
                    mention_norm = normalize_string(mention)
                    head_noun = list()
                    noun_chunks = list(nlp(mention_norm).noun_chunks)
                    for noun_chunk in noun_chunks: 
                        head_noun.append(noun_chunk.root.text)
                    entity_head_noun.append({"span": mention_norm, "hn": head_noun})
                doc_gold_entitys_head_noun[doc_id][key].append(entity_head_noun)

    ## report head noun
    print("===============Head Noun Match===============") 
    print("Prec, Recall, F-1")
    prec_macro, recall_macro = 0, 0
    for key in key2role:
        gold_entity_num, right_entity_num, pred_mention_num, right_mention_num = 0, 0, 0, 0
        for doc_id in doc_gold_entitys_head_noun:
            gold_entitys = doc_gold_entitys_head_noun[doc_id][key]
            if doc_id not in doc_pred_mentions_head_noun: continue
            pred_mentions = doc_pred_mentions_head_noun[doc_id][key]
            # for recall
            for entity in gold_entitys:
                gold_entity_num += 1
                if match_head_noun(pred_mentions, entity):
                    right_entity_num += 1
            # for prec
            for mention in pred_mentions:
                pred_mention_num += 1
                all_entitys = list()
                for entity in gold_entitys:
                    all_entitys += entity
                if match_head_noun([mention], all_entitys):
                    right_mention_num += 1

        if gold_entity_num: 
            recall = (right_entity_num + 0.0) / gold_entity_num
        else:
            recall = -1
        if pred_mention_num: 
            prec = (right_mention_num + 0.0) / pred_mention_num
        else:
            prec = -1
        if prec <= 0 or recall <= 0:
            f_measure = -1
        else:
            prec *= 100
            recall *= 100
            f_measure = 2*prec*recall/(prec+recall)

        print("%s\n%.4f %.4f %.4f"%(key2role[key], prec, recall, f_measure))
        prec_macro += prec
        recall_macro += recall

    prec_macro = prec_macro / len(key2role)
    recall_macro = recall_macro / len(key2role)
    f_measure_macro = 2*prec_macro*recall_macro/(prec_macro+recall_macro)
    print("MACRO average:")
    print("%.4f %.4f %.4f"%(prec_macro, recall_macro, f_measure_macro))



if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--predfile')
    p.add_argument('--goldfile')
    args = p.parse_args()

    # read pred and gold
    with open(args.predfile) as f_pred:
        doc_pred_mentions = json.loads(f_pred.read())
    with open(args.goldfile) as f_gold:
        doc_gold_entitys = json.loads(f_gold.read())     

    # obtain eval results
    eval_exact(copy.deepcopy(doc_pred_mentions), copy.deepcopy(doc_gold_entitys))
    eval_head_noun(copy.deepcopy(doc_pred_mentions), copy.deepcopy(doc_gold_entitys))