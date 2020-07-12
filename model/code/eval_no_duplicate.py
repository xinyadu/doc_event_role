import sys
import json
from pytorch_pretrained_bert import BertTokenizer
import spacy

nlp = spacy.load("en_core_web_sm") # for finding head noun
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased') # Load bert pre-trained model tokenizer (vocabulary)

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


def match_exact(preds, golds):
    for pred_span in preds:
        for gold_span in golds:
            if pred_span == gold_span:
                return True
    return False

def match_noun(preds, golds):
    for pred in preds:
        for gold in golds:
            # must have this line (no head noun in 'fmln')
            if pred['span'] == gold['span']: 
                return True
            if pred['hn'] and gold['hn']:
                for n1 in pred['hn']:
                    for n2 in gold['hn']:
                        if n1 == n2:
                            return True
    return False

def remove_duplicate(items):
    items_no_dup = list()
    for item in items:
        if item not in items_no_dup:
            items_no_dup.append(item)
    return items_no_dup

def remove_duplicate_head_noun(items):
    items_no_dup = list()
    items_no_dup_hn = list()
    for item in items:
        if item["hn"] not in items_no_dup_hn or not item["hn"]:
            items_no_dup.append(item)
            items_no_dup_hn.append(item["hn"])
    return items_no_dup

def get_eval_results(doc_pred_spans, doc_gold_events, to_print=False):
    ## get pred spans (tokenized and headnoun)
    doc_pred_spans_head_noun = dict()
    for doc_id in doc_pred_spans:
        doc_pred_spans_head_noun[doc_id] = dict()
        for tag in tags_to_extract:
            doc_pred_spans_head_noun[doc_id][tag] = list()
            for idx, span in enumerate(doc_pred_spans[doc_id][tag]):
                span_tokenized = tokenizer.tokenize(span) # to normalize (remove other diff between pred and gold)
                doc_pred_spans[doc_id][tag][idx] = span_tokenized

                head_noun = list()
                noun_chunks = list(nlp(" ".join(span_tokenized).replace(" ##", '')).noun_chunks)
                for noun_chunk in noun_chunks: 
                    head_noun.append(noun_chunk.root.text)

                doc_pred_spans_head_noun[doc_id][tag].append({"span": span_tokenized, "hn": head_noun})

            doc_pred_spans[doc_id][tag] = remove_duplicate(doc_pred_spans[doc_id][tag])
            doc_pred_spans_head_noun[doc_id][tag] = remove_duplicate_head_noun(doc_pred_spans_head_noun[doc_id][tag])


    ## get gold event (tokenized and headnoun)
    doc_gold_events_head_noun = dict()
    for doc_id in doc_gold_events:
        doc_gold_events_head_noun[doc_id] = dict()
        for tag in tags_to_extract:
            doc_gold_events_head_noun[doc_id][tag] = list()
            for idx, event in enumerate(doc_gold_events[doc_id]["roles"][tag]):
                event_tokenized = []
                event_head_noun = []

                for span in event:
                    span_tokenized = tokenizer.tokenize(span)
                    event_tokenized.append(span_tokenized)

                    head_noun = list()
                    noun_chunks = list(nlp(" ".join(span_tokenized).replace(" ##", '')).noun_chunks)
                    for noun_chunk in noun_chunks: 
                        head_noun.append(noun_chunk.root.text)
                    event_head_noun.append({"span": span_tokenized, "hn": head_noun})

                doc_gold_events[doc_id]["roles"][tag][idx] = event_tokenized
                doc_gold_events_head_noun[doc_id][tag].append(event_head_noun)

    # ## get gold event (tokenized and headnoun)
    # doc_gold_events_exact = dict()
    # doc_gold_events_head_noun = dict()
    # for doc_id in doc_gold_events:
    #     doc_gold_events_exact[doc_id] = dict()
    #     doc_gold_events_head_noun[doc_id] = dict()
    #     for tag in tags_to_extract:
    #         doc_gold_events_exact[doc_id][tag] = list()
    #         doc_gold_events_head_noun[doc_id][tag] = list()
    #         for event in doc_gold_events[doc_id]["roles"][tag]:
    #             event_tokenized = []
    #             event_head_noun = []

    #             for span in event:
    #                 span_tokenized = tokenizer.tokenize(span)
    #                 event_tokenized.append(span_tokenized)

    #                 head_noun = list()
    #                 noun_chunks = list(nlp(" ".join(span_tokenized).replace(" ##", '')).noun_chunks)
    #                 for noun_chunk in noun_chunks: 
    #                     head_noun.append(noun_chunk.root.text)
    #                 event_head_noun.append({"span": span_tokenized, "hn": head_noun})

    #             doc_gold_events_exact[doc_id][tag].append(event_tokenized)
    #             doc_gold_events_head_noun[doc_id][tag].append(event_head_noun)

    
    ##
    ## report exact
    ##
    prec_marco, recall_marco = 0, 0
    if to_print:
        print("Exact Match\n", "precision, recall, F-1")
    final_print = []
    for tag in tags_to_extract:
        gold_event_num, right_event_num, pred_span_num, right_span_num = 0, 0, 0, 0
        for doc_id in doc_gold_events:
            gold_events = doc_gold_events[doc_id]["roles"][tag]
            if doc_id not in doc_pred_spans: continue
            pred_spans = doc_pred_spans[doc_id][tag]

            # for recall
            for event in gold_events:
                gold_event_num += 1
                if match_exact(pred_spans, event):
                    right_event_num += 1
            # for prec
            for span in pred_spans:
                pred_span_num += 1
                all_events = list()
                for event in gold_events:
                    all_events += event
                if match_exact([span], all_events):
                    right_span_num += 1

        recall, prec = -1, -1
        if gold_event_num: recall = (right_event_num + 0.0) / gold_event_num
        if pred_span_num: prec = (right_span_num + 0.0) / pred_span_num

        if prec <= 0 or recall <= 0:
            f_measure = -1
        else:
            prec *= 100
            recall *= 100
            f_measure = 2*prec*recall/(prec+recall)

        prec_marco += prec
        recall_marco += recall

        if to_print:
            print("%s\n%.4f %.4f %.4f"%(tag2category[tag], prec, recall, f_measure))
        final_print += [prec, recall, f_measure]

    prec_marco = prec_marco / 5
    recall_marco = recall_marco / 5
    f_measure_marco = 2*prec_marco*recall_marco/(prec_marco+recall_marco)
    final_print += [prec_marco, recall_marco, f_measure_marco]
    # print("\n\nmarco avg")
    if to_print:
        print("Macro:")
        print("%.4f %.4f %.4f"%(prec_marco, recall_marco, f_measure_marco))
        print("\nfinal_print")
        for num in final_print: print(num, end = " ")


    ##
    ## report head noun
    ##
    doc_pred_spans, doc_gold_events = doc_pred_spans_head_noun, doc_gold_events_head_noun

    prec_marco, recall_marco = 0, 0
    if to_print:
        print("\n\nHead Noun Match\n", "precision, recall, F-1")
    final_print = []
    for tag in tags_to_extract:
        gold_event_num, right_event_num, pred_span_num, right_span_num = 0, 0, 0, 0
        for doc_id in doc_gold_events:
            gold_events = doc_gold_events[doc_id][tag]
            if doc_id not in doc_pred_spans: continue
            pred_spans = doc_pred_spans[doc_id][tag]

            # for recall
            for event in gold_events:
                gold_event_num += 1
                if match_noun(pred_spans, event):
                    right_event_num += 1
            # for prec
            for span in pred_spans:
                pred_span_num += 1
                all_events = list()
                for event in gold_events:
                    all_events += event
                if match_noun([span], all_events):
                    right_span_num += 1

        recall, prec = -1, -1
        if gold_event_num: recall = (right_event_num + 0.0) / gold_event_num
        if pred_span_num: prec = (right_span_num + 0.0) / pred_span_num

        if prec <= 0 or recall <= 0:
            f_measure = -1
        else:
            prec *= 100
            recall *= 100
            f_measure = 2*prec*recall/(prec+recall)

        prec_marco += prec
        recall_marco += recall

        if to_print:
            print("%s\n%.4f %.4f %.4f"%(tag2category[tag], prec, recall, f_measure))
        final_print += [prec, recall, f_measure]

    prec_marco = prec_marco / 5
    recall_marco = recall_marco / 5
    f_measure_marco = 2*prec_marco*recall_marco/(prec_marco+recall_marco)
    final_print += [prec_marco, recall_marco, f_measure_marco]
    if to_print:
        print("Macro:")
        print("%.4f %.4f %.4f"%(prec_marco, recall_marco, f_measure_marco))
        print("\nfinal_print")
        for num in final_print: print(num, end = " ")

    return prec_marco, recall_marco, f_measure_marco

def get_macro_avg(sequences, pred_results, doc_ids):
    # read pred spans
    doc_pred_spans = dict()
    for seq, pred_seq, doc_id in zip(sequences, pred_results, doc_ids):
        # instance_sent, pred_label_list, doc_id = instance[0], instance[1], instance[2]
        if doc_id not in doc_pred_spans:
            doc_pred_spans[doc_id] = dict()
            for tag in tags_to_extract:
                doc_pred_spans[doc_id][tag] = list()
        stand_matrix =  get_ner_BIO(pred_seq)
        if stand_matrix: 
            for item in stand_matrix:
                target_position = item.index(']')
                offsets_str = item[:target_position + 1]
                label_str = item[target_position + 1:].lower()
                offsets = json.loads(offsets_str)
                if len(offsets) == 2:
                    span = seq[offsets[0]: offsets[1] + 1]
                else:
                    span = seq[offsets[0]: offsets[0] + 1]
                span = " ".join(span).replace(' ##', '')
                doc_pred_spans[doc_id][label_str].append(span)

    # read gold events
    with open("../../data/processed/dev_full.json") as f_gold:
        doc_gold_events = json.loads(f_gold.read())

    # get_eval_results
    p, r, f = get_eval_results(doc_pred_spans, doc_gold_events)
    return p, r, f



if __name__ == '__main__':
    # init
    gold_file, pred_file = sys.argv[1], sys.argv[2]
    # print("gold_file: ", gold_file, "\n", "pred_file: ", pred_file)

    # read pred spans
    instance_texts = read_instance(pred_file)
    doc_pred_spans = dict()
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


    # read gold events
    with open(gold_file) as f_gold:
        doc_gold_events = json.loads(f_gold.read())     

    # get_eval_results
    _, _, _ = get_eval_results(doc_pred_spans, doc_gold_events, to_print=True)