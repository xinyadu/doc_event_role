"""
process the test file
"""

import json
from collections import OrderedDict
key2role = OrderedDict([('perp_individual_id', "PerpInd"), ('perp_organization_id',"PerpOrg"), ('phys_tgt_id',"Target"), ('hum_tgt_name',"Victim"), ('incident_instrument_id',"Weapon")])

def read_files(doc_file, keys_file):
    doc_dict = OrderedDict()
    keys_dict = OrderedDict()

    with open(doc_file) as f_doc:
        for line in f_doc:
            line_json = json.loads(line)
            doc_dict[line_json["docid"]] = line_json["text"]

    # read keys from files and merge the values for different templates of the same doc
    with open(keys_file) as f_keys:
        contents = f_keys.read()
        contents = contents.split("%%%")
        for content in contents:
            if not content: continue
            content = json.loads(content)
            key, doc_id = content[0][0], content[0][1]
            if key == "message_id":
                if doc_id not in keys_dict:
                    keys_dict[doc_id] = OrderedDict()
                    for key in key2role: keys_dict[doc_id][key] = list()

            for key in key2role:
                # 
                for keyval in content[1:]:
                    key_c, val = keyval[0], keyval[1]
                    if key_c == key:
                        if val:
                            entity = list()
                            for val_str in val["strings"]: 
                                entity.append(val_str)
                            if entity not in keys_dict[doc_id][key]:
                                keys_dict[doc_id][key].append(entity)
                            # for val_str in val["strings"]:
                            #     if val_str not in keys_dict[doc_id][role]:
                            #         keys_dict[doc_id][role].append(val_str)


    doc_keys = OrderedDict()
    for doc_id in doc_dict.keys():
        doc_keys[doc_id] = OrderedDict()
        doc_keys[doc_id]["doc"] = doc_dict[doc_id]
        doc_keys[doc_id]["roles"] = keys_dict[doc_id]

    return doc_keys


if __name__=='__main__':

    div = "test"
    doc_file = "./raw_muc/" + ".".join(["doc_" + div, "jsons", "txt"])
    keys_file = "./raw_muc/" + ".".join(["keys_" + div, "jsons", "txt"])
    doc_keys = read_files(doc_file, keys_file)

    doc_keys_file = open("processed/" + div + ".json", "w+")
    doc_keys_file.write(json.dumps(doc_keys, indent=4))
