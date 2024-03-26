import os
import math
import json
import pickle

import html
from bs4 import BeautifulSoup
from collections import Counter, defaultdict
from evaporate.utils import get_file_attribute
from evaporate.evaluate_synthetic_utils import text_f1


# Compute recall from two sets
def set_recall(pred, gt):
    return len(set(pred) & set(gt)) / len(set(gt))


# Compute precision from two sets
def set_precision(pred, gt):
    return len(set(pred) & set(gt)) / len(set(pred))


# Compute F1 from precision and recall
def compute_f1(precision, recall):
    if recall > 0. or precision > 0.:
        return 2. * (precision * recall) / (precision + recall)
    else:
        return 0.


def evaluate_schema_identification(run_string, args, group_name, train_size=-1):
    with open(f"{args.generative_index_path}/{run_string}_identified_schema.json") as f:
        most_common_fields = json.load(f)

    try:
        with open(args.gold_extractions_file) as f:
            gold_file2extractions = json.load(f)
    except:
        with open(args.gold_extractions_file, "rb") as f:
            gold_file2extractions = pickle.load(f)
    
    for file, dic in gold_file2extractions.items():
        gold_metadata = list(dic.keys())
        gold_metadata = [m for m in gold_metadata if m not in ['topic_entity_name']]
        break

    ctr =  Counter(most_common_fields)
    results = {}
    for k in [len(gold_metadata), 1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 100, len(most_common_fields)]:
        if not most_common_fields:
            results[k] = {
                "recall": 0,
                "precision": 0,
                "f1": 0,
                "num_gold_attributes": k,
            }
            continue
        gold_metadata = [item.lower() for item in gold_metadata]
        pred_metadata = ctr
        
        limit = k
        pred_metadata = sorted(pred_metadata.most_common(limit), key=lambda x: (x[1], x[0]), reverse=True)
        pred_metadata = [item[0].lower() for item in pred_metadata]
        cleaned_pred_metadata = set()
        for pred in pred_metadata:
            if not pred:
                continue
            cleaned_pred_metadata.add(pred)
        cleaned_gold_metadata = set()
        for gold in gold_metadata:
            cleaned_gold_metadata.add(gold)
        
        recall = [x for x in cleaned_gold_metadata if x in cleaned_pred_metadata]
        precision = [x for x in cleaned_pred_metadata if x in cleaned_gold_metadata]
        recall = len(recall) / len(cleaned_gold_metadata)
        precision = len(precision) / len(cleaned_pred_metadata)
        f1 = compute_f1(precision, recall)

        results[k] = {
            "recall": recall,
            "precision": precision,
            "f1": f1,
            "num_gold_attributes": limit,
        }

        print(f"@k = %d --- Recall: %.3f, Precision: %.3f, F1: %.3f" % (k, recall, precision, f1))
    print()
    return results


def clean_comparison(extraction, attribute='', exact_match=False):
    # formatting transformations 

    if type(extraction) == list:
        if extraction and type(extraction[0]) == list:
            full_answer = []
            for answer in extraction:
                if type(answer) == list:
                    dedup_list = []
                    for a in answer:
                        if a not in dedup_list:
                            dedup_list.append(a)
                    answer = dedup_list
                    answer = [str(a).strip().strip("\n") for a in answer]
                    full_answer.append(", ".join(answer))
                else:
                    full_answer.append(answer.strip().strip("\n"))
            full_answer = [a.strip() for a in full_answer]
            extraction = ", ".join(full_answer)
        else:
            dedup_list = []
            for a in extraction:
                if a not in dedup_list:
                    dedup_list.append(a)
            extraction = dedup_list
            extraction = [(str(e)).strip().strip("\n") for e in extraction if e]
            extraction = ", ".join(extraction)
    elif type(extraction) == "str" and (
        extraction.lower() == "none" or any(
            phrase in extraction.lower() for phrase in ["not reported", "none available", "n/a"]
        )
    ):
        extraction = ""
    if type(extraction) == float and math.isnan(extraction):
        extraction = ''
    if type(extraction) != str:
        extraction = str(extraction)

    if type(extraction) == str:
        if ("<" in extraction) and (">" in extraction):
            extraction = BeautifulSoup(extraction).text

    extraction = extraction.strip().replace("  ", " ").lower()
    attribute_variations = [f"{attribute}(s)".lower(), f"{attribute.strip()}(s)".lower(), attribute.lower(), attribute]
    for a in attribute_variations:
        extraction = extraction.replace(a, "").strip()
    for char in ["'", '"', "(", ")", ",", "/", "]", "[", ":"]:
        extraction = extraction.replace(char, "").strip()
    extraction = html.unescape(extraction)
    for char in ['&amp;', '&', "-", "_", "\n", "\t", "http:", "<", ">"]:
        extraction = extraction.replace(char, " ").strip()
    if exact_match:
        extraction = extraction.replace(" ", "")
    if extraction == " ":
        extraction = ""
    extraction = extraction.strip()
    return extraction


def evaluate_extraction_quality(run_string, args, gold_extractions_file, gold_extractions_file_dir, gold_attributes=None):
    all_attribute_f1 = 0
    all_attribute_total = 0
    total_runtime_overall_attributes = 0
    attribute2f1 = {}
    attribute2scripts = {}

    # load gold extractions file
    try:
        with open(gold_extractions_file) as f:
            gold_extractions_tmp = json.load(f)
    except:
        with open(gold_extractions_file, "rb") as f:
            gold_extractions_tmp = pickle.load(f)
    gold_extractions = {}
    for file, extractions in gold_extractions_tmp.items():
        gold_extractions[os.path.join(gold_extractions_file_dir, file.split('/')[-1])] = extractions
    
    for attribute in gold_attributes:
        attribute = attribute.lower()
        # load predicted extractions
        fileattribute = get_file_attribute(attribute)
        if not os.path.exists(f"{args.generative_index_path}/{run_string}_{fileattribute}_file2metadata.json"):
            print(f"Missing file for {attribute}")
            continue
        with open(f"{args.generative_index_path}/{run_string}_{fileattribute}_file2metadata.json") as f:
            file2metadata = json.load(f)
        
        try:
            with open(f"{args.generative_index_path}/{run_string}_{fileattribute}_functions.json") as f:
                function_dictionary = json.load(f)
            with open(f"{args.generative_index_path}/{run_string}_{fileattribute}_top_k_keys.json") as f:
                selected_keys = json.load(f)
            attribute2scripts[attribute] = selected_keys
        except:
            function_dictionary = {}
            selected_keys = []
            pass
        total_runtime = 0
        for key in selected_keys:
            if key in function_dictionary: 
                runtime = function_dictionary[key]['runtime']
                total_runtime += runtime

        preds = []
        golds = []
        for file, gold_entry in gold_extractions.items():
            for attr, gold_value in gold_entry.items():
                attr = clean_comparison(attr)
                attribute = clean_comparison(attribute)
                if attr.lower() != attribute.lower():
                    continue
                if file not in file2metadata:
                    continue
                pred_value = file2metadata[file]

                value_check = ''
                pred_value_check = ''
                if type(pred_value) == list and type(pred_value[0]) == str:
                    pred_value_check = sorted([p.strip() for p in pred_value])
                elif type(pred_value) == str and "," in pred_value:
                    pred_value_check = sorted([p.strip() for p in pred_value.split(",")])
                if type(gold_value) == list:
                    value_check = gold_value[0]
                if "," in gold_value:
                    value_check = sorted([p.strip() for p in gold_value.split(",")])
                if value_check and pred_value_check and value_check == pred_value_check:
                    gold_value = pred_value

                # SWDE doesn't include the full passage in many cases (e.g. "IMDB synopsis")
                pred_value = clean_comparison(pred_value, attribute=attribute)
                gold_value = clean_comparison(gold_value, attribute=attribute)
                if pred_value.lower().strip(".").startswith(gold_value.lower().strip(".")):
                    pred_value = " ".join(pred_value.split()[:len(gold_value.split())])
                preds.append(pred_value)
                golds.append(gold_value)
        
        if not preds:
            total_f1, total_f1_median = 0, 0
        if golds and preds:
            total_f1, total_f1_median = text_f1(preds, golds, attribute=attribute)
        else:
            print(f"Skipping eval of attribute: {attribute}")
            continue
        
        if preds:
            all_attribute_f1 += (total_f1)
            all_attribute_total += 1
            attribute2f1[attribute] = total_f1

        total_runtime_overall_attributes += total_runtime
    
    num_function_scripts = 0
    for k, v in attribute2f1.items():
        scripts = []
        if k in attribute2scripts:
            scripts = attribute2scripts[k]
        if any(s for s in scripts if "function" in s):
            num_function_scripts += 1
        print(f"{k}, text-f1 = {v} --- {scripts}")
    
    try:
        overall_f1 = all_attribute_f1 / all_attribute_total
        print(f"\nOverall f1 across %d attributes: %.3f" % (all_attribute_total, overall_f1))
        print(f"Used functions for {num_function_scripts} out of {len(attribute2f1)} attributes")
        print(f"Average time: {total_runtime_overall_attributes/all_attribute_total} seconds, {all_attribute_total} fns.\n\n")

        results = {
            "f1": all_attribute_f1 / all_attribute_total,
            "total_attributes": all_attribute_total,
            "attribute2f1": attribute2f1,
        }
    except:
        results = {
            "f1": 0,
            "total_attributes": all_attribute_total,
            "attribute2f1": attribute2f1,
        }
    
    return results


def determine_attribute_slices(gold_extractions, slice_results):
    num_occurences, num_characters = defaultdict(int), defaultdict(int)
    num_documents = len(gold_extractions)
    for file, extraction_dict in gold_extractions.items():
        for key, value in extraction_dict.items():
            if type(value) == str and value:
                num_occurences[key] += 1 
                num_characters[key] += len(value)
            elif type(value) == list and value[0]:
                num_occurences[key] += 1 
                num_characters[key] += len(value[0])
    
    # calculate the average length of the attribute
    for attr, total_len in num_characters.items():
        num_characters[attr] = total_len / num_occurences[attr]
        
    # split into the "head", "tail", and "unstructured"
    attribute_slices = defaultdict(set)
    for attr, num_occur in num_occurences.items():
        attribute_slices["all"].add(attr)
        
        # skip the rest if not slicing results
        if not slice_results: 
            continue
        
        num_char = num_characters[attr]
        if int(num_documents * 0.5) <= num_occur:
            attribute_slices["head"].add(attr)
        else:
            attribute_slices["tail"].add(attr)

        if num_char >= 20:
            attribute_slices["unstructured"].add(attr)
        else:
            attribute_slices["structured"].add(attr)
            
    return attribute_slices


def evaluate_openie_quality(
        run_string, 
        args, 
        gold_extractions_file, 
        sample_files=None, 
        slice_results=False, 
        mappings_names={}
):
    # load pred extractions file
    with open(f"{args.generative_index_path}/{run_string}_file2extractions.json") as f:
        pred_extractions = json.load(f)

    # alternate gold attribute naming
    if args.set_dicts:
        with open(args.set_dicts) as f:
            set_dicts = json.load(f)
    else:
        set_dicts = {}
    
    # load gold extractions file
    try:
        with open(gold_extractions_file) as f:
            gold_extractions = json.load(f) 
    except:
        with open(gold_extractions_file, "rb") as f:
            gold_extractions = pickle.load(f)

    pred_attributes = set()
    for file, extraction_dict in pred_extractions.items():
        for key, value in extraction_dict.items():
            pred_attributes.add(key)
    
    # split the attribute into slices -> "head", "tail", and "unstructured"
    attribute_slices = determine_attribute_slices(gold_extractions, slice_results)
    
    results = {}
    for attribute_slice, gold_attributes in attribute_slices.items():
        
        # lenient attribute scoring method: https://arxiv.org/pdf/2201.10608.pdf
        gold_attribute_mapping = {}
        for gold_attribute in gold_attributes:
            if gold_attribute in pred_attributes or not set_dicts:
                gold_attribute_mapping[gold_attribute] = gold_attribute
                continue
            if gold_attribute in set_dicts:
                alternate_golds = set_dicts[gold_attribute]
            else:
                alternate_golds = [gold_attribute]
            found = 0
            for alternate_gold in alternate_golds:
                if alternate_gold in pred_attributes:
                    gold_attribute_mapping[gold_attribute] = alternate_gold
                    found = 1
            if not found:
                if gold_attribute.strip('s') in pred_attributes:
                    gold_attribute_mapping[gold_attribute] = gold_attribute.strip('s')
                elif gold_attribute+"s" in pred_attributes:
                    gold_attribute_mapping[gold_attribute] = gold_attribute+"s"
                elif gold_attribute.strip('(s)') in pred_attributes:
                    gold_attribute_mapping[gold_attribute] = gold_attribute.strip('(s)')
                elif gold_attribute+"(s)" in pred_attributes:
                    gold_attribute_mapping[gold_attribute] = gold_attribute+"(s)"
                elif gold_attribute.replace(" ", "") in pred_attributes:
                    gold_attribute_mapping[gold_attribute] = gold_attribute.replace(" ", "")
                elif any(pred_attribute.replace(" ", "")  in gold_attributes for pred_attribute in pred_attributes):
                    for pred_attribute in pred_attributes:
                        if pred_attribute.replace(" ", "")  in gold_attributes:
                            gold_attribute_mapping[gold_attribute] = pred_attribute
                elif gold_attribute in mappings_names and mappings_names[gold_attribute] in pred_attributes:
                    gold_attribute_mapping[gold_attribute] = mappings_names[gold_attribute]
                else:
                    gold_attribute_mapping[gold_attribute] = gold_attribute
        
        pred_set = set()
        skipped = set()
        all_measurements = defaultdict(dict)
        for file, extraction_dict in pred_extractions.items():
            if sample_files and file not in sample_files:
                continue
            for key, value in extraction_dict.items():
                if key not in attribute_slices["all"]:
                    if key.replace(" ", "") in attribute_slices["all"]:
                        key = key.replace(" ", "")

                # skip predicted attributes that are in a different slice
                if key in attribute_slices["all"] and key not in gold_attributes:
                    skipped.add(key)
                    continue
                
                clean_key = clean_comparison(key, exact_match=True)
                clean_value = clean_comparison(value, attribute=key, exact_match=True)
                if clean_value:
                    pred_set.add((file, clean_key, clean_value))
                if file not in all_measurements[clean_key]:
                    all_measurements[clean_key][file] = {
                        "pred": "",
                        "gold": "",
                    }
                all_measurements[clean_key][file]['pred'] = clean_value

        clean_pred_attributes = set([x[1] for x in pred_set])
        # resolve mapping between gold and pred attributes
        gold_attribute_mapping = {}
        for gold_attribute in gold_attributes:
            if gold_attribute in clean_pred_attributes:
                gold_attribute_mapping[gold_attribute] = gold_attribute
                continue
            found = False
            if set_dicts and gold_attribute in set_dicts:
                alternate_golds = set_dicts[gold_attribute]
                for alternate_gold in alternate_golds:
                    if alternate_gold in clean_pred_attributes:
                        gold_attribute_mapping[gold_attribute] = alternate_gold
                        found = True
            if not found:
                if gold_attribute.strip('s') in clean_pred_attributes:
                    gold_attribute_mapping[gold_attribute] = gold_attribute.strip('s')
                elif gold_attribute+"s" in clean_pred_attributes:
                    gold_attribute_mapping[gold_attribute] = gold_attribute+"s"
                else:
                    gold_attribute_mapping[gold_attribute] = gold_attribute

        num_attributes = len(clean_pred_attributes)

        gold_set = set()
        for file, extraction_dict in gold_extractions.items(): 
            if sample_files and file not in sample_files:
                continue
            for key, value in extraction_dict.items():
                # ignore attributes in a different slice
                if key not in gold_attributes:
                    continue
                    
                if key == "topic_entity_name":
                    if "name" in pred_attributes:
                        gold_attribute_mapping[key] = "name"

                key = gold_attribute_mapping[key]
                if key not in pred_attributes:
                    if key.replace(" ", "") in pred_attributes:
                        key = key.replace(" ", "") 

                # sort list-based attribute values for consistency.
                if file in pred_extractions and key in pred_extractions[file]:
                    pred_value = pred_extractions[file][key]
                    value_check = ''
                    pred_value_check = ''
                    if type(pred_value) == list and type(pred_value[0]) == str:
                        pred_value_check = sorted([p.strip() for p in pred_value])
                    elif type(pred_value) == str and "," in pred_value:
                        pred_value_check = sorted([p.strip() for p in pred_value.split(",")])
                    if type(value) == list:
                        value_check = value[0]
                    if "," in value:
                        value_check = sorted([p.strip() for p in value.split(",")])
                    if value_check and pred_value_check and value_check == pred_value_check:
                        value = pred_value

                clean_key = clean_comparison(key, exact_match=True)
                clean_value = clean_comparison(value, attribute=key, exact_match=True)

                if clean_value:
                    gold_set.add((file, clean_key, clean_value))
                if file not in all_measurements[clean_key]:
                    all_measurements[clean_key][file] = {
                        "pred": "",
                        "gold": "",
                    }
                all_measurements[clean_key][file]['gold'] = clean_value

        if not pred_set or not gold_set:
            results[attribute_slice] = {
                "precision": 0,
                "recall": 0,
                "f1": 0,
                "num_files_evaluated": len(pred_extractions),
            }
        else:
            # exact match over all fields
            precision = set_precision(pred_set, gold_set)
            recall = set_recall(pred_set, gold_set)
            f1 = compute_f1(precision, recall)
        
            results[attribute_slice] = {
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "num_files_evaluated": len(pred_extractions),
            }
            print(f"[%s] OpenIE Precision (%d attributes): Precision: %.3f Recall: %.3f F1: %.3f" % (attribute_slice, num_attributes, precision, recall, f1))
    return results if slice_results else results["all"]


def main(
        run_string, 
        args, 
        profiler_args, 
        data_lake = "wiki_nba_players", 
        sample_files=None, 
        stage='', 
        gold_attributes=[], 
        mappings_names={}
):
    gold_extractions_file = args.gold_extractions_file
    train_size = profiler_args.train_size

    overall_results = {}

    if stage and stage != 'schema_id':
        pass
    else:
        schema_id_results = evaluate_schema_identification(
            run_string,
            args, 
            data_lake, 
            train_size=train_size,
        )
        overall_results["schema_id"] = schema_id_results

    if stage and stage != 'extract':
        pass
    else:
        extraction_results = evaluate_extraction_quality(
            run_string,
            args,       
            gold_extractions_file,
            profiler_args.data_dir,
            gold_attributes=gold_attributes
        )
        overall_results["extraction"] = extraction_results

    if stage and stage != 'openie':
        pass
    else:
        openie_results = evaluate_openie_quality(
            run_string,
            args, 
            gold_extractions_file,
            sample_files=sample_files,
            slice_results = profiler_args.slice_results,
            mappings_names = mappings_names
        )
        overall_results["openie"] = openie_results

    return overall_results


if __name__ == "__main__":
    main()