from collections import defaultdict, Counter
import numpy as np
from prompts import (PICK_VALUE, Step,)
from utils import apply_prompt


def clean_comparison(responses, field):
    clean_responses = []
    if type(responses) == str:
        responses = [responses]
    for response in responses:
        response = response.lower()
        field = field.lower()
        field_reformat = field.replace("_", "-")

        for char in ["'", field, field_reformat, ":", "<", ">", '"', "none"]:
            response = response.replace(char, " ")
        for char in [",", ".", "?", "!", ";", "(", ")", "[", "]", "{", "}", "-", "none", "\n", "\t", "\r"]: 
            response = response.replace(char, " ")
        response = response.replace("  ", " ")
        response = response.split()
        response = [r.strip() for r in response]
        response = [r for r in response if r]
        response = ' '.join(response)
        clean_responses.append(response)
    clean_responses = ", ".join(clean_responses)
    return clean_responses


def normalize_value_type(metadata, attribute):
    # make everything a list of strings since functions can return diverse types
    cleaned_items = [] 
    if type(metadata) == str:
        metadata = [metadata]
    for item in metadata:
        if type(item) == list:
            item = [str(i) for i in item]
            item = ", ".join(item)
        elif type(item) == tuple:
            item = list(item)
            item = [str(i) for i in item] 
            item = ", ".join(item)
        elif item is None:
            item = ''
        elif type(item) != str:
            item = [str(item)]
            item = ", ".join(item)
        if item: 
            cleaned_items.append(item)
    return cleaned_items


def pick_a_gold_label(golds, attribute="", manifest_session=None, overwrite_cache=False):
    """
    To counteract the large model hallucinating on various chunks affecting the evaluation of good functions.
    """

    pred_str = "- " + "\n- ".join(golds)

    prompt_template = PICK_VALUE[0]
    prompt = prompt_template.format(pred_str=pred_str, attribute=attribute)
    try:
        check, num_toks = apply_prompt(
            Step(prompt), 
            max_toks=100, 
            manifest=manifest_session,
            overwrite_cache=overwrite_cache
        )
    except:
        return golds, 0 
    check = check.split("\n")
    check = [c for c in check if c]
    if check:
        if "none" in check[0].lower():
            check = golds
        else:
            check = check[0]
    return check, num_toks


def text_f1(
    preds=[], 
    golds=[], 
    extraction_fraction=1.0, 
    attribute=None,
    extraction_fraction_thresh=0.8,
    use_abstension=True,
):
    """Compute average F1 of text spans.
    Taken from Squad without prob threshold for no answer.
    """
    total_f1 = 0
    total_recall = 0
    total_prec = 0
    f1s = []
    total = 0

    if extraction_fraction >= extraction_fraction_thresh and use_abstension:
        new_preds = []
        new_golds = []
        for pred, gold in zip(preds, golds):
            if pred:
                new_preds.append(pred)
                new_golds.append(gold)
        preds = new_preds
        golds = new_golds
        if not preds:
            return 0.0, 0.0
    for pred, gold in zip(preds, golds):
        if type(pred) == str:
            pred_toks = pred.split()
        else:
            pred_toks = pred
        if type(gold) == str:
            gold_toks_list = [gold.split()]
        else:
            assert 0, print(gold)
            gold_toks_list = gold
            
        if type(gold_toks_list) == list and gold_toks_list: 
            for gold_toks in gold_toks_list:

                # If both lists are lenght 1, split to account for example like:
                # ["a b"], ["a"] -> ["a","b"], ["a"]
                if len(gold_toks) == 1 and len(pred_toks) == 1:
                    gold_toks = gold_toks[0].split()
                    pred_toks = pred_toks[0].split()
                
                common = Counter(pred_toks) & Counter(gold_toks)
                num_same = sum(common.values())
                if len(gold_toks) == 0 or len(pred_toks) == 0:
                    # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
                    total_f1 += int(gold_toks == pred_toks)
                    f1s.append(int(gold_toks == pred_toks))
                    total_recall += int(gold_toks == pred_toks)
                elif num_same == 0:
                    total_f1 += 0
                    f1s.append(0)
                else:
                    precision = 1.0 * num_same / len(pred_toks)
                    recall = 1.0 * num_same / len(gold_toks)
                    f1 = (2 * precision * recall) / (precision + recall)
                    total_f1 += f1
                    total_recall += recall
                    total_prec += precision 
                    f1s.append(f1)

                total += 1
    if not total:
        return 0.0, 0.0
    f1_avg = total_f1 / total
    f1_median = np.percentile(f1s, 50)  
    return f1_avg, f1_median


def evaluate(
    all_extractions:list,
    gold_key:str, 
    field:str, 
    manifest_session=None, 
    overwrite_cache=False, 
    combiner_mode='mv',
    extraction_fraction_thresh=0.8,
    use_abstension=True,
):
    """
  Evaluate extraction quality against gold labels.

  Args:
    all_extractions: Dict of extractions from all models/functions.
    gold_key: Key of gold model in all_extractions. 
    field: Field name being extracted.
    manifest_session: API session for prompting.
    overwrite_cache: Whether to re-prompt API.
    combiner_mode: Aggregation method used.
    extraction_fraction_thresh: Threshold for abstentions.
    use_abstension: Whether to use abstentions.

  Returns:
    metrics: Dictionary of quality metrics per model/function.
    key2golds: Gold labels matched to models.
    total_tokens_prompted: Number of tokens used.

  Scores all model and function extractions against gold labels.
  Metrics include F1, extraction fraction, tokens used.

  Can apply prompts to pick best gold label when multiple exist.

    Key points:

    Purpose is evaluating extractions vs gold labels.
    Inputs include extraction dict, gold key, configs.
    Outputs are quality metrics and gold labels.
    Notes techniques like using abstentions.

    Q: https://github.com/HazyResearch/evaporate/issues/28
    """
    if combiner_mode == 'mv':
        key2golds = defaultdict(list)
        metrics = {} 
        total_tokens_prompted = 0
        return metrics, key2golds, total_tokens_prompted

    # -- Original Evaporate code --
    normalized_field_name = field
    for char in ["'", ":", "<", ">", '"', "_", "-", " ", "none"]:
        normalized_field_name = normalized_field_name.replace(char, "")

    key2golds = defaultdict(list)
    key2preds = defaultdict(list)
    total_tokens_prompted = 0

    # handle FM golds on D_eval
    gold_file2metadata = all_extractions[gold_key]
    cleaned_gold_metadata = {}
    for filepath, gold_metadata in gold_file2metadata.items():
        gold_metadata = normalize_value_type(gold_metadata, field)
        if len(gold_metadata) > 1:
            gold_metadata, num_toks = pick_a_gold_label(
                gold_metadata, 
                attribute=field, 
                manifest_session=manifest_session, 
                overwrite_cache=overwrite_cache
            )
            total_tokens_prompted += num_toks
        gold_metadata = clean_comparison(gold_metadata, field)
        cleaned_gold_metadata[filepath] = gold_metadata

    # handle function preds on D_eval
    for i, (key, file2metadata) in enumerate(all_extractions.items()):
        if key == gold_key:
            continue
        for filepath, metadata in file2metadata.items():
            gold_metadata = cleaned_gold_metadata[filepath]
            pred_metadata = normalize_value_type(metadata, field)
            pred_metadata = clean_comparison(pred_metadata, field)
            key2golds[key].append(gold_metadata)
            key2preds[key].append(pred_metadata)

    # Handling abstensions (refuses to give output correct values for attributed although present in doc)
    num_extractions = 0
    for golds in key2golds[key]:
        if golds and not any(golds.lower() == wd for wd in ['none']):
            num_extractions += 1
    extraction_fraction = float(num_extractions) / float(len(key2golds[key]))
    if combiner_mode == "top_k":
        # Don't use the extraction fraction in the naive setting for scoring
        extraction_fraction = 0.0
    print(f"Extraction fraction: {extraction_fraction}")

    metrics = {}
    for key, golds in key2golds.items():
        preds = key2preds[key]
        f1, f1_med = text_f1(
            preds, golds, 
            extraction_fraction=extraction_fraction, 
            attribute=field,
            extraction_fraction_thresh=extraction_fraction_thresh,
            use_abstension=use_abstension,
        )
        priorf1, priorf1_med = text_f1(preds, golds, extraction_fraction=0.0, attribute=field)
        metrics[key] = {
            "average_f1": f1,
            "median_f1": f1_med,
            "extraction_fraction": extraction_fraction,
            "prior_average_f1": priorf1,
            "prior_median_f1": priorf1_med,
        } 

    return metrics, key2golds, total_tokens_prompted


def get_topk_scripts_per_field(
    script2metrics, 
    function_dictionary, 
    all_extractions,
    gold_key='', 
    k=3, 
    do_end_to_end=False, 
    keep_thresh = 0.5, 
    cost_thresh = 1, 
    combiner_mode='mv',
): 
    script2avg = dict(
        sorted(script2metrics.items(), 
        reverse=True, 
        key=lambda x: (x[1]['average_f1'], x[1]['median_f1']))
    )

    top_k_scripts = [k for k, v in script2avg.items() if k != gold_key] 
    top_k_values = [
        max(v['average_f1'], v['median_f1']) for k, v in script2avg.items() if k != gold_key
    ]
    if not top_k_values:
        return []
    
    best_value = top_k_values[0]
    best_script = top_k_scripts[0]
    if best_value < keep_thresh and do_end_to_end:
        return []

    filtered_fn_scripts = {
        k:v for k, v in script2metrics.items() if (
            v['average_f1'] >= keep_thresh or v['median_f1'] >= keep_thresh
        ) and "function" in k
    }
    top_k_fns = []
    num_fns = 0
    if filtered_fn_scripts:
        script2avg = dict(
            sorted(filtered_fn_scripts.items(), 
            reverse=True, 
            key=lambda x: (x[1]['average_f1'], x[1]['median_f1']))
        )

        top_k_fns = [
            k for k, v in script2avg.items() if k != gold_key and abs(
                max(v['average_f1'], v['median_f1'])-best_value
            ) < cost_thresh
        ]
        num_fns = len(top_k_fns)
    
    if num_fns:
        top_k_scripts = top_k_scripts[0:min(k, num_fns)]
    else:
        return []

    # construct final set of functions
    final_set = []
    for key in top_k_scripts:
        if key in top_k_fns:
            final_set.append(key)

    if len(final_set) > k:
        final_set = final_set[:k]
    if not final_set and not do_end_to_end:
        return [top_k_scripts[0]]

    # print results
    print(f"Top {k} scripts:")
    for script in final_set:
        print(f"- {script}; Score: {script2metrics[script]}")
    print(f"Best script overall: {best_script}; Score: {script2metrics[best_script]}")
    return final_set

