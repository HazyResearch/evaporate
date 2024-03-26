import numpy as np
from collections import Counter, defaultdict

def text_f1(preds=[], golds=[], attribute= ''):
    """Compute average F1 of text spans.
    Taken from Squad without prob threshold for no answer.
    """
    total_f1 = 0
    total_recall = 0
    total_prec = 0
    f1s = []
    for pred, gold in zip(preds, golds):
        if isinstance(pred, list):
            pred = ' '.join(pred)  # Example way to convert list to string
        if isinstance(gold, list):
            gold = ' '.join(gold)  # Example way to convert list to string
        pred_toks = pred.split()
        gold_toks = gold.split()
        common = Counter(pred_toks) & Counter(gold_toks)
        num_same = sum(common.values())
        if len(gold_toks) == 0 or len(pred_toks) == 0:
            # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
            total_f1 += int(gold_toks == pred_toks)
            f1s.append(int(gold_toks == pred_toks))
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
    f1_avg = total_f1 / len(golds)
    f1_median = np.percentile(f1s, 50)     
    return f1_avg, f1_median

def get_file_attribute(attribute):
    attribute = attribute.lower()
    attribute = attribute.replace("/", "_").replace(")", "").replace("-", "_")
    attribute = attribute.replace("(", "").replace(" ", "_")
    if len(attribute) > 30:
        attribute = attribute[:30]
    return attribute