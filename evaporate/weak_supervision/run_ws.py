import argparse
import numpy as np
import json
import sys
import pickle
import random
import cvxpy as cp
import scipy as sp
from tqdm import tqdm
from evaporate.weak_supervision.methods import Aggregator
from metal.label_model import LabelModel
from collections import defaultdict, Counter

from evaporate.evaluate_synthetic import clean_comparison


def get_data(
    all_votes, 
    gold_extractions_file, 
    attribute='', 
    has_abstains=1.0, 
    num_elts = 5,
    extraction_fraction_thresh=0.9,
):
    """
    Load in dataset from task_name depending on where files are saved.

    - num_elts = number of ``choices'' to use in the multiple-choice setup.

    """       
    label_name_to_ints = []
    has_abstains = has_abstains >= extraction_fraction_thresh 

    try:
        with open(gold_extractions_file) as f:
            gold_extractions = json.load(f)
    except:
        with open(gold_extractions_file, "rb") as f:
            gold_extractions = pickle.load(f)
    
    test_votes = []
    test_golds = []
    total_abstains = []
    average_unique_votes = []
    missing_files = []
    random.seed(0)
    for file, extractions in tqdm(all_votes.items()):
        if file not in gold_extractions:
            missing_files.append(file)
            continue

        extractions = [clean_comparison(e) for e in extractions]
        if has_abstains:
            extractions = [e if e else 'abstain' for e in extractions]
        
        unique_votes = Counter(extractions).most_common(num_elts)
        unique_votes = [i for i, _ in unique_votes if i != 'abstain']
        average_unique_votes.append(len(unique_votes))
        if len(unique_votes) < num_elts:
            missing_elts = num_elts - len(unique_votes)
            for elt_num in range(missing_elts):
                unique_votes.append(f"dummy{elt_num}")

        random.shuffle(unique_votes)
        label_name_to_int = {elt: j for j, elt in enumerate(unique_votes)}
        label_name_to_ints.append(label_name_to_int)
        
        test_votes.append(np.array(
            [label_name_to_int[ans] if ans in label_name_to_int else -1 for ans in extractions]
        ))
        
        num_abstains = len([a for a in extractions if a not in label_name_to_int])
        total_abstains.append(num_abstains)

        # golds are just for class balance purposes
        if attribute in gold_extractions[file]:
            gold = gold_extractions[file][attribute]
        elif clean_comparison(attribute) in gold_extractions[file]:
            gold = gold_extractions[file][clean_comparison(attribute)]
        else:
            gold = ""
        gold = clean_comparison(gold)
        if gold in label_name_to_int:
            test_golds.append(label_name_to_int[gold]) 
        else:
            gold = random.sample(range(len(label_name_to_int)), 1)
            test_golds.append(gold[0])

    test_votes = np.array(test_votes)
    test_gold = np.array(test_golds)

    test_votes = test_votes.astype(int)
    test_gold = test_gold.astype(int)

    print(f"Average abstains across documents: {np.mean(total_abstains)}")
    print(f"Average unique votes per document: {np.mean(average_unique_votes)}")

    return test_votes, test_gold, label_name_to_ints, missing_files


def get_top_deps_from_inverse_sig(J, k):
    m = J.shape[0]
    deps = []
    sorted_idxs = np.argsort(np.abs(J), axis=None)
    n = m*m 
    idxs = sorted_idxs[-k:]
    for idx in idxs:
        i = int(np.floor(idx / m))
        j = idx % m 
        if (j, i) in deps:
            continue
        deps.append((i, j))
    return deps


def learn_structure(L):
    m = L.shape[1]
    n = float(np.shape(L)[0])
    sigma_O = (np.dot(L.T,L))/(n-1) -  np.outer(np.mean(L,axis=0), np.mean(L,axis=0))
    
    #bad code
    O = 1/2*(sigma_O+sigma_O.T)
    O_root = np.real(sp.linalg.sqrtm(O))

    # low-rank matrix
    L_cvx = cp.Variable([m,m], PSD=True)

    # sparse matrix
    S = cp.Variable([m,m], PSD=True)

    # S-L matrix
    R = cp.Variable([m,m], PSD=True)

    #reg params
    lam = 1/np.sqrt(m)
    gamma = 1e-8

    objective = cp.Minimize(0.5*(cp.norm(R @ O_root, 'fro')**2) - cp.trace(R) + lam*(gamma*cp.pnorm(S,1) + cp.norm(L_cvx, "nuc")))
    constraints = [R == S - L_cvx, L_cvx>>0]

    prob = cp.Problem(objective, constraints)
    result = prob.solve(verbose=False, solver=cp.SCS)
    opt_error = prob.value

    #extract dependencies
    J_hat = S.value
    
    if J_hat is None:
        raise ValueError("CVXPY failed to solve the structured learning problem, use result without dependencies.")
    
    for i in range(m):
        J_hat[i, i] = 0
    return J_hat


def learn_structure_multiclass(L, k):
    m = L.shape[1]
    J_hats = np.zeros((k, m, m))
    for c in range(k):

        all_votes_c = np.where(L == c, 1, 0)
        J_hats[c] = learn_structure(all_votes_c)

    return J_hats


def get_min_off_diagonal(J_hat):
    J_hat_copy = J_hat.copy()
    for i in range(len(J_hat_copy)):
        J_hat_copy[i, i] = np.inf
    return np.abs(J_hat_copy).min()


def run_ws(
    all_votes, 
    gold_extractions_file, 
    symmetric=True,
    attribute='', 
    has_abstains=1.0,
    extraction_fraction_thresh=0.9,
):
    test_votes, test_gold, label_name_to_ints, missing_files = get_data(
        all_votes, 
        gold_extractions_file, 
        attribute=attribute, 
        has_abstains=has_abstains,
        extraction_fraction_thresh=extraction_fraction_thresh,
    )

    classes = np.sort(np.unique(test_gold))
    vote_classes = np.sort(np.unique(test_votes))
    n_test, m = test_votes.shape
    k = len(classes)
    abstains = len(vote_classes) == len(classes) + 1
    print(f"Abstains: {abstains}")

    m = test_votes.shape[1]
    all_votes = test_votes

    label_model = LabelModel(k=k, seed=123)

    # scale to 0, 1, 2 (0 is abstain)
    test_votes_scaled = (test_votes + np.ones((n_test, m))).astype(int)
    test_gold_scaled = (test_gold + np.ones(n_test)).astype(int)
    all_votes_scaled = test_votes_scaled

    label_model.train_model(
        all_votes_scaled, 
        Y_dev=test_gold_scaled, 
        abstains=abstains, 
        symmetric=symmetric, 
        n_epochs=10000, 
        log_train_every=1000, 
        lr=0.00001
    )

    print('Trained Label Model Metrics (No deps):')
    scores, preds = label_model.score(
        (test_votes_scaled, test_gold_scaled), 
        metric=['accuracy','precision', 'recall', 'f1']
    )
    print(scores)
    all_votes_no_abstains = np.where(all_votes == -1, 0, all_votes)

    used_deps = False
    try:
        if len(classes) == 2:
            J_hat = learn_structure(all_votes_no_abstains)
        else:
            J_hats = learn_structure_multiclass(all_votes_no_abstains, len(classes))
            J_hat = J_hats.mean(axis=0)

        # if values in J are all too large, then everything is connected / structure learning isn't learning the right thing. Don't model deps then
        min_entry = get_min_off_diagonal(J_hat)
        if min_entry < 1:
            deps = get_top_deps_from_inverse_sig(J_hat, 1)
            print("Recovered dependencies: ", deps)

            label_model.train_model(
                all_votes_scaled, 
                Y_dev=test_gold_scaled, 
                abstains=abstains, 
                symmetric=symmetric, 
                n_epochs=80000, 
                log_train_every=1000, 
                lr=0.000001, 
                deps=deps
            )
            print('Trained Label Model Metrics (with deps):')
            scores, preds = label_model.score(
                (test_votes_scaled, test_gold_scaled), 
                metric=['accuracy', 'precision', 'recall', 'f1']
            )
            print(scores)
            used_deps = True
    except:
        print(f"Not modeling dependencies.")

    # convert the preds back
    mapped_preds = []
    for label_name_to_int, pred in tqdm(zip(label_name_to_ints, preds)):
        int_to_label_name = {v:k for k, v in label_name_to_int.items()}
        try:
            pred = int_to_label_name[pred-1]
        except:
            pred = ''
        mapped_preds.append(pred)
    return mapped_preds, used_deps, missing_files


if __name__ == "__main__":
    run_ws()
