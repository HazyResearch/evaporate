import os
import sys
import time
import random
import json
import datetime
from pathlib import Path
from tqdm import tqdm
import pickle
import argparse
from collections import defaultdict, Counter

from utils import get_structure, get_manifest_sessions, get_file_attribute
from profiler_utils import chunk_file, sample_scripts, set_profiler_args
from schema_identification import identify_schema
from profiler import run_profiler
from evaluate_synthetic import main as evaluate_synthetic_main

from pdb import set_trace as st

random.seed(0)
def get_data_lake_info(args, data_lake, DATA_DIR = "/data/evaporate"):
    extractions_file = None
    
    # if data_lake == "fda_510ks":
    DATA_DIR = args.data_dir
    file_groups = os.listdir(DATA_DIR)
    if not DATA_DIR.endswith("/"):
        DATA_DIR += "/"
    file_groups = [f"{DATA_DIR}{file_group}" for file_group in file_groups if not file_group.startswith(".")]
    full_file_groups = file_groups.copy()
    extractions_file = args.gold_extractions_file
    parser = "txt"

    return file_groups, extractions_file, parser, full_file_groups


def chunk_files(file_group, parser, chunk_size, remove_tables, max_chunks_per_file, body_only):
    print(f'Chunking files in func: {chunk_files=}')
    print(f'{parser=}')
    print(f'{chunk_size=}')
    file2chunks = {}
    file2contents = {}
    for file in tqdm(file_group, total=len(file_group), desc="Chunking files"):
        content, chunks = chunk_file(
            parser, 
            file, 
            chunk_size=chunk_size, 
            mode="train", 
            remove_tables=remove_tables,
            body_only=body_only
        )
        if max_chunks_per_file > 0:
            chunks = chunks[:max_chunks_per_file] 
        file2chunks[file] = chunks
        file2contents[file] = content
    return file2chunks, file2contents


# chunking & preparing data
def prepare_data(profiler_args, file_group, parser = "html"):
    print(f'Preparing data in func: {prepare_data=}')
    print(f'{parser=}')
    data_lake = profiler_args.data_lake
    if profiler_args.body_only:
        print(profiler_args.body_only)
        print('I think this does only the body an html based on profiler_args')
        body_only = profiler_args.body_only
        suffix = f"_bodyOnly{body_only}"
    else:
        suffix = ""

    # prepare the datalake: chunk all files
    manifest_sessions = get_manifest_sessions(profiler_args.MODELS, MODEL2URL=profiler_args.MODEL2URL, KEYS=profiler_args.KEYS)
    if os.path.exists(f".cache/{data_lake}_size{len(file_group)}_chunkSize{profiler_args.chunk_size}_{suffix}_file2chunks.pkl"):
        with open(f".cache/{data_lake}_size{len(file_group)}_chunkSize{profiler_args.chunk_size}_{suffix}_file2chunks.pkl", "rb") as f:
            file2chunks = pickle.load(f)
        with open(f".cache/{data_lake}_size{len(file_group)}_chunkSize{profiler_args.chunk_size}_{suffix}_file2contents.pkl", "rb") as f:
            file2contents = pickle.load(f)
    else:
        
        file2chunks, file2contents = chunk_files(
            file_group, 
            parser,
            profiler_args.chunk_size, 
            profiler_args.remove_tables, 
            profiler_args.max_chunks_per_file,
            profiler_args.body_only
        )
        with open(f".cache/{data_lake}_size{len(file_group)}_chunkSize{profiler_args.chunk_size}_removeTables{profiler_args.remove_tables}{suffix}_file2chunks.pkl", "wb") as f:
            pickle.dump(file2chunks, f)
        with open(f".cache/{data_lake}_size{len(file_group)}_chunkSize{profiler_args.chunk_size}_removeTables{profiler_args.remove_tables}{suffix}_file2contents.pkl", "wb") as f:
            pickle.dump(file2contents, f)

    return file2chunks, file2contents, manifest_sessions


def get_run_string(
    data_lake, today, 
    file_groups, profiler_args, 
    do_end_to_end, 
    train_size, dynamicbackoff, models
):
    body = profiler_args.body_only # Baseline systems only operate on the HTML body
    model_ct = len(models)
    if profiler_args.use_qa_model:
        model_ct += 1
    run_string = f"dataLake{data_lake}_date{today}_fileSize{len(file_groups)}_trainSize{train_size}_numAggregate{profiler_args.num_top_k_scripts}_chunkSize{profiler_args.chunk_size}_removeTables{profiler_args.remove_tables}_body{body}_cascading{do_end_to_end}_useBackoff{dynamicbackoff}_MODELS{model_ct}"
    return run_string


def get_gold_metadata(args):
    # get the list of gold metadata for closed-IE runs
    try:
        with open(args.gold_extractions_file) as f:
            gold_file2extractions = json.load(f)
    except:
        with open(args.gold_extractions_file, "rb") as f:
            gold_file2extractions = pickle.load(f)
    frequency = Counter()
    for file, dic in gold_file2extractions.items():
        for k, v in dic.items():
            if k != "topic_entity_name":
                if type(v) == str and v:
                    frequency[k] += 1
                elif type(v) == list and v and v[0]:
                    frequency[k] += 1
    sorted_frequency = sorted(frequency.items(), key=lambda x: x[1], reverse=True)
    gold_metadata = [x[0] for x in sorted_frequency]
    gold_attributes = [m.lower() for m in gold_metadata if m not in ['topic_entity_name']]
    return gold_attributes 


def determine_attributes_to_remove(attributes, args, run_string, num_attr_to_cascade):
    attributes_reordered = {}
    attributes_to_remove = []
    attributes_to_metrics = {}
    attribute_to_first_extractions = {}
    mappings_names = {}
    for num, attribute in enumerate(attributes):
        attribute = attribute.lower()
        file_attribute = get_file_attribute(attribute)
        if not os.path.exists(f"{args.generative_index_path}/{run_string}_{file_attribute}_all_metrics.json"):
            continue
        if not os.path.exists(f"{args.generative_index_path}/{run_string}_{file_attribute}_file2metadata.json"):
            continue
        if num >= num_attr_to_cascade:
            os.remove(f"{args.generative_index_path}/{run_string}_{file_attribute}_all_metrics.json")
            os.remove(f"{args.generative_index_path}/{run_string}_{file_attribute}_file2metadata.json")
            continue
        with open(f"{args.generative_index_path}/{run_string}_{file_attribute}_all_metrics.json") as f:
            metrics = json.load(f)
        with open(f"{args.generative_index_path}/{run_string}_{file_attribute}_top_k_keys.json") as f:
            selected_keys = json.load(f)
        with open(f"{args.generative_index_path}/{run_string}_{file_attribute}_file2metadata.json") as f:
            file2metadata = json.load(f)
        attributes_reordered[attribute] = metrics[selected_keys[0]]

        if selected_keys and metrics:
            for a, m in attributes_to_metrics.items():
                if attribute.lower() in a.lower() or a.lower() in attribute.lower():
                    if m == metrics[selected_keys[0]]['average_f1']:
                        attributes_to_remove.append(attribute)
                        mappings_names[a] = attribute
                        mappings_names[attribute] = a
                        break

        first_extractions = [m for i, (f, m) in enumerate(file2metadata.items()) if i < 5]
        if any(f != "" for f in first_extractions):
            first_extractions = " ".join(first_extractions)
            for a, m in attribute_to_first_extractions.items():
                if m == first_extractions:
                    attributes_to_remove.append(attribute)
                    mappings_names[a] = attribute
                    mappings_names[attribute] = a
                    break
                    
        if attribute in attributes_to_remove:
            continue
        if selected_keys:
            attributes_to_metrics[attribute] = metrics[selected_keys[0]]['average_f1']
        attribute_to_first_extractions[attribute] = first_extractions
    return attributes_to_remove, mappings_names, attributes


def measure_openie_results(
    attributes, 
    args, 
    profiler_args,
    run_string, 
    gold_attributes, 
    attributes_to_remove, 
    file_groups, 
    mappings_names
):
    file2extractions = defaultdict(dict)
    unique_attributes = set()
    num_extractions2results = {}
    data_lake = profiler_args.data_lake
    for attr_num, attribute in enumerate(attributes):
        attribute = attribute.lower()
        file_attribute = get_file_attribute(attribute)
        if os.path.exists(f"{args.generative_index_path}/{run_string}_{file_attribute}_file2metadata.json"):
            if attribute in attributes_to_remove:
                print(f"Removing: {attribute}")
                os.remove(f"{args.generative_index_path}/{run_string}_{file_attribute}_file2metadata.json")
                continue
                            
            with open(f"{args.generative_index_path}/{run_string}_{file_attribute}_file2metadata.json") as f:
                file2metadata = json.load(f)
                for file, extraction in file2metadata.items():
                    file2extractions[file][attribute] = extraction
        unique_attributes.add(attribute) 
                    
        if file2extractions:
            num_extractions = len(unique_attributes) 
        nums = [1, 2, 3, 4, len(attributes) - 1, len(gold_attributes)]
        if file2extractions and ((num_extractions) % 5 == 0 or num_extractions in nums) or attr_num == len(attributes) - 1:
            if num_extractions in num_extractions2results:
                continue
            with open(f"{args.generative_index_path}/{run_string}_file2extractions.json", "w") as f:
                json.dump(file2extractions, f, indent=4)

            results = evaluate_synthetic_main(
                run_string, 
                args, 
                profiler_args, 
                data_lake, 
                sample_files=file_groups,
                stage='openie',
                mappings_names=mappings_names
            )
            num_extractions2results[num_extractions] = results
    return num_extractions2results


def run_experiment(profiler_args): 
    print(f'Running run_experiment: {run_experiment=}')
    do_end_to_end = profiler_args.do_end_to_end
    num_attr_to_cascade = profiler_args.num_attr_to_cascade
    train_size = profiler_args.train_size
    data_lake = profiler_args.data_lake

    print(f"Data lake")
    today = datetime.datetime.today().strftime("%m%d%Y") 
    print(f'{today=}')
    
    setattr(profiler_args, 'chunk_size', 3000)
    _, _, _, _, args = get_structure(data_lake)
    print(f'{args=}')
    file_groups, extractions_file, parser, full_file_groups = get_data_lake_info(args, data_lake)
    file2chunks, file2contents, manifest_sessions = prepare_data(
        profiler_args, full_file_groups, parser
    )
    extraction_manifest_sessions = {
        k: v for k, v in manifest_sessions.items() if k in profiler_args.EXTRACTION_MODELS
    }
    # todo: perhaps hard code some attributes, perhaps robust llm works?
    # gold_attributes = get_gold_metadata(args)

    results_by_train_size = defaultdict(dict)
    total_time_dict = defaultdict(dict) 
    
    # -- Run the schema identification
    print(f"\n\n-- Run the schema identification")
    if 1:
        total_tokens_prompted = 0

        print(f"\n\nData-lake: {data_lake}, Train size: {train_size}")
        setattr(profiler_args, 'train_size', train_size)

        run_string = get_run_string(
            data_lake, today, full_file_groups, profiler_args, 
            do_end_to_end, train_size, 
            profiler_args.use_dynamic_backoff,
            profiler_args.EXTRACTION_MODELS,
        )
        print(f'{run_string=}')
        
        sample_files = sample_scripts(
            file_groups,  
            train_size=profiler_args.train_size,
        )

        # top-level schema identification
        print('\n\n- top-level schema identification')
        if do_end_to_end:
            print('\n\nIdentifying schema')
            t0 = time.time()
            num_toks = identify_schema(
                run_string,
                args, 
                file2chunks, 
                file2contents, 
                sample_files, 
                extraction_manifest_sessions, 
                data_lake, 
                profiler_args
            )
            t1 = time.time()
            total_time = t1-t0
            total_tokens_prompted += num_toks
            print(f"Total tokens prompted: {total_tokens_prompted=}")
            total_time_dict[f'schemaId'][f'totalTime_trainSize{train_size}'] = int(total_time)

            # results = evaluate_synthetic_main(
            #     run_string,
            #     args, 
            #     profiler_args, 
            #     data_lake, 
            #     stage='schema_id'
            # )
            # results_by_train_size[train_size]['schema_id'] = results

        # -- run the extraction
        print(f"\n\n-- Run the extraction")
        if 1:
            if do_end_to_end:
                with open(f"{args.generative_index_path}/{run_string}_identified_schema.json") as f:
                    print(f"Identified schema: {f.name=}")
                    most_common_fields = json.load(f)
                with open(f"{args.generative_index_path}/{run_string}_order_of_addition.json") as f:
                    print(f"Order of addition (order to populate data frame): {f.name=}")
                    order_of_addition = json.load(f)
                    order = {item: (len(order_of_addition) - i) for i, item in enumerate(order_of_addition)}
                ctr =  Counter(most_common_fields)
                pred_metadata = sorted(
                    ctr.most_common(num_attr_to_cascade), 
                    key=lambda x: (x[1], order[x[0]]), 
                    reverse=True
                )
                attributes = [item[0].lower() for item in pred_metadata]
                print(f'Attributes: {attributes=}')
            else:
                attributes = gold_attributes

            # - top-level information extraction
            print('\n\n- top-level information extraction')
            num_collected = 0
            for i, attribute in enumerate(attributes):
                print(f'{attribute=}')
                print(f"Extracting {attribute} ({i+1} / {len(attributes)})")
                t0 = time.time()
                num_toks, success = run_profiler(
                    run_string,
                    args, 
                    file2chunks, 
                    file2contents, 
                    sample_files, 
                    full_file_groups,
                    extraction_manifest_sessions, 
                    attribute, 
                    profiler_args
                ) 
                print(f'current number of tokens prompted/used: {num_toks=}')
                t1 = time.time()
                total_time = t1-t0
                total_tokens_prompted += num_toks 
                print(f"Total tokens prompted: {total_tokens_prompted=}")
                total_time_dict[f'extract'][f'totalTime_trainSize{train_size}'] = int(total_time)
                if success:
                    num_collected += 1
                if num_collected >= num_attr_to_cascade:
                    # break if we have collected enough attributes
                    break
            
            # run closed ie eval
            # results = evaluate_synthetic_main(
            #     run_string, 
            #     args, 
            #     profiler_args, 
            #     data_lake,
            #     gold_attributes=gold_attributes, 
            #     stage='extract'
            # )
            # results_by_train_size[train_size]['extract'] = results

            # # Determine whether to remove any attributes based on the extractions
            # # Potentially can rerank the attributes based on the metric comparison to big model
            # if do_end_to_end:
            #     attributes_to_remove, mappings_names, attributes = determine_attributes_to_remove(
            #         attributes, 
            #         args, 
            #         run_string, 
            #         num_attr_to_cascade, 
            #     )
            #     # numextractions2results = measure_openie_results(
            #     #     attributes, 
            #     #     args, 
            #     #     profiler_args,
            #     #     run_string, 
            #     #     gold_attributes, 
            #     #     attributes_to_remove, 
            #     #     full_file_groups, 
            #     #     mappings_names
            #     # )
            #     if 'openie' not in results_by_train_size[train_size]:
            #         results_by_train_size[train_size]['openie'] = {}
            #     # results_by_train_size[train_size]['openie'] = numextractions2results

        results_by_train_size[train_size]['total_tokens_prompted'] = total_tokens_prompted
        results_by_train_size[train_size]['num_total_files'] = len(full_file_groups)
        results_by_train_size[train_size]['num_sample_files'] = len(sample_files)
        results_path = Path('~/data/evaporate/results_dumps').expanduser()
        if not os.path.exists(results_path):
            os.mkdir(results_path)
            print(results_path)
        print(run_string)
        with open(results_path / f"{run_string}_results_by_train_size.pkl", "wb") as f:
            pickle.dump(results_by_train_size, f)
            print(f.name)
            print(f"Saved!")

        print(f"Total tokens prompted: {total_tokens_prompted}")


def get_experiment_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_lake", 
        type=str,
        help="Name of the data lake to operate over. Must be in configs.py"
    )

    parser.add_argument(
        "--do_end_to_end", 
        type=bool,
        default=True,
        help="True for OpenIE, False for ClosedIE"
    )

    parser.add_argument(
        "--num_attr_to_cascade", 
        type=int,
        default=35,
        help="Number of attributes to generate functions for"
    )

    parser.add_argument(
        "--num_top_k_scripts",
        type=int,
        default=10,
        help="Number of generated functions to combine over for each attribute"
    )

    parser.add_argument(
        "--train_size",
        type=int,
        default=10,
        help="Number of files to prompt on"
    )

    parser.add_argument(
        "--combiner_mode",
        type=str,
        default='ws',
        help="Combiner mode for combining the outputs of the generated functions",
        choices=['ws', 'mv', 'top_k']
    )

    parser.add_argument(
        "--use_dynamic_backoff",
        type=bool,
        default=True,
        help="Whether to generate functions or do Evaporate-Direct",
    )

    parser.add_argument(
        "--KEYS",
        type=str,
        default=[],
        help="List of keys to use the model api",
        nargs='*'
    )

    experiment = parser.parse_args()
    return experiment


def main():
    print(f'Running main: {main=}')
    print(f"{os.getenv('CONDA_DEFAULT_ENV')=}")
    # print(f'{sys.path=}')
    experiment_args = get_experiment_args()
    profiler_args = {}
    profiler_args = set_profiler_args(profiler_args)

    model_dict = {
        'MODELS': ["text-davinci-003"],
        'EXTRACTION_MODELS':  ["text-davinci-003"],  
        'GOLD_KEY': "text-davinci-003",
    }
    # Example of how to use a locally-hosted FM
    # model_dict = {
    #     'MODELS': [" EleutherAI/gpt-j-6B"],
    #     'EXTRACTION_MODELS': [" EleutherAI/gpt-j-6B"],
    #     'GOLD_KEY': " EleutherAI/gpt-j-6B",
    #     'MODEL2URL': {
    #         " EleutherAI/gpt-j-6B": "http://127.0.0.1:5000"
    #     },
    # }
            
    for k, v in model_dict.items():
        print(f'model_dict: {k=} {v=}')
        setattr(profiler_args, k, v)

    for k in vars(experiment_args):
        v = getattr(experiment_args, k)
        print(f'experiment_args: {k=} {v=}')
        setattr(profiler_args, k, v)

    # for k, v in vars(profiler_args).items():
    #     print(f'profiler_args: {k=} {v=}')
    print(f'{profiler_args=}')
    # st()
    run_experiment(profiler_args)


if __name__ == "__main__":
    main()
