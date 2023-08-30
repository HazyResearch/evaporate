"""
So in essence, "profiling" here means building a concise profile or summary of the key information in documents 
by extracting fields and values. https://claude.ai/chat/8c19bc99-b203-44b2-a0b6-782c07b62f65

Abstention commonly means deliberate refusal/failure, whereas in Evaporate it refers to a failure/refusal to extract an attribute value from a document.

-- Overview of codebase
get_experiment_args - def in run_profiler.py
get_profiler_args - def in profiler_utils.py
get_structure in utils.py calls -> get_args - def in configs.py

# -- Running run_profiler.py notes
- data_lake: Name of the data lake dataset.  
- do_end_to_end: Whether to perform OpenIE (learn schema) or ClosedIE (use predefined schema).
- num_attr_to_cascade: Number of attributes to extract. One set of extractor functions per attribute.
- num_top_k_scripts: Number of extraction scripts to ensemble.
- train_size: Number of sample files to use for training.
- combiner_mode: How to combine multiple extractions.
- use_dynamic_backoff: Use extraction functions or LLM to get values/profile/summary for attributes.
    - True = Generate and use extraction functions
    - False = No function generation, rely on direct LLM extraction
- KEYS: API keys for models.
"""
import os
import sys
import time
import random
import json
import yaml
import datetime
from pathlib import Path
from tqdm import tqdm
import pickle
import argparse
from collections import defaultdict, Counter

from utils import get_structure, get_manifest_sessions, get_file_attribute
from profiler_utils import chunk_file, sample_scripts, set_profiler_args, filter_file2chunks
from schema_identification import identify_schema
from profiler import run_profiler, get_all_extractions
from evaluate_synthetic import main as evaluate_synthetic_main

from pdb import set_trace as st

random.seed(0)

def get_data_lake_info(args, data_lake, DATA_DIR = "/data/evaporate"):
    """
    Gets file groups and metadata for a data lake.

    Args:
    args: Contains data directory and extractions file info.
    data_lake: Name of the data lake.

    Returns:
    file_groups: List of file paths in the data lake.
    extractions_file: Path to extractions file.
    parser: Parser for the file type.
    full_file_groups: Copy of file groups.

    Gets the file groups by listing the data directory. Also returns
    extractions file path and parser for the data lake format.

    Provides input data and metadata to prepare the data lake for extraction.
    """
    extractions_file = None
    
    # if data_lake == "fda_510ks":
    DATA_DIR = args.data_dir
    file_groups = os.listdir(DATA_DIR)
    if not DATA_DIR.endswith("/"):
        DATA_DIR += "/"
    file_groups = [f"{DATA_DIR}{file_group}" for file_group in file_groups if not file_group.startswith(".")]
    file_groups_new = []
    # for file_group in file_groups:
    #     if not file_group.startswith("."):
    #         file_group.append(f"{DATA_DIR}{file_group}")
    #     # if os.path.isdir(f"{DATA_DIR}{file_group}"):
    #     #     continue
    full_file_groups = file_groups.copy()
    extractions_file = args.gold_extractions_file
    parser = "txt"

    return file_groups, extractions_file, parser, full_file_groups


def chunk_files(file_group, parser, chunk_size, remove_tables, max_chunks_per_file, body_only):
    print(f'{parser=}')
    print(f'{chunk_size=}')
    file2chunks = {}
    file2contents = {}
    for file in tqdm(file_group, total=len(file_group), desc="Chunking files"):
        # # print(f'{file=}')
        # # - skip if directory
        # if os.path.isdir(file):
        #     continue
        # # - process if text file
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
    # print(f'{file2contents.keys()=}')
    return file2chunks, file2contents


# chunking & preparing data
def prepare_data(profiler_args, file_group, parser = "html"):
    """
    Prepares data lake documents by chunking and caching.

    Args:
    profiler_args: Arguments like chunk size and models.
    file_group: List of file paths to prepare.
    parser: Parser based on file type.

    Returns:
    file2chunks: Mapping of files to chunks.
    file2contents: Mapping of files to contents.

    Chunks documents based on chunk size. Caches chunks and contents
    if not already cached. Returns mappings of files to chunks and
    contents, to be used for extraction.

    Prepares documents by chunking and caching for efficient extraction.
    """
    print(f'{parser=}')
    data_lake = profiler_args.data_lake
    print(f'{profiler_args.body_only=}')
    if profiler_args.body_only:
        print('I think this does only the body an html based on profiler_args')
        body_only = profiler_args.body_only
        suffix = f"_bodyOnly{body_only}"
    else:
        suffix = ""

    # prepare the datalake: chunk all files
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

    return file2chunks, file2contents 


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
    """
    From gold_extractions_file i.e. gold file -> attr: extracted_val infer a solid
    list of attributes by counting the most frequent attributes.
    """
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
    # ---- Running run_experiment
    print(f'\n---- Running run_experiment...')
    do_end_to_end = profiler_args.do_end_to_end
    num_attr_to_cascade = profiler_args.num_attr_to_cascade
    train_size = profiler_args.train_size
    data_lake = profiler_args.data_lake 
    chunk_size = 3000
    print(f'{do_end_to_end=}. If True then OpenIE (learn schema) else ClosedIE (ground truth/human specified).')
    print(f'{chunk_size=}')
    
    # -- Get args for files in data lake to enable processing and chunking
    setattr(profiler_args, 'chunk_size', chunk_size)
    _, _, _, _, args = get_structure(data_lake, profiler_args=profiler_args, exist_ok=True)  # Get args for data lake from data lake config. In config.py calls get_args
    print(f'{args=}')
    file_groups, extractions_file, parser, full_file_groups = get_data_lake_info(args, data_lake)
    print(f'{full_file_groups=}')
    # Get sample/subset of files that will be used to learn/infer schema gen, fun gen & fun scoring.
    sample_files = sample_scripts(file_groups,  train_size=profiler_args.train_size,) 
    print(f'{sample_files=}')
    today = datetime.datetime.today().strftime("today_M%mD%dY%Y") 
    print(f'{today=}')
    run_string = get_run_string(
        data_lake, today, full_file_groups, profiler_args, 
        do_end_to_end, train_size, 
        profiler_args.use_dynamic_backoff,
        profiler_args.EXTRACTION_MODELS,
    )
    print(f'{run_string=}') 
    # NOTE: you might have to reorder this if you have multiple files for 1 data lake (but likely 1 file per textbook so this is likely fine)
    # NOTE: you might have to make chunking more robust later to not accidentally chunk/chop a theorem/definition in the middle.
    print(f'{full_file_groups=}')
    print(f'{parser=}')
    file2chunks: dict = {}  # {filename: chunks}
    file2chunks, file2contents = prepare_data(profiler_args, full_file_groups, parser)
    manifest_sessions = get_manifest_sessions(profiler_args.MODELS, MODEL2URL=profiler_args.MODEL2URL, KEYS=profiler_args.KEYS)
    # Get dict of {model_name: manifest_session}
    extraction_manifest_sessions: dict = {mdl_name: manifest_session for mdl_name, manifest_session in manifest_sessions.items() if mdl_name in profiler_args.EXTRACTION_MODELS}

    # - Load gold attributes; do_end_to_end == OpenIE
    if not do_end_to_end:  # closed ie (do_end_to_end <-> don't learn schema)
        if 'gold_attributes.yaml' in getattr(args, 'gold_attributes_file', ''):
            data = yaml.safe_load(open(Path(args.gold_attributes_file).expanduser(), 'r'))
            gold_attributes = data['gold_attributes']
        else:
            gold_attributes = get_gold_metadata(args)  # original evaporate code
    assert not do_end_to_end, f"Only OpenIE is supported right now {do_end_to_end=}"
    attributes = gold_attributes
    print(f'{attributes=}')

    # -- Get curriculum of mathematics from textbook in order (for now 1 textbook 1 file == 1 data lake)
    assert len(file2chunks.keys()) == 1, f"If more than 1 file for 1 textbook, then you need to sort the files so curriculum for that textbook is respected {file2chunks=}"
    assert set(sample_files) == set(file2chunks.keys()), f'Currently all files are used to learn schema & do processing.'
    # TODO: figure out the sample_files role better (later) 
    assert file2chunks is not None, f"Err: {file2chunks=}"
    filename: str = list(file2chunks.keys())[0]
    print(f'{filename=}')
    chunks: list[str] = file2chunks[filename]
    print(f'{len(chunks)=}')
    print(f'{chunks=}')

    # total_tokens_prompted = 0 update later
    for filename, chunk in file2chunks.items():
        print(f'{filename=}')
        print(f'{chunk=}')
        # -- Get all candidate extractd data wrt attributes in current chunk
        extractions = []  # {extracted_data: , attr: , method_or_fun_name:, file_name:}
        for attribute in attributes:
            print(f'{attribute=}, {attribute.lower()=}')
            attribute = attribute.lower()
            file2chunks = filter_file2chunks(file2chunks, sample_files, attribute) # filter out file chunks that don't have the attribute 
            file_attribute = get_file_attribute(attribute)
            # save_path = f"{args.generative_index_path}/{run_string}_{file_attribute}_file2metadata.json"
            # extractions: list[dict] = get_all_extractions(chunk, attr)  # [{extracted_data: , attr: , method_or_fun_name:, file_name: }]
            # PREDICT: get extractions using the synthesized functions and the LM on the sample documents
            all_extractions, function_dictionary, num_toks = get_all_extractions(
                file2chunks,
                file2contents,
                sample_files,
                attribute,
                manifest_sessions,
                profiler_args.EXTRACTION_MODELS,
                profiler_args.GOLD_KEY,
                args,
                use_qa_model=profiler_args.use_qa_model,
                overwrite_cache=profiler_args.overwrite_cache,
            )
            # type(all_extractions type) == {extraction_method: {file_names: -> list[data]}}
            print(f'{type(all_extractions)=}')
            for extraction_method, filename2extraction_data in all_extractions.items():
                # extraction_method: str, filename2extraction_data: dict
                for filename, extraction_data_list in filename2extraction_data.items():
                    # filename: str, extraction_data_list: list
                    for ex_data in extraction_data_list:
                        # extraction_data: str
                        extraction_row = {'extraction_data': ex_data, 'attribute': attribute, 'extraction_method': extraction_method, 'filename':filename}
                        extraction_row.update({"1st_idx_in_chunk": chunk.find(extraction_data)})
                        extractions.append(extraction_row)
            # Sort extractec data by their order of appearance in the chunk.
            extractions.sort(key=lambda row: raw[first_idx_in_chunk])
        #
    for row in extractions:
        print(row)

    # 
    results_by_train_size = defaultdict(dict)
    total_time_dict = defaultdict(dict) 
        
    # - Get sample/subset of files that will be used to learn/infer schema gen, fun gen & fun scoring.
    # sample_files = sample_scripts(
    #     file_groups,  
    #     train_size=profiler_args.train_size,
    # )
    # print(f'{sample_files=}')

    # # -- Run the extraction (profiling a document ~ extracting fields and values to summarize document)
    # print(f'{attributes=}')
    # # - top-level information extraction
    # print('\n\n- top-level information extraction')
    # num_collected = 0
    # for i, attribute in enumerate(attributes):
    #     print(f'{attribute=}')
    #     print(f"Extracting {attribute} ({i+1} / {len(attributes)})")
    #     t0 = time.time()
                
    #         # - "profiling" here means building a concise profile or summary of the key information in documents
    #         num_toks, success = run_profiler(
    #             run_string,
    #             args, 
    #             file2chunks, 
    #             file2contents, 
    #             sample_files, 
    #             full_file_groups,
    #             extraction_manifest_sessions, 
    #             attribute, 
    #             profiler_args
    #         ) 
    #         print(f'current number of tokens prompted/used: {num_toks=}')
    #         t1 = time.time()
    #         total_time = t1-t0
    #         total_tokens_prompted += num_toks 
    #         print(f"Total tokens prompted: {total_tokens_prompted=}")
    #         total_time_dict[f'extract'][f'totalTime_trainSize{train_size}'] = int(total_time)
    #         if success:
    #             num_collected += 1
    #         if num_collected >= num_attr_to_cascade:
    #             # break if we have collected enough attributes
    #             break

    # results_by_train_size[train_size]['total_tokens_prompted'] = total_tokens_prompted
    # results_by_train_size[train_size]['num_total_files'] = len(full_file_groups)
    # results_by_train_size[train_size]['num_sample_files'] = len(sample_files)
    # results_path = Path('~/data/evaporate/results_dumps').expanduser()
    # if not os.path.exists(results_path):
    #     os.mkdir(results_path)
    #     print(results_path)
    # print(run_string)
    # with open(results_path / f"{run_string}_results_by_train_size.pkl", "wb") as f:
    #     pickle.dump(results_by_train_size, f)
    #     print(f.name)
    #     print(f"Saved!")

    # print(f"Total tokens prompted: {total_tokens_prompted}")


def get_experiment_args():
    """
    The experiment settings for how the Machine Learning will behave when profiling/summarizing/extracting
    information from the documents in the data lake. 
    i.e., how the ML experiment will run: what data_lake, if to learn the schema (OpenIE/ClosedIE), type of combiner, 
    train_size/num. files to use, etc.

    get_experiment_args vs set_profiler_args: The distinction is between configuring the profiler vs controlling the overall experiment.

    - data_lake: Name of the data lake dataset.  
    - do_end_to_end: Whether to perform OpenIE (learn schema) or ClosedIE (use predefined schema).
    - num_attr_to_cascade: Number of attributes to extract. One set of extractor functions per attribute.
    - num_top_k_scripts: Number of extraction scripts to ensemble.
    - train_size: Number of sample files to use for training.
    - combiner_mode: How to combine multiple extractions.
    - use_dynamic_backoff: Use extraction functions or LLM to get values/profile/summary for attributes.
        - True = Generate and use extraction functions
        - False = No function generation, rely on direct LLM extraction
    - KEYS: API keys for models.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_lake", 
        type=str,
        help="Name of the data lake to operate over. Must be in configs.py"
    )

    parser.add_argument(
        "--do_end_to_end",  
        type=str,
        default="True", 
        help="True for generating schema from data/OpenIE, False for ClosedIE/given schema. Default is True genererate schema/OpenIE.",
    )

    parser.add_argument(
        "--num_attr_to_cascade", 
        type=int,
        default=35,
        help="Number of attributes to generate functions for. "
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
        default='mv',
        help="Combiner mode for combining the outputs of the generated functions",
        choices=['ws', 'mv', 'top_k']
    )

    parser.add_argument(
        "--use_dynamic_backoff",
        type=str,
        default="True",  
        help="True (default) uses generated functions for extraction. Else, False uses evaporate direct/LLM cfor extraction."
    )

    parser.add_argument(
        "--KEYS",
        type=str,
        default=[],
        help="List of keys to use the model api",
        nargs='*'
    )

    print(f'{sys.argv=}')
    experiment_args = parser.parse_args(args=sys.argv[1:])
    print(f'{experiment_args.do_end_to_end=}')
    print(f'{type(experiment_args.do_end_to_end)=}')
    # - process non-standard args to behave correctly
    experiment_args.do_end_to_end = True if experiment_args.do_end_to_end.lower() == 'true' else False
    print(f'{experiment_args.do_end_to_end=} (learn schema/openie or not)')
    experiment_args.use_dynamic_backoff = True if experiment_args.use_dynamic_backoff.lower() == 'true' else False
    print(f'{experiment_args.use_dynamic_backoff=} (use generated functions or not)')
    # - return expt args
    print(f'{experiment_args=}')
    return experiment_args


def main():
    print(f'Running main: {main=}')
    print(f"{os.getenv('CONDA_DEFAULT_ENV')=}")
    
    # - Hack, hardcode openai key so that vscode debugger works
    keys = open(Path("~/keys/openai_api_key_brandos_koyejolab.txt").expanduser()).read().strip()
    # keys = open(Path("~/keys/openai_api_brandos_personal_key.txt").expanduser()).read().strip()
    # - put the opeanai api key in the sys.argv so code works
    # sys.argv[-1] = keys
    sys.argv = [arg.replace('HARDCODE_IN_PYTHON', keys) for arg in sys.argv]
    
    # - Get args
    experiment_args = get_experiment_args()  # args for the ML e.g., combiner_mode. In run_profiler.py
    profiler_args = set_profiler_args(profiler_args={})  # args for what profile/summary to extract from doc. In profiler_utils.py
    
    # -- model dict
    # model_dict = {
    #     'MODELS': ["text-davinci-003"],
    #     'EXTRACTION_MODELS':  ["text-davinci-003"],  
    #     'GOLD_KEY': "text-davinci-003",
    # }
    model_dict = {
        'MODELS': ["gpt-3.5-turbo"],
        'EXTRACTION_MODELS':  ["gpt-3.5-turbo"],  
        'GOLD_KEY': "gpt-3.5-turbo",
    }
    # model_dict = {
    #     'MODELS': ["gpt-4"],
    #     'EXTRACTION_MODELS':  ["gpt-4"],  
    #     'GOLD_KEY': "gpt-4",
    # }
    # load model dict from yaml file
    print(f'{model_dict=}')
            
    for k, v in model_dict.items():
        print(f'model_dict: {k=} {v=}')
        setattr(profiler_args, k, v)

    for k in vars(experiment_args):
        v = getattr(experiment_args, k)
        print(f'experiment_args: {k=} {v=}')
        setattr(profiler_args, k, v)

    print(f'{profiler_args=}')
    if profiler_args.use_dynamic_backoff:
        print(f"Using dynamic backoff ==> generating functions (not Evaporate Direct)")
    # profiler_args.overwrite_cache = 1
    print(f'{profiler_args.overwrite_cache=}')
    run_experiment(profiler_args)


if __name__ == "__main__":
    main()