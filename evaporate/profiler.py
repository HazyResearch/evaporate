"""
So in essence, "profiling" here means building a concise profile or summary of the key information in documents by extracting fields and values. https://claude.ai/chat/8c19bc99-b203-44b2-a0b6-782c07b62f65
"""
import os
import random
from pathlib import Path
import pickle
from tqdm import tqdm
from functools import partial
from multiprocessing import Pool
from collections import Counter, defaultdict
import signal
from contextlib import contextmanager

import re
import json
import math
import time
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import warnings

from bs4 import GuessedAtParserWarning
warnings.filterwarnings('ignore', category=GuessedAtParserWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="bs4")
warnings.filterwarnings("ignore", category=UserWarning, module="BeautifulSoup")
warnings.filterwarnings("ignore", category=UserWarning, module="lxml")

# from prompts import (METADATA_GENERATION_FOR_FIELDS, EXTRA_PROMPT, METADATA_EXTRACTION_WITH_LM, METADATA_EXTRACTION_WITH_LM_ZERO_SHOT, IS_VALID_ATTRIBUTE, Step,)
from prompts_math import (METADATA_GENERATION_FOR_FIELDS, EXTRA_PROMPT, METADATA_EXTRACTION_WITH_LM, METADATA_EXTRACTION_WITH_LM_ZERO_SHOT, IS_VALID_ATTRIBUTE, Step,)
from utils import apply_prompt, get_file_attribute
from evaluate_profiler import get_topk_scripts_per_field, evaluate
from profiler_utils import filter_file2chunks, check_vs_train_extractions, clean_function_predictions

# import sys
# sys.path.append(f"./weak_supervision/")
# from evaporate.run_ws import run_ws
from evaporate.weak_supervision.run_ws import run_ws

class TimeoutException(Exception): pass


@contextmanager
def time_limit(seconds):
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)


def check_remove_attribute(
    all_extractions, 
    attribute, 
    topic, 
    train_extractions={},
    manifest_session=None,
    overwrite_cache=False,
    all_metrics={},
):

    extraction_fraction = 1.0
    for key, info in all_metrics.items():
        extraction_fraction = info['extraction_fraction']
        break

    values = []
    num_toks = 0
    has_non_none = False
    for i, (file, metadata) in enumerate(all_extractions.items()):
        if metadata and (metadata.lower() not in ["none"]) and metadata != '':
            has_non_none = True
        if len(values) < 3 and metadata and metadata.lower() != "none" and metadata != '':
            values.append(metadata)
    
    if not has_non_none and extraction_fraction > 0.5:
        return False, num_toks
    elif not has_non_none and extraction_fraction <= 0.5:
        return True, num_toks

    extractions = [m for f, m in all_extractions.items()]
    if len(set(extractions)) == 1 or (len(set(extractions)) == 2 and "" in set(extractions)):
        keys = list(train_extractions.keys())
        gold_extractions = train_extractions[keys[0]]
        if Counter(gold_extractions).most_common(1)[0][0].lower() != Counter(extractions).most_common(1)[0][0].lower():
            return False, num_toks
        else:
            return True, num_toks

    attr_str = f"{attribute}" 
    
    prompt_template = IS_VALID_ATTRIBUTE[0]

    votes = Counter()
    for value in values:
        prompt = prompt_template.format(value=value, attr_str=attr_str, topic=topic)
        try:
            check, num_toks = apply_prompt(
                Step(prompt), 
                max_toks=10, 
                manifest=manifest_session,
                overwrite_cache=overwrite_cache
            )
            check = check.split("----")[0]
            if "yes" in check.lower():
                votes["yes"] += 1
            elif 'no' in check.lower():
                votes["no"] += 1
        except:
            print(f"Rate limited...")

    keep = False
    if votes['yes']:
        keep = True
    return keep, num_toks


def combine_extractions(
    args,
    all_extractions, 
    all_metrics, 
    combiner_mode = "mv", 
    attribute=None, 
    train_extractions = None,
    gold_key = None,
    extraction_fraction_thresh=0.8,
):
    final_extractions = {}

    extraction_fraction = 0.0
    for key, info in all_metrics.items():
        extraction_fraction = info['extraction_fraction']
        break

    # collect all values by file
    all_file2extractions = defaultdict(list)
    total_tokens_prompted = 0
    num_keys = all_extractions.keys()
    for key, file2extractions in all_extractions.items():
        for i, (file, extraction) in tqdm(enumerate(
            file2extractions.items()), 
            total=len(file2extractions), 
            desc=f"Applying key {key}"
        ):
            extraction = clean_function_predictions(
                extraction, 
                attribute=attribute
            )
            all_file2extractions[file].append(extraction)

    # if combiner_mode == "mv" or combiner_mode == "top_k":
    #     for file, extractions in all_file2extractions.items():
    #         if extraction_fraction >= extraction_fraction_thresh:
    #             extractions = [e for e in extractions if e]
    #             if not extractions:
    #                 extractions = ['']
    #         final_extractions[file] = str(Counter(extractions).most_common(1)[0][0])
    if combiner_mode == "mv" or combiner_mode == "top_k":
        for file, extractions in all_file2extractions.items():
            # Filter out empty extractions if fraction above threshold 
            if extraction_fraction >= extraction_fraction_thresh:  
                extractions = [e for e in extractions if e]
                if not extractions:
                    extractions = ['']
            # Tally up votes for each value using Counter
            counted_extractions = Counter(extractions) 
            # Get the most common value
            top_value = counted_extractions.most_common(1)[0][0]
            # Set the top voted value as the final extraction
            final_extractions[file] = top_value

    elif combiner_mode == "ws":
        preds, used_deps, missing_files = run_ws(
            all_file2extractions, 
            args.gold_extractions_file, 
            attribute=attribute, 
            has_abstains=extraction_fraction,
            extraction_fraction_thresh=extraction_fraction_thresh,
        )

        for i, (file, extractions) in enumerate(all_file2extractions.items()):
            if file in missing_files:
                continue
            if len(extractions)== 1:
                if type(extractions) == list:
                    extractions = extractions[0]
                pred = extractions
                final_extractions[file] = pred
            elif len(Counter(extractions)) == 1:
                pred = str(Counter(extractions).most_common(1)[0][0])
                final_extractions[file] = pred
            else:
                pred = preds[len(final_extractions)]
                if not pred: 
                    final_extractions[file] = str(Counter(extractions).most_common(1)[0][0])
                else:
                    final_extractions[file] = pred

    if train_extractions:
        final_extractions = check_vs_train_extractions(train_extractions, final_extractions, gold_key)
    
    return final_extractions, total_tokens_prompted


def apply_final_ensemble(
    group_files,
    file2chunks,
    file2contents,
    selected_keys,
    all_metrics,
    attribute,
    function_dictionary,
    data_lake='',
    function_cache=False,
    manifest_sessions=[],
    MODELS=[],
    overwrite_cache=False,
    do_end_to_end=False,
    args = None,
):
    print(f'{function_cache=}')
    all_extractions = {}
    total_tokens_prompted = 0
    for key in selected_keys:
        if "function" in key:
            t0 = time.time()
            print(f"Applying function {key}...")
            extractions, num_function_errors = apply_final_profiling_functions(
                file2contents,
                group_files,
                function_dictionary[key]['function'],
                attribute,
                data_lake=data_lake,
                function_cache=function_cache,
                args=args
            )
            t1 = time.time()
            total_time = t1 - t0
            all_extractions[key] = extractions
            function_dictionary[key]['runtime'] = total_time
        elif key in MODELS:
            manifest_session = manifest_sessions[key]
            extractions, num_toks, errored_out = get_model_extractions(
                file2chunks,
                group_files,
                attribute,
                manifest_session,
                key,
                overwrite_cache=overwrite_cache,
            )
            total_tokens_prompted += num_toks
            if not errored_out:
                all_extractions[key] = extractions
        else:
            raise ValueError(f"Key {key} not supported.")


    if not do_end_to_end and not all_extractions:
        default = {}
        for file, _ in file2contents.items():
            default[file] = ['']
        all_extractions['default'] = default
    
    return all_extractions, total_tokens_prompted


def apply_final_profiling_functions(
    files2contents,
    sample_files,
    fn,
    attribute,
    data_lake='',
    function_cache=False,
    args = None,
    ):
    print(f'--> {args=}')
    function_cache = True  # hardcoded TODO
    print(f'{function_cache=} (hardcoded)') 
    # perhaps fix later to store fn's always original https://github.com/HazyResearch/evaporate/blob/83204a54dd97fb0f51a01643b4fc16c97fc5e472/evaporate/profiler.py#L249
    if function_cache:
        original_fn = fn
        file_attribute = attribute.replace(" ", "_").replace("/", "_").lower()
        # - set cach_dir (for /function_cache/)
        if args is None:
            cache_dir = "./function_cache/"
        else:
            # cache_dir = "./function_cache/"
            cache_dir = args.cache_dir + '/function_cache/'
        print(f'{cache_dir=}')
        # - create function_cache dir
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        cache_path = f"{cache_dir}function_cache_{file_attribute}_{data_lake}.pkl" 
        if os.path.exists(cache_path):
            try:
                with open(cache_path, "rb") as f:
                    function_cache_dict = pickle.load(f)
            except:
                function_cache_dict = defaultdict(dict)
        else:
            function_cache_dict = defaultdict(dict)

    all_extractions = {}
    num_function_errors = 0
    num_timeouts = 0
    for i, (file) in enumerate(sample_files):
        content = files2contents[file]
        extractions = []

        global result
        global text 
        global preprocessed_text 
        text = content
        preprocessed_text = text.replace(">\n", ">")

        if num_timeouts > 1:
            all_extractions[file] = deduplicate_extractions(extractions)
            continue

        if function_cache and file in function_cache_dict and original_fn in function_cache_dict[file]:
            extractions = function_cache_dict[file][original_fn]
        else:
            if type(fn) != str:
                # Function is defined in code and not prompt-generated (e.g. for QA)
                # So no need for special parsing and error handling
                result = fn(text)
                extractions.append(result)
            else:
                fn = "\n".join([l for l in fn.split("\n") if "print(" not in l])
                fn = "\n".join([l for l in fn.split("\n") if not l.startswith("#")])
                function_field = get_function_field_from_attribute(attribute)

                err = 0
                try:
                    try:
                        with time_limit(1):
                            exec(fn, globals())
                            exec(f"result = get_{function_field}_field(text)", globals())
                    except TimeoutException as e:
                        print(f"Timeout {num_timeouts}")
                        num_timeouts += 1
                        raise e
                    extractions.append(result)
                except Exception as e:
                    # This error is due to compilation and execution errors in the synthesized functions
                    err = 1
                    pass

                if err:
                    # applied to preprocessed text
                    try:
                        try:
                            with time_limit(1):
                                exec(fn, globals())
                                exec(f"result = get_{function_field}_field(preprocessed_text)", globals())
                        except TimeoutException as e:
                            print("Timeout")
                            raise e
                        extractions.append(result)
                        err = 0
                    except Exception as e:
                        # This error is due to compilation and execution errors in the synthesized functions
                        pass

                if err:
                    num_function_errors = 1

            if function_cache:
                function_cache_dict[file][original_fn] = extractions
        all_extractions[file] = deduplicate_extractions(extractions)

    if function_cache:
        try:
            with open(cache_path, "wb") as f:
                pickle.dump(function_cache_dict, f)
        except Exception as e:
            pass
    return all_extractions, num_function_errors


def get_function_field_from_attribute(attribute):
    return re.sub(r"[^A-Za-z0-9]", "_", attribute)


def get_functions(
    file2chunks,
    sample_files,
    attribute,
    manifest_session,
    overwrite_cache=False,
):
    total_tokens_prompted = 0
    functions = {}
    function_promptsource = {}
    for i, (file) in tqdm(
        enumerate(sample_files),
        total=len(sample_files),
        desc=f"Generating functions for attribute {attribute}",
    ):
        print(f'{attribute=}')
        chunks = file2chunks[file]
        for chunk in chunks:
            function_field = get_function_field_from_attribute(attribute)
            for prompt_num, prompt_template in enumerate(METADATA_GENERATION_FOR_FIELDS):
                print(f'{prompt_template=}')
                prompt = prompt_template.format(
                    attribute=attribute, 
                    function_field=function_field, 
                    chunk=chunk, 
                ) 
                try:
                    script, num_toks = apply_prompt(
                        Step(prompt), 
                        max_toks=500, 
                        manifest=manifest_session,
                        overwrite_cache=overwrite_cache
                    )
                    total_tokens_prompted += num_toks
                except Exception as e:
                    print(e)
                    print(f"Failed to generate function for {attribute}")
                    continue

                if "def" not in script:
                    script = \
f"""def get_{function_field}_field(text: str):
    \"""
    Function to extract {attribute}. 
    \"""
    {script}
"""             
                return_idx = [i for i, s in enumerate(script.split("\n")) if "return" in s]
                if not return_idx:
                    continue
                return_idx = return_idx[0]
                script = "\n".join(script.split("\n")[: return_idx + 1])
                script = "\n".join([s for s in script.split("\n") if "print(" not in s])
                script = "\n".join([s for s in script.split("\n") if s.startswith(" ") or s.startswith("\t") or s.startswith("def")])
                fn_num = len(functions)
                functions[f"function_{fn_num}"] = script
                function_promptsource[f"function_{fn_num}"] = prompt_num
    return functions, function_promptsource, total_tokens_prompted


def trim_chunks(chunk, attribute, window=20):
    # Handling context length issues.

    tokenized_chunk = chunk.lower().split()
    indices = [i for i, s in enumerate(tokenized_chunk) if attribute.lower() in s]
    if indices:
        index = indices[0]
        lb = max(0, index-window)
        ub = min(len(chunk), index)
        trimmed_chunk = " ".join(tokenized_chunk[lb:ub])
    else:
        # split tokenized_chunk into groups of 50 tokens
        mini_chunks = []
        for i in range(0, len(tokenized_chunk), 50):
            mini_chunks.append(" ".join(tokenized_chunk[i:i+50]))

        # find the mini chunk with the most attribute tokens
        max_num_attr_tokens = 0
        max_num_attr_tokens_idx = 0
        for i, mini_chunk in enumerate(mini_chunks):
            num_attr_tokens = len([s for s in attribute.lower().split() if s in mini_chunk])
            if num_attr_tokens > max_num_attr_tokens:
                max_num_attr_tokens = num_attr_tokens
                max_num_attr_tokens_idx = i
        trimmed_chunk = mini_chunks[max_num_attr_tokens_idx]
    
    return trimmed_chunk


def deduplicate_extractions(extractions):
    deduplicated_extractions = []
    for extraction in extractions:
        duplicate = False
        for prev_extraction in deduplicated_extractions:
            if extraction == prev_extraction:
                duplicate = True
        if not duplicate:
            deduplicated_extractions.append(extraction)
    return deduplicated_extractions


def get_model_extractions(
    file2chunks, 
    sample_files, 
    attribute, 
    manifest_session,
    model_name,
    overwrite_cache=False,
    collecting_preds=False,
):
    """ Extract value for attribute given using LM. """

    num_errors = 0
    total_prompts = 0
    total_tokens_prompted = 0
    has_context_length_error = False
    file2results = {}
    errored_out = False
    for i, (file) in tqdm(
        enumerate(sample_files),
        total=len(sample_files),
        desc=f"Extracting attribute {attribute} using LM",
    ):
        if num_errors > 10 and num_errors == total_prompts:
            print(f"All errorring out.. moving on.")
            errored_out = True
            continue
        
        chunks = file2chunks[file] 
        extractions = []
        for chunk_num, chunk in enumerate(chunks):
            if "flan" in model_name:
                PROMPTS = METADATA_EXTRACTION_WITH_LM_ZERO_SHOT
            else:
                PROMPTS = METADATA_EXTRACTION_WITH_LM

            if has_context_length_error:
                    chunk = trim_chunks(chunk, attribute)
            for prompt_template in PROMPTS:
                prompt = prompt_template.format(attribute=attribute, chunk=chunk)
                err = 0
                total_prompts += 1
                try:
                    extraction, num_toks = apply_prompt(
                        Step(prompt), 
                        max_toks=100, 
                        manifest=manifest_session,
                        overwrite_cache=overwrite_cache
                    )
                    total_tokens_prompted += num_toks
                except:
                    err = 1
                    num_errors += err
                    print(f"Failed to extract {attribute} for {file}")
                    has_context_length_error = True
                    continue
                extraction = extraction.split("---")[0].strip("\n")
                extraction = extraction.split("\n")[-1].replace("[", "").replace("]", "").replace("'", "").replace('"', '')
                extraction = extraction.split(", ")
                extractions.append(extraction)
            if collecting_preds and (not any(e for e in extractions) or not any(e[0] for e in extractions)):
                for prompt_template in EXTRA_PROMPT:
                    prompt = prompt_template.format(attribute=attribute, chunk=chunk)
                    err = 0
                    total_prompts += 1
                    try:
                        extraction, num_toks = apply_prompt(
                            Step(prompt), 
                            max_toks=100, 
                            manifest=manifest_session,
                            overwrite_cache=overwrite_cache
                        )
                        total_tokens_prompted += num_toks
                    except:
                        err = 1
                        num_errors += err
                        print(f"Failed to extract {attribute} for {file}")
                        has_context_length_error = True
                        continue
                    extraction = extraction.split("---")[0].strip("\n")
                    extraction = extraction.split("\n")[-1].replace("[", "").replace("]", "").replace("'", "").replace('"', '')
                    extraction = extraction.split(", ")
                    extractions.append(extraction)
            
        file2results[file] = deduplicate_extractions(extractions)
    return file2results, total_tokens_prompted, errored_out


def get_all_extractions(
        file2chunks: dict[str, list[str]],  # {filename: [chunk1, chunk2, ...]}
        file2contents: dict[str, str],  # {filename: content}
        sample_files,
        attribute: str, # e.g., definition, example, theorem, proof, etc.
        manifest_sessions,
        MODELS,
        GOLD_KEY, 
        args,
        use_qa_model=False,
        overwrite_cache=False,
    ):
    """
  Extract attribute values from sample files using models and synthesized functions.

  Args:
    file2chunks: Mapping of files to chunks.
    file2contents: Mapping of files to full contents.
    sample_files: List of sample file names.
    attribute: Name of attribute to extract.
    manifest_sessions: API sessions for models. 
    MODELS: List of model names to apply.
    GOLD_KEY: Name of gold model.
    args: Config arguments.
    use_qa_model: If True, apply QA model.
    overwrite_cache: Whether to re-prompt API.

  Returns:
    all_extractions: Dict of extractions from all models and functions.
    function_dict: attribute values on generated functions.
    total_tokens: Number of tokens prompted.

  Gets extractions for the attribute from:
    - Models in MODELS using prompt engineering
    - Synthesized functions based on sample files
  
  Saves the generated functions in function_dict.

  Returns all model and function extractions before aggregation.
    """
    total_tokens_prompted = 0
    all_extractions = {}

    #-- Get extractions from LLM models directly
    # same as get_extractions_directly_from_LLM_model (todo re-use code?)
    for model in MODELS:
        manifest_session = manifest_sessions[model]
        # Extract values for attribute given using LM from file chunks.
        extractions, num_toks, errored_out = get_model_extractions(
            file2chunks, 
            sample_files, 
            attribute,  
            manifest_session,
            model,
            overwrite_cache=overwrite_cache,
            collecting_preds=True,
        )

        total_tokens_prompted += num_toks
        if not errored_out:
            all_extractions[model] = extractions
        else:
            print(f"Not applying {model} extractions")
            return 0, 0, total_tokens_prompted
    print(f'{all_extractions=}')

    # -- Generate extraction functions using LLM model
    manifest_session = manifest_sessions[GOLD_KEY]
    functions, function_promptsource, num_toks = get_functions(
        file2chunks, 
        sample_files, 
        attribute, 
        manifest_session,
        overwrite_cache=overwrite_cache,
    )
    print(f"Number of functions: {len(functions)}")
    total_tokens_prompted += num_toks

    # -- Get extractions from synthesized functions
    function_dictionary = defaultdict(dict)
    for fn_key, fn in functions.items():
        print(f'\n{args=}')
        all_extractions[fn_key], num_function_errors = apply_final_profiling_functions(
            file2contents, 
            sample_files,
            fn,
            attribute,
            args=args 
        )
        function_dictionary[fn_key]['function'] = fn
        function_dictionary[fn_key]['promptsource'] = function_promptsource[fn_key]
    return all_extractions, function_dictionary, total_tokens_prompted


def run_profiler(run_string, args, file2chunks, file2contents, sample_files, group_files, manifest_sessions, attribute, profiler_args):
    """
  Run profiling pipeline to extract attr data for a single attribute.

  Args:
    run_string: Unique ID string for this run.
    args: Arguments for data set and file paths.
    file2chunks: Mapping of files to chunked content.
    file2contents: Mapping of files to raw content.
    sample_files: List of sample file names.
    group_files: List of all file names.
    manifest_sessions: Dictionary of model API sessions.
    attribute: Attribute name to extract.
    profiler_args: Configurable arguments for profiler.

  Returns:
    total_tokens_prompted: Number of tokens used.
    success: Whether extraction was successful.

  This executes the full profiling pipeline for a single attribute:
    - Generates candidate extraction functions
    - Evaluates and selects best functions
    - Applies functions to full data set
    - Aggregates outputs

  Output is the extracted metadata for the attribute across all files.
    """
    print(f'{run_profiler=}')
    total_tokens_prompted = 0
    
    attribute = attribute.lower()
    file_attribute = get_file_attribute(attribute)
    save_path = f"{args.generative_index_path}/{run_string}_{file_attribute}_file2metadata.json"

    # FILTER: filter out file chunks that don't have the attribute
    file2chunks = filter_file2chunks(file2chunks, sample_files, attribute)
    if file2chunks is None:
        return total_tokens_prompted, 0

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
    # print(f'{all_extractions=}')
    total_tokens_prompted += num_toks
    if not all_extractions:
        return total_tokens_prompted, 0

    # SCORE: Determine a set of functions to utilize for the full data lake.
    # note: for our theorem proving applicaiton, we could choose the function/output that actually translates to valid ITP code
    all_metrics, key2golds, num_toks = evaluate(
        all_extractions,
        profiler_args.GOLD_KEY,
        field=attribute,
        manifest_session=manifest_sessions[profiler_args.GOLD_KEY], 
        overwrite_cache=profiler_args.overwrite_cache,
        combiner_mode=profiler_args.combiner_mode,
        extraction_fraction_thresh=profiler_args.extraction_fraction_thresh,
        use_abstension=profiler_args.use_abstension,
    )
    total_tokens_prompted += num_toks

    selected_keys = get_topk_scripts_per_field(
        all_metrics,
        function_dictionary,
        all_extractions,
        gold_key = profiler_args.GOLD_KEY, 
        k=profiler_args.num_top_k_scripts,
        do_end_to_end=profiler_args.do_end_to_end,
        combiner_mode=profiler_args.combiner_mode,
    )

    if not selected_keys and profiler_args.do_end_to_end:
        print(f"Removing {file_attribute}")
        if os.path.exists(save_path):
            os.remove(save_path)
        return total_tokens_prompted, 0

    # APPLY: Run the best performing functions on the data lake.
    print(f"Apply the scripts to the data lake and save the metadata. Taking the top {profiler_args.num_top_k_scripts} scripts per field.")
    top_k_extractions, num_toks = apply_final_ensemble(
        group_files,
        file2chunks,
        file2contents,
        selected_keys,
        all_metrics,
        attribute,
        function_dictionary,
        data_lake=args.data_lake,
        manifest_sessions=manifest_sessions,
        function_cache=True,
        MODELS=profiler_args.EXTRACTION_MODELS,
        overwrite_cache=profiler_args.overwrite_cache,
        do_end_to_end=profiler_args.do_end_to_end,
    )
    total_tokens_prompted += num_toks

    file2metadata, num_toks = combine_extractions(
        args,
        top_k_extractions, 
        all_metrics, 
        combiner_mode=profiler_args.combiner_mode,
        train_extractions=all_extractions,
        attribute=attribute, 
        gold_key = profiler_args.GOLD_KEY,
        extraction_fraction_thresh=profiler_args.extraction_fraction_thresh,
    )
    total_tokens_prompted += num_toks

    # FINAL CHECK: Ensure that the metadata is valid (Skip this for ClosedIE).
    if profiler_args.do_end_to_end:
        keep_attribute, num_toks = check_remove_attribute(
            file2metadata, 
            attribute, 
            args.topic, 
            train_extractions=key2golds,
            manifest_session=manifest_sessions[profiler_args.GOLD_KEY], 
            overwrite_cache=profiler_args.overwrite_cache,
            all_metrics=all_metrics,
        )
        total_tokens_prompted += num_toks
        if not keep_attribute:
            print(f"Removing {file_attribute}")
            if os.path.exists(save_path):
                os.remove(save_path)
            return total_tokens_prompted, 0

    # Save: all_extractions, functions, all_metrics, top_k_keys, file2metadata and top_k_extractions.
    print('Save: all_extractions, functions, all_metrics, top_k_keys, file2metadata and top_k_extractions.')
    current_file_path = None # just so the error tells me which path failed
    exist_ok = True
    try:
        print('args.generative_index_path}/{run_string}_{file_attribute}_')
        print(f'{args.generative_index_path}/{run_string}_{file_attribute}_')

        current_file_path = f"{args.generative_index_path}/{run_string}_{file_attribute}_all_extractions.json"
        os.makedirs(args.generative_index_path, exist_ok=exist_ok)
        with open(current_file_path, "w") as f:
            json.dump(all_extractions, f, indent=4)

        current_file_path = f"{args.generative_index_path}/{run_string}_{file_attribute}_functions.json"
        with open(current_file_path, "w") as f:
            json.dump(function_dictionary, f, indent=4)

        current_file_path = f"{args.generative_index_path}/{run_string}_{file_attribute}_all_metrics.json"
        with open(current_file_path, "w") as f:
            json.dump(all_metrics, f, indent=4)

        current_file_path = f"{args.generative_index_path}/{run_string}_{file_attribute}_top_k_keys.json"
        with open(current_file_path, "w") as f:
            json.dump(selected_keys, f, indent=4)

        current_file_path = f"{args.generative_index_path}/{run_string}_{file_attribute}_file2metadata.json"
        with open(current_file_path, "w") as f:
            json.dump(file2metadata, f, indent=4)

        current_file_path = f"{args.generative_index_path}/{run_string}_{file_attribute}_top_k_extractions.json"
        with open(current_file_path, "w") as f:
            json.dump(top_k_extractions, f, indent=4)

        return total_tokens_prompted, 1

    except Exception as e:
        import traceback
        traceback.print_exc()  # This will print the full stack trace
        raise Exception(f'Error saving to file {e}:\n\t{current_file_path}\n\tWith exception.')

    try:
        clean_file2metadata = {}
        for file, metadata in file2metadata.items():
            clean_file2metadata[file] = str(metadata)
        with open(f"{args.generative_index_path}/{run_string}_{file_attribute}_file2metadata.json", "w") as f:
            json.dump(clean_file2metadata, f, indent=4)
        with open(f"{args.generative_index_path}/{run_string}_{file_attribute}_all_metrics.json", "w") as f:
            json.dump(all_metrics, f, indent=4)
        with open(f"{args.generative_index_path}/{run_string}_{file_attribute}_top_k_keys.json", "w") as f:
            json.dump(selected_keys, f, indent=4)
    except Exception as e:
        pass


    return total_tokens_prompted, 0

def get_extractions_directly_from_LLM_model(
                                file2chunks: dict[str, list[str]], # for 1 chunk do {filename: [chunk1]}
                                attribute: str, # e.g., definition, example, theorem, proof, etc.
                                manifest_sessions,  # dict[str, Session]
                                MODELS: list[str],  # e.g., ['gpt-3.5-turbo] 
                                sample_files: list[str],  # given code I read, it seems to restrict which chunks to look at from file2chunks, thus semantically it limits which chunks we extract data by using filename 
                                overwrite_cache: bool = False,
                                ) -> tuple[dict[str, dict[dict, list[str]]], int]]:
    """
    For current attribute, extract data from chunks using LLM models directly.
    Set file2chunks = {filename -> [chunk]} to process a single chunk. 

    """
    total_tokens_prompted = 0
    all_extractions: dict[str, dict[dict, list[str]]]  # {mdl_name, {filename, [extractions]}}, attr
    #-- Get extractions from LLM models directly
    for model in MODELS:
        manifest_session = manifest_sessions[model]
        # Extract values for attribute given using LM from file chunks.
        extractions, num_toks, errored_out = get_model_extractions(
            file2chunks, 
            sample_files, 
            attribute,  
            manifest_session,
            model,
            overwrite_cache=overwrite_cache,
            collecting_preds=True,
        )

        total_tokens_prompted += num_toks
        if not errored_out:
            all_extractions[model] = extractions
        else:
            print(f"Not applying {model} extractions")
            return 0, 0, total_tokens_prompted
    # print(f'{all_extractions=}') 
    return all_extractions, total_tokens_prompted

 def get_extractions_from_synthesized_functions(
                                                file2chunks: dict[str, list[str]], # for 1 chunk do {filename: [chunk1]}
                                                attribute: str, # e.g., definition, example, theorem, proof, etc.
                                                manifest_sessions,  # dict[str, Session]
                                                MODELS: list[str],  # e.g., ['gpt-3.5-turbo'] 
                                                GOLD_KEY: str , # e.g. gpt-3.5-turbo, profiler_args.GOLD_KEY
                                                sample_files: list[str],  # in this context it does limit which files from data labe/textbook LLM uses to synthesize extraction functions functions
                                                overwrite_cache: bool = False,
    ):
    """ 
    """
    # -- Generate extraction functions using LLM model
    manifest_session = manifest_sessions[GOLD_KEY]
    functions, function_promptsource, num_toks = get_functions(
        file2chunks, 
        sample_files, 
        attribute, 
        manifest_session,
        overwrite_cache=overwrite_cache,
    )
    print(f"Number of functions: {len(functions)}")
    total_tokens_prompted += num_toks

    # -- Get extractions from synthesized functions
    function_dictionary = defaultdict(dict)
    for fn_key, fn in functions.items():
        print(f'\n{args=}')
        all_extractions[fn_key], num_function_errors = apply_final_profiling_functions(
            file2contents 
            sample_files,
            fn,
            attribute,
            args=args 
        )
        function_dictionary[fn_key]['function'] = fn
        function_dictionary[fn_key]['promptsource'] = function_promptsource[fn_key]
    return all_extractions, function_dictionary, total_tokens_prompted

def f(file2contents, sample_files, functions, attribute, args):
    # -- Get extractions from synthesized functions
    function_dictionary = defaultdict(dict)
    all_extractions: dict[str, dict[dict, list[str]]]  # {fn_key, {filename, [extractions]}}, attr
    for fn_key, fn in functions.items():
        print(f'\n{args=}')
        all_extractions[fn_key], num_function_errors = apply_final_profiling_functions(
            file2contents 
            sample_files,
            fn,
            attribute,
            args=args 
        )
        function_dictionary[fn_key]['function'] = fn
        function_dictionary[fn_key]['promptsource'] = function_promptsource[fn_key]
    return all_extractions, function_dictionary, total_tokens_prompted