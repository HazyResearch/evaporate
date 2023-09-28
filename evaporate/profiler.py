"""
So in essence, "profiling" here means building a concise profile or summary of the key information in documents by extracting fields and values. https://claude.ai/chat/8c19bc99-b203-44b2-a0b6-782c07b62f65

This is a summary of evaporate:
Evaporate is a system that can automatically extract structured data from unstructured or semi-structured documents like HTML pages, PDFs, and text files. Based on the paper, Evaporate takes a collection of documents as input and outputs a structured table with columns corresponding to fields or attributes extracted from the documents.

The key capabilities of Evaporate that allow it to extract fields/attributes automatically are:

It uses large language models (LLMs) like GPT-3 that are pretrained on large amounts of text data. By prompts such as natural language descriptions and examples, the LLMs can be adapted to extract fields and values from documents without any customization or training data.
It has strategies to synthesize extraction code from the LLMs that can be applied to process documents at scale after training on just a small sample. This makes it efficient compared to running the LLM on every single document.
It generates multiple candidate extraction functions and aggregates their outputs using techniques like weak supervision to improve quality.
It requires no human effort or labeling. The prompts used are generic and not customized to any domain. The same prompts work across different document types like HTML, PDFs, text etc.
So in summary, by leveraging the versatility of LLMs and code synthesis techniques, Evaporate is able to automatically extract fields from heterogeneous document collections without any customization for different domains or document types. The paper shows it achieves high accuracy across diverse real-world datasets compared to prior specialized systems.
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
    files2contents: dict[str, str], # {filename: content} but can also be file2current_chunk = {filename: chunk}  # hack to trick API to only extract things from 1 chunk
    sample_files: list[str],  # e.g., ['/lfs/ampere1/0/brand...b74ebg.tex'] if textbook is a single file
    fn: str,  # function as a string e.g., def get_definition(text:str ... etc.
    attribute: str,  # e.g., definition, example, theorem, proof, etc.
    data_lake='',
    function_cache=False,
    args = None,
    ):
    """
    Applies the given extraction function fn to each file in sample_files 
    to extract the specified attribute.
    The files2conents in the original evaporate was the whole file, but it can also be a single 
    chunk of text/str, so this function would do extract the attribute in the content string for
    the designated strings in sample_files. 
    Also save functions to cache

    Args:
        files2contents (dict): Mapping of filenames to file contents
        sample_files (list[str]): List of sample file names
        fn (str): Extraction function code 
        attribute (str): Attribute to extract
        data_lake (str): Optional name of data lake
        function_cache (bool): Whether to cache extraction results
        args (Namespace): Optional namespace with args like cache_dir

    Returns:
        all_extractions (dict): Extractions indexed by file name
        num_function_errors (int): Number of errors executing functions

    The function handles caching extractions if function_cache is True.

    Extractions are returned in all_extractions structured as:
    
    {
        "file1": ["extracted_val1"],
        "file2": ["extracted_val2"],
        ...
    }

    The key points are:
        It applies the extraction function fn to each file to get attribute values
        Caches extractions if function_cache enabled
        Returns extractions indexed by file name in all_extractions
        Also returns number of errors executing functions
    """
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

# get fns from LLM using chunks for attr
def get_functions(
    file2chunks: dict[str, list[str]],  # {filename: chunks} to use to generate extraction functions
    sample_files: list[str],  # for now my textbooks only have 1 file, so we use all files to generate extraction functions for each attribute
    # sample_files: list[str],  # in this context it does limit which files from data lake/textbook LLM uses to synthesize extraction functions functions
    attribute: str,
    manifest_session,  # single LLM session from manifest_sessions: dict[str, Session]
    overwrite_cache=False,
) -> tuple[dict[str, str], dict[str, str], int]:
    """
    Generates candidate extraction functions for the given attribute 
    using the provided manifest session and sample files. 
    It creates a function for each chunk and prompt template for the given attribute.
    Since the functions are synthesized with an LLM, each chunk gives a function from the LLM with
    fixed context length (full file to generate extractor function for attribute dont fit).

        # ideally if chunk = whole content, then perhaps a more robust extractio fn for attr could be made using LLL, but LLM is limited in context length
        ex_fn_att: str = LLM(chunk, prompt_template, attribute)

    Synthesizes multiple candidate functions by prompting the LLM
    with different chunks and prompt templates.
    Creates different extraction functions for depending on the chunk and prompt template.
    It uses all chunks (to process all file) and gives a bunch of extraction functions based on each chunk (and prompt template)

    Args:
        file2chunks (dict[str, list[str]]): Filenames to chunks mapping
        sample_files (list[str]): Sample filenames
        attribute (str): Attribute to extract
        manifest_session (Session): Manifest session
        overwrite_cache (bool): Whether to overwrite cache
    
    Returns:
        functions (dict): Generated functions dict, {function_{fn_num}} : script
        function_promptsource (dict): Maps function to prompt
        total_tokens_prompted (int): Tokens prompted

    Functions are returned in functions dict structured as:

        {
            "function_1": "def function1...", 
            "function_2": "def function2...",
        }

    This function gets file2chunks because it needs to pass example chunks to the LLM to synthesize the extraction functions.

    The key points on when functions are synthesized:

        Functions are synthesized at the start for each attribute
        All candidate functions for an attribute are synthesized before extracting values
        The functions are then re-used for all files when extracting that attribute
        So for a new data lake, Evaporate would recommend synthesizing functions once at the start using a sample of files/chunks. The same functions can then be applied to all files for efficiency.
    """
    total_tokens_prompted = 0
    functions = {}  # {function_{fn_num}} : script
    function_promptsource = {}
    for i, (file) in tqdm(
        enumerate(sample_files),
        total=len(sample_files),
        desc=f"Generating functions for attribute {attribute}",
    ):
        # print(f'{attribute=}')
        chunks = file2chunks[file]
        for ii, chunk in enumerate(chunks):
            function_field = get_function_field_from_attribute(attribute)
            for prompt_num, prompt_template in enumerate(METADATA_GENERATION_FOR_FIELDS):
                # print(f'{prompt_template=}')
                prompt = prompt_template.format(
                    attribute=attribute, 
                    function_field=function_field, 
                    chunk=chunk, 
                ) 
                try:
                    script, num_toks = apply_prompt(
                        Step(prompt), 
                        max_toks=500, 
                        # max_toks=2000, 
                        # max_toks=4097, 
                        manifest=manifest_session,
                        overwrite_cache=overwrite_cache
                    )
                    total_tokens_prompted += num_toks
                    print(f'{ii=} {prompt_num=}')
                    # time.sleep(0.1)
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
    file2chunks: dict[str, list[str]],  # {filename: chunks}
    sample_files: list[str], # note all files since our data lake is 1 file
    attribute: str,  # e.g., definition, example, theorem, proof, etc. 
    manifest_session,  # API session for model
    model_name: str,  # e.g., "gpt4" "gpt3.5-turbo"  
    overwrite_cache=False,
    collecting_preds=False,
):
    """
    Extracts values for the given attribute from sample_files
    using the LLM model in the provided manifest session.

    Handles prompting the model with each chunk in file2chunks
    and accumulating the extractions.

    Args:
        file2chunks (dict[str, list[str]]): File paths to chunks
        sample_files (list[str]): Sample file names
        attribute (str): Attribute to extract
        manifest_session (Session): Manifest session 
        model_name (str): Name of LLM model
        overwrite_cache (bool): Whether to overwrite cache
        collecting_preds (bool): Whether to collect more preds

    Returns:
        file2results (dict): Extractions indexed by file name
        total_tokens_prompted (int): Total tokens prompted
        errored_out (bool): Whether model errored out

    Extractions are returned in file2results structured as:

        {
            "file1": ["extracted_val1"],
            "file2": ["extracted_val2"],
        }

    The key reason file2chunks is used here rather than file contents is because 
    the prompts are applied on a per-chunk basis.
    E.g., exctracting all attributes for the entire file isn't possible since the LLM
    has a fixed input context length, so every chunk has to be fed to LLM to make sure
    all attributes are extracted (assuming you want to extract all attributes using the
    LLM, which is expensive. In textbook case with a single file this is likely fine 
    if we have a good prompt!). 
    So we need the chunks to loop through (all chunks if you want all attributes to be 
    extracted correctly with the LLM).

    In contrast, apply_final_profiling_functions operates on the full file contents 
    because it is applying the synthesized functions which are designed to parse 
    the entire content.
    The synthesized functions were built with a subset of the files but all the chunks
    in the selected files. A function is built for each chunk and prompt template (for
    the sample_files selected).

    And get_functions uses file2chunks because it needs a chunk to include as an example in the function synthesis prompt.

    So in summary:

        file2chunks used when operating on individual chunks
        file contents used when operating on full file
        Different granularity depending on the task

    Main concepts:
        - LLMs have limited input, so chunks have to be given if you want to generate
            - generate the attribute extracted form chunk (llm acts as extractor)
            - generate the extractor function that can be re-used "cheaply" on the entire document
                - but the extractor function would be generate per prompt synthesize template per chunk (for all chunks for a subset of files given in sample_files)
    
    For your problem, it might be fine to only use LLM because:
    - we are evaluating on a single textbook with a single file
    (but it is possible to generate functions for subset of files...all files all chunks for that file)
    (possible to do both LLM extraction and function extraction in my case)
    """
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
                                manifest_sessions: dict,  # dict[str, Session]
                                MODELS: list[str],  # e.g., ['gpt-3.5-turbo] 
                                sample_files: list[str],  # given code I read, it seems to restrict which chunks to look at from file2chunks, thus semantically it limits which chunks we extract data by using filename 
                                overwrite_cache: bool = False,
                                ) -> tuple[dict[str, dict[dict, list[str]]], int]:
    """
    For current attribute, extract data from chunks using LLM models directly.
    Set file2chunks = {filename -> [chunk]} to process a single chunk. 
    Since the LLM is a fixed context length, it can only extract data from a single chunk at a time,
    so all chunks for the current file in question have to be given for the LLM to be 
    able to extract correctly given the extraction prompt.
    This is basically forcing the LLM to act as an extractor function (~parser) by 
    conditioning it with a prompt that is designed to extract the attribute from the chunk.

    Extract relevant attributes from text chunks using direct prompts to LLM models without relying on synthesized extraction functions.
    Given the versatility of LLMs, this function leverages their capability to extract information directly from prompts. 
    Chunks are presented to the LLM with a conditioned extraction prompt designed to retrieve the desired attribute. 
    Chunks have to be used since LLMs have a fixed context length and can only extract information from a single chunk at a time, 
    but to extract all attributes from entire doc feed all chunks one at a time.
    This mimics a parser's behavior but utilizes the LLM's vast knowledge and flexibility to follow instructions in a prompt with few shot example.


    sample_files = all files to extract all attribute data from all files. 
    You can also limit the chunks list[str] in file2chunks {str, list[str]} if you want a susbet of the file. 

    Parameters:
        file2chunks (dict): A mapping from filenames to their corresponding text chunks. For processing only one chunk, format it as {filename: [chunk]}.
        attribute (str): The specific type of extraction required, such as "definition" or "theorem".
        manifest_sessions (dict): Contains session-related data for each manifest, facilitating interactions with the LLM.
        MODELS (list): List of LLM model names to be utilized for extraction, such as ['gpt-3.5-turbo'].
        sample_files (list): Provides a means to limit which chunks from 'file2chunks' are to be examined based on their filenames.
        overwrite_cache (bool, optional): If set to True, it will overwrite any existing cached results. Defaults to False.

    Returns:
        tuple:
            - all_extractions (dict): A nested dictionary containing the extracted information, organized first by model name, then by filename.
            - total_tokens_prompted (int): Total number of tokens processed across all models.

    Note:
        This function embodies the direct application of Evaporate's extraction capabilities, bypassing the code synthesis techniques and extracting data straight from the models using conditioned prompts.
    """
    total_tokens_prompted = 0
    #-- Get extractions from LLM models directly
    all_extractions: dict[str, dict[dict, list[str]]] = {} # {mdl_name, {filename, [extractions]}}, attr
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

def get_extractions_using_functions(
                                    functions: dict[str, str], # {fn_key: fn}
                                    files2contents: dict[str, str], # {filename: content} but can also be file2current_chunk = {filename: chunk}  # hack to trick API to only extract things from 1 chunk
                                    attribute: str, # e.g., definition, example, theorem, proof, etc.
                                    manifest_sessions: dict,  # dict[str, Session]
                                    sample_files: list[str],  # in this context it does limit which files from data labe/textbook LLM uses to synthesize extraction functions functions
                                    args,  # for now needed to set the cache directory path for function caching 
                                    overwrite_cache: bool = False,
                                    function_promptsource: dict = defaultdict(str),  # this is actually optional but useful so that code still looks like evaporate
    ):
    """ 
    Get extractions from current file to content (or subset but needs to be a single string) using the LLM extraction functions. 
    Extract relevant information from provided text contents using synthesized functions from LLM.

    Parameters:
        files2contents (dict): Mapping from filenames to their content or a specific chunk of content. If extracting from a specific chunk, the content should represent that chunk.
        attribute (str): Desired extraction type (e.g., definition, theorem).
        manifest_sessions (dict): Sessions associated with each manifest. Presumably used for LLM communication.
        sample_files (list): Specifies the subset of files from the data lake/textbook to be used for function synthesis.
        args: Miscellaneous arguments, notably used for cache directory setup.
        overwrite_cache (bool, optional): If True, overwrites existing cached functions. Defaults to False.
        function_promptsource (dict, optional): Mapping from function names to their source prompts. Provides traceability.

    Returns:
        tuple: 
            - all_extractions (dict): Contains the extracted information, organized by function name, then by filename.
            - function_dictionary (dict): Provides additional metadata about the functions used for extraction.

    Notes:
        The function assumes the presence of another function 'apply_final_profiling_functions' which is not defined in this context. The exact behavior and requirements of this function should be documented separately.
    """
    # -- Get extractions from synthesized functions
    all_extractions: dict[str, dict[str, list[str]]] = defaultdict(dict)  #  {fn_key, {filename, [extractions]}}
    function_dictionary: dict[str, dict[str, str]] = defaultdict(dict)  # {fn_key, {"function", fn_str}
    fn_key: str  # e.g., "function_6211"
    fn: str # function as a string e.g., def get_definition(text:str ... etc.
    for fn_key, fn in functions.items():
        print(f'\n{args=}')
        # remember extractor functions can take much larger file, not necessrily only chunk
        all_extractions[fn_key], num_function_errors = apply_final_profiling_functions(
            files2contents, 
            sample_files,
            fn,
            attribute,
            args=args 
        )
        # code bellow could be removed but decided to leave it to 
        function_dictionary[fn_key]['function'] = fn  
        function_dictionary[fn_key]['promptsource'] = function_promptsource[fn_key]
    return all_extractions, function_dictionary
