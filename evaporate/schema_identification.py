import json
import math
import statistics
import random
from tqdm import tqdm
from collections import Counter, defaultdict
from typing import List, Dict, Tuple, Set

from prompts import Step, SCHEMA_ID_PROMPTS
from utils import apply_prompt
from profiler_utils import clean_metadata


def directly_extract_from_chunks_w_value(
    file2chunks, 
    sample_files, 
    manifest_session,
    overwrite_cache=False,
    topic=None,
    use_dynamic_backoff=True,
):
    """
    Directly extract schema fields and values from chunks of sample files.

    Args:
        file2chunks: Dictionary mapping files to chunked content.
        sample_files: List of sample file names to extract from.
        manifest_session: (OpenAI API?) session for prompting.
        overwrite_cache: Whether to re-prompt API if results cached.
        topic: Topic description to provide context in prompts.
        use_dynamic_backoff: Whether to stop prompting chunk if field detected.

    Returns: 
        field2value: Dictionary mapping extracted fields to values.
        field2count: Dictionary with count of extractions per field. 
        total_tokens_prompted: Total number of API tokens used.

    This uses the language model to directly extract fields and values from
    the chunked sample files, without synthesizing any extraction functions.
    It prompts each chunk independently to discover schema attributes.
    "use_dynamic_backoff" stops prompting once a field is found in the chunk.

    Output dictionaries capture the discovered schema fields and values.
    """
    total_tokens_prompted = 0
    field2value = defaultdict(list)
    field2count = Counter()
    file2results = defaultdict()
    num_chunks_per_file = [len(file2chunks[file]) for file in file2chunks]
    avg_num_chunks_per_file = statistics.mean(num_chunks_per_file)
    stdev_num_chunks_per_file = statistics.stdev(num_chunks_per_file)

    for i, file in enumerate(sample_files):
        chunks = file2chunks[file]
        print(f"Chunks in sample file {file}: {len(chunks)}")

    for i, file in tqdm(
        enumerate(sample_files),
        total=len(sample_files),
        desc="Directly extracting metadata from chunks",
    ):
        chunks = file2chunks[file]
        extractionset = set()
        file_results = {}
        for chunk_num, chunk in enumerate(chunks):
            if (chunk_num > avg_num_chunks_per_file + stdev_num_chunks_per_file) and use_dynamic_backoff:
                break
            prompt_template = SCHEMA_ID_PROMPTS[0]
            prompt = prompt_template.format(chunk=chunk, topic=topic)
            try: 
                result, num_toks = apply_prompt(
                    Step(prompt), 
                    max_toks=500, 
                    manifest=manifest_session, 
                    overwrite_cache=overwrite_cache
                )
            except:
                print("Failed to apply prompt to chunk.")
                continue
            total_tokens_prompted  += num_toks
            result = result.split("---")[0].strip("\n")
            results = result.split("\n")
            results = [r.strip("-").strip() for r in results]
            results = [r[2:].strip() if len(r) > 2 and r[1] == "." else r for r in results ]
            for result in results:
                try:
                    field = result.split(": ")[0].strip(":")
                    value = ": ".join(result.split(": ")[1:])
                except:
                    print(f"Skipped: {result}")
                    continue
                field_versions = [
                    field,
                    field.replace(" ", ""),
                    field.replace("-", ""),
                    field.replace("_", ""),
                ]
                if not any([f.lower() in chunk.lower() for f in field_versions]) and use_dynamic_backoff:
                    continue
                if not value and use_dynamic_backoff:
                    continue
                field = field.lower().strip("-").strip("_").strip(" ").strip(":") 
                if field in extractionset and use_dynamic_backoff:
                    continue
                field2value[field].append(value)
                extractionset.add(field)
                field2count[field] += 1
                file_results[field] = value
        file2results[file] = file_results
    return field2value, field2count, total_tokens_prompted


def get_metadata_string_w_value(field2value, exclude=[], key=0):
    field2num_extractions = Counter()
    for field, values in field2value.items():
        field2num_extractions[field] += len(values)

    reranked_metadata = {}
    try:
        max_count = field2num_extractions.most_common(1)[0][1]
    except:
        return ''
    fields = []
    sort_field2num_extractions = sorted(
        field2num_extractions.most_common(), 
        key=lambda x: (x[1], x[0]), 
        reverse=True
    )
    for item in sort_field2num_extractions:
        field, count = item[0], item[1]
        if field.lower() in exclude:
            continue
        if count == 1 and max_count > 1:
            continue
        idx = min(key, len(field2value[field]) - 1)
        values = [field2value[field][idx]]
        if idx < len(field2value[field]) - 1:
            values.append(field2value[field][idx + 1])
        reranked_metadata[field] = values
        if len(reranked_metadata) > 200:
            break
        fields.append(field)

    random.seed(key)
    keys=reranked_metadata.keys() 
    random.shuffle(list(keys))
    reordered_dict = {}
    for key in keys:
        reordered_dict[key] = reranked_metadata[key]
    reranked_metadata_str = str(reordered_dict)
    return reranked_metadata_str


def rerank(
    field2value, exclude, cleaned_counter, order_of_addition, base_extraction_count,
    most_in_context_example, topic, manifest_session, overwrite_cache=False
):
    total_tokens_prompted = 0
    votes_round1 = Counter()
    for i in range(3):
        reranked_metadata_str = get_metadata_string_w_value(field2value, exclude=exclude, key=i)
        if not reranked_metadata_str:
            continue

        prompt = \
f"""{most_in_context_example}Attributes:
{reranked_metadata_str}

List the most useful keys to include in a SQL database about "{topic}", if any.
Answer:"""
        try:
                result, num_toks = apply_prompt(
                    Step(prompt), 
                    max_toks=500, 
                    manifest=manifest_session, 
                    overwrite_cache=overwrite_cache,
                )
        except:
                print("Failed to apply prompt")
                continue
        total_tokens_prompted += num_toks
        result = result.split("---")[0].strip("\n")
        results = result.split("\n")
        result = results[0].replace("[", "").replace("]", "").replace("'", "").replace('"', '')
        result = result.split(", ")
        result = [r.lower() for r in result]

        indices = [idx for idx, r in enumerate(result) if not r]
        if result and indices:
                result = result[:indices[0]]
            
        # Deduplicate but preserve order
        result = list(dict.fromkeys(result))
        for r in result:
                r = r.strip("_").strip("-")
                r = r.strip("'").strip('"').strip()
                if not r or r in exclude or r not in base_extraction_count:
                    continue
                votes_round1[r] += 2

    fields = sorted(list(votes_round1.keys()))
    for r in fields:
        r = r.strip("_").strip("-")
        r = r.strip("'").strip('"').strip()
        if not r or r in exclude or r not in base_extraction_count:
            continue
        if votes_round1[r] > 1:
            cleaned_counter[r] = votes_round1[r] * base_extraction_count[r]
            order_of_addition.append(r)
        else:
            cleaned_counter[r] = base_extraction_count[r]
            order_of_addition.append(r)
        exclude.append(r)

    return cleaned_counter, order_of_addition, exclude, total_tokens_prompted


def rerank_metadata(
        base_extraction_count, field2value, topic, manifest_session, overwrite_cache
    ):

    most_in_context_example = \
"""Attributes:
{'name': 'Jessica', 'student major': 'Computer Science', 'liscense': 'accredited', 'college name': 'University of Michigan', ''GPA': '3.9', 'student email': 'jess@umich.edu', 'rating': '42', 'title': 'details'}

List the most useful keys to include in a SQL database for "students", if any.
Answer: ['name', 'student major', 'college name', 'GPA', 'student email']

----

"""

    total_tokens_prompted = 0
    cleaned_counter = Counter()
    exclude = []
    order_of_addition = []

    cleaned_counter, order_of_addition, exclude, total_tokens_prompted = rerank(
        field2value, exclude, cleaned_counter, order_of_addition, base_extraction_count,
        most_in_context_example, topic, manifest_session, overwrite_cache=overwrite_cache
    )

    cleaned_counter, order_of_addition, exclude, total_tokens_prompted = rerank(
        field2value, exclude, cleaned_counter, order_of_addition, base_extraction_count,
        most_in_context_example, topic, manifest_session, overwrite_cache=overwrite_cache
    )
    
    fields = sorted(list(base_extraction_count.keys()))
    for field in fields:
        if field not in cleaned_counter:
            cleaned_counter[field] = base_extraction_count[field] / 2
            order_of_addition.append(field)
    return cleaned_counter, total_tokens_prompted, order_of_addition


#################### SAVE GENERATIVE INDEX OF FILE BASED METADATA #########################
def identify_schema(run_string, args, file2chunks: Dict, file2contents: Dict, sample_files: List, manifest_sessions: Dict, group_name: str, profiler_args):
    print(f'{identify_schema=}')
    # get sample and eval files, convert the sample scripts to chunks
    random.seed(0)
    total_tokens_prompted = 0

    field2value, extract_w_value, num_toks = directly_extract_from_chunks_w_value(
        file2chunks,
        sample_files,
        manifest_sessions[profiler_args.GOLD_KEY],
        overwrite_cache=profiler_args.overwrite_cache,
        topic=args.topic,
        use_dynamic_backoff=profiler_args.use_dynamic_backoff,
    )
    total_tokens_prompted += num_toks

    base_extraction_count, num_toks, order_of_addition = rerank_metadata( 
        extract_w_value,
        field2value,
        args.topic, 
        manifest_sessions[profiler_args.GOLD_KEY], 
        profiler_args.overwrite_cache,
    )  
    total_tokens_prompted += num_toks

    with open(f"{args.generative_index_path}/{run_string}_identified_schema.json", "w") as f:
        json.dump(base_extraction_count, f)

    with open(f"{args.generative_index_path}/{run_string}_order_of_addition.json", "w") as f:
        json.dump(order_of_addition, f)

    return total_tokens_prompted
