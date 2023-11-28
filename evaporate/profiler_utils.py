import re
import os
import argparse
import random
from bs4 import BeautifulSoup
from collections import Counter, defaultdict


def set_profiler_args(profiler_args):
    """
    Configures the profiler (=summarizer) pipeline itself e.g., size of chunk, eval_size, if remove_tables or not etc.
    How the profile/summary of the document will look like. 
    While the experiment_args is the hyperparameters for the actual experiment 
    e.g., if the combiner is weak-supervised (ws) or majority-vote (mv), etc.

    I removed all args that were part of the experiment to make this less confusing. 
    This will be as much as I can only about the profiling/summarization pipeline and not how the experiment
    works becuase of the ML.

    So crucial to merge the two args into one properly before running experiment. 
    """

    parser = argparse.ArgumentParser(
        "LLM profiler.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--chunk_size",
        type=int,
        default=5000
    )

    # parser.add_argument(
    #     "--train_size",
    #     type=int,
    #     default=15,
    # )

    parser.add_argument(
        "--eval_size",
        type=int,
        default=15,
    )

    parser.add_argument(
        "--max_chunks_per_file",
        type=int,
        default=-1,
    )

    # parser.add_argument(
    #     "--num_top_k_scripts",
    #     type=int,
    #     default=1,
    #     help="of all the scripts we generate for the metadata fields, number to retain after scoring their qualities",
    # )

    parser.add_argument(
        "--extraction_fraction_thresh",
        type=int,
        default=0.9,
        help="for abstensions (=miss/omission/failure to extract) approach",
    )

    parser.add_argument(
        "--remove_tables",
        type=bool,
        default=False,
        help="Remove tables from the html files?",
    )

    parser.add_argument(
        "--body_only",
        type=bool,
        default=False,
        help="Only use HTML body",
    )

    parser.add_argument(
        "--max_metadata_fields",
        type=int,
        default=15,
    )

    # parser.add_argument(
    #     "--use_dynamic_backoff",
    #     type=bool,
    #     default=True,
    #     help="Whether to do the extraction function generation workflow or directly extract using LLM from chunks, to get data from doc for attributes.",
    # )

    parser.add_argument(
        "--use_qa_model",
        type=bool,
        default=False,
        help="Whether to apply the span-extractor QA model.",
    )

    parser.add_argument(
        "--overwrite_cache", 
        type=int, 
        default=0,
        help="overwrite the manifest cache"
    )

    # models to use in the pipeline
    parser.add_argument(
        "--MODELS", 
        type=list, 
        help="models to use in the pipeline"
    )

    parser.add_argument(
        "--KEYS", 
        type=list, 
        help="keys for openai models"
    )

    parser.add_argument(
        "--GOLDKEY", 
        type=str, 
        help="models to use in the pipeline"
    )

    parser.add_argument(
        "--MODEL2URL", 
        type=dict, 
        default={},
        help="models to use in the pipeline"
    )

    parser.add_argument(
        "--swde_plus",
        type=bool,
        default=False,
        help="Whether to use the extended SWDE dataset to measure OpenIE performance",
    )

    parser.add_argument(
        "--schema_id_sizes",
        type=int,
        default=0,
        help="Number of documents to use for schema identification stage, if it differs from extraction",
    )

    parser.add_argument(
        "--slice_results",
        type=bool,
        default=False,
        help="Whether to measure the results by attribute-slice",
    )

    parser.add_argument(
        "--fn_generation_prompt_num",
        type=int,
        default=-1,
        help="For ablations on function generation with diversity, control which prompt we use. Default is all.",
    )

    parser.add_argument(
        "--upper_bound_fns",
        type=bool,
        default=False,
        help="For ablations that select functions using ground truth instead of the FM.",
    )

    # parser.add_argument(
    #     "--combiner_mode",
    #     type=str,
    #     default='mv',
    #     help="For ablations that select functions using ground truth instead of the FM.",
    # )

    parser.add_argument(
        "--use_alg_filtering",
        type=str,
        default=True,
        help="Whether to filter functions based on quality.",
    )

    parser.add_argument(
        "--use_abstension",
        type=str,
        default=True,
        help="Whether to use the abstensions approach.",
    )

    args = parser.parse_args(args=[])
    for arg, val in profiler_args.items():
        setattr(args, arg, val)
    return args


#################### GET SOME SAMPLE FILES TO SEED THE METADATA SEARCH #########################

def sample_scripts(files, train_size=5):
    # "Train" split
    random.seed(0)
    if train_size <= len(files):
        sample_files = random.sample(files, train_size)
    else:
        sample_files = files
    sample_contents = []
    for sample_file in sample_files:
        # if os.path.isdir(sample_file):
        #     continue
        with open(sample_file, 'r') as f:
            sample_contents.append(f.read())
    return sample_files
 

#################### BOILERPLATE CHUNKING CODE, CRITICAL FOR LONG SEUQENCES ####################
def chunk_file(
    parser, file, chunk_size=5000, mode="train", remove_tables=False, body_only=False
):
    content =  get_file_contents(file)
    if "html" in parser:
        content, chunks = get_html_parse(
            content, 
            chunk_size=chunk_size, 
            mode=mode, 
            remove_tables=remove_tables,
            body_only=body_only
        )
    else:
        content, chunks = get_txt_parse(content, chunk_size=chunk_size, mode=mode)
    return content, chunks


# HTML --> CHUNKS
def clean_html(content):
    for tag in ['script', 'style', 'svg']:
        content = content.split("\n")
        clean_content = []
        in_script = 0
        for c in content:
            if c.strip().strip("\t").startswith(f"<{tag}"):
                in_script = 1
            endstr = "</" + tag # + ">"
            if endstr in c or "/>" in c:
                in_script = 0
            if not in_script:
                clean_content.append(c)
        content = "\n".join(clean_content)
    return content


def get_flattened_items(content, chunk_size=500):
    flattened_divs = str(content).split("\n")
    flattened_divs = [ch for ch in flattened_divs if ch.strip() and ch.strip("\n").strip()]

    clean_flattened_divs = []
    for div in flattened_divs:
        if len(str(div)) > chunk_size:
            sub_divs = div.split("><")
            if len(sub_divs) == 1:
                clean_flattened_divs.append(div)
            else:
                clean_flattened_divs.append(sub_divs[0] + ">")
                for sd in sub_divs[1:-1]:
                    clean_flattened_divs.append("<" + sd + ">")
                clean_flattened_divs.append("<" + sub_divs[-1])
        else:
            clean_flattened_divs.append(div)
    return clean_flattened_divs
 

def get_html_parse(content, chunk_size=5000, mode="train", remove_tables=False, body_only=False):  
    if remove_tables:
        soup = BeautifulSoup(content)
        tables = soup.find_all("table")
        for table in tables:
            if "infobox" not in str(table):
                content = str(soup)
                content = content.replace(str(table), "")
                soup = BeautifulSoup(content)

    if body_only:
        soup = BeautifulSoup(content)
        content = str(soup.find("body"))
        soup = BeautifulSoup(content)

    else:
        content = clean_html(content)
        clean_flattened_divs = []
        flattened_divs = get_flattened_items(content, chunk_size=chunk_size)
        for i, div in enumerate(flattened_divs):
            new_div = re.sub(r'style="[^"]*"', '', str(div))
            new_div = re.sub(r'<style>.*?</style>', '', str(new_div))
            new_div = re.sub(r'<style.*?/style>', '', str(new_div))
            new_div = re.sub(r'<meta.*?/>', '', str(new_div))
            new_div = "\n".join([l for l in new_div.split("\n") if l.strip() and l.strip("\n").strip()])
            # new_div = BeautifulSoup(new_div) #.fsind_all("div")[0]
            if new_div:
                clean_flattened_divs.append(new_div)

        if mode == "eval":
            return []

    grouped_divs = []
    current_div = []
    current_length = 0
    max_length = chunk_size
    join_str = " " if use_raw_text else "\n"
    for div in clean_flattened_divs:
        str_div = str(div)
        len_div = len(str_div)
        if (current_length + len_div > max_length):
            grouped_divs.append(join_str.join(current_div))
            current_div = []
            current_length = 0
        elif not current_div and (current_length + len_div > max_length):
            grouped_divs.append(str_div)
            continue
        current_div.append(str_div)
        current_length += len_div

    return content, grouped_divs


# GENERIC TXT --> CHUNKS
def get_txt_parse(content, chunk_size=5000, mode="train"):
    # convert to chunks
    if mode == "train":
        chunks = content.split("\n")
        clean_chunks = []
        for chunk in chunks:
            if len(chunk) > chunk_size:
                sub_chunks = chunk.split(". ")
                clean_chunks.extend(sub_chunks)
            else:
                clean_chunks.append(chunk)

        chunks = clean_chunks.copy()
        clean_chunks = []
        for chunk in chunks:
            if len(chunk) > chunk_size:
                sub_chunks = chunk.split(", ")
                clean_chunks.extend(sub_chunks)
            else:
                clean_chunks.append(chunk)

        final_chunks = []
        cur_chunk = []
        cur_chunk_size = 0
        for chunk in clean_chunks:
            if cur_chunk_size + len(chunk) > chunk_size:
                final_chunks.append("\n".join(cur_chunk))
                cur_chunk = []
                cur_chunk_size = 0
            cur_chunk.append(chunk)
            cur_chunk_size += len(chunk)
        if cur_chunk:
            final_chunks.append("\n".join(cur_chunk))
    else:
        final_chunks = []
    return content, final_chunks


def get_file_contents(file):
    text = ''
    if file.endswith(".swp"):
        return text
    try:
        with open(file) as f:
            text = f.read()
    except:
        with open(file, "rb") as f:
            text = f.read().decode("utf-8", "ignore")
    return text


def clean_metadata(field):
    return field.replace("\t", " ").replace("\n", " ").strip().lower()


def filter_file2chunks(file2chunks, sample_files, attribute):
    """
  Filter chunks in file2chunks based on relevance to attribute/Keeps only chunks containing the attribute name as a keyword, removing those that do not.

  Args:
    file2chunks: Dict mapping files to lists of chunks.
    sample_files: List of sample file names. 
    attribute: Attribute name to filter for.

  Returns:
    filtered_file2chunks: Dict with filtered chunks for each file.

  Filters the chunk lists in file2chunks to only include chunks 
  relevant to the specified attribute. 

  - Removes files with no matching chunks
  - Keeps top 1-2 chunks per file based on keyword search

  This focuses the chunks on the attribute of interest to improve
  signal for synthesizing extraction functions.

    Key points:

    Purpose is to filter chunks by attribute
    Inputs are original chunks, sample files, attribute name
    Output is filtered chunk mapping
    Briefly explains filtering process

  Returns filtered mapping of files to filtered chunk lists.
    """
    def get_attribute_parts(attribute):
        for char in ["/", "-", "(", ")", "[", "]", "{", "}", ":"]:
            attribute = attribute.replace(char, " ")
        attribute_parts = attribute.lower().split()
        return attribute_parts

    # filter chunks with simple keyword search
    attribute_chunks = defaultdict(list)
    starting_num_chunks = 0
    ending_num_chunks = 0
    ending_in_sample_chunks = 0
    starting_in_sample_chunks = 0
    for file, chunks in file2chunks.items():
        starting_num_chunks += len(chunks)
        if file in sample_files:
            starting_in_sample_chunks += len(chunks)
        cleaned_chunks = []
        for chunk in chunks:
            # key line: check if attribute is in chunk and only append relevant chunk
            if attribute.lower() in chunk.lower():
                cleaned_chunks.append(chunk)
        if len(cleaned_chunks) == 0:
            for chunk in chunks:
                if attribute.lower().replace(" ", "") in chunk.lower().replace(" ", ""):
                    cleaned_chunks.append(chunk)
        if len(cleaned_chunks) == 0:
            chunk2num_word_match = Counter()
            for chunk_num, chunk in enumerate(chunks):
                attribute_parts = get_attribute_parts(attribute.lower())
                for wd in attribute_parts:
                    if wd.lower() in chunk.lower():
                        chunk2num_word_match[chunk_num] += 1
            # sort chunks by number of words that match
            sorted_chunks = sorted(chunk2num_word_match.items(), key=lambda x: x[1], reverse=True)
            if len(sorted_chunks) > 0:
                cleaned_chunks.append(chunks[sorted_chunks[0][0]])
            if len(sorted_chunks) > 1:
                cleaned_chunks.append(chunks[sorted_chunks[1][0]])
        ending_num_chunks += len(cleaned_chunks)
        num_chunks = len(cleaned_chunks)
        num_chunks = min(num_chunks, 2)

        # key line: only keep relevant clean chunks
        cleaned_chunks = cleaned_chunks[:num_chunks]
        attribute_chunks[file] = cleaned_chunks  # key line
        if file in sample_files:
            ending_in_sample_chunks += len(attribute_chunks[file])
    file2chunks = attribute_chunks
    if ending_num_chunks == 0 or ending_in_sample_chunks == 0:
        print(f"Removing because no chunks for attribute {attribute} in any file")
        return None
    print(f"For attribute {attribute}\n-- Starting with {starting_num_chunks} chunks\n-- Ending with {ending_num_chunks} chunks")
    print(f"-- {starting_in_sample_chunks} starting chunks in sample files\n-- {ending_in_sample_chunks} chunks in sample files")

    return file2chunks


def clean_function_predictions(extraction, attribute=None):
        if extraction is None:
            return ''
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
            elif extraction and len(extraction) == 1 and extraction[0] is None:
                extraction = ''
            else:
                dedup_list = []
                for a in extraction:
                    if a not in dedup_list:
                        dedup_list.append(a)
                extraction = dedup_list
                extraction = [(str(e)).strip().strip("\n") for e in extraction]
                extraction = ", ".join(extraction)
        if type(extraction) == "str" and extraction.lower() == "none":
            extraction = ""
        extraction = extraction.strip().replace("  ", " ")
        if extraction.lower().startswith(attribute.lower()):
            idx = extraction.lower().find(attribute.lower())
            extraction = extraction[idx+len(attribute):].strip()
        for char in [':', ","]:
            extraction = extraction.strip(char).strip()
        extraction  = extraction.replace(",", ", ").replace("  ", " ")
        return extraction


def check_vs_train_extractions(train_extractions, final_extractions, gold_key):
        clean_final_extractions = {}

        gold_values = train_extractions[gold_key]
        modes = []
        start_toks = []
        end_toks = []
        for file, gold in gold_values.items():
            if type(gold) == list:
                if gold and type(gold[0]) == list:
                    gold = [g[0] for g in gold]
                    gold = ", ".join(gold)
                else:
                    gold = ", ".join(gold)
            gold = gold.lower()
            pred = final_extractions[file].lower()
            if not pred or not gold:
                continue
            if ("<" in pred and "<" not in gold) or (">" in pred and ">" not in gold):
                check_pred =  BeautifulSoup(pred).text
                if check_pred in gold or gold in check_pred:
                    modes.append("soup")
            elif gold in pred and len(pred) > len(gold):
                modes.append("longer")
                idx = pred.index(gold)
                if idx > 0:
                    start_toks.append(pred[:idx-1])
                end_idx = idx + len(gold)
                if end_idx < len(pred):
                    end_toks.append(pred[end_idx:])

        def long_substr(data):
            substr = ''
            if len(data) > 1 and len(data[0]) > 0:
                for i in range(len(data[0])):
                    for j in range(len(data[0])-i+1):
                        if j > len(substr) and is_substr(data[0][i:i+j], data):
                            substr = data[0][i:i+j]
            return substr

        def is_substr(find, data):
            if len(data) < 1 and len(find) < 1:
                return False
            for i in range(len(data)):
                if find not in data[i]:
                    return False
            return True
        
        longest_end_tok = long_substr(end_toks)
        longest_start_tok = long_substr(start_toks)
        if len(set(modes)) == 1:
            num_golds = len(gold_values)
            for file, extraction in final_extractions.items():
                if "longer" in modes:
                    # gold longer than pred
                    if len(end_toks) == num_golds and longest_end_tok in extraction and extraction.count(longest_end_tok) == 1:
                        idx = extraction.index(longest_end_tok)
                        extraction = extraction[:idx] 
                    if len(start_toks) == num_golds and longest_start_tok in extraction and extraction.count(longest_start_tok) == 1:
                        idx = extraction.index(longest_start_tok)
                        extraction = extraction[idx:] 
                elif "soup" in modes:
                    extraction = BeautifulSoup(extraction).text
                clean_final_extractions[file] = extraction
        else:
            clean_final_extractions = final_extractions
        return clean_final_extractions
