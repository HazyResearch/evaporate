import os
import json
from collections import Counter, defaultdict

from manifest import Manifest
from evaporate.configs import get_args
from evaporate.prompts import Step
from openai import OpenAI

cur_idx = 0
TOGETHER_API_KEY = os.environ.get("TOGETHER_API_KEY")
#If using together AI, you will need to set the TOGETHER_API_KEY to your API key


def together_call(prompt, model, streaming = False, max_tokens = 1024):
    client = OpenAI(
        api_key=TOGETHER_API_KEY,
        base_url='https://api.together.xyz',

    )
    messages = [{
        "role": "system",
        "content": "You are an AI assistant",
    }, {
        "role": "user",
        "content": prompt,
    }]
    chat_completion = client.chat.completions.create(messages=messages,
                                                    model=model,
                                                    max_tokens=max_tokens,
                                                    #response_format={ "type": "json_object" },
                                                    stream=streaming)
    response = chat_completion.choices[0].message.content
    return response

def apply_prompt(step : Step, max_toks = 50, do_print=False, manifest=None, overwrite_cache=False):
    global cur_idx 
    manifest_lst = manifest.copy()
    if len(manifest) == 1:
        manifest = manifest_lst[0]
    else:
        manifest = manifest_lst[cur_idx]

    # sometimes we want to rotate keys
    cur_idx = cur_idx + 1
    if cur_idx >= len(manifest_lst)-1:
        cur_idx = 0

    prompt = step.prompt
    response, num_tokens = get_response(
        prompt, 
        manifest, 
        max_toks = max_toks, 
        overwrite=overwrite_cache,
        stop_token="---"
    )
    step.response = response
    if do_print:
        print(response)
    return response, num_tokens


def get_file_attribute(attribute):
    attribute = attribute.lower()
    attribute = attribute.replace("/", "_").replace(")", "").replace("-", "_")
    attribute = attribute.replace("(", "").replace(" ", "_")
    if len(attribute) > 30:
        attribute = attribute[:30]
    return attribute


def get_all_files(data_dir):
    files = []
    for file in os.listdir(data_dir):
        if os.path.isfile(os.path.join(data_dir, file)):
            files.append(os.path.join(data_dir, file))
        else:
            files.extend(get_all_files(os.path.join(data_dir, file)))
    return files


def get_directory_hierarchy(data_dir):
    if not data_dir.endswith("/") and os.path.isdir(data_dir):
        data_dir = data_dir + "/"
    directories2subdirs = defaultdict(list)
    for file in os.listdir(data_dir):
        new_dir = os.path.join(data_dir, file)
        if not new_dir.endswith("/") and os.path.isdir(new_dir):
            new_dir = new_dir + "/"
        if os.path.isdir(new_dir):
            directories2subdirs[data_dir].append(new_dir)
            if os.listdir(new_dir):
                more_subdirs = get_directory_hierarchy(new_dir)
                for k, v in more_subdirs.items():
                    directories2subdirs[k].extend(v)
            else:
                directories2subdirs[new_dir] = []
        else:
            directories2subdirs[data_dir].append(new_dir)
    return directories2subdirs


def get_unique_file_types(files):
    suffix2file = {}
    suffix2count = Counter()
    for file in files:
        suffix = file.split(".")[-1]
        if not suffix:
            suffix = "txt"
        suffix2count[suffix] += 1
        if suffix not in suffix2file:
            suffix2file[suffix] = file
    return suffix2file, suffix2count


def get_structure(dataset_name, profiler_args):
    args = get_args(profiler_args)
    if not os.path.exists(args.cache_dir):
        os.makedirs(args.cache_dir)
        
    if not os.path.exists(args.generative_index_path):
        os.makedirs(args.generative_index_path)

    if not os.path.exists(args.generative_index_path):
        os.makedirs(args.generative_index_path)
    
    # all files
    cache_path = f"{args.cache_dir}/all_files.json"
    if not os.path.exists(cache_path) or args.overwrite_cache:
        files = get_all_files(args.data_dir)
        with open(cache_path, "w") as f:
            json.dump(files, f)
    else:
        with open(cache_path) as f:
            files = json.load(f)

    # all directories
    cache_path = f"{args.cache_dir}/all_dirs.json"
    if not os.path.exists(cache_path) or args.overwrite_cache:
        directory_hierarchy = get_directory_hierarchy(args.data_dir)
        with open(cache_path, "w") as f:
            json.dump(directory_hierarchy, f)
    else:
        with open(cache_path) as f:
            directory_hierarchy = json.load(f)
    
    suffix2file, suffix2count = get_unique_file_types(files)
    file_examples = "\n".join(list(suffix2file.values()))
    file_types = ", ".join((suffix2file.keys()))
    return directory_hierarchy, files, file_examples, file_types, args


def get_files_in_group(dir_path):
    file_group = []
    for i, (root,dirs,files) in enumerate(os.walk(dir_path, topdown=True)):
        files = [f"{root}/{f}" for f in files] 
        file_group.extend(files)
    print(f"Working with a sample size of : {len(file_group)} files.")
    return file_group


# MANIFEST
def get_manifest_sessions(MODELS, MODEL2URL=None, KEYS=[]):
    manifest_sessions = defaultdict(list)
    for model in MODELS: 
        if any(kwd in model for kwd in ["davinci", "curie", "babbage", "ada", "cushman"]):
            if not KEYS:
                raise ValueError("You must provide a list of keys to use these models.")
            for key in KEYS:
                manifest, model_name = get_manifest_session(
                    client_name="openai", 
                    client_engine=model, 
                    client_connection=key,
                )
                manifest_sessions[model].append(manifest)
        elif any(kwd in model for kwd in ["gpt-4", "gpt-3.5"]):
            if not KEYS:
                raise ValueError("You must provide a list of keys to use these models.")
            for key in KEYS:
                manifest, model_name = get_manifest_session(
                    client_name="openaichat", 
                    client_engine=model, 
                    client_connection=key,
                )
                manifest_sessions[model].append(manifest)
        else:
            if(model not in MODEL2URL):
                manifest = {}
                manifest["__name"] = model
                print("using together AI")
            else:
                print("using huggingface")
                manifest, model_name = get_manifest_session(
                    client_name="huggingface", 
                    client_engine=model, 
                    client_connection=MODEL2URL[model],
                )
            manifest_sessions[model].append(manifest)
    return manifest_sessions


def get_manifest_session(
    client_name="huggingface",
    client_engine=None,
    client_connection="http://127.0.0.1:5000",
    cache_connection=None,
    temperature=0,
    top_p=1.0,
):
    if client_name == "huggingface" and temperature == 0:
        params = {
            "temperature": 0.001,
            "do_sample": False,
            "top_p": top_p,
        }
    elif client_name in {"openai", "ai21", "openaichat"}:
        params = {
            "temperature": temperature,
            "top_p": top_p,
            "engine": client_engine,
        }
    else:
        raise ValueError(f"{client_name} is not a valid client name")
    
    cache_params = {
        "cache_name": "sqlite",
        "cache_connection": cache_connection,
    }

    manifest = Manifest(
        client_name=client_name,
        client_connection=client_connection,
        **params,
        **cache_params,
    )
    
    params = manifest.client_pool.get_current_client().get_model_params()
    model_name = params["model_name"]
    if "engine" in params:
        model_name += f"_{params['engine']}"
    return manifest, model_name


def get_response(
    prompt,
    manifest,
    overwrite=False,
    max_toks=10,
    stop_token=None,
    gold_choices=[],
    verbose=False,
):
    prompt = prompt.strip()
    if gold_choices:
        gold_choices = [" " + g.strip() for g in gold_choices]
        if type(manifest) == dict and manifest["__name"] != "openai":
            response = together_call(prompt, manifest["__name"])
            num_tokens = 0
        else:
            response_obj = manifest.run(
                prompt, 
                gold_choices=gold_choices, 
                overwrite_cache=overwrite, 
                return_response=True,
            )
            response_obj = response_obj.get_json_response()["choices"][0]
            log_prob = response_obj["text_logprob"]
            response = response_obj["text"]
            num_tokens = response_obj['usage']['total_tokens']
    else:
        if type(manifest) == dict and manifest["__name"] != "openai":
            response = together_call(prompt, manifest["__name"])
            num_tokens = 0
        else:
            response_obj = manifest.run(
                prompt,
                max_tokens=max_toks,
                stop_token=stop_token,
                overwrite_cache=overwrite,
                return_response=True
            )
            num_tokens = -1
            try:
                num_tokens = response_obj.get_usage_obj().usages[0].total_tokens
            except:
                num_tokens = 0
                print("Fail to get total tokens used")
            response_obj = response_obj.get_json_response()
            response = response_obj["choices"][0]["text"]
        stop_token = "---"
        response = response.strip().split(stop_token)[0].strip() if stop_token else response.strip()
        log_prob = None
    if verbose:
        print("\n***Prompt***\n", prompt)
        print("\n***Response***\n", response)
    if log_prob:
        return response, log_prob
    return response, num_tokens

