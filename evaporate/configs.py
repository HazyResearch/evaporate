import argparse
import os
import datetime

def get_run_string(
    data_lake, today, file_groups, profiler_args, do_end_to_end, 
    train_size, dynamicbackoff, models
):
    body = profiler_args.body_only # Baseline systems only operate on the HTML body
    model_ct = len(models)
    if profiler_args.use_qa_model:
        model_ct += 1
    run_string = f"dataLake{data_lake}_date{today}_fileSize{len(file_groups)}_trainSize{train_size}_numAggregate{profiler_args.num_top_k_scripts}_chunkSize{profiler_args.chunk_size}_removeTables{profiler_args.remove_tables}_body{body}_cascading{do_end_to_end}_useBackoff{dynamicbackoff}_MODELS{model_ct}"
    return run_string

def get_data_lake_info(args):
    extractions_file = None
    
    if 1:
        DATA_DIR = args.data_dir
        file_groups = os.listdir(args.data_dir)
        if not DATA_DIR.endswith("/"):
            DATA_DIR += "/"
        file_groups = [f"{DATA_DIR}{file_group}" for file_group in file_groups if not file_group.startswith(".")]
        full_file_groups = file_groups.copy()
        extractions_file = args.gold_extractions_file
        parser = "txt"

    return file_groups, extractions_file, parser, full_file_groups

#default args for the experiment
def get_experiment_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_lake", 
        type=str,
        help="Name of the data lake to operate over. Must be in configs.py"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        help="Path to raw data-lake documents",
    )
    parser.add_argument(
        "--base_data_dir",
        type=str,
        default="/",
        help="Path to save intermediate result",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default= "",
        help = "Path to cache intermediate files during system execution",
    )
    parser.add_argument(
        "--generative_index_path",
        type=str,
        default= "",
        help = "Path to store the generated structured view of the data lake",
    )
    parser.add_argument(
        "--gold_extractions_file",
        type=str,
        default= "",
        help = "Path to store the generated structured view of the data lake",
    )
    parser.add_argument(
        "--do_end_to_end", 
        action='store_true',
        default=False,
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
        action="store_true",
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
    parser.add_argument(
        "--MODELS",
        type=str,
        default=["gpt-4"],
        help="List of models to use for the extraction step"
    )
    parser.add_argument(
        "--EXTRACTION_MODELS",
        type=str,
        default=["gpt-4"],
        help="List of models to use for the extraction step"
    )
    parser.add_argument(
        "--MODEL2URL",
        type=str,
        default={},
        help="Dict mapping models to their urls"
    )
    parser.add_argument(
        "--use_qa_model",
        action="store_true",
        default=False,
        help="Whether to use a QA model for the extraction step"
    )
    parser.add_argument(
        "--GOLD_KEY",
        type=str,
        default="gpt-4",
        help="Key to use for the gold standard"
    )

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

    parser.add_argument(
        "--chunk_size",
        type=int,
        default=3000,
    )

    parser.add_argument(
        "--extraction_fraction_thresh",
        type=int,
        default=0.9,
        help="for abstensions approach",
    )

    parser.add_argument(
        "--remove_tables",
        action="store_true",
        default=False,
        help="Remove tables from the html files?",
    )

    parser.add_argument(
        "--body_only",
        action="store_true",
        default=False,
        help="Only use HTML body",
    )

    parser.add_argument(
        "--max_metadata_fields",
        type=int,
        default=15,
    )

    parser.add_argument(
        "--overwrite_cache", 
        action='store_true',
        default=False,
        help="overwrite the manifest cache"
    )
    parser.add_argument(
        "--swde_plus",
        action="store_true",
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
        action="store_true",
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
        action="store_true",
        default=False,
        help="For ablations that select functions using ground truth instead of the FM.",
    )

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

    parser.add_argument(
        "--set_dicts",
        type=str,
        default='',
        help="Alternate valid names for the SWDE attributes as provided in the benchmark.",
    )

    parser.add_argument(
        "--topic",
        type=list,
        default=[],
        help="Topic of the data lake",
    )
    experiment = parser.parse_args()
    return experiment

#get args related to data storage paths
def get_args(profiler_args):
    if(profiler_args.generative_index_path == ""):
        profiler_args.generative_index_path = os.path.join(profiler_args.base_data_dir, "generative_indexes/", profiler_args.data_lake)
    if(profiler_args.cache_dir == ""):
        profiler_args.cache_dir = os.path.join(profiler_args.base_data_dir, "cache/", profiler_args.data_lake)
    if(profiler_args.gold_extractions_file == ""):
        profiler_args.gold_extractions_file = os.path.join(profiler_args.base_data_dir, "gold_extractions.json" )
    parser = argparse.ArgumentParser(
        "LLM explorer.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--overwrite_cache",
        type=bool,
        default=profiler_args.overwrite_cache,
        help="Whether to overwrite the caching for prompts."
    )

    parser.add_argument(
        "--data_lake",
        type=str,
        default=profiler_args.data_lake,
        help="Name of the data lake"
    )

    parser.add_argument(
        "--data_dir",
        type=str,
        default = profiler_args.data_dir,
        help="Path to raw data-lake documents",
    )

    parser.add_argument(
        "--generative_index_path",
        type=str,
        default = profiler_args.generative_index_path,
        help="Path to store the generated structured view of the data lake",
    )

    parser.add_argument(
        "--cache_dir",
        type=str,
        default=profiler_args.cache_dir,
        help="Path to cache intermediate files during system execution",
    )

    parser.add_argument(
        "--set_dicts",
        type=str,
        default=profiler_args.set_dicts,
        help="Alternate valid names for the SWDE attributes as provided in the benchmark.",
    )

    parser.add_argument(
        "--gold_extractions_file",
        type=str,
        default=profiler_args.gold_extractions_file,
        help="Path to store the generated structured view of the data lake",
    )
    parser.add_argument(
        "--topic",
        type=list,
        default=profiler_args.topic,
        help="Topic of the data lake",
    )

    args = parser.parse_args(args=[])
    return args

#for notebook settings
def set_profiler_args(information):
    parser = argparse.ArgumentParser(
        "LLM explorer.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data_lake", 
        type=str,
        help="Name of the data lake to operate over. Must be in configs.py"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        help="Path to raw data-lake documents",
    )
    parser.add_argument(
        "--base_data_dir",
        type=str,
        default="/",
        help="Path to save intermediate result",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default= "",
        help = "Path to cache intermediate files during system execution",
    )
    parser.add_argument(
        "--generative_index_path",
        type=str,
        default= "",
        help = "Path to store the generated structured view of the data lake",
    )
    parser.add_argument(
        "--gold_extractions_file",
        type=str,
        default= "",
        help = "Path to store the generated structured view of the data lake",
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
        "--combiner_mode",
        type=str,
        default='ws',
        help="Combiner mode for combining the outputs of the generated functions",
        choices=['ws', 'mv', 'top_k']
    )

    parser.add_argument(
        "--use_dynamic_backoff",
        action="store_true",
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
    parser.add_argument(
        "--MODELS",
        type=str,
        default=["gpt-4"],
        help="List of models to use for the extraction step"
    )
    parser.add_argument(
        "--EXTRACTION_MODELS",
        type=str,
        default=["gpt-4"],
        help="List of models to use for the extraction step"
    )
    parser.add_argument(
        "--MODEL2URL",
        type=str,
        default={},
        help="Dict mapping models to their urls"
    )
    parser.add_argument(
        "--use_qa_model",
        action="store_true",
        default=False,
        help="Whether to use a QA model for the extraction step"
    )
    parser.add_argument(
        "--GOLD_KEY",
        type=str,
        default="gpt-4",
        help="Key to use for the gold standard"
    )

    parser.add_argument(
        "--eval_size",
        type=int,
        default=15,
    )

    parser.add_argument(
        "--chunk_size",
        type=int,
        default=3000,
    )

    parser.add_argument(
        "--max_chunks_per_file",
        type=int,
        default=-1,
    )


    parser.add_argument(
        "--extraction_fraction_thresh",
        type=int,
        default=0.9,
        help="for abstensions approach",
    )

    parser.add_argument(
        "--remove_tables",
        action="store_true",
        default=False,
        help="Remove tables from the html files?",
    )

    parser.add_argument(
        "--body_only",
        action="store_true",
        default=False,
        help="Only use HTML body",
    )

    parser.add_argument(
        "--max_metadata_fields",
        type=int,
        default=15,
    )

    parser.add_argument(
        "--overwrite_cache", 
        action='store_true',
        default=False,
        help="overwrite the manifest cache"
    )
    parser.add_argument(
        "--swde_plus",
        action="store_true",
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
        action="store_true",
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
        action="store_true",
        default=False,
        help="For ablations that select functions using ground truth instead of the FM.",
    )

    parser.add_argument(
        "--use_alg_filtering",
        action='store_true',
        default=False,
        help="Whether to filter functions based on quality.",
    )

    parser.add_argument(
        "--use_abstension",
        action='store_true',
        default=False,
        help="Whether to use the abstensions approach.",
    )

    parser.add_argument(
        "--do_end_to_end", 
        action='store_true',
        default=False,
        help="True for OpenIE, False for ClosedIE"
    )
    
    parser.add_argument(
        "--set_dicts",
        type=str,
        default='',
        help="Alternate valid names for the SWDE attributes as provided in the benchmark.",
    )

    parser.add_argument(
        "--topic",
        type=list,
        default=[],
        help="Topic of the data lake",
    )

    args = parser.parse_args(args=[])
    for key, value in information.items():
        setattr(args, key, value)
    today = datetime.datetime.today().strftime("%m%d%Y")
    file_groups, extractions_file, parser, full_file_groups = get_data_lake_info(args)
    setattr(args, "file_groups", file_groups)
    setattr(args, "extractions_file", extractions_file)
    setattr(args, "parser", parser)
    setattr(args, "full_file_groups", full_file_groups)
    if("train_size" not in args):
        args.train_size = 10
    if "use_dynamic_backoff" not in args:
        args.use_dynamic_backoff = True
    setattr(args, "run_string", get_run_string(args.data_lake, today, args.full_file_groups, args, args.do_end_to_end, args.train_size, args.use_dynamic_backoff, args.EXTRACTION_MODELS))
    if(args.generative_index_path == ""):
        args.generative_index_path = os.path.join(args.base_data_dir, "generative_indexes/", args.data_lake)
    if(args.cache_dir == ""):
        args.cache_dir = os.path.join(args.base_data_dir, "cache/", args.data_lake)
    if(args.gold_extractions_file == ""):
        args.gold_extractions_file = os.path.join(args.base_data_dir, "gold_extractions.json" )
    return args


