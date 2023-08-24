import argparse
import os
from pathlib import Path

# original
# BASE_DATA_DIR = "/data/evaporate/"
# Path("~/data/evaporate/").expanduser()
# BASE_DATA_DIR = Path("~/data/evaporate/data/").expanduser()
# HOME = Path('~/').expanduser()

# called in utils.py by get_structure which is called by run_profiler.py
def get_args(database_name):
    
    parser = argparse.ArgumentParser(
        "LLM explorer.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--overwrite_cache",
        type=bool,
        default=0,
        help="Whether to overwrite the caching for prompts."
    )

    parser.add_argument(
        "--data_lake",
        type=str,
        default="fda_510ks",
        help="Name of the data lake"
    )

    parser.add_argument(
        "--data_dir",
        type=str,
        help="Path to raw data-lake documents",
    )

    parser.add_argument(
        "--generative_index_path",
        type=str,
        help="Path to store the generated structured view of the data lake",
    )

    parser.add_argument(
        "--cache_dir",
        type=str,
        default=".cache/",
        help="Path to cache intermediate files during system execution",
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

    # CONSTANTS = {
    #     "fda_510ks": {
    #         "data_dir": str(BASE_DATA_DIR / "fda_510ks/data/evaporate/fda-ai-pmas/510k"),  # raw data
    #         "database_name": "fda_510ks",
    #         "cache_dir": ".cache/fda_510ks/",
    #         "generative_index_path": str(BASE_DATA_DIR / "generative_indexes/fda_510ks/"),  # 
    #         "gold_extractions_file": str(BASE_DATA_DIR / "ground_truth/fda_510ks_gold_extractions.json"),  # todo
    #         "topic": "fda 510k device premarket notifications",
    #     },
    # }
    BASE_PATH = Path("~/data/maf_data").expanduser()
    HOME = Path('~/').expanduser()
    CONSTANTS = {
        "fda_510ks": {
            "data_dir": str(BASE_PATH / "fda_510ks/data/evaporate/fda-ai-pmas/510k"),  # raw data
            "database_name": "fda_510ks",
            "cache_dir": ".cache/fda_510ks/",
            "generative_index_path": str(BASE_PATH / "generative_indexes/fda_510ks/"),  # 
            "gold_extractions_file": str(BASE_PATH / "ground_truth/fda_510ks_gold_extractions.json"),  # todo
            "topic": "fda 510k device premarket notifications",
        }
    }
    # - get from yaml
    data_lakes_config_path = '~/massive-autoformalization-maf/configs/data_lakes/data_lakes.yaml'
    data_lakes_config_path = Path(data_lakes_config_path).expanduser()
    import yaml
    with open(data_lakes_config_path, 'r') as f:
        data_lakes_config = yaml.load(f, Loader=yaml.FullLoader)
        data_lakes_config = dict(data_lakes_config)  # {data_lake_name: config_dict}
    for data_lake_name, data_lake_config_dict in data_lakes_config.items():
        base_name = data_lake_config_dict['base_name']
        base_path = os.path.expanduser(data_lake_config_dict['base_path'])
        database_name = os.path.expanduser(data_lake_config_dict['database_name'])
        assert database_name == data_lake_name, f'Err: {database_name} != {data_lake_name}'
        # fill the strings with the base path and dattabase_name
        for config_key, config_value in data_lake_config_dict.items():
            # filled_string = extractions_base_path.format(base_path=base_path, database_name=database_name)
            config_value = config_value.format(base_path=base_path, database_name=database_name, base_name=base_name)
            data_lake_config_dict[config_key] = config_value
        data_lakes_config[data_lake_name] = data_lake_config_dict
        CONSTANTS[data_lake_name] = data_lake_config_dict
        # CONSTANTS[data_lake_name]['data_dir'] = data_lake_config_dict[data_lake_name]['data_dir']
        # CONSTANTS[data_lake_name]['cache_dir'] = data_lake_config_dict[data_lake_name]['cache_dir']
        # CONSTANTS[data_lake_name]['generative_index_path'] = data_lake_config_dict[data_lake_name]['generative_index_path']
        # CONSTANTS[data_lake_name]['topic'] = data_lake_config_dict[data_lake_name]['topic']
        # # note might be gold attributes as a hack
        # CONSTANTS[data_lake_name]['gold_extractions_file'] = data_lake_config_dict[data_lake_name]['gold_extractions_file']

    # - load the data lake config
    args = parser.parse_args(args=[])

    args_fill = CONSTANTS[database_name]
    print(f'{args_fill=}')
    args.data_dir = args_fill["data_dir"]
    args.cache_dir = args_fill["cache_dir"]
    args.generative_index_path = args_fill["generative_index_path"]
    args.topic = args_fill['topic']
    args.gold_extractions_file = args_fill['gold_extractions_file']
    args.gold_attributes_file = args_fill['gold_attributes_file']
    args.data_lake = database_name
    print(f'{args.data_lake=}')
    if 'set_dicts' in args_fill:
        args.set_dicts = args_fill['set_dicts']

    # -- load my data lakes
    return args
