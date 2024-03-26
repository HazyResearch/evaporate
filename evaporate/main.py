import time
from tqdm import tqdm
from evaporate.run_profiler import prerun_profiler, identify_attributes, get_attribute_function
from evaporate.profiler import get_model_extractions
from evaporate.configs import  set_profiler_args
from evaporate.evaluate_synthetic_utils import text_f1, get_file_attribute
from evaporate.evaluate_profiler import pick_a_gold_label, evaluate, get_topk_scripts_per_field
from evaporate.retrieval import get_most_similarity
from evaporate.profiler import get_functions, apply_final_profiling_functions,apply_final_ensemble,combine_extractions
import os
import json
from collections import defaultdict, Counter
import numpy as np
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class EvaporateData:
    def __init__(self, profiler_args):
        self.GOLD_MODEL = profiler_args["direct_extract_model"]
        profiler_args["GOLD_KEY"] = "gold_extraction_file"
        self.GOLD_KEY = "gold_extraction_file"

        self.profiler_args= set_profiler_args(profiler_args)
        self.data_dict = prerun_profiler(self.profiler_args)
        self.runtime = {}
        self.token_used = {}
        self.accuracy = {}
        self.attributes = []
        self.direct_result = {}
        self.function_dictionary = {}
        self.all_extractions = {}
        self.manifest_sessions = self.data_dict["manifest_sessions"]
        self.selected_func_key = {}
        self.extract_result = {}
        self.all_metrics = None
        self.gold_extractions = self.data_dict["gold_extractions"]

    def save_results():
        pass

    def get_attribute(self, do_end_to_end = False):
        if  do_end_to_end or self.profiler_args.do_end_to_end:
            self.attributes, total_time, num_toks, evaluation_result = identify_attributes(self.profiler_args, self.data_dict, evaluation=True)
            self.runtime["get_attribute"] = total_time
            self.token_used["get_attribute"] = num_toks
            self.accuracy["get_attribute"] = evaluation_result
        else:
            self.attributes = self.data_dict["gold_attributes"]
        return self.attributes


    def direct_extract(self, use_retrieval_model = True, is_getting_sample = False, gold = ""):
        if(self.attributes == []):
            print("Please run get_attribute first")
            return
        files = list(self.data_dict["file2chunks"].keys())
        if(is_getting_sample):
            files = self.data_dict["sample_files"]
        time_begin = time.time()
        token_used = 0
        for attribute in self.attributes:
            if(attribute in self.direct_result):
                print("already extract ", attribute)
                #continue
            new_file_chunk_dict = {}
            if(use_retrieval_model):
                baseline_sentence = attribute + ":"+  gold[attribute]
                for file in files:
                    sentences = self.data_dict["file2chunks"][file]
                    new_file_chunk_dict[file] = [get_most_similarity(baseline_sentence, sentences)]
            else:
                new_file_chunk_dict =  self.data_dict["file2chunks"]
            extractions, num_toks, errored_out = get_model_extractions(
                new_file_chunk_dict, 
                files,
                attribute,  
                self.manifest_sessions[self.GOLD_MODEL],
                self.GOLD_MODEL,
                overwrite_cache=self.profiler_args.overwrite_cache,
                collecting_preds=True,
            )
            token_used += num_toks
            self.direct_result[attribute] = {}
            for file in extractions:
                golds = []
                for tmp in extractions[file]:
                    golds.append( "- " + "\n- ".join(tmp))
                golds = "- " + "\n- ".join(golds)
                if(use_retrieval_model):
                    try:
                        self.direct_result[attribute][file] = extractions[file][0]
                    except:
                        print("error in ", attribute, file, extractions[file])
                else:
                    self.direct_result[attribute][file] = pick_a_gold_label(golds, attribute, self.manifest_session)
            print("finish extract ", attribute)
        self.runtime["direct_extract"] = time.time() - time_begin
        self.token_used["direct_extract"] = token_used
        return self.direct_result, self.evaluate(self.direct_result)
    
    def get_extract_functions(self):
        self.runtime["get_extract_functions"] = 0
        self.token_used["get_extract_functions"] = 0
        begin_time = time.time()
        total_tokens_prompted = 0
        for attribute in self.attributes:
            if attribute in self.function_dictionary:
                print("already generate ", attribute, " function")
                continue
            self.all_extractions[attribute] = {}
            self.function_dictionary[attribute] = {}
            for model in self.profiler_args.EXTRACTION_MODELS:
                manifest_session = self.manifest_sessions[model]
                functions, function_promptsource, num_toks = get_functions(
                    self.data_dict["file2chunks"], 
                    self.data_dict["sample_files"], 
                    {},
                    attribute, 
                    manifest_session,
                    overwrite_cache=self.profiler_args.overwrite_cache,
                )
                total_tokens_prompted += num_toks
                for fn_key, fn in functions.items():
                    self.all_extractions[attribute][fn_key], num_function_errors = apply_final_profiling_functions(
                        self.data_dict["file2contents"], 
                        self.data_dict["sample_files"],
                        fn,
                        attribute,
                    )
                    self.function_dictionary[attribute][fn_key] = {}
                    self.function_dictionary[attribute][fn_key]['function'] = fn
                    self.function_dictionary[attribute][fn_key]['promptsource'] = function_promptsource[fn_key]
                    self.function_dictionary[attribute][fn_key]['extract_model'] = model
        self.runtime["get_extract_functions"] = time.time() - begin_time
        self.token_used["get_extract_functions"] = total_tokens_prompted
        return self.function_dictionary
    
    def weak_supervision(self, use_gold_key = False):
        result = self.direct_extract
        self.runtime["weak_supervision"] = 0
        self.token_used["weak_supervision"] = 0
        begin_time = time.time()
        total_tokens_prompted = 0
        for attribute in self.attributes:
            if attribute in self.selected_func_key:
                print("already weak supervision ", attribute)
                continue
            self.all_extractions[attribute]["gold_extraction_file"] = self.gold_extractions
            if(use_gold_key):
                self.GOLD_KEY = "gold_extraction_file"
            else:
                if(result == {}):
                    print("Please run direct_extract first")
                    return
                self.all_extractions[attribute]["gold-key"] = result[attribute]
                self.GOLD_KEY = "gold-key"
            self.all_metrics, key2golds, num_toks = evaluate(
                    self.all_extractions[attribute],
                    self.GOLD_KEY,
                    field=attribute,
                    manifest_session=self.manifest_sessions[self.profiler_args.GOLD_KEY], 
                    overwrite_cache=self.profiler_args.overwrite_cache,
                    combiner_mode=self.profiler_args.combiner_mode,
                    extraction_fraction_thresh=self.profiler_args.extraction_fraction_thresh,
                    use_abstension=self.profiler_args.use_abstension,
                )
            total_tokens_prompted += num_toks
            selected_keys = get_topk_scripts_per_field(
                self.all_metrics,
                self.function_dictionary[attribute],
                self.all_extractions,
                self.GOLD_KEY,
                k=self.profiler_args.num_top_k_scripts,
                do_end_to_end=self.profiler_args.do_end_to_end,
                combiner_mode=self.profiler_args.combiner_mode,
                keep_thresh = 0.00
            )
            self.selected_func_key[attribute] = selected_keys
        self.runtime["weak_supervision"] = time.time() - begin_time
        self.token_used["weak_supervision"] = total_tokens_prompted
        return self.selected_func_key
    
    def apply_functions(self):
        total_tokens_prompted = 0
        self.runtime["apply_functions"] = 0
        self.token_used["apply_functions"] = 0
        self.extract_result = {}
        begin_time = time.time()
        for attribute in self.attributes:
            print(f"Apply the scripts to the data lake and save the metadata. Taking the top {self.profiler_args.num_top_k_scripts} scripts per field.")
            top_k_extractions, num_toks = apply_final_ensemble(
                self.profiler_args.full_file_groups,
                self.data_dict["file2chunks"],
                self.data_dict['file2contents'],
                self.selected_func_key[attribute],
                self.all_metrics,
                attribute,
                self.function_dictionary[attribute],
                data_lake=self.profiler_args.data_lake,
                manifest_sessions=self.manifest_sessions,
                function_cache=True,
                MODELS=self.profiler_args.EXTRACTION_MODELS,
                overwrite_cache=self.profiler_args.overwrite_cache,
                do_end_to_end=self.profiler_args.do_end_to_end,
            )
            total_tokens_prompted += num_toks

            file2metadata, num_toks = combine_extractions(
                self.profiler_args,
                top_k_extractions, 
                self.all_metrics, 
                combiner_mode=self.profiler_args.combiner_mode,
                train_extractions=self.all_extractions[attribute],
                attribute=attribute, 
                gold_key = self.GOLD_KEY,
                extraction_fraction_thresh=self.profiler_args.extraction_fraction_thresh,
            )
            total_tokens_prompted += num_toks
            self.extract_result[attribute] = file2metadata
        self.runtime["apply_functions"] = time.time() - begin_time
        self.token_used["apply_functions"] = total_tokens_prompted
        return self.extract_result, self.evaluate(self.extract_result)
    
    def evaluate(self, result):
        f1_pred = {}
        for attribute in self.attributes:
            attribute2 = get_file_attribute(attribute)
            preds = []
            golds = []
            for file in result[attribute]:
                if(file in self.gold_extractions.keys()):
                    preds.append(result[attribute][file])
                    golds.append(self.gold_extractions[file][attribute])
            try:
                f1_pred[attribute] = text_f1(preds, golds)
            except:
                print(attribute)
        #turn to f1_pred[extraction_name] to a list of f1 scores
        f1_result = np.array(list(f1_pred.values()))
        return {"mean":f1_result.mean(), "std":f1_result.std(),"result":f1_pred}