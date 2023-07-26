# keys="PLACEHOLDER" # INSERT YOUR API KEY(S) HERE
keys=$(cat ~/data/openai_api_key.txt)
echo $keys

# cd ~/evaporate/evaporate
# sh ~/evaporate/install_deps.sh
# conda activate evaporate
conda activate maf

# evaporate code closed ie (Closed Information Extraction) [predefined schema]
# python ~/evaporate/evaporate/run_profiler.py \
#     --data_lake fda_510ks \
#     --do_end_to_end False \
#     --num_attr_to_cascade 50 \
#     --num_top_k_scripts 10 \
#     --train_size 10 \
#     --combiner_mode ws \
#     --use_dynamic_backoff True \
#     --KEYS ${keys}

# evaporate code open ie (Open Information Extraction) [learns schema]
python ~/evaporate/evaporate/run_profiler.py \
    --data_lake fda_510ks \
    --do_end_to_end True \
    --num_attr_to_cascade 2 \
    --num_top_k_scripts 3 \
    --train_size 1 \
    --combiner_mode mv \
    --use_dynamic_backoff True \
    --KEYS ${keys}

python ~/evaporate/evaporate/run_profiler.py \
    --data_lake small_synth_linalg_textbook \
    --do_end_to_end True \
    --num_attr_to_cascade 2 \
    --num_top_k_scripts 3 \
    --train_size 1 \
    --combiner_mode mv \
    --use_dynamic_backoff True \
    --KEYS ${keys}

# --- Original code

# keys="PLACEHOLDER" # INSERT YOUR API KEY(S) HERE

# # evaporate code closed ie
# python run_profiler.py \
#     --data_lake fda_510ks \
#     --do_end_to_end False \
#     --num_attr_to_cascade 50 \
#     --num_top_k_scripts 10 \
#     --train_size 10 \
#     --combiner_mode ws \
#     --use_dynamic_backoff True \
#     --KEYS ${keys}

# # evaporate code open ie
# python run_profiler.py \
#     --data_lake fda_510ks \
#     --do_end_to_end True \
#     --num_attr_to_cascade 1 \
#     --num_top_k_scripts 10 \
#     --train_size 10 \
#     --combiner_mode ws \
#     --use_dynamic_backoff True \
#     --KEYS ${keys}