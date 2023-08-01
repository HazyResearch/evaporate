# keys="PLACEHOLDER" # INSERT YOUR API KEY(S) HERE
keys=$(cat ~/data/openai_api_key.txt)
echo $keys

# cd ~/evaporate/evaporate
# sh ~/evaporate/install_deps.sh
# conda activate evaporate
conda activate maf

# -- Run data extraction
# num_attr_to_cascade --> number of attributes to extract functions for.
# do_end_to_end --> if set means learns schema/OpenIE.
# num_top_k_scripts --> number of functions to extract for each attribute.
# use_dynamic_backoff --> if set (store True) means use generated function to extract.

python ~/evaporate/evaporate/run_profiler.py \
    --data_lake small_debug_lin_alg_textbook \
    --do_end_to_end False \
    --num_attr_to_cascade 15 \
    --num_top_k_scripts 5 \
    --train_size 10 \
    --combiner_mode ws \
    --use_dynamic_backoff True \
    --KEYS ${keys}

# python ~/evaporate/evaporate/run_profiler.py \
#     --data_lake small_debug_lin_alg_textbook \
#     --num_attr_to_cascade 15 \
#     --num_top_k_scripts 5 \
#     --train_size 10 \
#     --combiner_mode mv \
#     --KEYS ${keys}
