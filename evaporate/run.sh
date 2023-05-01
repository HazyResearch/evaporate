keys="PLACEHOLDER" # INSERT YOUR API KEY(S) HERE

# evaporate code closed ie
python run_profiler.py \
    --data_lake fda_510ks \
    --num_attr_to_cascade 50 \
    --num_top_k_scripts 10 \
    --train_size 10 \
    --combiner_mode ws \
    --use_dynamic_backoff \
    --KEYS ${keys}

# evaporate code open ie
python run_profiler.py \
    --data_lake fda_510ks \
    --do_end_to_end \
    --num_attr_to_cascade 1 \
    --num_top_k_scripts 10 \
    --train_size 10 \
    --combiner_mode ws \
    --use_dynamic_backoff \
    --KEYS ${keys}
